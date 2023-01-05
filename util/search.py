"""Plan search: dynamic programming, beam search, etc."""
import collections
import copy
import math
import random
import time

import numpy as np
import torch
from sklearn.cluster import KMeans

import encoding
import util.plans_lib as plans_lib
from encoding import TreeConvFeaturize
from util import costing
from util import hyperparams
from util import postgres, envs

# Nest Loop lhs/rhs whitelist. Empirically determined from Postgres plans.  A
# more general solution is to delve into PG source code.
_NL_WHITE_LIST = set([
    ('Nested Loop', 'Index Scan'),
    ('Nested Loop', 'Seq Scan'),  # NOTE: SeqScan needs to be "small".
    ('Nested Loop', 'Index Only Scan'),
    ('Hash Join', 'Index Scan'),
    ('Hash Join', 'Index Only Scan'),
    ('Merge Join', 'Index Scan'),
    ('Seq Scan', 'Index Scan'),
    ('Seq Scan', 'Nested Loop'),  # NOTE: SeqScan needs to be "small".
    ('Seq Scan', 'Index Only Scan'),
    ('Index Scan', 'Index Scan'),
    ('Index Scan', 'Seq Scan'),  # NOTE: SeqScan needs to be "small".
])

trainBuffer = []
DEVICE = 'cuda:2' if torch.cuda.is_available() else 'cpu'


def collectSubplans(root, subplans_fin, workload, exp):
    allPlans = [root]
    # print('collect')
    currentChild = root
    temlevel = currentChild.info.get("currentLevel")
    if (not temlevel == None) and temlevel > 1:

        temsql = currentChild.to_sql(currentChild.info["join_conds"],
                                     with_select_exprs=True)

        temhint = currentChild.hint_str()
        found = False
        for i in subplans_fin[temlevel]:
            if (i[1] == temsql and i[2] == temhint):
                found = True
                break
        if not found:
            tem = []
            tem.append(math.log(currentChild.info["cost"]))
            tem.append(temsql)
            tem.append(temhint)
            nodelatency = currentChild.info.get("latency")

            if nodelatency == None:
                for i in exp[currentChild.info.get("currentLevel")]:
                    if (i[1] == temsql and i[2] == temhint):
                        nodelatency = i[3]

                        break

            if nodelatency == None:
                nodelatency = postgres.GetLatencyFromPg(temsql, temhint, verbose=False, check_hint_used=False,
                                                        timeout=10000, dropbuffer=False)
                tem.append(nodelatency)
                tem.append(encoding.getencoding_Balsa(temsql, temhint, workload))
                subplans_fin[temlevel].append(copy.deepcopy(tem))
                exp[temlevel].append(copy.deepcopy(tem))
            else:
                tem.append(nodelatency)
                tem.append(encoding.getencoding_Balsa(temsql, temhint, workload))
                subplans_fin[temlevel].append(copy.deepcopy(tem))
    while (allPlans):
        currentNode = allPlans.pop()
        allPlans.extend(currentNode.children)
        for currentChild in currentNode.children:
            temlevel = currentChild.info.get("currentLevel")
            # print(temlevel)
            if (not temlevel == None) and temlevel > 1:
                #  print(currentChild)
                temsql = currentChild.to_sql(currentChild.info["join_conds"],
                                             with_select_exprs=True)
                #    print(temsql)
                temhint = currentChild.hint_str()
                found = False
                for i in subplans_fin[temlevel]:
                    if (i[1] == temsql and i[2] == temhint):
                        found = True
                        break
                if not found:
                    tem = []
                    tem.append(math.log(currentChild.info["cost"]))
                    tem.append(temsql)
                    tem.append(temhint)
                    nodelatency = currentChild.info.get("latency")

                    if nodelatency == None:
                        for i in exp[currentChild.info.get("currentLevel")]:
                            if (i[1] == temsql and i[2] == temhint):
                                nodelatency = i[3]
                                break

                    if nodelatency == None:
                        nodelatency = postgres.GetLatencyFromPg(temsql, temhint, verbose=False, check_hint_used=False,
                                                                timeout=30000, dropbuffer=False)
                        tem.append(nodelatency)
                        tem.append(encoding.getencoding_Balsa(temsql, temhint, workload))
                        subplans_fin[temlevel].append(copy.deepcopy(tem))
                        exp[temlevel].append(copy.deepcopy(tem))
                    else:
                        tem.append(nodelatency)
                        tem.append(encoding.getencoding_Balsa(temsql, temhint, workload))
                        subplans_fin[temlevel].append(copy.deepcopy(tem))


def slackTimeout(exp):
    num = 0
    for i in exp:
        if not (i[3] == 90000):
            num = num + 1
        if num > 2:
            return False
    return True


def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic


def IsJoinCombinationOk(join_type,
                        left,
                        right,
                        avoid_eq_filters=False,
                        engine='postgres',
                        use_plan_restrictions=True):
    """Checks whether hinting a join would be accepted by Postgres.

    Due to Postgres' internal implementations, pg_hint_plan would pass through
    all hints but Postgres may still silently reject and rewrite them.  Here we
    guard against this using empirical checks (which may be more conservative
    than the exact criteria that PG uses).

    Args:
      join_type: str, the op of the planned join.
      left: Node, left child of the planned join with its scan type assigned.
      right: Node, rihgt child of the planned join with its scan type assigned.
      avoid_eq_filters: bool, whether to avoid for certain equality filter scans
          required for Ext-JOB planning to work.
    Returns:
      bool, whether the planned join is going to be respected.
    """
    if not use_plan_restrictions:
        return True

    if join_type == 'Nested Loop':
        return IsNestLoopOk(left, right)

    # Avoid problematic info_type + filter predicate combination for Ext-JOB
    # hints.
    if avoid_eq_filters:
        if right.table_name == 'info_type' and _IsFilterScan(right):
            return False
        if left.table_name == 'info_type' and _IsFilterScan(left):
            return False

    if join_type == 'Hash Join':
        return IsHashJoinOk(left, right)

    return True


def _IsFilterScan(n):
    return n.HasEqualityFilters() and n.IsScan()


# @profile
def _IsSmallScan(n):
    # Special case: check this leaf is "small".  Here we use a hack and treat
    # the small dim tables *_type as small.  A general solution is to delve
    # into PG code and figure out how many rows/how much cost are deemed as
    # small (likely need to compare with work_mem, etc).
    return n.table_name.endswith('_type')


# @profile
def IsNestLoopOk(left, right):
    """Nested Loop hint is only respected by PG in some scenarios."""
    l_op = left.node_type
    r_op = right.node_type
    if (l_op, r_op) not in _NL_WHITE_LIST:
        return False
    # Special cases: check the Seq Scan side is "small".
    if (l_op, r_op) == ('Seq Scan', 'Nested Loop'):
        return _IsSmallScan(left)
    if (l_op, r_op) in [
        ('Index Scan', 'Seq Scan'),
        ('Nested Loop', 'Seq Scan'),
    ]:
        return _IsSmallScan(right)
    # All other cases OK.
    return True


# @profile
def IsHashJoinOk(left, right):
    """Hash Join hint is only respected by PG in some scenarios."""
    l_op = left.node_type
    r_op = right.node_type

    if (l_op, r_op) == ('Index Scan', 'Hash Join'):
        # Allows iff (1) LHS is small and (2) there are select exprs.  This
        # rule is empirically determined.
        l_exprs = left.GetSelectExprs()
        r_exprs = right.GetSelectExprs()
        return _IsSmallScan(left) and (len(l_exprs) or len(r_exprs))
    # All other cases OK.
    return True


def EnumerateScanOps(node, scan_ops):
    if not node.IsScan():
        yield node
    else:
        for scan_op in scan_ops:
            if scan_op == 'Index Only Scan':
                continue
            yield node.ToScanOp(scan_op)


# @profile
def EnumerateJoinWithOps(left,
                         right,
                         join_ops,
                         scan_ops,
                         avoid_eq_filters=False,
                         engine='postgres',
                         use_plan_restrictions=True):
    """Yields all valid JoinOp(ScanOp(left), ScanOp(right))."""
    random.shuffle(join_ops)
    random.shuffle(scan_ops)
    for join_op in join_ops:
        for l in EnumerateScanOps(left, scan_ops):
            for r in EnumerateScanOps(right, scan_ops):
                if not IsJoinCombinationOk(join_op, l, r, avoid_eq_filters,
                                           engine, use_plan_restrictions):
                    continue
                join = plans_lib.Node(join_op)
                join.children = [l, r]
                yield join


class DynamicProgramming(object):
    """Bottom-up dynamic programming plan search."""

    @classmethod
    def Params(cls):
        p = hyperparams.InstantiableParams(cls)
        # PostgresCost ,MinCardCost
        p.Define('cost_model', costing.PostgresCost.Params(),
                 'Params of the cost model to use.')
        p.Define('search_space', 'bushy',
                 'Options: bushy, dbmsx, bushy_norestrict.')

        # Physical planning.
        p.Define('plan_physical_ops', True, 'Do we plan physical joins/scans?')

        # On enumeration hook.
        p.Define(
            'collect_data_include_suboptimal', True, 'Call on enumeration'
                                                     ' hooks on suboptimal plans for each k-relation?')
        return p

    def __init__(self, params):
        self.params = params.Copy()
        p = self.params
        self.cost_model = p.cost_model.cls(p.cost_model)
        self.on_enumerated_hooks = []

        assert p.search_space in ('bushy', 'dbmsx',
                                  'bushy_norestrict'), 'Not implemented.'

        self.join_ops = ['Join']
        self.scan_ops = ['Scan']
        self.use_plan_restrictions = (p.search_space != 'bushy_norestrict')

    def SetPhysicalOps(self, join_ops, scan_ops):
        """Must be called once if p.plan_physical_ops is true."""
        p = self.params
        assert p.plan_physical_ops
        self.join_ops = copy.deepcopy(join_ops)
        self.scan_ops = copy.deepcopy(scan_ops)

    def PushOnEnumeratedHook(self, func):
        """Executes func(Node, cost) on each enumerated and costed subplan.

        This can be useful for, e.g., collecting value function training data.

        The subplan does not have to be an optimal one.
        """
        self.on_enumerated_hooks.append(func)

    def PopOnEnumeratedHook(self):
        self.on_enumerated_hooks.pop()

    def Run(self, query_node, query_str, model, exp):
        """Executes DP planning for a given query node/string.

        Returns:
           A tuple of:
             best_node: balsa.Node;
             dp_tables: dict of size N (number of table in query), where
               dp_table[i] is a dict mapping a sorted string of a relation set
               (e.g., 'mi,t'), to (cost, the best plan that joins this set).
        """

        p = self.params
        join_graph, all_join_conds = query_node.GetOrParseSql()
        assert len(join_graph.edges) == len(all_join_conds)
        # Base tables to join.
        query_leaves = query_node.CopyLeaves()
        dp_tables = collections.defaultdict(dict)  # level -> dp_table

        # Fill in level 1.
        for leaf_node in query_leaves:
            dp_tables[1][leaf_node.table_alias] = (0, leaf_node)

        fns = {
            'bushy': self._dp_getTrainData,
            'dbmsx': self._dp_dbmsx_search_space,
            'bushy_norestrict': self._dp_bushy_search_space,
        }
        fn = fns[p.search_space]
        return fn(query_node, join_graph, all_join_conds, query_leaves,
                  dp_tables, model, exp)

    def _dp_bushy_search_space(self, original_node, join_graph, all_join_conds,
                               query_leaves, dp_tables):
        p = self.params
        #  b=0
        #   alltime = 0
        # num_rels = len(join_graph.nodes)
        num_rels = len(query_leaves)
        for level in range(2, num_rels + 1):
            dp_table = dp_tables[level]
            for level_i in range(1, level):
                level_j = level - level_i
                dp_table_i = dp_tables[level_i]
                dp_table_j = dp_tables[level_j]
                #                print('dp items'+dp_table_i.items)
                for l_ids, l_tup in dp_table_i.items():
                    for r_ids, r_tup in dp_table_j.items():
                        l = l_tup[1]
                        r = r_tup[1]
                        if not plans_lib.ExistsJoinEdgeInGraph(
                                l, r, join_graph):
                            # No join clause linking two sides.  Skip.
                            continue
                        l_ids_splits = l_ids.split(',')
                        r_ids_splits = r_ids.split(',')
                        if len(np.intersect1d(l_ids_splits, r_ids_splits)) > 0:
                            # A relation exists in both sides.  Skip.
                            continue
                        join_ids = ','.join(sorted(l_ids_splits + r_ids_splits))

                        # Otherwise, form a new join.
                        for join in EnumerateJoinWithOps(
                                l,
                                r,
                                self.join_ops,
                                self.scan_ops,
                                use_plan_restrictions=self.use_plan_restrictions
                        ):

                            join_conds = join.KeepRelevantJoins(all_join_conds)
                            #  b=b+1
                            #  print(b)
                            #  st=time.time()
                            cost, sql, hint = self.cost_model(join, join_conds)
                            # costbais = test.getCostbais(sql, hint)
                            # print(costbais+1)
                            origincost = cost

                            # cost = cost * costbais
                            #  en =time.time()
                            #  tt=en-st
                            #  print('time =' ,str(tt))
                            #  alltime =alltime +tt
                            if p.collect_data_include_suboptimal:
                                # Call registered hooks on the costed subplan.
                                for hook in self.on_enumerated_hooks:
                                    hook(join, cost)

                            # Record if better cost.
                            if join_ids not in dp_table or dp_table[join_ids][
                                0] > cost:
                                dp_table[join_ids] = (cost, join)
        # print(alltime/b)
        return list(dp_tables[num_rels].values())[0][1], dp_tables

    def _dp_getTrainData(self, original_node, join_graph, all_join_conds,
                         query_leaves, dp_tables, model, exp):

        ALL = []
        num = 0

        # model = torch.load('testmodel1r.pth')

        # balsa model
        # model = torch.load('/home/ht/PycharmProjects/pythonProject3/Balsa_model10a_train6.pth')
        # print(model)

        #  exp =pickle.load(open("/home/ht/PycharmProjects/pythonProject3/exp.pkl", "rb"))

        workload = envs.JoinOrderBenchmark(envs.JoinOrderBenchmark.Params())
        workload.workload_info.table_num_rows = postgres.GetAllTableNumRows(
            workload.workload_info.rel_names)
        num_rels = len(query_leaves)
        trainBuffer.append([] * (num_rels + 1))
        for i in range(0, num_rels):
            trainBuffer.append([])
        for level in range(2, num_rels + 1):
            dp_table = dp_tables[level]
            # print(dp_table)
            for level_i in range(1, level):
                level_j = level - level_i
                dp_table_i = dp_tables[level_i]
                dp_table_j = dp_tables[level_j]

                for l_ids, l_tup in dp_table_i.items():

                    for r_ids, r_tup in dp_table_j.items():
                        l = l_tup[1]
                        r = r_tup[1]
                        if not plans_lib.ExistsJoinEdgeInGraph(
                                l, r, join_graph):
                            # No join clause linking two sides.  Skip.
                            continue
                        l_ids_splits = l_ids.split(',')
                        r_ids_splits = r_ids.split(',')
                        if len(np.intersect1d(l_ids_splits, r_ids_splits)) > 0:
                            # A relation exists in both sides.  Skip.
                            continue
                        join_ids = ','.join(sorted(l_ids_splits + r_ids_splits))

                        # Otherwise, form a new join.
                        for join in EnumerateJoinWithOps(
                                l,
                                r,
                                self.join_ops,
                                self.scan_ops,
                                use_plan_restrictions=self.use_plan_restrictions
                        ):

                            join_conds = join.KeepRelevantJoins(all_join_conds)

                            cost, sql, hint = self.cost_model(join, join_conds)
                            origincost = math.log(cost)
                            # use model to dp

                            data = encoding.getencoding_Balsa(sql, hint, workload)
                            costbais = torch.tanh(model(data[0], data[1], data[2])) + 1
                            # # print(costbais)
                            # # print(costbais)
                            cost = math.log(cost) * costbais

                            if join_ids not in dp_table or dp_table[join_ids][
                                0] > cost:

                                # print(num)
                                tem = []
                                tem.append(origincost)
                                tem.append(sql)
                                tem.append(hint)
                                # # # collect train data (latency)
                                usebuffer = False
                                for j in exp[level]:
                                    if (j[2] == hint and j[1] == sql):
                                        # latency=j[3]
                                        usebuffer = True
                                        # print('use buffer')
                                        # tem.append(latency)
                                        break
                                if (usebuffer == False):
                                    num = num + 1
                                    # latency=postgres.GetLatencyFromPg(sql, hint, verbose=False, check_hint_used=False)
                                    # tem.append(latency)
                                    exp[level].append(tem)
                                # print('latency = ',end='')
                                #  print(latency)
                                #   trainBuffer[level].append(tem)

                                dp_table[join_ids] = (cost, join)
        # save train data ?
        #  a_file = open("data10_0.pkl", "wb")
        # b_file =open('exp.pkl','wb')
        # pickle.dump(trainBuffer, a_file)
        # pickle.dump(exp,b_file)
        # print(trainBuffer)
        #  print(num)
        # return list(dp_tables[num_rels].values())[0][1], dp_tables
        return num

    def _autoGetData(self, join_graph, all_join_conds, query_leaves, dp_tables, workload, exp, FirstTrain, model,
                     timeout, dropbuffer):
        num_rels = len(query_leaves)
        trainBuffer.append([] * (num_rels + 1))
        num = 0
        for i in range(0, num_rels):
            trainBuffer.append([])
        for level in range(2, num_rels + 1):
            dp_table = dp_tables[level]
            for level_i in range(1, level):
                level_j = level - level_i
                dp_table_i = dp_tables[level_i]
                dp_table_j = dp_tables[level_j]
                for l_ids, l_tup in dp_table_i.items():
                    for r_ids, r_tup in dp_table_j.items():
                        l = l_tup[1]
                        r = r_tup[1]
                        if not plans_lib.ExistsJoinEdgeInGraph(
                                l, r, join_graph):
                            # No join clause linking two sides.  Skip.
                            continue
                        l_ids_splits = l_ids.split(',')
                        r_ids_splits = r_ids.split(',')
                        if len(np.intersect1d(l_ids_splits, r_ids_splits)) > 0:
                            # A relation exists in both sides.  Skip.
                            continue
                        join_ids = ','.join(sorted(l_ids_splits + r_ids_splits))
                        # Otherwise, form a new join.
                        for join in EnumerateJoinWithOps(
                                l,
                                r,
                                self.join_ops,
                                self.scan_ops,
                                use_plan_restrictions=self.use_plan_restrictions
                        ):

                            join_conds = join.KeepRelevantJoins(all_join_conds)
                            cost, sql, hint = self.cost_model(join, join_conds)
                            logcost = math.log(cost)

                            data = encoding.getencoding_Balsa(sql, hint, workload)
                            if not FirstTrain:
                                costbais = torch.tanh(model(data[0], data[1], data[2])) + 1
                                cost = math.log(cost) * costbais
                            if join_ids not in dp_table or dp_table[join_ids][
                                0] > cost:

                                tem = []
                                tem.append(logcost)
                                tem.append(sql)
                                tem.append(hint)
                                # # # collect train data (latency)
                                usebuffer = False
                                for j in exp[level]:
                                    if (j[2] == hint and j[1] == sql):
                                        usebuffer = True
                                        latency = j[3]
                                        # print('use buffer')
                                        break
                                if (usebuffer == False):
                                    num = num + 1
                                    latency = postgres.GetLatencyFromPg(sql, hint, verbose=False, check_hint_used=False,
                                                                        timeout=timeout, dropbuffer=dropbuffer)
                                    tem.append(latency)
                                    tem.append(data)
                                    exp[level].append(tem)
                                    trainBuffer[level].append(tem)
                                dp_table[join_ids] = (cost, join)
        if timeout > latency:
            timeout = latency
        # print('dp now timeout = '+str(timeout))
        bestplanhint = list(dp_tables[num_rels].values())[0][1].hint_str()
        return trainBuffer, bestplanhint, num, timeout

    def _autoGetDataForbatch(self, join_graph, all_join_conds, query_leaves, dp_tables, workload, exp, FirstTrain,
                             model, timeout, dropbuffer):
        num_rels = len(query_leaves)
        num = 0
        for i in range(0, num_rels + 1):
            trainBuffer.append([])
        for level in range(2, num_rels + 1):
            dp_table = dp_tables[level]
            for level_i in range(1, level):
                level_j = level - level_i
                dp_table_i = dp_tables[level_i]
                dp_table_j = dp_tables[level_j]

                for l_ids, l_tup in dp_table_i.items():
                    for r_ids, r_tup in dp_table_j.items():
                        l = l_tup[1]
                        r = r_tup[1]
                        if not plans_lib.ExistsJoinEdgeInGraph(
                                l, r, join_graph):
                            # No join clause linking two sides.  Skip.
                            continue
                        l_ids_splits = l_ids.split(',')
                        r_ids_splits = r_ids.split(',')
                        if len(np.intersect1d(l_ids_splits, r_ids_splits)) > 0:
                            # A relation exists in both sides.  Skip.
                            continue
                        join_ids = ','.join(sorted(l_ids_splits + r_ids_splits))
                        # Otherwise, form a new join.

                        for join in EnumerateJoinWithOps(
                                l,
                                r,
                                self.join_ops,
                                self.scan_ops,
                                use_plan_restrictions=self.use_plan_restrictions
                        ):

                            join_conds = join.KeepRelevantJoins(all_join_conds)
                            cost, sql, hint = self.cost_model(join, join_conds)
                            logcost = math.log(cost)

                            data = encoding.getencoding_Balsa(sql, hint, workload)
                            assert len(data) == 2
                            if not FirstTrain:
                                costbais = torch.tanh(model(data[0], data[1], data[2])) + 1
                                cost = math.log(cost) * costbais
                            if join_ids not in dp_table or dp_table[join_ids][
                                0] > cost:

                                tem = []
                                tem.append(logcost)
                                tem.append(sql)
                                tem.append(hint)
                                # # # collect train data (latency)
                                usebuffer = False
                                for j in exp[level]:
                                    if (j[2] == hint and j[1] == sql):
                                        usebuffer = True
                                        latency = j[3]
                                        # print('use buffer')
                                        break
                                if (usebuffer == False):
                                    num = num + 1
                                    latency = postgres.GetLatencyFromPg(sql, hint, verbose=False, check_hint_used=False,
                                                                        timeout=timeout, dropbuffer=dropbuffer)
                                    tem.append(latency)
                                    tem.append(data)
                                    exp[level].append(tem)
                                    trainBuffer[level].append(tem)
                                dp_table[join_ids] = (cost, join)

        if timeout > latency:
            timeout = latency
        # print('dp now timeout = ' + str(timeout))
        print('new experience num =' + str(num))
        bestplanhint = list(dp_tables[num_rels].values())[0][1].hint_str()
        return trainBuffer, bestplanhint, num, timeout

    def _batch_DP(self, join_graph, all_join_conds, query_leaves, dp_tables, workload, exp, FirstTrain,
                  model, timeout, dropbuffer, nodeFeaturizer):
        num_rels = len(query_leaves)
        num = 0
        latency = 0
        for i in range(0, num_rels + 1):
            trainBuffer.append([])
        for level in range(2, num_rels + 1):
            dp_table = dp_tables[level]
            # print(level)
            for level_i in range(1, level):
                level_j = level - level_i
                dp_table_i = dp_tables[level_i]
                dp_table_j = dp_tables[level_j]

                for l_ids, l_tup in dp_table_i.items():
                    for r_ids, r_tup in dp_table_j.items():
                        l = l_tup[1]
                        r = r_tup[1]
                        if not plans_lib.ExistsJoinEdgeInGraph(
                                l, r, join_graph):
                            # No join clause linking two sides.  Skip.
                            continue
                        l_ids_splits = l_ids.split(',')
                        r_ids_splits = r_ids.split(',')
                        if len(np.intersect1d(l_ids_splits, r_ids_splits)) > 0:
                            # A relation exists in both sides.  Skip.
                            continue
                        join_ids = ','.join(sorted(l_ids_splits + r_ids_splits))
                        # Otherwise, form a new join.
                        dp_costs = []
                        dp_query_encodings = []
                        dp_nodes = []
                        dp_hints_sqls = []
                        dp_join = []
                        for join in EnumerateJoinWithOps(
                                l,
                                r,
                                self.join_ops,
                                self.scan_ops,
                                use_plan_restrictions=self.use_plan_restrictions
                        ):
                            join_conds = join.KeepRelevantJoins(all_join_conds)
                            cost, sql, hint = self.cost_model(join, join_conds)
                            logcost = math.log(cost)
                            data = encoding.getencoding_Balsa(sql, hint, workload)
                            dp_join.append(join)
                            dp_costs.append(logcost)
                            dp_query_encodings.append(data[0])
                            dp_nodes.append(data[1])
                            dp_hints_sqls.append([hint, sql])
                            assert len(data) == 2
                        if not FirstTrain:
                            query_feats = torch.cat(dp_query_encodings, dim=0)
                            trees, indexes = TreeConvFeaturize(nodeFeaturizer, dp_nodes)
                            if torch.cuda.is_available():
                                trees = trees.cuda()
                                indexes = indexes.cuda()
                                torch_dpcosts = torch.tensor(dp_costs)
                                torch_dpcosts = torch_dpcosts.cuda()
                            costbais = torch.tanh(model(query_feats, trees, indexes)).add(1).squeeze(1)
                            costlist = torch.mul(costbais, torch_dpcosts).tolist()
                        else:
                            costlist = dp_costs
                        # print(costlist)
                        for i in range(0, len(costlist)):
                            if join_ids not in dp_table or dp_table[join_ids][
                                0] > costlist[i]:
                                tem = []
                                tem.append(dp_costs[i])
                                tem.append(dp_hints_sqls[i][1])
                                tem.append(dp_hints_sqls[i][0])
                                # # # collect train data (latency)
                                usebuffer = False
                                for j in exp[level]:
                                    if (j[2] == dp_hints_sqls[i][0] and j[1] == dp_hints_sqls[i][1]):
                                        usebuffer = True
                                        latency = j[3]
                                        # print('use buffer')
                                        break
                                if (usebuffer == False):
                                    num = num + 1
                                    latency = postgres.GetLatencyFromPg(dp_hints_sqls[i][1], dp_hints_sqls[i][0],
                                                                        verbose=False, check_hint_used=False,
                                                                        timeout=timeout, dropbuffer=dropbuffer)
                                    # latency = random.random()
                                    tem.append(latency)
                                    tem.append([dp_query_encodings[i], dp_nodes[i]])
                                    exp[level].append(tem)
                                    trainBuffer[level].append(tem)
                                dp_table[join_ids] = (costlist[i], dp_join[i])

        if timeout > latency:
            timeout = latency
        # print('dp now timeout = ' + str(timeout))
        print('new experience num =' + str(num))
        bestplanhint = list(dp_tables[num_rels].values())[0][1].hint_str()
        return trainBuffer, bestplanhint, num, timeout

    def _batch_DP_level(self, join_graph, all_join_conds, query_leaves, dp_tables, workload, exp, FirstTrain,
                        model, timeout, dropbuffer, nodeFeaturizer, greedy):
        num_rels = len(query_leaves)
        num = 0
        latency = 0
        for i in range(0, num_rels + 1):
            trainBuffer.append([])
        for level in range(2, num_rels + 1):
            dp_table = dp_tables[level]
            # print(level)
            for level_i in range(1, level):
                level_j = level - level_i
                dp_table_i = dp_tables[level_i]
                dp_table_j = dp_tables[level_j]

                for l_ids, l_tup in dp_table_i.items():
                    for r_ids, r_tup in dp_table_j.items():
                        l = l_tup[1]
                        r = r_tup[1]
                        if not plans_lib.ExistsJoinEdgeInGraph(
                                l, r, join_graph):
                            # No join clause linking two sides.  Skip.
                            continue
                        l_ids_splits = l_ids.split(',')
                        r_ids_splits = r_ids.split(',')
                        if len(np.intersect1d(l_ids_splits, r_ids_splits)) > 0:
                            # A relation exists in both sides.  Skip.
                            continue
                        join_ids = ','.join(sorted(l_ids_splits + r_ids_splits))
                        # Otherwise, form a new join.
                        dp_costs = []
                        dp_query_encodings = []
                        dp_nodes = []
                        dp_hints_sqls = []
                        dp_join = []
                        for join in EnumerateJoinWithOps(
                                l,
                                r,
                                self.join_ops,
                                self.scan_ops,
                                use_plan_restrictions=self.use_plan_restrictions
                        ):
                            join_conds = join.KeepRelevantJoins(all_join_conds)
                            cost, sql, hint = self.cost_model(join, join_conds)

                            logcost = math.log(cost)
                            data = encoding.getencoding_Balsa(sql, hint, workload)
                            dp_join.append(join)
                            dp_costs.append(logcost)
                            dp_query_encodings.append(data[0])
                            dp_nodes.append(data[1])
                            dp_hints_sqls.append([hint, sql])
                            assert len(data) == 2
                        if not FirstTrain:
                            query_feats = (torch.cat(dp_query_encodings, dim=0)).to(DEVICE)
                            trees, indexes = TreeConvFeaturize(nodeFeaturizer, dp_nodes)
                            if torch.cuda.is_available():
                                trees = trees.to(DEVICE)
                                indexes = indexes.to(DEVICE)
                                torch_dpcosts = (torch.tensor(dp_costs)).to(DEVICE)
                            costbais = torch.tanh(model[level](query_feats, trees, indexes).to(DEVICE)).add(1).squeeze(
                                1)
                            costlist = torch.mul(costbais, torch_dpcosts).tolist()
                        else:
                            costlist = dp_costs
                        # print(costlist)
                        for i in range(0, len(costlist)):
                            if (random.random() < greedy):
                                tem = []
                                tem.append(dp_costs[i])
                                tem.append(dp_hints_sqls[i][1])
                                tem.append(dp_hints_sqls[i][0])
                                # # # collect train data (latency)
                                usebuffer = False
                                for j in exp[level]:
                                    if (j[2] == dp_hints_sqls[i][0] and j[1] == dp_hints_sqls[i][1]):
                                        usebuffer = True
                                        glatency = j[3]
                                        # print('use buffer')
                                        break
                                if (usebuffer == False):
                                    num = num + 1
                                    glatency = postgres.GetLatencyFromPg(dp_hints_sqls[i][1], dp_hints_sqls[i][0],
                                                                         verbose=False, check_hint_used=False,
                                                                         timeout=timeout, dropbuffer=dropbuffer)
                                    tem.append(glatency)
                                    tem.append([dp_query_encodings[i], dp_nodes[i]])
                                    exp[level].append(tem)
                                    trainBuffer[level].append(tem)
                            if join_ids not in dp_table or dp_table[join_ids][
                                0] > costlist[i]:
                                tem = []
                                tem.append(dp_costs[i])
                                tem.append(dp_hints_sqls[i][1])
                                tem.append(dp_hints_sqls[i][0])
                                # # # collect train data (latency)
                                usebuffer = False
                                for j in exp[level]:
                                    if (j[2] == dp_hints_sqls[i][0] and j[1] == dp_hints_sqls[i][1]):
                                        usebuffer = True
                                        latency = j[3]
                                        # print('use buffer')
                                        break
                                if (usebuffer == False):
                                    num = num + 1
                                    latency = postgres.GetLatencyFromPg(dp_hints_sqls[i][1], dp_hints_sqls[i][0],
                                                                        verbose=False, check_hint_used=False,
                                                                        timeout=timeout, dropbuffer=dropbuffer)
                                    tem.append(latency)
                                    tem.append([dp_query_encodings[i], dp_nodes[i]])
                                    exp[level].append(tem)
                                    trainBuffer[level].append(tem)
                                dp_table[join_ids] = (costlist[i], dp_join[i])

        if timeout > latency:
            timeout = latency
        bestplanhint = list(dp_tables[num_rels].values())[0][1].hint_str()
        return trainBuffer, bestplanhint, num, timeout

    def _DP_TEST(self, join_graph, all_join_conds, query_leaves, dp_tables, workload
                 , model, nodeFeaturizer, justDP):
        num_rels = len(query_leaves)
        num = 0
        latency = 0
        for level in range(2, num_rels + 1):
            dp_table = dp_tables[level]
            # print(level)
            for level_i in range(1, level):
                level_j = level - level_i
                dp_table_i = dp_tables[level_i]
                dp_table_j = dp_tables[level_j]

                for l_ids, l_tup in dp_table_i.items():
                    for r_ids, r_tup in dp_table_j.items():
                        l = l_tup[1]
                        r = r_tup[1]
                        if not plans_lib.ExistsJoinEdgeInGraph(
                                l, r, join_graph):
                            # No join clause linking two sides.  Skip.
                            continue
                        l_ids_splits = l_ids.split(',')
                        r_ids_splits = r_ids.split(',')
                        if len(np.intersect1d(l_ids_splits, r_ids_splits)) > 0:
                            # A relation exists in both sides.  Skip.
                            continue
                        join_ids = ','.join(sorted(l_ids_splits + r_ids_splits))
                        # Otherwise, form a new join.
                        dp_costs = []
                        dp_query_encodings = []
                        dp_nodes = []
                        dp_hints_sqls = []
                        dp_join = []
                        for join in EnumerateJoinWithOps(
                                l,
                                r,
                                self.join_ops,
                                self.scan_ops,
                                use_plan_restrictions=self.use_plan_restrictions
                        ):
                            join_conds = join.KeepRelevantJoins(all_join_conds)
                            cost, sql, hint = self.cost_model(join, join_conds)

                            logcost = math.log(cost)
                            data = encoding.getencoding_Balsa(sql, hint, workload)
                            dp_join.append(join)
                            dp_costs.append(logcost)
                            dp_query_encodings.append(data[0])
                            dp_nodes.append(data[1])
                            dp_hints_sqls.append([hint, sql])
                            assert len(data) == 2
                        query_feats = (torch.cat(dp_query_encodings, dim=0)).to(DEVICE)
                        trees, indexes = TreeConvFeaturize(nodeFeaturizer, dp_nodes)
                        if justDP:
                            costlist = dp_costs
                        else:
                            if torch.cuda.is_available():
                                trees = trees.to(DEVICE)
                                indexes = indexes.to(DEVICE)
                                torch_dpcosts = (torch.tensor(dp_costs)).to(DEVICE)
                            costbais = torch.tanh(model[level](query_feats, trees, indexes).to(DEVICE)).add(1).squeeze(
                                1)
                            costlist = torch.mul(costbais, torch_dpcosts).tolist()
                        for i in range(0, len(costlist)):
                            if join_ids not in dp_table or dp_table[join_ids][
                                0] > costlist[i]:
                                dp_table[join_ids] = (costlist[i], dp_join[i])
        bestplanhint = list(dp_tables[num_rels].values())[0][1].hint_str()
        return bestplanhint

    def _batch_DP_level_left(self, join_graph, all_join_conds, query_leaves, dp_tables, workload, exp, FirstTrain,
                             model, timeout, dropbuffer, nodeFeaturizer, greedy=0):
        num_rels = len(query_leaves)
        num = 0
        latency = 0
        for i in range(0, num_rels + 1):
            trainBuffer.append([])
        for level in range(2, num_rels + 1):
            dp_table = dp_tables[level]
            print(level)
            # for level_i in range(1, level):
            # level_j = level - level_i
            dp_table_i = random_dic(dp_tables[level - 1])
            dp_table_j = random_dic(dp_tables[1])
            # limit = int(len(dp_table_i) * 0.2)
            #    nowplansnum = 0
            for l_ids, l_tup in dp_table_i.items():
                for r_ids, r_tup in dp_table_j.items():
                    l = l_tup[1]
                    r = r_tup[1]
                    if not plans_lib.ExistsJoinEdgeInGraph(
                            l, r, join_graph):
                        # No join clause linking two sides.  Skip.
                        continue
                    l_ids_splits = l_ids.split(',')
                    r_ids_splits = r_ids.split(',')
                    if len(np.intersect1d(l_ids_splits, r_ids_splits)) > 0:
                        # A relation exists in both sides.  Skip.
                        continue
                    join_ids = ','.join(sorted(l_ids_splits + r_ids_splits))
                    # Otherwise, form a new join.
                    dp_costs = []
                    dp_query_encodings = []
                    dp_nodes = []
                    dp_hints_sqls = []
                    dp_join = []
                    for join in EnumerateJoinWithOps(
                            l,
                            r,
                            self.join_ops,
                            self.scan_ops,
                            use_plan_restrictions=self.use_plan_restrictions
                    ):
                        join_conds = join.KeepRelevantJoins(all_join_conds)
                        cost, sql, hint = self.cost_model(join, join_conds)
                        logcost = math.log(cost)
                        data = encoding.getencoding_Balsa(sql, hint, workload)
                        dp_join.append(join)
                        dp_costs.append(logcost)
                        dp_query_encodings.append(data[0])
                        dp_nodes.append(data[1])
                        dp_hints_sqls.append([hint, sql])
                        assert len(data) == 2
                    if not FirstTrain:
                        query_feats = (torch.cat(dp_query_encodings, dim=0)).to(DEVICE)
                        trees, indexes = TreeConvFeaturize(nodeFeaturizer, dp_nodes)
                        if torch.cuda.is_available():
                            trees = trees.to(DEVICE)
                            indexes = indexes.to(DEVICE)
                            torch_dpcosts = (torch.tensor(dp_costs)).to(DEVICE)
                        costbais = torch.tanh(model[level](query_feats, trees, indexes).to(DEVICE)).add(1).squeeze(1)
                        costlist = torch.mul(costbais, torch_dpcosts).tolist()
                    else:
                        costlist = dp_costs
                    #                        for i in range(0,len(costlist)):
                    #                            if (random.random()<greedy):
                    #                                tem = []
                    #                                tem.append(dp_costs[i])
                    #                                tem.append(dp_hints_sqls[i][1])
                    #                                tem.append(dp_hints_sqls[i][0])
                    #                                # # # collect train data (latency)
                    #                                usebuffer = False
                    #                                for j in exp[level]:
                    #                                    if (j[2] == dp_hints_sqls[i][0] and j[1] == dp_hints_sqls[i][1]):
                    #                                        usebuffer = True
                    #                                        glatency = j[3]
                    #                                        # print('use buffer')
                    #                                        break
                    #                                if (usebuffer == False):
                    #                                    num = num + 1
                    #                                    glatency = postgres.GetLatencyFromPg(dp_hints_sqls[i][1], dp_hints_sqls[i][0], verbose=False, check_hint_used=False,
                    #                                                                       timeout=timeout, dropbuffer=dropbuffer)
                    #                                    tem.append(glatency)
                    #                                    tem.append([dp_query_encodings[i],dp_nodes[i]])
                    #                                    exp[level].append(tem)
                    #                                    trainBuffer[level].append(tem)
                    # print(costlist)
                    for i in range(0, len(costlist)):
                        if join_ids not in dp_table or dp_table[join_ids][
                            0] > costlist[i]:
                            tem = []
                            tem.append(dp_costs[i])
                            tem.append(dp_hints_sqls[i][1])
                            tem.append(dp_hints_sqls[i][0])
                            # # # collect train data (latency)
                            usebuffer = False
                            for j in exp[level]:
                                if (j[2] == dp_hints_sqls[i][0] and j[1] == dp_hints_sqls[i][1]):
                                    usebuffer = True
                                    latency = j[3]
                                    # print('use buffer')
                                    break
                            if (usebuffer == False):
                                num = num + 1
                                latency = postgres.GetLatencyFromPg(dp_hints_sqls[i][1], dp_hints_sqls[i][0],
                                                                    verbose=False, check_hint_used=False,
                                                                    timeout=timeout, dropbuffer=dropbuffer)
                                # latency = random.random()
                                tem.append(latency)
                                tem.append([dp_query_encodings[i], dp_nodes[i]])
                                exp[level].append(tem)
                                trainBuffer[level].append(tem)
                            dp_table[join_ids] = (costlist[i], dp_join[i])
            #  nowplansnum = nowplansnum + 1
            #   if nowplansnum > limit and level > 2:
            #   print('plan nums = ',num)
            #    break
            if level > 3 and level < 14:
                sortlist = dict(sorted(dp_table.items(), key=lambda x: x[1][0])[:math.ceil(len(dp_table) * 0.3)])
                dp_tables[level] = sortlist
        if timeout > latency:
            timeout = latency
        bestplanhint = list(dp_tables[num_rels].values())[0][1].hint_str()
        return trainBuffer, bestplanhint, num, timeout

    def UCB_left_KL_prune(self, join_graph, all_join_conds, query_leaves, dp_tables, workload, exp, FirstTrain,
                          model, timeout, dropbuffer, nodeFeaturizer, greedy, subplans_fin, finsql, levelList):
        num_rels = len(query_leaves)
        num = 0
        latency = 0
        for i in range(0, num_rels + 1):
            trainBuffer.append([])
        for level in range(2, num_rels + 1):
            dp_table = dp_tables[level]
            bayes_tep = []
            bayes_list = []
            dp_table_i = random_dic(dp_tables[level - 1])
            dp_table_j = random_dic(dp_tables[1])
            # 按层维护一个字典，字典key为等价类的表名，value为list，存不同的plan的信息
            # [  [ {},{} ....], [ {}...] ....]
            # levelList[level]={}
            for l_ids, l_tup in dp_table_i.items():
                for r_ids, r_tup in dp_table_j.items():
                    l = l_tup[1]
                    r = r_tup[1]
                    if not plans_lib.ExistsJoinEdgeInGraph(
                            l, r, join_graph):
                        # No join clause linking two sides.  Skip.
                        continue
                    l_ids_splits = l_ids.split(',')
                    r_ids_splits = r_ids.split(',')
                    if len(np.intersect1d(l_ids_splits, r_ids_splits)) > 0:
                        # A relation exists in both sides.  Skip.
                        continue
                    join_ids = ','.join(sorted(l_ids_splits + r_ids_splits))
                    # Otherwise, form a new join.
                    dp_costs = []
                    dp_query_encodings = []
                    dp_nodes = []
                    dp_hints_sqls = []
                    dp_join = []
                    for join in EnumerateJoinWithOps(
                            l,
                            r,
                            self.join_ops,
                            self.scan_ops,
                            use_plan_restrictions=self.use_plan_restrictions
                    ):
                        join.info["currentLevel"] = level

                        join_conds = join.KeepRelevantJoins(all_join_conds)
                        cost, sql, hint = self.cost_model(join, join_conds)
                        join.info["cost"] = cost
                        logcost = math.log(cost)
                        data = encoding.getencoding_Balsa(sql, hint, workload)
                        join.info["encoding"] = data[0]
                        join.info["node"] = data[1]
                        dp_join.append(join)
                        dp_costs.append(logcost)
                        dp_query_encodings.append(data[0])
                        dp_nodes.append(data[1])
                        dp_hints_sqls.append([hint, sql])
                        tem = []
                        tem.append(logcost)
                        tem.append(sql)
                        tem.append(hint)
                        tem.append(0)
                        tem.append([data[0], data[1]])

                        bayes_tep.append(tem)
                        assert len(data) == 2
                    levelList[level][join_ids] = [dp_costs, dp_query_encodings, dp_nodes]
                    if not FirstTrain:
                        query_feats = (torch.cat(dp_query_encodings, dim=0)).to(DEVICE)
                        trees, indexes = TreeConvFeaturize(nodeFeaturizer, dp_nodes)
                        if torch.cuda.is_available():
                            trees = trees.to(DEVICE)
                            indexes = indexes.to(DEVICE)
                            torch_dpcosts = (torch.tensor(dp_costs)).to(DEVICE)
                        costbais = []
                        for i in range(10):
                            costbais.append(torch.tanh(model[level](query_feats, trees, indexes).to(DEVICE)).add(1))
                        costbais = torch.cat(costbais, 1)
                        costbais_mean = torch.mean(costbais, dim=1)
                        cost_min, _ = torch.min(costbais, dim=1)
                        var = torch.sum(torch.pow(costbais - costbais_mean.unsqueeze(1), 2), dim=1)
                        ucb = var / var.max() - cost_min / cost_min.max()
                        # print('ucb ')
                        # print(ucb)
                        bayes_list.extend(ucb.tolist())
                        costlist = torch.mul(costbais_mean, torch_dpcosts).tolist()
                    else:
                        costlist = dp_costs
                    #                        for i in range(0,len(costlist)):
                    #                            if (random.random()<greedy):
                    #                                tem = []
                    #                                tem.append(dp_costs[i])
                    #                                tem.append(dp_hints_sqls[i][1])
                    #                                tem.append(dp_hints_sqls[i][0])
                    #                                # # # collect train data (latency)
                    #                                usebuffer = False
                    #                                for j in exp[level]:
                    #                                    if (j[2] == dp_hints_sqls[i][0] and j[1] == dp_hints_sqls[i][1]):
                    #                                        usebuffer = True
                    #                                        glatency = j[3]
                    #                                        # print('use buffer')
                    #                                        break
                    #                                if (usebuffer == False):
                    #                                    num = num + 1
                    #                                    glatency = postgres.GetLatencyFromPg(dp_hints_sqls[i][1], dp_hints_sqls[i][0], verbose=False, check_hint_used=False,
                    #                                                                       timeout=timeout, dropbuffer=dropbuffer)
                    #                                    tem.append(glatency)
                    #                                    tem.append([dp_query_encodings[i],dp_nodes[i]])
                    #                                    exp[level].append(tem)
                    #                                    trainBuffer[level].append(tem)
                    # print(costlist)
                    for i in range(0, len(costlist)):
                        if join_ids not in dp_table or dp_table[join_ids][
                            0] > costlist[i]:
                            if FirstTrain:
                                tem = []
                                tem.append(dp_costs[i])
                                tem.append(dp_hints_sqls[i][1])
                                tem.append(dp_hints_sqls[i][0])
                                # # # collect train data (latency)
                                usebuffer = False
                                for j in exp[level]:
                                    if (j[2] == dp_hints_sqls[i][0] and j[1] == dp_hints_sqls[i][1]):
                                        usebuffer = True
                                        latency = j[3]
                                        dp_nodes[i].info["latency"] = latency
                                        # print('use buffer')
                                        break
                                if (usebuffer == False):
                                    num = num + 1
                                    if ((level > num_rels - 1) and slackTimeout(exp[level])):
                                        latency = postgres.GetLatencyFromPg(dp_hints_sqls[i][1], dp_hints_sqls[i][0],
                                                                            verbose=False, check_hint_used=False,
                                                                            timeout=12000, dropbuffer=dropbuffer)
                                        print("slack")
                                        #                                   else:
                                        #                                          latency =postgres.GetLatencyFromPg(dp_hints_sqls[i][1], dp_hints_sqls[i][0], verbose=False, check_hint_used=False,timeout=timeout*2.0, dropbuffer=dropbuffer)
                                        print(latency, 'level', level)
                                    else:
                                        latency = postgres.GetLatencyFromPg(dp_hints_sqls[i][1], dp_hints_sqls[i][0],
                                                                            verbose=False, check_hint_used=False,
                                                                            timeout=timeout, dropbuffer=dropbuffer)
                                    # latency = random.random()
                                    dp_nodes[i].info["latency"] = latency
                                    tem.append(latency)
                                    tem.append([dp_query_encodings[i], dp_nodes[i]])
                                    exp[level].append(tem)
                                    trainBuffer[level].append(tem)
                                    if level == num_rels:
                                        allPlans = [dp_nodes[i]]
                                        while (allPlans):
                                            currentNode = allPlans.pop()
                                            allPlans.extend(currentNode.children)
                                            for currentChild in currentNode.children:
                                                temlevel = currentChild.info.get("currentLevel")
                                                if (not temlevel == None) and temlevel > 1:
                                                    #  print(currentChild)
                                                    temsql = currentChild.to_sql(currentChild.info["join_conds"],
                                                                                 with_select_exprs=True)
                                                    #    print(temsql)
                                                    temhint = currentChild.hint_str()
                                                    found = False
                                                    for i in subplans_fin:
                                                        if (i[1] == dp_hints_sqls[i][1], i[2] == dp_hints_sqls[i][0]):
                                                            found = True
                                                            break
                                                    if not found:
                                                        tem = []
                                                        tem.append(currentChild.info["cost"])
                                                        tem.append(temsql)
                                                        tem.append(temhint)
                                                        tem.append(currentChild.info["latency"])
                                                        tem.append([currentChild.info["encoding"], currentChild])
                                                        subplans_fin[temlevel].append(tem)
                            dp_table[join_ids] = (costlist[i], dp_join[i])
            #  nowplansnum = nowplansnum + 1
            #   if nowplansnum > limit and level > 2:
            #   print('plan nums = ',num)
            #    break
            # print('num = ',len(dp_table))
            # todo:根据方差收集数据

            if not FirstTrain:
                # print('bayes_list len :',len(bayes_list))
                # print('bayes tep num :',len(bayes_tep))
                bayes_list = torch.tensor(bayes_list)
                ucb_argsort = torch.argsort(bayes_list, descending=True)
                if level < 4:
                    p = 0.4
                else:
                    p = 0.2
                n = math.ceil(p * len(bayes_tep))
                for i in range(n):
                    bayes_plan = bayes_tep[ucb_argsort[i]]
                    usebuffer = False
                    for j in exp[level]:
                        if (j[2] == bayes_plan[2] and j[1] == bayes_plan[1]):
                            usebuffer = True
                            blatency = j[3]
                            break
                    if (usebuffer == False):
                        num = num + 1
                        if ((level > num_rels - 1) and slackTimeout(exp[level])):
                            blatency = postgres.GetLatencyFromPg(bayes_plan[1], bayes_plan[2], verbose=False,
                                                                 check_hint_used=False, timeout=12000,
                                                                 dropbuffer=dropbuffer)
                            print("slack")
                            #                                   else:
                            #                                          latency =postgres.GetLatencyFromPg(dp_hints_sqls[i][1], dp_hints_sqls[i][0], verbose=False, check_hint_used=False,timeout=timeout*2.0, dropbuffer=dropbuffer)
                            print(latency, 'level', level)
                        else:
                            blatency = postgres.GetLatencyFromPg(bayes_plan[1], bayes_plan[2], verbose=False,
                                                                 check_hint_used=False, timeout=timeout,
                                                                 dropbuffer=dropbuffer)
                    bayes_plan[3] = blatency
                    if usebuffer == False:
                        trainBuffer[level].append(copy.deepcopy(bayes_plan))
                        exp[level].append(copy.deepcopy(bayes_plan))

            if level > 4 and level < 15 and level < num_rels:
                temtable = copy.deepcopy(dp_table)
                if not FirstTrain:
                    temcost = []
                    temnodes = []
                    tem_query_encodings = []
                    for key, values in temtable.items():
                        temcost.append(values[0])
                        temnodes.append(values[1].info["node"])
                        tem_query_encodings.append(values[1].info["encoding"])
                    #  print("nodes num = ",len(temnodes))
                    temquery_feats = (torch.cat(tem_query_encodings, dim=0)).to(DEVICE)
                    temtrees, temindexes = TreeConvFeaturize(nodeFeaturizer, temnodes)
                    if torch.cuda.is_available():
                        temtrees = temtrees.to(DEVICE)
                        temindexes = temindexes.to(DEVICE)
                        torch_costs = (torch.tensor(temcost)).to(DEVICE)
                    temcostbais = []
                    for i in range(10):
                        temcostbais.append(
                            torch.tanh(model[-1](temquery_feats, temtrees, temindexes).to(DEVICE)).add(1))
                    temcostbais = torch.cat(temcostbais, 1)
                    temcostbais_mean = torch.mean(temcostbais, dim=1)
                    temcostlist = torch.mul(temcostbais_mean, torch_costs).tolist()
                    count = 0
                    for key in temtable:
                        temtable[key] = (temcostlist[count], temtable[key][1])
                        count = count + 1
                sortlist = dict(sorted(temtable.items(), key=lambda x: x[1][0])[math.ceil(len(dp_table) * 0.3):])
                for key in sortlist:
                    dp_table.pop(key)
                dp_tables[level] = dp_table
        # print('level:',level,'subplans num = ',len(dp_tables[level]))
        # print('level:',level,'exp num = ',len(exp[level]))
        bestplanhint = list(dp_tables[num_rels].values())[0][1].hint_str()
        nowlatency = postgres.GetLatencyFromPg(finsql, bestplanhint, verbose=False, check_hint_used=False,
                                               timeout=timeout, dropbuffer=dropbuffer)
        if timeout > nowlatency:
            timeout = nowlatency
        return trainBuffer, bestplanhint, num, timeout

    def _batch_DP_level_left_prune(self, join_graph, all_join_conds, query_leaves, dp_tables, workload, exp, FirstTrain,
                                   model, timeout, dropbuffer, nodeFeaturizer, greedy, subplans_fin):
        num_rels = len(query_leaves)
        num = 0
        latency = 0
        for i in range(0, num_rels + 1):
            trainBuffer.append([])
        for level in range(2, num_rels + 1):
            dp_table = dp_tables[level]
            # for level_i in range(1, level):
            # level_j = level - level_i
            dp_table_i = random_dic(dp_tables[level - 1])
            dp_table_j = random_dic(dp_tables[1])
            # limit = int(len(dp_table_i) * 0.2)
            #    nowplansnum = 0
            for l_ids, l_tup in dp_table_i.items():
                for r_ids, r_tup in dp_table_j.items():
                    l = l_tup[1]
                    r = r_tup[1]
                    if not plans_lib.ExistsJoinEdgeInGraph(
                            l, r, join_graph):
                        # No join clause linking two sides.  Skip.
                        continue
                    l_ids_splits = l_ids.split(',')
                    r_ids_splits = r_ids.split(',')
                    if len(np.intersect1d(l_ids_splits, r_ids_splits)) > 0:
                        # A relation exists in both sides.  Skip.
                        continue
                    join_ids = ','.join(sorted(l_ids_splits + r_ids_splits))
                    # Otherwise, form a new join.
                    dp_costs = []
                    dp_query_encodings = []
                    dp_nodes = []
                    dp_hints_sqls = []
                    dp_join = []
                    for join in EnumerateJoinWithOps(
                            l,
                            r,
                            self.join_ops,
                            self.scan_ops,
                            use_plan_restrictions=self.use_plan_restrictions
                    ):
                        join.info["currentLevel"] = level

                        join_conds = join.KeepRelevantJoins(all_join_conds)
                        join.info["join_conds"] = join_conds

                        cost, sql, hint = self.cost_model(join, join_conds)
                        join.info["cost"] = cost
                        logcost = math.log(cost)
                        data = encoding.getencoding_Balsa(sql, hint, workload)
                        join.info["encoding"] = data[0]
                        join.info["node"] = data[1]
                        dp_join.append(join)
                        dp_costs.append(logcost)
                        dp_query_encodings.append(data[0])
                        dp_nodes.append(data[1])
                        dp_hints_sqls.append([hint, sql])
                        assert len(data) == 2
                    if not FirstTrain:
                        query_feats = (torch.cat(dp_query_encodings, dim=0)).to(DEVICE)
                        trees, indexes = TreeConvFeaturize(nodeFeaturizer, dp_nodes)
                        if torch.cuda.is_available():
                            trees = trees.to(DEVICE)
                            indexes = indexes.to(DEVICE)
                            torch_dpcosts = (torch.tensor(dp_costs)).to(DEVICE)
                        costbais = torch.tanh(model[level](query_feats, trees, indexes).to(DEVICE)).add(1).squeeze(1)
                        costlist = torch.mul(costbais, torch_dpcosts).tolist()
                    else:
                        costlist = dp_costs
                    #                        for i in range(0,len(costlist)):
                    #                            if (random.random()<greedy):
                    #                                tem = []
                    #                                tem.append(dp_costs[i])
                    #                                tem.append(dp_hints_sqls[i][1])
                    #                                tem.append(dp_hints_sqls[i][0])
                    #                                # # # collect train data (latency)
                    #                                usebuffer = False
                    #                                for j in exp[level]:
                    #                                    if (j[2] == dp_hints_sqls[i][0] and j[1] == dp_hints_sqls[i][1]):
                    #                                        usebuffer = True
                    #                                        glatency = j[3]
                    #                                        # print('use buffer')
                    #                                        break
                    #                                if (usebuffer == False):
                    #                                    num = num + 1
                    #                                    glatency = postgres.GetLatencyFromPg(dp_hints_sqls[i][1], dp_hints_sqls[i][0], verbose=False, check_hint_used=False,
                    #                                                                       timeout=timeout, dropbuffer=dropbuffer)
                    #                                    tem.append(glatency)
                    #                                    tem.append([dp_query_encodings[i],dp_nodes[i]])
                    #                                    exp[level].append(tem)
                    #                                    trainBuffer[level].append(tem)
                    # print(costlist)
                    for i in range(0, len(costlist)):
                        if join_ids not in dp_table or dp_table[join_ids][
                            0] > costlist[i]:
                            tem = []
                            tem.append(dp_costs[i])
                            tem.append(dp_hints_sqls[i][1])
                            tem.append(dp_hints_sqls[i][0])
                            # # # collect train data (latency)
                            usebuffer = False
                            for j in exp[level]:
                                if (j[2] == dp_hints_sqls[i][0] and j[1] == dp_hints_sqls[i][1]):
                                    usebuffer = True
                                    latency = j[3]
                                    dp_nodes[i].info["latency"] = latency
                                    # print('use buffer')
                                    break
                            if (usebuffer == False):
                                num = num + 1
                                if ((level > num_rels - 1) and slackTimeout(exp[level])):
                                    latency = postgres.GetLatencyFromPg(dp_hints_sqls[i][1], dp_hints_sqls[i][0],
                                                                        verbose=False, check_hint_used=False,
                                                                        timeout=12000, dropbuffer=dropbuffer)
                                    print("slack")
                                    #                                   else:
                                    #                                          latency =postgres.GetLatencyFromPg(dp_hints_sqls[i][1], dp_hints_sqls[i][0], verbose=False, check_hint_used=False,timeout=timeout*2.0, dropbuffer=dropbuffer)
                                    print(latency, 'level', level)
                                else:
                                    latency = postgres.GetLatencyFromPg(dp_hints_sqls[i][1], dp_hints_sqls[i][0],
                                                                        verbose=False, check_hint_used=False,
                                                                        timeout=timeout, dropbuffer=dropbuffer)
                                # latency = random.random()
                                dp_nodes[i].info["latency"] = latency
                                tem.append(latency)
                                tem.append([dp_query_encodings[i], dp_nodes[i]])
                                exp[level].append(tem)
                                trainBuffer[level].append(tem)
                                if level == num_rels:
                                    allPlans = [dp_nodes[i]]
                                    while (allPlans):
                                        currentNode = allPlans.pop()
                                        allPlans.extend(currentNode.children)
                                        for currentChild in currentNode.children:
                                            temlevel = currentChild.info.get("currentLevel")
                                            if (not temlevel == None) and temlevel > 1:
                                                #  print(currentChild)
                                                temsql = currentChild.to_sql(currentChild.info["join_conds"],
                                                                             with_select_exprs=True)
                                                #    print(temsql)
                                                temhint = currentChild.hint_str()
                                                found = False
                                                for i in subplans_fin:
                                                    if (i[1] == dp_hints_sqls[i][1], i[2] == dp_hints_sqls[i][0]):
                                                        found = True
                                                        break
                                                if not found:
                                                    tem = []
                                                    tem.append(currentChild.info["cost"])
                                                    tem.append(temsql)
                                                    tem.append(temhint)
                                                    tem.append(currentChild.info["latency"])
                                                    tem.append([currentChild.info["encoding"], currentChild])
                                                    subplans_fin[temlevel].append(tem)
                            dp_table[join_ids] = (costlist[i], dp_join[i])
            #  nowplansnum = nowplansnum + 1
            #   if nowplansnum > limit and level > 2:
            #   print('plan nums = ',num)
            #    break
            # print('num = ',len(dp_table))
            if level > 4 and level < 15 and level < num_rels:
                temtable = copy.deepcopy(dp_table)
                if not FirstTrain:

                    temcost = []
                    temnodes = []
                    tem_query_encodings = []
                    for key, values in temtable.items():
                        temcost.append(values[0])
                        temnodes.append(values[1].info["node"])
                        tem_query_encodings.append(values[1].info["encoding"])
                    #  print("nodes num = ",len(temnodes))
                    temquery_feats = (torch.cat(tem_query_encodings, dim=0)).to(DEVICE)
                    temtrees, temindexes = TreeConvFeaturize(nodeFeaturizer, temnodes)
                    if torch.cuda.is_available():
                        temtrees = temtrees.to(DEVICE)
                        temindexes = temindexes.to(DEVICE)
                        torch_costs = (torch.tensor(temcost)).to(DEVICE)
                    temcostbais = torch.tanh(model[-1](temquery_feats, temtrees, temindexes).to(DEVICE)).add(1).squeeze(
                        1)
                    temcostlist = torch.mul(temcostbais, torch_costs).tolist()
                    count = 0

                    for key in temtable:
                        temtable[key] = (temcostlist[count], temtable[key][1])
                        count = count + 1
                sortlist = dict(sorted(temtable.items(), key=lambda x: x[1][0])[math.ceil(len(dp_table) * 0.3):])
                for key in sortlist:
                    dp_table.pop(key)
                dp_tables[level] = dp_table
            print('level:', level, 'subplans num = ', len(dp_tables[level]))
            print('level:', level, 'exp num = ', len(exp[level]))
        if timeout > latency:
            timeout = latency
        bestplanhint = list(dp_tables[num_rels].values())[0][1].hint_str()
        return trainBuffer, bestplanhint, num, timeout

    def UCB_left_prune(self, join_graph, all_join_conds, query_leaves, dp_tables, workload, exp, FirstTrain,
                       model, timeout, dropbuffer, nodeFeaturizer, greedy, subplans_fin, finsql, costCache):
        num_rels = len(query_leaves)
        num = 0
        latency = 0
        for i in range(0, num_rels + 1):
            trainBuffer.append([])
        for level in range(2, num_rels + 1):
            dp_table = dp_tables[level]
            bayes_tep = []
            bayes_list = []
            dp_table_i = random_dic(dp_tables[level - 1])
            dp_table_j = random_dic(dp_tables[1])
            for l_ids, l_tup in dp_table_i.items():
                for r_ids, r_tup in dp_table_j.items():
                    l = l_tup[1]
                    r = r_tup[1]
                    if not plans_lib.ExistsJoinEdgeInGraph(
                            l, r, join_graph):
                        # No join clause linking two sides.  Skip.
                        continue
                    l_ids_splits = l_ids.split(',')
                    r_ids_splits = r_ids.split(',')
                    if len(np.intersect1d(l_ids_splits, r_ids_splits)) > 0:
                        # A relation exists in both sides.  Skip.
                        continue
                    join_ids = ','.join(sorted(l_ids_splits + r_ids_splits))
                    # Otherwise, form a new join.
                    dp_costs = []
                    dp_query_encodings = []
                    dp_nodes = []
                    dp_hints_sqls = []
                    dp_join = []
                    for join in EnumerateJoinWithOps(
                            l,
                            r,
                            self.join_ops,
                            self.scan_ops,
                            use_plan_restrictions=self.use_plan_restrictions
                    ):
                        join.info["currentLevel"] = level

                        join_conds = join.KeepRelevantJoins(all_join_conds)
                        join.info["join_conds"] = join_conds

                        cost, sql, hint = self.cost_model.getCost_cache(join, join_conds, costCache)
                        join.info["cost"] = cost
                        logcost = math.log(cost)
                        data = encoding.getencoding_Balsa(sql, hint, workload)
                        join.info["encoding"] = data[0]
                        join.info["node"] = data[1]
                        dp_join.append(join)
                        dp_costs.append(logcost)
                        dp_query_encodings.append(data[0])
                        dp_nodes.append(data[1])
                        dp_hints_sqls.append([hint, sql])
                        tem = []
                        tem.append(logcost)
                        tem.append(sql)
                        tem.append(hint)
                        tem.append(0)
                        tem.append([data[0], data[1]])
                        btem = copy.deepcopy(tem[:] + [join])
                        bayes_tep.append(btem)
                        assert len(data) == 2
                    if not FirstTrain:
                        query_feats = (torch.cat(dp_query_encodings, dim=0)).to(DEVICE)
                        trees, indexes = TreeConvFeaturize(nodeFeaturizer, dp_nodes)
                        if torch.cuda.is_available():
                            trees = trees.to(DEVICE)
                            indexes = indexes.to(DEVICE)
                            torch_dpcosts = (torch.tensor(dp_costs)).to(DEVICE)
                        model[level].train()
                        costbais = []
                        for i in range(10):
                            costbais.append(torch.tanh(model[level](query_feats, trees, indexes).to(DEVICE)).add(1))
                        costbais = torch.cat(costbais, 1)
                        costbais_mean = torch.mean(costbais, dim=1)
                        cost_min, _ = torch.min(costbais, dim=1)
                        var = torch.sum(torch.pow(costbais - costbais_mean.unsqueeze(1), 2), dim=1)
                        ucb = var / var.max() - cost_min / cost_min.max()
                        # print('ucb ')
                        # print(ucb)
                        bayes_list.extend(ucb.tolist())
                        costlist = torch.mul(costbais_mean, torch_dpcosts).tolist()
                    else:
                        costlist = dp_costs
                    for i in range(0, len(costlist)):
                        if join_ids not in dp_table or dp_table[join_ids][
                            0] > costlist[i]:
                            if FirstTrain:
                                tem = []
                                tem.append(dp_costs[i])
                                tem.append(dp_hints_sqls[i][1])
                                tem.append(dp_hints_sqls[i][0])
                                # # # collect train data (latency)
                                usebuffer = False
                                for j in exp[level]:
                                    if (j[2] == dp_hints_sqls[i][0] and j[1] == dp_hints_sqls[i][1]):
                                        usebuffer = True
                                        latency = j[3]
                                        dp_nodes[i].info["latency"] = latency
                                        break
                                if (usebuffer == False):
                                    if ((level > num_rels - 1) and slackTimeout(exp[level])):
                                        latency = postgres.GetLatencyFromPg(dp_hints_sqls[i][1], dp_hints_sqls[i][0],
                                                                            verbose=False, check_hint_used=False,
                                                                            timeout=12000, dropbuffer=dropbuffer)
                                    else:
                                        latency = postgres.GetLatencyFromPg(dp_hints_sqls[i][1], dp_hints_sqls[i][0],
                                                                            verbose=False, check_hint_used=False,
                                                                            timeout=timeout, dropbuffer=dropbuffer)
                                    dp_nodes[i].info["latency"] = latency
                                    dp_join[i].info["latency"] = latency
                                    tem.append(latency)
                                    tem.append([dp_query_encodings[i], dp_nodes[i]])
                                    exp[level].append(tem)
                                    trainBuffer[level].append(tem)
                                    if level == num_rels:
                                        pass
                                        # collectSubplans(dp_join[i],subplans_fin,num_rels,workload)

                            dp_table[join_ids] = (costlist[i], dp_join[i])
            # todo:根据方差收集数据

            if not FirstTrain:
                # print('bayes_list len :',len(bayes_list))
                # print('bayes tep num :',len(bayes_tep))
                bayes_list = torch.tensor(bayes_list)
                ucb_argsort = torch.argsort(bayes_list, descending=True)
                if level < 4:
                    p = 0.25
                else:
                    p = 0.1
                n = math.ceil(p * len(bayes_tep))
                for i in range(n):
                    bayes_plan = bayes_tep[ucb_argsort[i]]
                    usebuffer = False
                    for j in exp[level]:
                        if (j[2] == bayes_plan[2] and j[1] == bayes_plan[1]):
                            usebuffer = True
                            blatency = j[3]
                            break
                    if (usebuffer == False):
                        num = num + 1
                        if ((level > num_rels - 1) and slackTimeout(exp[level])):
                            blatency = postgres.GetLatencyFromPg(bayes_plan[1], bayes_plan[2], verbose=False,
                                                                 check_hint_used=False, timeout=12000,
                                                                 dropbuffer=dropbuffer)
                        else:
                            blatency = postgres.GetLatencyFromPg(bayes_plan[1], bayes_plan[2], verbose=False,
                                                                 check_hint_used=False, timeout=timeout,
                                                                 dropbuffer=dropbuffer)
                    bayes_plan[3] = blatency
                    bayes_plan[4][1].info["latency"] = blatency
                    bayes_plan[5].info["latency"] = blatency
                    if usebuffer == False:
                        trainBuffer[level].append(copy.deepcopy(bayes_plan))
                        exp[level].append(copy.deepcopy(bayes_plan))
                        if level == num_rels:
                            pass
                        # collectSubplans(bayes_plan[5],subplans_fin,num_rels,workload)
            if level > 4 and level < 15 and level < num_rels:
                temtable = copy.deepcopy(dp_table)
                if not FirstTrain:
                    temcost = []
                    temnodes = []
                    tem_query_encodings = []
                    for key, values in temtable.items():
                        temcost.append(values[0])
                        temnodes.append(values[1].info["node"])
                        tem_query_encodings.append(values[1].info["encoding"])
                    #  print("nodes num = ",len(temnodes))
                    temquery_feats = (torch.cat(tem_query_encodings, dim=0)).to(DEVICE)
                    temtrees, temindexes = TreeConvFeaturize(nodeFeaturizer, temnodes)
                    if torch.cuda.is_available():
                        temtrees = temtrees.to(DEVICE)
                        temindexes = temindexes.to(DEVICE)
                        torch_costs = (torch.tensor(temcost)).to(DEVICE)
                    model[num_rels].eval()
                    temcostbais = model[num_rels](temquery_feats, temtrees, temindexes).to(DEVICE).add(1)
                    #                    temcostbais = []
                    #                    for i in range(10):
                    #                        temcostbais.append(
                    #                            torch.tanh(model[-1](temquery_feats, temtrees, temindexes).to(DEVICE)).add(1))
                    #                    temcostbais = torch.cat(temcostbais, 1)
                    #                    temcostbais = torch.mean(temcostbais, dim=1)
                    temcostlist = torch.mul(temcostbais, torch_costs).tolist()
                    count = 0
                    for key in temtable:
                        temtable[key] = (temcostlist[count], temtable[key][1])
                        count = count + 1
                sortlist = dict(sorted(temtable.items(), key=lambda x: x[1][0])[math.ceil(len(dp_table) * 0.3):])
                for key in sortlist:
                    dp_table.pop(key)
                dp_tables[level] = dp_table
        bestplanhint = list(dp_tables[num_rels].values())[0][1].hint_str()
        nowlatency = postgres.GetLatencyFromPg(finsql, bestplanhint, verbose=False, check_hint_used=False,
                                               timeout=timeout, dropbuffer=dropbuffer)
        if timeout > nowlatency:
            timeout = nowlatency

        return trainBuffer, bestplanhint, num, timeout

    def UCB_left_prune_replay(self, join_graph, all_join_conds, query_leaves, dp_tables, workload, exp, FirstTrain,
                              model, timeout, dropbuffer, nodeFeaturizer, greedy, subplans_fin, finsql, costCache,
                              dpsign):
        num_rels = len(query_leaves)
        num = 0
        latency = 0
        for i in range(0, num_rels + 1):
            trainBuffer.append([])
        for level in range(2, num_rels + 1):
            dp_table = dp_tables[level]
            bayes_tep = []
            bayes_list = []
            dp_table_i = random_dic(dp_tables[level - 1])
            dp_table_j = random_dic(dp_tables[1])
            for l_ids, l_tup in dp_table_i.items():
                for r_ids, r_tup in dp_table_j.items():
                    l = l_tup[1]
                    r = r_tup[1]
                    if not plans_lib.ExistsJoinEdgeInGraph(
                            l, r, join_graph):
                        # No join clause linking two sides.  Skip.
                        continue
                    l_ids_splits = l_ids.split(',')
                    r_ids_splits = r_ids.split(',')
                    if len(np.intersect1d(l_ids_splits, r_ids_splits)) > 0:
                        # A relation exists in both sides.  Skip.
                        continue
                    join_ids = ','.join(sorted(l_ids_splits + r_ids_splits))
                    # Otherwise, form a new join.
                    dp_costs = []
                    dp_query_encodings = []
                    dp_nodes = []
                    dp_hints_sqls = []
                    dp_join = []
                    for join in EnumerateJoinWithOps(
                            l,
                            r,
                            self.join_ops,
                            self.scan_ops,
                            use_plan_restrictions=self.use_plan_restrictions
                    ):
                        join.info["currentLevel"] = level

                        join_conds = join.KeepRelevantJoins(all_join_conds)
                        join.info["join_conds"] = join_conds

                        cost, sql, hint = self.cost_model.getCost_cache(join, join_conds, costCache)
                        join.info["cost"] = cost
                        logcost = math.log(cost)
                        data = encoding.getencoding_Balsa(sql, hint, workload)
                        join.info["encoding"] = data[0]
                        join.info["node"] = data[1]
                        dp_join.append(join)
                        dp_costs.append(logcost)
                        dp_query_encodings.append(data[0])
                        dp_nodes.append(data[1])
                        dp_hints_sqls.append([hint, sql])
                        tem = []
                        tem.append(logcost)
                        tem.append(sql)
                        tem.append(hint)
                        tem.append(0)
                        tem.append([data[0], data[1]])
                        btem = copy.deepcopy(tem[:] + [join])
                        bayes_tep.append(btem)
                        assert len(data) == 2
                    if not FirstTrain:
                        query_feats = (torch.cat(dp_query_encodings, dim=0)).to(DEVICE)
                        trees, indexes = TreeConvFeaturize(nodeFeaturizer, dp_nodes)
                        if torch.cuda.is_available():
                            trees = trees.to(DEVICE)
                            indexes = indexes.to(DEVICE)
                            torch_dpcosts = (torch.tensor(dp_costs)).to(DEVICE)
                        model[level].train()
                        costbais = []
                        for i in range(10):
                            costbais.append(torch.tanh(model[level](query_feats, trees, indexes).to(DEVICE)).add(1))
                        costbais = torch.cat(costbais, 1)
                        costbais_mean = torch.mean(costbais, dim=1)
                        cost_min, _ = torch.min(costbais, dim=1)
                        var = torch.sum(torch.pow(costbais - costbais_mean.unsqueeze(1), 2), dim=1)
                        ucb = var / var.max() - cost_min / cost_min.max()
                        # print('ucb ')
                        # print(ucb)
                        bayes_list.extend(ucb.tolist())
                        costlist = torch.mul(costbais_mean, torch_dpcosts).tolist()
                    else:
                        costlist = dp_costs
                    for i in range(0, len(costlist)):
                        if join_ids not in dp_table or dp_table[join_ids][
                            0] > costlist[i]:
                            if FirstTrain or dpsign:
                                tem = []
                                tem.append(dp_costs[i])
                                tem.append(dp_hints_sqls[i][1])
                                tem.append(dp_hints_sqls[i][0])
                                # # # collect train data (latency)
                                usebuffer = False
                                for j in exp[level]:
                                    if (j[2] == dp_hints_sqls[i][0] and j[1] == dp_hints_sqls[i][1]):
                                        usebuffer = True
                                        latency = j[3]
                                        dp_nodes[i].info["latency"] = latency
                                        break
                                coll = False
                                if (usebuffer == False):
                                    if (slackTimeout(exp[level])):
                                        print('slack')
                                        latency = postgres.GetLatencyFromPg(dp_hints_sqls[i][1], dp_hints_sqls[i][0],
                                                                            verbose=False, check_hint_used=False,
                                                                            timeout=12000, dropbuffer=dropbuffer)
                                        coll = True
                                    else:
                                        if random.random() > 0.2:
                                            latency = postgres.GetLatencyFromPg(dp_hints_sqls[i][1],
                                                                                dp_hints_sqls[i][0],
                                                                                verbose=False, check_hint_used=False,
                                                                                timeout=timeout, dropbuffer=dropbuffer)
                                            coll = True
                                    if coll:
                                        dp_nodes[i].info["latency"] = latency
                                        dp_join[i].info["latency"] = latency
                                        tem.append(latency)
                                        tem.append([dp_query_encodings[i], dp_nodes[i]])
                                        exp[level].append(tem)
                                        trainBuffer[level].append(tem)
                                    if level == num_rels:
                                        pass
                                        # collectSubplans(dp_join[i],subplans_fin,num_rels,workload)

                            dp_table[join_ids] = (costlist[i], dp_join[i])
            # todo:根据方差收集数据

            if not FirstTrain and not dpsign:
                # print('bayes_list len :',len(bayes_list))
                # print('bayes tep num :',len(bayes_tep))
                bayes_list = torch.tensor(bayes_list)
                ucb_argsort = torch.argsort(bayes_list, descending=True)
                if level < 4:
                    p = 0.2
                else:
                    p = 0.1
                n = math.ceil(p * len(bayes_tep))
                for i in range(n):
                    bayes_plan = bayes_tep[ucb_argsort[i]]
                    usebuffer = False
                    for j in exp[level]:
                        if (j[2] == bayes_plan[2] and j[1] == bayes_plan[1]):
                            usebuffer = True
                            blatency = j[3]
                            break
                    if (usebuffer == False):
                        num = num + 1
                        if ((level > num_rels - 1) and slackTimeout(exp[level])):
                            blatency = postgres.GetLatencyFromPg(bayes_plan[1], bayes_plan[2], verbose=False,
                                                                 check_hint_used=False, timeout=12000,
                                                                 dropbuffer=dropbuffer)
                        else:
                            blatency = postgres.GetLatencyFromPg(bayes_plan[1], bayes_plan[2], verbose=False,
                                                                 check_hint_used=False, timeout=timeout,
                                                                 dropbuffer=dropbuffer)
                    bayes_plan[3] = blatency
                    bayes_plan[4][1].info["latency"] = blatency
                    bayes_plan[5].info["latency"] = blatency
                    if usebuffer == False:
                        trainBuffer[level].append(copy.deepcopy(bayes_plan))
                        exp[level].append(copy.deepcopy(bayes_plan))
                        if level == num_rels:
                            pass
                        # collectSubplans(bayes_plan[5],subplans_fin,num_rels,workload)
            if level > 4 and level < 15 and level < num_rels - 1:
                temtable = copy.deepcopy(dp_table)
                if not FirstTrain:
                    temcost = []
                    temnodes = []
                    tem_query_encodings = []
                    for key, values in temtable.items():
                        temcost.append(values[0])
                        temnodes.append(values[1].info["node"])
                        tem_query_encodings.append(values[1].info["encoding"])
                    #  print("nodes num = ",len(temnodes))
                    temquery_feats = (torch.cat(tem_query_encodings, dim=0)).to(DEVICE)
                    temtrees, temindexes = TreeConvFeaturize(nodeFeaturizer, temnodes)
                    if torch.cuda.is_available():
                        temtrees = temtrees.to(DEVICE)
                        temindexes = temindexes.to(DEVICE)
                        torch_costs = (torch.tensor(temcost)).to(DEVICE)
                    model[num_rels].eval()
                    temcostbais = model[num_rels](temquery_feats, temtrees, temindexes).to(DEVICE).add(1)
                    #                    temcostbais = []
                    #                    for i in range(10):
                    #                        temcostbais.append(
                    #                            torch.tanh(model[-1](temquery_feats, temtrees, temindexes).to(DEVICE)).add(1))
                    #                    temcostbais = torch.cat(temcostbais, 1)
                    #                    temcostbais = torch.mean(temcostbais, dim=1)
                    temcostlist = torch.mul(temcostbais, torch_costs).tolist()
                    count = 0
                    for key in temtable:
                        temtable[key] = (temcostlist[count], temtable[key][1])
                        count = count + 1
                sortlist = dict(sorted(temtable.items(), key=lambda x: x[1][0])[math.ceil(len(dp_table) * 0.3):])
                for key in sortlist:
                    dp_table.pop(key)
                dp_tables[level] = dp_table
        bestplanhint = list(dp_tables[num_rels].values())[0][1].hint_str()
        # print(list(dp_tables[num_rels].values())[0][1].info)
        if dpsign:
            collectSubplans(list(dp_tables[num_rels].values())[0][1], subplans_fin, workload, exp)
        nowlatency = postgres.GetLatencyFromPg(finsql, bestplanhint, verbose=False, check_hint_used=False,
                                               timeout=timeout, dropbuffer=dropbuffer)
        if timeout > nowlatency:
            timeout = nowlatency

        return trainBuffer, bestplanhint, num, timeout

    def UCB_left_prune_replay_fix_kl(self, join_graph, all_join_conds, query_leaves, dp_tables, workload, exp,
                                     FirstTrain,
                                     model, timeout, dropbuffer, nodeFeaturizer, greedy, subplans_fin, finsql,
                                     costCache,
                                     dpsign, levelList):
        num_rels = len(query_leaves)
        num = 0
        latency = 0
        for i in range(0, num_rels + 1):
            trainBuffer.append([])
        for level in range(2, num_rels + 1):
            dp_table = dp_tables[level]
            dp_table_i = random_dic(dp_tables[level - 1])
            dp_table_j = random_dic(dp_tables[1])
            for l_ids, l_tup in dp_table_i.items():
                for r_ids, r_tup in dp_table_j.items():
                    l = l_tup[1]
                    r = r_tup[1]
                    if not plans_lib.ExistsJoinEdgeInGraph(
                            l, r, join_graph):
                        # No join clause linking two sides.  Skip.
                        continue
                    l_ids_splits = l_ids.split(',')
                    r_ids_splits = r_ids.split(',')
                    if len(np.intersect1d(l_ids_splits, r_ids_splits)) > 0:
                        # A relation exists in both sides.  Skip.
                        continue
                    join_ids = ','.join(sorted(l_ids_splits + r_ids_splits))
                    # Otherwise, form a new join.
                    dp_costs = []
                    dp_query_encodings = []
                    dp_nodes = []
                    dp_hints_sqls = []
                    dp_join = []
                    bayes_tep = []
                    bayes_list = []
                    for join in EnumerateJoinWithOps(
                            l,
                            r,
                            self.join_ops,
                            self.scan_ops,
                            use_plan_restrictions=self.use_plan_restrictions
                    ):
                        join.info["currentLevel"] = level

                        join_conds = join.KeepRelevantJoins(all_join_conds)
                        join.info["join_conds"] = join_conds
                        cost, sql, hint = self.cost_model.getCost_cache(join, join_conds, costCache)
                        join.info["cost"] = cost
                        logcost = math.log(cost)
                        data = encoding.getencoding_Balsa(sql, hint, workload)
                        join.info["encoding"] = data[0]
                        join.info["node"] = data[1]
                        dp_join.append(join)
                        dp_costs.append(logcost)
                        dp_query_encodings.append(data[0])
                        dp_nodes.append(data[1])
                        dp_hints_sqls.append([hint, sql])
                        tem = []
                        tem.append(logcost)
                        tem.append(sql)
                        tem.append(hint)
                        tem.append(0)
                        tem.append([data[0], data[1]])
                        btem = copy.deepcopy(tem[:] + [join] + [join_ids])
                        bayes_tep.append(btem)
                    #                    if level > num_rels - 5:
                    #                        levelList[level][join_ids] = [dp_costs, dp_query_encodings, dp_nodes]
                    if not FirstTrain and level > num_rels - 4:
                        query_feats = (torch.cat(dp_query_encodings, dim=0)).to(DEVICE)
                        trees, indexes = TreeConvFeaturize(nodeFeaturizer, dp_nodes)
                        if torch.cuda.is_available():
                            trees = trees.to(DEVICE)
                            indexes = indexes.to(DEVICE)
                            torch_dpcosts = (torch.tensor(dp_costs)).to(DEVICE)
                        model[level].train()
                        costbais = []
                        for i in range(10):
                            with torch.no_grad():
                                costbais.append(torch.tanh(model[level](query_feats, trees, indexes).to(DEVICE)).add(1))
                        costbais = torch.cat(costbais, 1)
                        costbais_mean = torch.mean(costbais, dim=1)
                        cost_t = torch.mul(costbais_mean, torch_dpcosts)
                        costlist = cost_t.tolist()
                        cost_min, _ = torch.min(cost_t, dim=0)
                        var = torch.var(costbais, dim=1)
                        ucb = var / var.max() - cost_min / cost_min.max()
                        bayes_list.extend(ucb.tolist())

                        if not FirstTrain and not dpsign and level > num_rels - 4:
                            # print('bayes_list len :',len(bayes_list))
                            # print('bayes tep num :',len(bayes_tep))
                            bayes_list = torch.tensor(bayes_list)
                            # ucb_argsort = torch.argsort(bayes_list, descending=True)
                            ucb_argsort = torch.argsort(bayes_list, descending=True)
                            p = 0.1
                            n = math.ceil(p * len(bayes_tep))
                            for i in range(n):
                                bayes_plan = bayes_tep[ucb_argsort[i]]
                                usebuffer = False
                                for j in exp[level]:
                                    if (j[2] == bayes_plan[2] and j[1] == bayes_plan[1]):
                                        usebuffer = True
                                        blatency = j[3]
                                        break
                                if (usebuffer == False):
                                    num = num + 1
                                    if slackTimeout(exp[level]):
                                        blatency = postgres.GetLatencyFromPg(bayes_plan[1], bayes_plan[2],
                                                                             verbose=False,
                                                                             check_hint_used=False, timeout=12000,
                                                                             dropbuffer=dropbuffer)
                                    else:
                                        blatency = postgres.GetLatencyFromPg(bayes_plan[1], bayes_plan[2],
                                                                             verbose=False,
                                                                             check_hint_used=False, timeout=timeout,
                                                                             dropbuffer=dropbuffer)
                                    # if blatency == 90000:
                                    #  continue
                                    bayes_plan[3] = blatency
                                    bayes_plan[4][1].info["latency"] = blatency
                                    bayes_plan[5].info["latency"] = blatency
                                    bayes_plan[4][1].info["join_ids"] = join_ids
                                    bayes_plan[5].info["join_ids"] = join_ids
                                    trainBuffer[level].append(copy.deepcopy(bayes_plan))
                                    exp[level].append(copy.deepcopy(bayes_plan))

                    else:
                        costlist = dp_costs
                    for i in range(0, len(costlist)):
                        if join_ids not in dp_table or dp_table[join_ids][
                            0] > costlist[i]:
                            if (FirstTrain or dpsign) and level > num_rels - 4:
                                tem = []
                                tem.append(dp_costs[i])
                                tem.append(dp_hints_sqls[i][1])
                                tem.append(dp_hints_sqls[i][0])
                                # # # collect train data (latency)
                                usebuffer = False
                                for j in exp[level]:
                                    if (j[2] == dp_hints_sqls[i][0] and j[1] == dp_hints_sqls[i][1]):
                                        usebuffer = True
                                        latency = j[3]
                                        dp_nodes[i].info["latency"] = latency
                                        break
                                coll = False
                                if (usebuffer == False):
                                    if (slackTimeout(exp[level])):
                                        print('slack')
                                        latency = postgres.GetLatencyFromPg(dp_hints_sqls[i][1], dp_hints_sqls[i][0],
                                                                            verbose=False, check_hint_used=False,
                                                                            timeout=12000, dropbuffer=dropbuffer)
                                        coll = True
                                    else:
                                        if random.random() > -1:
                                            latency = postgres.GetLatencyFromPg(dp_hints_sqls[i][1],
                                                                                dp_hints_sqls[i][0],
                                                                                verbose=False, check_hint_used=False,
                                                                                timeout=timeout, dropbuffer=dropbuffer)
                                            coll = True
                                    if coll:
                                        dp_nodes[i].info["latency"] = latency
                                        dp_join[i].info["latency"] = latency
                                        dp_nodes[i].info["join_ids"] = join_ids
                                        dp_join[i].info["join_ids"] = join_ids

                                        tem.append(latency)
                                        tem.append([dp_query_encodings[i], dp_nodes[i]])
                                        tem.append(dp_join[i])
                                        tem.append(join_ids)
                                        exp[level].append(tem)
                                        trainBuffer[level].append(tem)

                            dp_table[join_ids] = (costlist[i], dp_join[i])

            if level > 6 and level < 15 and level < num_rels - 1:
                temtable = copy.deepcopy(dp_table)
                if not FirstTrain:
                    temcost = []
                    temnodes = []
                    tem_query_encodings = []
                    for key, values in temtable.items():
                        temcost.append(values[0])
                        temnodes.append(values[1].info["node"])
                        tem_query_encodings.append(values[1].info["encoding"])
                    #  print("nodes num = ",len(temnodes))
                    temquery_feats = (torch.cat(tem_query_encodings, dim=0)).to(DEVICE)
                    temtrees, temindexes = TreeConvFeaturize(nodeFeaturizer, temnodes)
                    if torch.cuda.is_available():
                        temtrees = temtrees.to(DEVICE)
                        temindexes = temindexes.to(DEVICE)
                        torch_costs = (torch.tensor(temcost)).to(DEVICE)
                    # temcostbais = model[num_rels](temquery_feats, temtrees, temindexes).to(DEVICE).add(1)
                    temcostbais = []
                    for i in range(10):
                        with torch.no_grad():
                            temcostbais.append(
                                torch.tanh(model[-1](temquery_feats, temtrees, temindexes).to(DEVICE)).add(1))
                    temcostbais = torch.cat(temcostbais, 1)
                    temcostbais = torch.mean(temcostbais, dim=1)
                    temcostlist = torch.mul(temcostbais, torch_costs).tolist()
                    count = 0
                    for key in temtable:
                        temtable[key] = (temcostlist[count], temtable[key][1])
                        count = count + 1
                sortlist = dict(sorted(temtable.items(), key=lambda x: x[1][0])[math.ceil(len(dp_table) * 0.3):])
                for key in sortlist:
                    dp_table.pop(key)
                dp_tables[level] = dp_table
        bestplanhint = list(dp_tables[num_rels].values())[0][1].hint_str()
        # print(list(dp_tables[num_rels].values())[0][1].info)
        #        if dpsign:
        #            collectSubplans(list(dp_tables[num_rels].values())[0][1], subplans_fin, workload, exp)
        nowlatency = postgres.GetLatencyFromPg(finsql, bestplanhint, verbose=False, check_hint_used=False,
                                               timeout=timeout, dropbuffer=dropbuffer)
        if timeout > nowlatency:
            timeout = nowlatency

        return trainBuffer, bestplanhint, num, timeout

    def test_speed(self, join_graph, all_join_conds, query_leaves, dp_tables, workload, exp, FirstTrain,
                   model, timeout, dropbuffer, nodeFeaturizer, greedy, subplans_fin, finsql, costCache,
                   dpsign, levelList):
        num_rels = len(query_leaves)
        num = 0
        latency = 0
        plans_num = 0
        tui_time = 0
        tui_num = 0
        for i in range(0, num_rels + 1):
            trainBuffer.append([])
        for level in range(2, num_rels + 1):
            dp_table = dp_tables[level]
            dp_table_i = random_dic(dp_tables[level - 1])
            dp_table_j = random_dic(dp_tables[1])
            for l_ids, l_tup in dp_table_i.items():
                for r_ids, r_tup in dp_table_j.items():
                    l = l_tup[1]
                    r = r_tup[1]
                    if not plans_lib.ExistsJoinEdgeInGraph(
                            l, r, join_graph):
                        # No join clause linking two sides.  Skip.
                        continue
                    l_ids_splits = l_ids.split(',')
                    r_ids_splits = r_ids.split(',')
                    if len(np.intersect1d(l_ids_splits, r_ids_splits)) > 0:
                        # A relation exists in both sides.  Skip.
                        continue
                    join_ids = ','.join(sorted(l_ids_splits + r_ids_splits))
                    # Otherwise, form a new join.
                    dp_costs = []
                    dp_query_encodings = []
                    dp_nodes = []
                    dp_hints_sqls = []
                    dp_join = []
                    bayes_tep = []
                    bayes_list = []
                    for join in EnumerateJoinWithOps(
                            l,
                            r,
                            self.join_ops,
                            self.scan_ops,
                            use_plan_restrictions=self.use_plan_restrictions
                    ):
                        #                        join.info["currentLevel"] = level

                        join_conds = join.KeepRelevantJoins(all_join_conds)
                        #                        join.info["join_conds"] = join_conds
                        #                        cost, sql, hint = self.cost_model.getCost_cache(join, join_conds, costCache)
                        cost = 1
                        plans_num = plans_num + 1
                        if level > num_rels - 3:
                            tui_num = tui_num + 1
                        #                        join.info["cost"] = cost
                        #                        logcost = math.log(cost)
                        #                        data = encoding.getencoding_Balsa(sql, hint, workload)
                        #                        join.info["encoding"] = data[0]
                        #                        join.info["node"] = data[1]
                        dp_join.append(join)
                        #                        dp_costs.append(logcost)
                        dp_costs.append(cost)
                    #                        dp_query_encodings.append(data[0])
                    #                        dp_nodes.append(data[1])
                    #                        dp_hints_sqls.append([hint, sql])
                    #                        tem = []
                    #                        tem.append(logcost)
                    #                        tem.append(sql)
                    #                        tem.append(hint)
                    #                        tem.append(0)
                    #                        tem.append([data[0], data[1]])
                    #                        btem = copy.deepcopy(tem[:] + [join]+[join_ids])
                    #                        bayes_tep.append(btem)
                    #                    if level > num_rels - 5:
                    #                        levelList[level][join_ids] = [dp_costs, dp_query_encodings, dp_nodes]
                    #                    if level > num_rels - 3 :
                    #
                    #                        query_feats = (torch.cat(dp_query_encodings, dim=0)).to(DEVICE)
                    #                        trees, indexes = TreeConvFeaturize(nodeFeaturizer, dp_nodes)
                    #                        if torch.cuda.is_available():
                    #                            trees = trees.to(DEVICE)
                    #                            indexes = indexes.to(DEVICE)
                    #                            torch_dpcosts = (torch.tensor(dp_costs)).to(DEVICE)
                    #                        model[level].train()
                    #                        costbais = []
                    #                        stime = time.time()
                    #                        with torch.no_grad():
                    #                            tem_tor = torch.tanh(model[level](query_feats, trees, indexes).to(DEVICE)).add(1)
                    #                        tui_time = tui_time+time.time()-stime
                    #                        costbais.append(tem_tor)
                    #
                    #                        costbais = torch.cat(costbais, 1)
                    #                        costbais_mean = torch.mean(costbais, dim=1)
                    #                        cost_t = torch.mul(costbais_mean, torch_dpcosts)
                    #
                    #                        costlist = cost_t.tolist()
                    #                        cost_min, _ = torch.min(cost_t, dim=0)
                    #                        var = torch.var(costbais, dim=1)
                    #                        ucb = var / var.max() - cost_min / cost_min.max()
                    #                        bayes_list.extend(ucb.tolist())

                    # else:
                    costlist = dp_costs
                    for i in range(0, len(costlist)):
                        if join_ids not in dp_table or dp_table[join_ids][
                            0] > costlist[i]:
                            dp_table[join_ids] = (costlist[i], dp_join[i])

        #            if level > 6 and level < 15 and level < num_rels - 1:
        #                temtable = copy.deepcopy(dp_table)
        ##                if not FirstTrain:
        ##                    temcost = []
        ##                    temnodes = []
        ##                    tem_query_encodings = []
        ##                    for key, values in temtable.items():
        ##                        temcost.append(values[0])
        ##                        temnodes.append(values[1].info["node"])
        ##                        tem_query_encodings.append(values[1].info["encoding"])
        ##                    #  print("nodes num = ",len(temnodes))
        ##                    temquery_feats = (torch.cat(tem_query_encodings, dim=0)).to(DEVICE)
        ##                    temtrees, temindexes = TreeConvFeaturize(nodeFeaturizer, temnodes)
        ##                    if torch.cuda.is_available():
        ##                        temtrees = temtrees.to(DEVICE)
        ##                        temindexes = temindexes.to(DEVICE)
        ##                        torch_costs = (torch.tensor(temcost)).to(DEVICE)
        ##                    # temcostbais = model[num_rels](temquery_feats, temtrees, temindexes).to(DEVICE).add(1)
        ##                    temcostbais = []
        ##                    for i in range(10):
        ##                        with torch.no_grad():
        ##                            temcostbais.append(
        ##                               torch.tanh(model[-1](temquery_feats, temtrees, temindexes).to(DEVICE)).add(1))
        ##                    temcostbais = torch.cat(temcostbais, 1)
        ##                    temcostbais = torch.mean(temcostbais, dim=1)
        ##                    temcostlist = torch.mul(temcostbais, torch_costs).tolist()
        ##                    count = 0
        ##                    for key in temtable:
        ##                        temtable[key] = (temcostlist[count], temtable[key][1])
        ##                        count = count + 1
        #                sortlist = dict(sorted(temtable.items(), key=lambda x: x[1][0])[math.ceil(len(dp_table) * 0.3):])
        #                for key in sortlist:
        #                    dp_table.pop(key)
        #                dp_tables[level] = dp_table
        bestplanhint = list(dp_tables[num_rels].values())[0][1].hint_str()
        #        # print(list(dp_tables[num_rels].values())[0][1].info)
        #        #        if dpsign:
        #        #            collectSubplans(list(dp_tables[num_rels].values())[0][1], subplans_fin, workload, exp)
        #        nowlatency = postgres.GetLatencyFromPg(finsql, bestplanhint, verbose=False, check_hint_used=False,
        #                                               timeout=timeout, dropbuffer=dropbuffer)
        #        if timeout > nowlatency:
        #            timeout = nowlatency
        print('plans num :', plans_num, 'tui_num :', tui_num)
        return trainBuffer, bestplanhint, num, timeout

    def UCB_left_prune_replay_fix_kl_1(self, join_graph, all_join_conds, query_leaves, dp_tables, workload, exp,
                                       FirstTrain,
                                       model, timeout, dropbuffer, nodeFeaturizer, greedy, subplans_fin, finsql,
                                       costCache,
                                       dpsign, levelList, epoch):
        num_rels = len(query_leaves)
        num = 0
        latency = 0
        time_all = 0
        for i in range(0, num_rels + 1):
            trainBuffer.append([])
        for level in range(2, num_rels + 1):
            dp_table = dp_tables[level]
            dp_table_i = random_dic(dp_tables[level - 1])
            dp_table_j = random_dic(dp_tables[1])
            for l_ids, l_tup in dp_table_i.items():
                for r_ids, r_tup in dp_table_j.items():
                    l = l_tup[1]
                    r = r_tup[1]
                    if not plans_lib.ExistsJoinEdgeInGraph(
                            l, r, join_graph):
                        # No join clause linking two sides.  Skip.
                        continue
                    l_ids_splits = l_ids.split(',')
                    r_ids_splits = r_ids.split(',')
                    if len(np.intersect1d(l_ids_splits, r_ids_splits)) > 0:
                        # A relation exists in both sides.  Skip.
                        continue
                    join_ids = ','.join(sorted(l_ids_splits + r_ids_splits))
                    # Otherwise, form a new join.
                    dp_costs = []
                    dp_query_encodings = []
                    dp_nodes = []
                    dp_hints_sqls = []
                    dp_join = []
                    bayes_tep = []
                    bayes_list = []
                    for join in EnumerateJoinWithOps(
                            l,
                            r,
                            self.join_ops,
                            self.scan_ops,
                            use_plan_restrictions=self.use_plan_restrictions
                    ):
                        join.info["currentLevel"] = level

                        join_conds = join.KeepRelevantJoins(all_join_conds)
                        join.info["join_conds"] = join_conds
                        cost, sql, hint = self.cost_model.getCost_cache(join, join_conds, costCache)
                        join.info["cost"] = cost
                        logcost = math.log(cost)
                        data = encoding.getencoding_Balsa(sql, hint, workload)
                        join.info["encoding"] = data[0]
                        join.info["node"] = data[1]
                        dp_join.append(join)
                        dp_costs.append(logcost)
                        dp_query_encodings.append(data[0])
                        dp_nodes.append(data[1])
                        dp_hints_sqls.append([hint, sql])
                        tem = []
                        tem.append(logcost)
                        tem.append(sql)
                        tem.append(hint)
                        tem.append(0)
                        tem.append([data[0], data[1]])
                        btem = copy.deepcopy(tem[:] + [join] + [join_ids])
                        bayes_tep.append(btem)
                    #                    if level > num_rels - 5:
                    #                        levelList[level][join_ids] = [dp_costs, dp_query_encodings, dp_nodes]
                    # if level > num_rels - 4 :
                    if not FirstTrain and level > num_rels - 4:
                        times = time.time()
                        query_feats = (torch.cat(dp_query_encodings, dim=0)).to(DEVICE)
                        trees, indexes = TreeConvFeaturize(nodeFeaturizer, dp_nodes)
                        if torch.cuda.is_available():
                            trees = trees.to(DEVICE)
                            indexes = indexes.to(DEVICE)
                            torch_dpcosts = (torch.tensor(dp_costs)).to(DEVICE)
                        model[level].train()
                        costbais = []
                        for i in range(1):
                            with torch.no_grad():
                                costbais.append(torch.tanh(model[level](query_feats, trees, indexes).to(DEVICE)).add(1))
                        costbais = torch.cat(costbais, 1)
                        costbais_mean = torch.mean(costbais, dim=1)
                        timee = time.time() - times
                        time_all = time_all + timee
                        cost_t = torch.mul(costbais_mean, torch_dpcosts)
                        costlist = cost_t.tolist()
                        cost_min, _ = torch.min(cost_t, dim=0)
                        if epoch > 9:
                            var = - torch.var(costbais, dim=1)
                            ucb = var / var.max() + cost_min / cost_min.max()
                        else:
                            ucb = cost_min / cost_min.max()
                        bayes_list.extend(ucb.tolist())

                        # if False:
                        if not FirstTrain and not dpsign:
                            # print('bayes_list len :',len(bayes_list))
                            # print('bayes tep num :',len(bayes_tep))
                            bayes_list = torch.tensor(bayes_list)
                            # ucb_argsort = torch.argsort(bayes_list, descending=True)
                            ucb_argsort = torch.argsort(bayes_list)
                            p = 0.1
                            n = math.ceil(p * len(bayes_tep))
                            for i in range(n):
                                bayes_plan = bayes_tep[ucb_argsort[i]]
                                usebuffer = False
                                for j in exp[level]:
                                    if (j[2] == bayes_plan[2] and j[1] == bayes_plan[1]):
                                        usebuffer = True
                                        blatency = j[3]
                                        break
                                if (usebuffer == False):
                                    num = num + 1
                                    if slackTimeout(exp[level]):
                                        blatency = postgres.GetLatencyFromPg(bayes_plan[1], bayes_plan[2],
                                                                             verbose=False,
                                                                             check_hint_used=False, timeout=12000,
                                                                             dropbuffer=dropbuffer)
                                    else:
                                        blatency = postgres.GetLatencyFromPg(bayes_plan[1], bayes_plan[2],
                                                                             verbose=False,
                                                                             check_hint_used=False, timeout=timeout,
                                                                             dropbuffer=dropbuffer)
                                    # if blatency == 90000:
                                    #  continue
                                    bayes_plan[3] = blatency
                                    bayes_plan[4][1].info["latency"] = blatency
                                    bayes_plan[5].info["latency"] = blatency
                                    bayes_plan[4][1].info["join_ids"] = join_ids
                                    bayes_plan[5].info["join_ids"] = join_ids
                                    trainBuffer[level].append(copy.deepcopy(bayes_plan))
                                    exp[level].append(copy.deepcopy(bayes_plan))

                    else:
                        costlist = dp_costs
                    for i in range(0, len(costlist)):
                        if join_ids not in dp_table or dp_table[join_ids][
                            0] > costlist[i]:
                            if False:
                                # if (FirstTrain or dpsign) and level > num_rels -3:
                                tem = []
                                tem.append(dp_costs[i])
                                tem.append(dp_hints_sqls[i][1])
                                tem.append(dp_hints_sqls[i][0])
                                # # # collect train data (latency)
                                usebuffer = False
                                for j in exp[level]:
                                    if (j[2] == dp_hints_sqls[i][0] and j[1] == dp_hints_sqls[i][1]):
                                        usebuffer = True
                                        latency = j[3]
                                        dp_nodes[i].info["latency"] = latency
                                        break
                                coll = False
                                if (usebuffer == False):
                                    if (slackTimeout(exp[level])):
                                        print('slack')
                                        latency = postgres.GetLatencyFromPg(dp_hints_sqls[i][1], dp_hints_sqls[i][0],
                                                                            verbose=False, check_hint_used=False,
                                                                            timeout=12000, dropbuffer=dropbuffer)
                                        coll = True
                                    else:
                                        if random.random() > -1:
                                            latency = postgres.GetLatencyFromPg(dp_hints_sqls[i][1],
                                                                                dp_hints_sqls[i][0],
                                                                                verbose=False, check_hint_used=False,
                                                                                timeout=timeout, dropbuffer=dropbuffer)
                                            coll = True
                                    if coll:
                                        dp_nodes[i].info["latency"] = latency
                                        dp_join[i].info["latency"] = latency
                                        dp_nodes[i].info["join_ids"] = join_ids
                                        dp_join[i].info["join_ids"] = join_ids

                                        tem.append(latency)
                                        tem.append([dp_query_encodings[i], dp_nodes[i]])
                                        tem.append(dp_join[i])
                                        tem.append(join_ids)
                                        exp[level].append(tem)
                                        trainBuffer[level].append(tem)

                            dp_table[join_ids] = (costlist[i], dp_join[i])

            if level > 6 and level < 15 and level < num_rels - 1:
                temtable = copy.deepcopy(dp_table)
                if True:
                    # if not FirstTrain:
                    temcost = []
                    temnodes = []
                    tem_query_encodings = []
                    for key, values in temtable.items():
                        temcost.append(values[0])
                        temnodes.append(values[1].info["node"])
                        tem_query_encodings.append(values[1].info["encoding"])
                    #  print("nodes num = ",len(temnodes))
                    temquery_feats = (torch.cat(tem_query_encodings, dim=0)).to(DEVICE)
                    temtrees, temindexes = TreeConvFeaturize(nodeFeaturizer, temnodes)
                    if torch.cuda.is_available():
                        temtrees = temtrees.to(DEVICE)
                        temindexes = temindexes.to(DEVICE)
                        torch_costs = (torch.tensor(temcost)).to(DEVICE)
                    # temcostbais = model[num_rels](temquery_feats, temtrees, temindexes).to(DEVICE).add(1)
                    temcostbais = []
                    for i in range(10):
                        with torch.no_grad():
                            temcostbais.append(
                                torch.tanh(model[-1](temquery_feats, temtrees, temindexes).to(DEVICE)).add(1))
                    temcostbais = torch.cat(temcostbais, 1)
                    temcostbais = torch.mean(temcostbais, dim=1)
                    temcostlist = torch.mul(temcostbais, torch_costs).tolist()
                    count = 0
                    for key in temtable:
                        temtable[key] = (temcostlist[count], temtable[key][1])
                        count = count + 1
                sortlist = dict(sorted(temtable.items(), key=lambda x: x[1][0])[math.ceil(len(dp_table) * 0.3):])
                for key in sortlist:
                    dp_table.pop(key)
                dp_tables[level] = dp_table
        bestplanhint = list(dp_tables[num_rels].values())[0][1].hint_str()
        # print(list(dp_tables[num_rels].values())[0][1].info)
        #        if dpsign:
        #            collectSubplans(list(dp_tables[num_rels].values())[0][1], subplans_fin, workload, exp)
        nowlatency = postgres.GetLatencyFromPg(finsql, bestplanhint, verbose=False, check_hint_used=False,
                                               timeout=timeout, dropbuffer=dropbuffer)
        if timeout > nowlatency:
            timeout = nowlatency
        print('*****time:', time_all)
        return trainBuffer, bestplanhint, num, timeout

    def KM_left_prune(self, join_graph, all_join_conds, query_leaves, dp_tables, workload, exp, FirstTrain,
                      model, timeout, dropbuffer, nodeFeaturizer, greedy, subplans_fin, finsql):
        num_rels = len(query_leaves)
        num = 0
        latency = 0
        for i in range(0, num_rels + 1):
            trainBuffer.append([])
        for level in range(2, num_rels + 1):
            dp_table = dp_tables[level]
            bayes_tep = []
            bayes_list = []
            # for level_i in range(1, level):
            # level_j = level - level_i
            dp_table_i = random_dic(dp_tables[level - 1])
            dp_table_j = random_dic(dp_tables[1])
            # limit = int(len(dp_table_i) * 0.2)
            #    nowplansnum = 0
            for l_ids, l_tup in dp_table_i.items():
                for r_ids, r_tup in dp_table_j.items():
                    l = l_tup[1]
                    r = r_tup[1]
                    if not plans_lib.ExistsJoinEdgeInGraph(
                            l, r, join_graph):
                        # No join clause linking two sides.  Skip.
                        continue
                    l_ids_splits = l_ids.split(',')
                    r_ids_splits = r_ids.split(',')
                    if len(np.intersect1d(l_ids_splits, r_ids_splits)) > 0:
                        # A relation exists in both sides.  Skip.
                        continue
                    join_ids = ','.join(sorted(l_ids_splits + r_ids_splits))
                    # Otherwise, form a new join.
                    dp_costs = []
                    dp_query_encodings = []
                    dp_nodes = []
                    dp_hints_sqls = []
                    dp_join = []
                    for join in EnumerateJoinWithOps(
                            l,
                            r,
                            self.join_ops,
                            self.scan_ops,
                            use_plan_restrictions=self.use_plan_restrictions
                    ):
                        join.info["currentLevel"] = level

                        join_conds = join.KeepRelevantJoins(all_join_conds)
                        join.info["join_conds"] = join_conds

                        cost, sql, hint = self.cost_model(join, join_conds)
                        join.info["cost"] = cost
                        logcost = math.log(cost)
                        data = encoding.getencoding_Balsa(sql, hint, workload)
                        join.info["encoding"] = data[0]
                        join.info["node"] = data[1]
                        dp_join.append(join)
                        dp_costs.append(logcost)
                        dp_query_encodings.append(data[0])
                        dp_nodes.append(data[1])
                        dp_hints_sqls.append([hint, sql])
                        tem = []
                        tem.append(logcost)
                        tem.append(sql)
                        tem.append(hint)
                        tem.append(0)
                        tem.append([data[0], data[1]])

                        bayes_tep.append(tem)
                        assert len(data) == 2
                    if not FirstTrain:
                        query_feats = (torch.cat(dp_query_encodings, dim=0)).to(DEVICE)
                        trees, indexes = TreeConvFeaturize(nodeFeaturizer, dp_nodes)
                        if torch.cuda.is_available():
                            trees = trees.to(DEVICE)
                            indexes = indexes.to(DEVICE)
                            torch_dpcosts = (torch.tensor(dp_costs)).to(DEVICE)
                        costbais = []
                        for i in range(10):
                            costbais.append(torch.tanh(model[level](query_feats, trees, indexes).to(DEVICE)).add(1))
                        costbais = torch.cat(costbais, 1)
                        costbais_mean = torch.mean(costbais, dim=1)
                        # cost_min, _ = torch.min(costbais, dim=1)
                        var = torch.sum(torch.pow(costbais - costbais_mean.unsqueeze(1), 2), dim=1)
                        # ucb = var / var.max() - cost_min / cost_min.max()
                        # print('ucb ')
                        # print(ucb)
                        bayes_list.extend(var.tolist())
                        costlist = torch.mul(costbais_mean, torch_dpcosts).tolist()
                    else:
                        costlist = dp_costs
                    #                        for i in range(0,len(costlist)):
                    #                            if (random.random()<greedy):
                    #                                tem = []
                    #                                tem.append(dp_costs[i])
                    #                                tem.append(dp_hints_sqls[i][1])
                    #                                tem.append(dp_hints_sqls[i][0])
                    #                                # # # collect train data (latency)
                    #                                usebuffer = False
                    #                                for j in exp[level]:
                    #                                    if (j[2] == dp_hints_sqls[i][0] and j[1] == dp_hints_sqls[i][1]):
                    #                                        usebuffer = True
                    #                                        glatency = j[3]
                    #                                        # print('use buffer')
                    #                                        break
                    #                                if (usebuffer == False):
                    #                                    num = num + 1
                    #                                    glatency = postgres.GetLatencyFromPg(dp_hints_sqls[i][1], dp_hints_sqls[i][0], verbose=False, check_hint_used=False,
                    #                                                                       timeout=timeout, dropbuffer=dropbuffer)
                    #                                    tem.append(glatency)
                    #                                    tem.append([dp_query_encodings[i],dp_nodes[i]])
                    #                                    exp[level].append(tem)
                    #                                    trainBuffer[level].append(tem)
                    # print(costlist)
                    for i in range(0, len(costlist)):
                        if join_ids not in dp_table or dp_table[join_ids][
                            0] > costlist[i]:
                            if FirstTrain:
                                tem = []
                                tem.append(dp_costs[i])
                                tem.append(dp_hints_sqls[i][1])
                                tem.append(dp_hints_sqls[i][0])
                                # # # collect train data (latency)
                                usebuffer = False
                                for j in exp[level]:
                                    if (j[2] == dp_hints_sqls[i][0] and j[1] == dp_hints_sqls[i][1]):
                                        usebuffer = True
                                        latency = j[3]
                                        dp_nodes[i].info["latency"] = latency
                                        # print('use buffer')
                                        break
                                if (usebuffer == False):
                                    num = num + 1
                                    if ((level > num_rels - 1) and slackTimeout(exp[level])):
                                        latency = postgres.GetLatencyFromPg(dp_hints_sqls[i][1],
                                                                            dp_hints_sqls[i][0], verbose=False,
                                                                            check_hint_used=False, timeout=12000,
                                                                            dropbuffer=dropbuffer)
                                        print("slack")
                                        #                                   else:
                                        #                                          latency =postgres.GetLatencyFromPg(dp_hints_sqls[i][1], dp_hints_sqls[i][0], verbose=False, check_hint_used=False,timeout=timeout*2.0, dropbuffer=dropbuffer)
                                        print(latency, 'level', level)
                                    else:
                                        latency = postgres.GetLatencyFromPg(dp_hints_sqls[i][1],
                                                                            dp_hints_sqls[i][0], verbose=False,
                                                                            check_hint_used=False, timeout=timeout,
                                                                            dropbuffer=dropbuffer)
                                    # latency = random.random()
                                    dp_nodes[i].info["latency"] = latency
                                    tem.append(latency)
                                    tem.append([dp_query_encodings[i], dp_nodes[i]])
                                    exp[level].append(tem)
                                    trainBuffer[level].append(tem)
                                    if level == num_rels:
                                        allPlans = [dp_nodes[i]]
                                        while (allPlans):
                                            currentNode = allPlans.pop()
                                            allPlans.extend(currentNode.children)
                                            for currentChild in currentNode.children:
                                                temlevel = currentChild.info.get("currentLevel")
                                                if (not temlevel == None) and temlevel > 1:
                                                    #  print(currentChild)
                                                    temsql = currentChild.to_sql(currentChild.info["join_conds"],
                                                                                 with_select_exprs=True)
                                                    #    print(temsql)
                                                    temhint = currentChild.hint_str()
                                                    found = False
                                                    for i in subplans_fin:
                                                        if (
                                                                i[1] == dp_hints_sqls[i][1],
                                                                i[2] == dp_hints_sqls[i][0]):
                                                            found = True
                                                            break
                                                    if not found:
                                                        tem = []
                                                        tem.append(currentChild.info["cost"])
                                                        tem.append(temsql)
                                                        tem.append(temhint)
                                                        tem.append(currentChild.info["latency"])
                                                        tem.append([currentChild.info["encoding"], currentChild])
                                                        subplans_fin[temlevel].append(tem)
                            dp_table[join_ids] = (costlist[i], dp_join[i])
            #  nowplansnum = nowplansnum + 1
            #   if nowplansnum > limit and level > 2:
            #   print('plan nums = ',num)
            #    break
            # print('num = ',len(dp_table))
            # todo:根据方差收集数据

            if not FirstTrain:
                # print('bayes_list len :',len(bayes_list))
                # print('bayes tep num :',len(bayes_tep))
                # bayes_list 存储的方差
                bayes_list = torch.tensor(bayes_list)
                # bayes_tep 所有的计划
                # item : [logcost ,sql ,hint ,latency,[queryencoding,nodes]]
                km_query_encodings = []
                km_nodes = []
                for item in bayes_tep:
                    km_query_encodings.append(item[4][0])
                    km_nodes.append(item[4][1])
                query_feats = (torch.cat(km_query_encodings, dim=0)).to(DEVICE)
                # print(query_feats.shape)
                trees, indexes = TreeConvFeaturize(nodeFeaturizer, km_nodes)
                if torch.cuda.is_available():
                    trees = trees.to(DEVICE)
                    indexes = indexes.to(DEVICE)
                final_idxes = []  # selected index
                trees = trees.reshape((trees.shape[0], -1))
                indexes = indexes.reshape((indexes.shape[0], -1))
                pool_encodings = torch.cat([query_feats, trees, indexes], dim=1)
                pool_encodings = pool_encodings.cpu().numpy()
                n_classes = 10
                if n_classes > len(pool_encodings):
                    n_classes = len(pool_encodings)
                kmeans = KMeans(n_clusters=n_classes, random_state=123).fit(pool_encodings)
                # print('pool_encodings num',len(pool_encodings))
                # print('bayes list num ',len(bayes_list))
                #  print('bayes_tep num ',len(bayes_tep))
                labels = list(kmeans.labels_)
                #  print(labels)
                classes = dict()
                if level < 4:
                    p = 0.4
                else:
                    p = 0.15
                for i in range(n_classes):
                    classes[i] = []
                    idxs = []
                    v = []
                    for idx, label in enumerate(labels):
                        if label == i:
                            idxs.append(idx)
                            v.append(bayes_list[idx])
                    v = torch.tensor(v)
                    v_argsort = torch.argsort(v, descending=True)
                    n = math.ceil(p * len(idxs))
                    for j in range(n):
                        final_idxes.append(idxs[v_argsort[j]])

                for i in final_idxes:
                    bayes_plan = bayes_tep[i]
                    usebuffer = False
                    for j in exp[level]:
                        if (j[2] == bayes_plan[2] and j[1] == bayes_plan[1]):
                            usebuffer = True
                            blatency = j[3]
                            break
                    if (usebuffer == False):
                        num = num + 1
                        if ((level > num_rels - 1) and slackTimeout(exp[level])):
                            blatency = postgres.GetLatencyFromPg(bayes_plan[1], bayes_plan[2], verbose=False,
                                                                 check_hint_used=False, timeout=12000,
                                                                 dropbuffer=dropbuffer)
                            print("slack")
                            #                                   else:
                            #                                          latency =postgres.GetLatencyFromPg(dp_hints_sqls[i][1], dp_hints_sqls[i][0], verbose=False, check_hint_used=False,timeout=timeout*2.0, dropbuffer=dropbuffer)
                            print(latency, 'level', level)
                        else:
                            blatency = postgres.GetLatencyFromPg(bayes_plan[1], bayes_plan[2], verbose=False,
                                                                 check_hint_used=False, timeout=timeout,
                                                                 dropbuffer=dropbuffer)
                    bayes_plan[3] = blatency
                    if usebuffer == False:
                        trainBuffer[level].append(copy.deepcopy(bayes_plan))
                        exp[level].append(copy.deepcopy(bayes_plan))

            if level > 4 and level < 15 and level < num_rels:
                temtable = copy.deepcopy(dp_table)
                if not FirstTrain:
                    temcost = []
                    temnodes = []
                    tem_query_encodings = []
                    for key, values in temtable.items():
                        temcost.append(values[0])
                        temnodes.append(values[1].info["node"])
                        tem_query_encodings.append(values[1].info["encoding"])
                    #  print("nodes num = ",len(temnodes))
                    temquery_feats = (torch.cat(tem_query_encodings, dim=0)).to(DEVICE)
                    temtrees, temindexes = TreeConvFeaturize(nodeFeaturizer, temnodes)
                    if torch.cuda.is_available():
                        temtrees = temtrees.to(DEVICE)
                        temindexes = temindexes.to(DEVICE)
                        torch_costs = (torch.tensor(temcost)).to(DEVICE)
                    temcostbais = []
                    for i in range(10):
                        temcostbais.append(
                            torch.tanh(model[-1](temquery_feats, temtrees, temindexes).to(DEVICE)).add(1))
                    temcostbais = torch.cat(temcostbais, 1)
                    temcostbais_mean = torch.mean(temcostbais, dim=1)
                    temcostlist = torch.mul(temcostbais_mean, torch_costs).tolist()
                    count = 0
                    for key in temtable:
                        temtable[key] = (temcostlist[count], temtable[key][1])
                        count = count + 1
                sortlist = dict(sorted(temtable.items(), key=lambda x: x[1][0])[math.ceil(len(dp_table) * 0.45):])
                for key in sortlist:
                    dp_table.pop(key)
                dp_tables[level] = dp_table
        # print('level:',level,'subplans num = ',len(dp_tables[level]))
        # print('level:',level,'exp num = ',len(exp[level]))
        bestplanhint = list(dp_tables[num_rels].values())[0][1].hint_str()
        nowlatency = postgres.GetLatencyFromPg(finsql, bestplanhint, verbose=False, check_hint_used=False,
                                               timeout=timeout, dropbuffer=dropbuffer)
        if timeout > nowlatency:
            timeout = nowlatency

        return trainBuffer, bestplanhint, num, timeout

    def TEST_left_prune_bayes(self, join_graph, all_join_conds, query_leaves, dp_tables, workload, model,
                              nodeFeaturizer, costCache):

        num_rels = len(query_leaves)

        for level in range(2, num_rels + 1):
            dp_table = dp_tables[level]

            dp_table_i = random_dic(dp_tables[level - 1])
            dp_table_j = random_dic(dp_tables[1])

            for l_ids, l_tup in dp_table_i.items():
                for r_ids, r_tup in dp_table_j.items():
                    l = l_tup[1]
                    r = r_tup[1]
                    if not plans_lib.ExistsJoinEdgeInGraph(
                            l, r, join_graph):
                        # No join clause linking two sides.  Skip.
                        continue
                    l_ids_splits = l_ids.split(',')
                    r_ids_splits = r_ids.split(',')
                    if len(np.intersect1d(l_ids_splits, r_ids_splits)) > 0:
                        # A relation exists in both sides.  Skip.
                        continue
                    join_ids = ','.join(sorted(l_ids_splits + r_ids_splits))
                    # Otherwise, form a new join.
                    dp_costs = []
                    dp_query_encodings = []
                    dp_nodes = []
                    dp_hints_sqls = []
                    dp_join = []
                    for join in EnumerateJoinWithOps(
                            l,
                            r,
                            self.join_ops,
                            self.scan_ops,
                            use_plan_restrictions=self.use_plan_restrictions
                    ):
                        join.info["currentLevel"] = level
                        join_conds = join.KeepRelevantJoins(all_join_conds)
                        join.info["join_conds"] = join_conds
                        join.info["join_ids"] = join_ids
                        cost, sql, hint = self.cost_model.getCost_cache(join, join_conds, costCache)
                        join.info["cost"] = cost
                        logcost = math.log(cost)
                        dp_costs.append(logcost)
                        dp_hints_sqls.append([hint, sql])
                        dp_join.append(join)
                        data = encoding.getencoding_Balsa(sql, hint, workload)
                        join.info["encoding"] = data[0]
                        join.info["node"] = data[1]
                        dp_query_encodings.append(data[0])
                        dp_nodes.append(data[1])

                    # assert len(data) == 2
                    # level > num_rels -3
                    costlist = dp_costs
                    if level > num_rels - 4:
                        query_feats = (torch.cat(dp_query_encodings, dim=0)).to(DEVICE)
                        trees, indexes = TreeConvFeaturize(nodeFeaturizer, dp_nodes)
                        if torch.cuda.is_available():
                            trees = trees.to(DEVICE)
                            indexes = indexes.to(DEVICE)
                            torch_dpcosts = (torch.tensor(dp_costs)).to(DEVICE)

                        #  costbais = torch.tanh(model[level](query_feats, trees, indexes).to(DEVICE)).add(1).squeeze(1)

                        costbais = []

                        for i in range(10):
                            with torch.no_grad():
                                costbais.append(
                                    torch.tanh(model[level](query_feats, trees, indexes).to(DEVICE)).add(1).detach())
                        costbais = torch.cat(costbais, 1)
                        costbais = torch.mean(costbais, dim=1)
                        # cost_min, _ = torch.min(costbais, dim=1)
                        #    var = torch.sum(torch.pow(costbais - costbais_mean.unsqueeze(1), 2), dim=1)
                        # ucb = var / var.max() - cost_min / cost_min.max()
                        # print('ucb ')
                        # print(ucb)
                        # bayes_list.extend(var.tolist())
                        costlist = torch.mul(costbais, torch_dpcosts).tolist()

                    for i in range(0, len(costlist)):
                        if join_ids not in dp_table or dp_table[join_ids][
                            0] > costlist[i]:
                            dp_table[join_ids] = (costlist[i], dp_join[i])
            if level > 6 and level < 15 and level < num_rels:
                temtable = copy.deepcopy(dp_table)
                temcost = []
                temnodes = []
                tem_query_encodings = []
                for key, values in temtable.items():
                    temcost.append(values[0])
                    temnodes.append(values[1].info["node"])
                    tem_query_encodings.append(values[1].info["encoding"])
                #  print("nodes num = ",len(temnodes))
                temquery_feats = (torch.cat(tem_query_encodings, dim=0)).to(DEVICE)
                temtrees, temindexes = TreeConvFeaturize(nodeFeaturizer, temnodes)
                if torch.cuda.is_available():
                    temtrees = temtrees.to(DEVICE)
                    temindexes = temindexes.to(DEVICE)
                    torch_costs = (torch.tensor(temcost)).to(DEVICE)
                # temcostbais = torch.tanh(model[num_rels](temquery_feats, temtrees, temindexes).to(DEVICE)).add(1)
                temcostbais = []
                for i in range(10):
                    with torch.no_grad():
                        temcostbais.append(
                            torch.tanh(model[-1](temquery_feats, temtrees, temindexes).to(DEVICE)).add(1))
                temcostbais = torch.cat(temcostbais, 1)
                temcostbais = torch.mean(temcostbais, dim=1)
                temcostlist = torch.mul(temcostbais, torch_costs).tolist()
                count = 0
                for key in temtable:
                    temtable[key] = (temcostlist[count], temtable[key][1])
                    count = count + 1
                sortlist = dict(sorted(temtable.items(), key=lambda x: x[1][0])[math.ceil(len(dp_table) * 0.3):])
                for key in sortlist:
                    dp_table.pop(key)
                dp_tables[level] = dp_table
        bestplanhint = list(dp_tables[num_rels].values())[0][1].hint_str()
        return bestplanhint, list(dp_tables[num_rels].values())[0][1]

    def TEST_left_prune_bayes1(self, join_graph, all_join_conds, query_leaves, dp_tables, workload, model,
                               nodeFeaturizer):
        num_rels = len(query_leaves)

        for level in range(2, num_rels + 1):
            dp_table = dp_tables[level]

            dp_table_i = random_dic(dp_tables[level - 1])
            dp_table_j = random_dic(dp_tables[1])

            for l_ids, l_tup in dp_table_i.items():
                for r_ids, r_tup in dp_table_j.items():
                    l = l_tup[1]
                    r = r_tup[1]
                    if not plans_lib.ExistsJoinEdgeInGraph(
                            l, r, join_graph):
                        # No join clause linking two sides.  Skip.
                        continue
                    l_ids_splits = l_ids.split(',')
                    r_ids_splits = r_ids.split(',')
                    if len(np.intersect1d(l_ids_splits, r_ids_splits)) > 0:
                        # A relation exists in both sides.  Skip.
                        continue
                    join_ids = ','.join(sorted(l_ids_splits + r_ids_splits))
                    # Otherwise, form a new join.
                    dp_costs = []
                    dp_query_encodings = []
                    dp_nodes = []
                    dp_hints_sqls = []
                    dp_join = []
                    for join in EnumerateJoinWithOps(
                            l,
                            r,
                            self.join_ops,
                            self.scan_ops,
                            use_plan_restrictions=self.use_plan_restrictions
                    ):
                        join.info["currentLevel"] = level

                        join_conds = join.KeepRelevantJoins(all_join_conds)
                        join.info["join_conds"] = join_conds
                        cost, sql, hint = self.cost_model(join, join_conds)
                        join.info["cost"] = cost
                        logcost = math.log(cost)
                        data = encoding.getencoding_Balsa(sql, hint, workload)
                        join.info["encoding"] = data[0]
                        join.info["node"] = data[1]
                        dp_join.append(join)
                        dp_costs.append(logcost)
                        dp_query_encodings.append(data[0])
                        dp_nodes.append(data[1])
                        dp_hints_sqls.append([hint, sql])
                        assert len(data) == 2
                    query_feats = (torch.cat(dp_query_encodings, dim=0)).to(DEVICE)
                    trees, indexes = TreeConvFeaturize(nodeFeaturizer, dp_nodes)
                    if torch.cuda.is_available():
                        trees = trees.to(DEVICE)
                        indexes = indexes.to(DEVICE)
                        torch_dpcosts = (torch.tensor(dp_costs)).to(DEVICE)
                    # costbais = torch.tanh(model[level](query_feats, trees, indexes).to(DEVICE)).add(1).squeeze(1)

                    costbais = []
                    for i in range(10):
                        costbais.append(
                            torch.tanh(model[level](query_feats, trees, indexes).to(DEVICE)).add(1).detach())
                    costbais = torch.cat(costbais, 1)
                    costbais = torch.mean(costbais, dim=1)
                    # cost_min, _ = torch.min(costbais, dim=1)
                    #    var = torch.sum(torch.pow(costbais - costbais_mean.unsqueeze(1), 2), dim=1)
                    # ucb = var / var.max() - cost_min / cost_min.max()
                    # print('ucb ')
                    # print(ucb)
                    # bayes_list.extend(var.tolist())
                    costlist = torch.mul(costbais, torch_dpcosts).tolist()

                    for i in range(0, len(costlist)):
                        if join_ids not in dp_table or dp_table[join_ids][
                            0] > costlist[i]:
                            dp_table[join_ids] = (costlist[i], dp_join[i])
            if level > 4 and level < 15 and level < num_rels:
                temtable = copy.deepcopy(dp_table)
                temcost = []
                temnodes = []
                tem_query_encodings = []
                for key, values in temtable.items():
                    temcost.append(values[0])
                    temnodes.append(values[1].info["node"])
                    tem_query_encodings.append(values[1].info["encoding"])
                #  print("nodes num = ",len(temnodes))
                temquery_feats = (torch.cat(tem_query_encodings, dim=0)).to(DEVICE)
                temtrees, temindexes = TreeConvFeaturize(nodeFeaturizer, temnodes)
                if torch.cuda.is_available():
                    temtrees = temtrees.to(DEVICE)
                    temindexes = temindexes.to(DEVICE)
                    torch_costs = (torch.tensor(temcost)).to(DEVICE)
                # temcostbais = torch.tanh(model[num_rels](temquery_feats, temtrees, temindexes).to(DEVICE)).add(1)
                temcostbais = []
                for i in range(10):
                    temcostbais.append(torch.tanh(model[-1](temquery_feats, temtrees, temindexes).to(DEVICE)).add(1))
                temcostbais = torch.cat(temcostbais, 1)
                temcostbais = torch.mean(temcostbais, dim=1)
                temcostlist = torch.mul(temcostbais, torch_costs).tolist()
                count = 0
                for key in temtable:
                    temtable[key] = (temcostlist[count], temtable[key][1])
                    count = count + 1
                sortlist = dict(sorted(temtable.items(), key=lambda x: x[1][0])[math.ceil(len(dp_table) * 0.3):])
                for key in sortlist:
                    dp_table.pop(key)
                dp_tables[level] = dp_table
        bestplanhint = list(dp_tables[num_rels].values())[0][1].hint_str()
        return bestplanhint

    def _dp_dbmsx_search_space(self, original_node, join_graph, all_join_conds,
                               query_leaves, dp_tables):
        """For Dbmsx."""
        raise NotImplementedError

    def isloadTraindata(self, level, hint):
        flag = True
        for i in trainBuffer[level]:
            if hint == i[2]:
                return False
        return flag


if __name__ == '__main__':
    print('')
