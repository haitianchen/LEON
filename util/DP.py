import collections

import reSQL
from util import plans_lib, postgres, search

sqlFiles = ''
p = search.DynamicProgramming.Params()
dp = p.cls(p)
join_ops = ['NestLoop', 'HashJoin', 'MergeJoin']
scan_ops = ['SeqScan', 'IndexScan']
dp.SetPhysicalOps(join_ops=join_ops, scan_ops=scan_ops)


def oneSqlDP(sqlFiles, model, exp):
    sql = ''
    with open(sqlFiles, 'r') as f:
        data = f.read().splitlines()
        sql = ' '.join(data)
    rootNode = plans_lib.Node('root', cost=0)
    rootNode.info['sql_str'] = sql
    joinGraph, _ = rootNode.GetOrParseSql()
    tableAndAlias = getTableAndAlias(sql)
    Jointables = getJoinTable(joinGraph.nodes, tableAndAlias)

    andlist = reSQL.getFliters(sqlFiles)
    selectlist = reSQL.getSelectExp(sqlFiles)
    for i in Jointables:
        temnode = plans_lib.Node('Scan', table_name=i[0], cost=0).with_alias(i[1])
        temandlist = []
        temselelist = []
        for j in andlist:
            if i[1] == j.split('.')[0] or i[1] == j.split('.')[0].replace('(', ''):
                temandlist.append(j)
        if len(temandlist) > 0:
            temnode.info['filter'] = ' AND '.join(temandlist)
        for j in selectlist:
            if i[1] == j[j.index('(') + 1:j.index(')')].split('.')[0]:
                temselelist.append(j)
        if len(temselelist) > 0:
            temnode.info['select_exprs'] = ' , '.join(temselelist)
        rootNode.children.append(temnode)
    num = dp.Run(rootNode, rootNode.info['sql_str'], model, exp)
    return num


def getPreCondition(sqlFiles):
    with open(sqlFiles, 'r') as f:
        data = f.read().splitlines()
        sql = ' '.join(data)
    rootNode = plans_lib.Node('root', cost=0)
    rootNode.info['sql_str'] = sql
    joinGraph, _ = rootNode.GetOrParseSql()
    tableAndAlias = getTableAndAlias(sql)
    Jointables = getJoinTable(joinGraph.nodes, tableAndAlias)
    andlist = reSQL.getFliters(sqlFiles)
    selectlist = reSQL.getSelectExp(sqlFiles)
    for i in Jointables:
        temnode = plans_lib.Node('Scan', table_name=i[0], cost=0).with_alias(i[1])
        temandlist = []
        temselelist = []
        for j in andlist:
            if i[1] == j.split('.')[0] or i[1] == j.split('.')[0].replace('(', ''):
                temandlist.append(j)
        if len(temandlist) > 0:
            temnode.info['filter'] = ' AND '.join(temandlist)
        for j in selectlist:
            if i[1] == j[j.index('(') + 1:j.index(')')].split('.')[0]:
                temselelist.append(str(j))
        if len(temselelist) > 0:
            temnode.info['select_exprs'] = ' , '.join(temselelist)
        rootNode.children.append(temnode)
    join_graph, all_join_conds = rootNode.GetOrParseSql()
    assert len(join_graph.edges) == len(all_join_conds)
    # Base tables to join.
    query_leaves = rootNode.CopyLeaves()
    dp_tables = collections.defaultdict(dict)  # level -> dp_table
    # Fill in level 1.
    for leaf_node in query_leaves:
        leaf_node.info["currentLevel"] = 1
        dp_tables[1][leaf_node.table_alias] = (0, leaf_node)
    return join_graph, all_join_conds, query_leaves, dp_tables


def getJoinTable(someAlias, tableAndAlias):
    Jointables = []
    for j in someAlias:
        for i in tableAndAlias:
            if (j == i[1]):
                Jointables.append(i)
                break
    return Jointables


def getTableAndAlias(sql):
    begin = sql.index('FROM')
    end = sql.index('WHERE')
    Table_alias = []
    for i in sql[begin:end].replace('FROM', '').split(','):
        tabAli = i.split('AS')
        t = (tabAli[0].strip(), tabAli[1].strip())
        Table_alias.append(t)
    return Table_alias


if __name__ == '__main__':
    sqlFiles = './join-order-benchmark/1a.sql'
    print("sqlFile is:", sqlFiles)
    usebuffer = False
    with open(sqlFiles, 'r') as f:
        data = f.read().splitlines()
        sql = ' '.join(data)
    # get init actual time
    timeout = postgres.GetLatencyFromPg(sql, None, verbose=False, check_hint_used=False, timeout=90000,
                                        dropbuffer=usebuffer)
