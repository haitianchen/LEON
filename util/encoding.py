import sqlparse
import torch

from util import postgres, pg_executor, plans_lib, treeconv


def getCostbais(sql, hint=None):
    sql = sqlparse.format(sql, reindent=True, keyword_case="upper")
    SQL_encode = pq.travesal_SQL(sql)
    query_encode = SQL_encode
    query_encode = replace_alias(query_encode)
    query_vector = get_vector(query_encode)
    result = postgres.getPlans(sql, hint, verbose=False, check_hint_used=False)[0][0][0]
    node = traversal_plan_tree_cost(result['Plan'], 17, query_vector)[2][-1]
    return node


def TreeConvFeaturize(plan_featurizer, subplans):
    """Returns (featurized plans, tree conv indexes) tensors."""
    assert len(subplans) > 0
    trees, indexes = treeconv.make_and_featurize_trees(subplans,
                                                       plan_featurizer)
    return trees, indexes


def getencoding_Balsa(sql, hint, workload):
    with pg_executor.Cursor() as cursor:
        node0 = postgres.SqlToPlanNode(sql, comment=hint, verbose=False,
                                       cursor=cursor)[0]
    node = plans_lib.FilterScansOrJoins([node0])[0]
    node.info['sql_str'] = sql
    plans_lib.GatherUnaryFiltersInfo(node)
    postgres.EstimateFilterRows(node)
    queryFeaturizer = plans_lib.QueryFeaturizer(workload.workload_info)
    query_vecs = torch.from_numpy(queryFeaturizer(node)).unsqueeze(0)
    if torch.cuda.is_available():
        return [query_vecs.cuda(), node]
    return [query_vecs, node]


if __name__ == '__main__':
    # N.model.train()
    with open('../join-order-benchmark/1a.sql', "r") as f:
        data = f.readlines()
        sql0 = ' '.join(data)
    bais = getCostbais(sql0)
    print(bais)
