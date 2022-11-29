import copy
import datetime
import logging
import math
import os
import pickle
import random
import time
import DP
from util import encoding
import numpy as np
import torch
from torch import nn

from util import treeconv, postgres, envs

DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'


def getexpnum(exp):
    num = 0
    for i in exp:
        num = num + len(i)
    return num


def getNodesNum(nodes):
    num = 0
    for i in nodes:
        num = num + len(i)
    return num


def slackTimeout(exp):
    num = 0
    for i in exp:
        if not (i.info['latency'] == 90000):
            num = num + 1
        if num > 1:
            return False
    return True


def get_logger(filename, verbosity=1, name=None):
    filename = filename + '.txt'
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def setInitialTimeout(sqls: list, dropbuffer, testtime=3):
    """
    :param sqls: list of sql string
    :return: timeout list
    """
    timeoutlist = []
    for i in sqls:
        tem1 = 0
        for j in range(0, testtime):
            _, _, latency = postgres.GetLatencyFromPg(i, None, verbose=False, check_hint_used=False, timeout=90000,
                                                      dropbuffer=dropbuffer, ml_clf_off=True)
            tem1 = tem1 + latency
        timeout = tem1 / float(testtime)
        timeoutlist.append(round(timeout, 3))
    return timeoutlist


def load_sql_Files(sql_list: list):
    """
    :param sql_list: list of sql template name
    :return: list of path of sql query file path
    """
    sqllist = []
    for i in range(0, len(sql_list)):
        sqlFiles = '/data0/chenx/cht/join-order-benchmark/' + sql_list[i] + '.sql'
        if not os.path.exists(sqlFiles):
            raise IOError("File Not Exists!")
        sqllist.append(sqlFiles)
    return sqllist


def load_sql(sql_list: list):
    """
    :param sql_list: list of sql file path
    :return: list of sql query string
    """
    sqls = []
    for i in sql_list:
        with open(i, 'r') as f:
            data = f.read().splitlines()
            sql = ' '.join(data)
        sqls.append(sql)
        f.close()
    return sqls


def calculateLossForBatch(latencies: list, costs: list, calibration: torch.Tensor):
    """
    :param latencies: real latency
    :param costs: PG estimated cost
    :param calibration: ML model's calibration for cost
    :return: loss to learn the ML model
    """
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    calibration = calibration.view(-1, 2)
    costs = torch.tensor(costs, device=DEVICE).view(-1, 2)
    calibratedCost = calibration * costs
    softm = nn.functional.softmax(calibratedCost, dim=1)
    assert (2 * len(costs) == len(latencies)) and (len(latencies) % 2 == 0)
    res = []
    for i in range(0, len(latencies), 2):
        if latencies[i] > latencies[i + 1]:
            res.append(0)
        else:
            res.append(1)
    res = torch.tensor(res, device=DEVICE)
    return loss_fn(softm, res)


def load_nodes(nodesPath):
    if not os.path.exists(nodesPath):
        raise IOError("nodes files Not Exists!", nodesPath)
    else:
        nodesFile = open(nodesPath, 'rb')
        nodes = pickle.load(nodesFile)
        nodesFile.close()
        return nodes
    return None


def isTheNodeRep(node, nodesList):
    for i in nodesList:
        if i.info['sql_str'] == node.info['sql_str'] and i.info['hint'] == node.info['hint']:
            return True
    return False


def getNodesEncoding(nodes, nodeFeaturizer, queryFeaturizer):
    queryencoding = []
    for i in nodes:
        tem = torch.from_numpy(queryFeaturizer(i)).unsqueeze(0)
        queryencoding.append(tem)
        i.info['query_encoding'] = copy.deepcopy(tem)
    trees, indexes = encoding.TreeConvFeaturize(nodeFeaturizer, nodes)
    tensor_query_encoding = (torch.cat(queryencoding, dim=0))
    return trees.to(DEVICE), indexes.to(DEVICE), tensor_query_encoding.to(DEVICE)


def pick_nodes(nodes, expnodes, FirstTrain, models, nodeFeaturizer, queryFeaturizer, timeoutlist, trainquery,
               epsilon=0.2):
    newTrainnodes = [[] for _ in range(20)]
    for i in range(2, len(nodes)):
        explor = False
        nowlevNodesLimit = int(getNodesNum(list(nodes[i].values())) * 0.3) if int(
            getNodesNum(list(nodes[i].values())) * 0.3) > 2 else 2
        # print(nowlevNodesLimit)
        nodenums = 0
        for k, v in nodes[i].items():
            random.shuffle(v)
            dp_cost = []
            for node in v:
                dp_cost.append(math.log(node.cost))
            torch_dpcosts = (torch.tensor(dp_cost)).to(DEVICE)
            trees, indexes, queryencoding = getNodesEncoding(v, nodeFeaturizer, queryFeaturizer)
            costbais = torch.tanh(models[i](queryencoding, trees, indexes).to(DEVICE)).add(1).squeeze(1)
            costlist = torch.mul(costbais, torch_dpcosts).tolist()
            bestcost = float('inf')
            if nodenums > nowlevNodesLimit:
                break
            for index in range(len(costlist)):
                if costlist[index] < bestcost:
                    bestcost = costlist[index]
                    if (nodenums < nowlevNodesLimit) and (not isTheNodeRep(v[index], expnodes[v[index].info['level']])):
                        timeout = 0
                        for timeIndex in range(len(trainquery)):
                            if v[index].info['sqlname'] == trainquery[timeIndex]:
                                timeout = timeoutlist[timeIndex]
                                if FirstTrain:
                                    timeout = timeout * 3.0
                                # print(slackTimeout(newTrainnodes[i]))
                                if slackTimeout(newTrainnodes[i]):
                                    timeout = 12000.0 if 12000.0 > timeout * 15.0 else timeout * 15.0
                                break
                        _, _, Latency = postgres.GetLatencyFromPg(v[index].info['sql_str'], v[index].info['hint'],
                                                                  verbose=False, check_hint_used=False,
                                                                  timeout=timeout, gethint=False,
                                                                  dropbuffer=False, ml_clf_off=True)
                        nodenums = nodenums + 1
                        # print(Latency)
                        if Latency < timeout:
                            explor = False
                        v[index].info['latency'] = Latency
                        newTrainnodes[i].append(v[index])
                        expnodes[i].append(v[index])
                else:
                    if FirstTrain and explor and random.random() < epsilon:
                        if not isTheNodeRep(v[index], expnodes[v[index].info['level']]):
                            timeout = 0
                            for timeIndex in range(len(trainquery)):
                                if v[index].info['sqlname'] == trainquery[timeIndex]:
                                    timeout = timeoutlist[timeIndex]
                                    if FirstTrain:
                                        timeout = timeout * 3.0
                                    break
                            _, _, Latency = postgres.GetLatencyFromPg(v[index].info['sql_str'], v[index].info['hint'],
                                                                      verbose=False, check_hint_used=False,
                                                                      timeout=timeout, gethint=False,
                                                                      dropbuffer=False, ml_clf_off=True)
                            v[index].info['latency'] = Latency
                            newTrainnodes[i].append(v[index])
                            expnodes[i].append(v[index])
    return newTrainnodes


def getTrainpair(newTrainnodes, expnodes, trainpairs):
    for i in range(2, len(newTrainnodes)):
        for newnode in newTrainnodes[i]:
            for expnode in expnodes[i]:
                if newnode.info['latency'] == expnode.info['latency']:
                    continue
                if newnode.info['sql_str'] == expnode.info['sql_str'] and newnode.info['hint'] == expnode.info['hint']:
                    continue
                tem = []
                tem.append((newnode.info['query_encoding'], newnode))
                tem.append(newnode.info['latency'])
                tem.append(math.log(newnode.cost))
                tem.append((expnode.info['query_encoding'], expnode))
                tem.append(expnode.info['latency'])
                tem.append(math.log(expnode.cost))
                trainpairs[i].append(tem)


def getmodels(maxlevel, modelpath):
    models = [[] for _ in range(maxlevel + 1)]
    for level in range(2, maxlevel + 1):
        if not os.path.exists(modelpath + str(level) + '.pth'):
            model = treeconv.TreeConvolution(820, 123, 1).to(DEVICE if torch.cuda.is_available() else 'cpu')
            torch.save(model, modelpath + str(level) + '.pth')
        else:
            model = torch.load(modelpath + str(level) + '.pth').to(DEVICE if torch.cuda.is_available() else 'cpu')
        models[level] = model
    return models


if __name__ == '__main__':

    # delete experience
    #path  : node.pkl
    if os.path.exists('./node.pkl'):
        os.remove('./node.pkl')
    logs_name = 'pg_dp'
    ISOTIMEFORMAT = '%m%d-%H%M%S'
    config = {'log_path': 'log_path '}
    timestamp = str(datetime.datetime.now().strftime(ISOTIMEFORMAT))
    loglogs = '_'.join((logs_name, timestamp))
    log_dir = os.path.join(config['log_path'], loglogs)
    os.makedirs(log_dir)
    log_file_name = os.path.join(log_dir, "running_log")
    logger = get_logger(log_file_name)
    logger.info(config)
    allstime = time.time()
    # log_dir path
    with open("./log_dir.txt", 'w') as file:
        file.write(log_dir)
        file.close()
    # workload path
    workloadpath = './workload.pkl'
    if not os.path.exists(workloadpath):
        workload = envs.JoinOrderBenchmark(envs.JoinOrderBenchmark.Params())
        workload.workload_info.table_num_rows = postgres.GetAllTableNumRows(
            workload.workload_info.rel_names)
        workloadfile = open(workloadpath, "wb")
        pickle.dump(workload, workloadfile)
        workloadfile.close()
    else:
        workloadfile = open(workloadpath, 'rb')
        workload = pickle.load(workloadfile)
    from util import plans_lib

    nodeFeaturizer = plans_lib.PhysicalTreeNodeFeaturizer(workload.workload_info)
    queryFeaturizer = plans_lib.QueryFeaturizer(workload.workload_info)
    batchsize = 1024
    FirstTrain = True
    dropbuffer = False
    # train queries

    trainquery = ['7a']
    sqllist = load_sql_Files(trainquery)
    sqls = load_sql(sqllist)
    logger.info("Train SQL List {}".format(sqllist))
    iteration_num = 15
    # initial timeout and it will update in dp
    timeoutlist = setInitialTimeout(sqls, dropbuffer, testtime=3)
    logger.info("timeoutList:{}".format(timeoutlist))
    expnodes = [[] for _ in range(20)]
    trainpair = [[] for _ in range(20)]
    bestplanslist = [[] for _ in range(len(sqls))]
    epsilon = 0.2
    maxlevel = 0
    for i in range(0, len(sqls)):
        join_graph, all_join_conds, query_leaves, origin_dp_tables = DP.getPreCondition(sqllist[i])
        maxlevel = maxlevel if maxlevel > len(query_leaves) else len(query_leaves)
    tempath = log_dir + '/model_'
    models = getmodels(maxlevel, tempath)
    for iter in range(0, iteration_num):
        logger.info('iter {} start!'.format(str(iter)))
        stime = time.time()
        for i in range(0, len(sqls)):
            # print(sqls[i])
            # sqlname path  uesed for store the sqlname, to communicate with postgreSQL
            with open("./sqlname.txt", 'w') as nowsql:
                nowsql.write(trainquery[i])
                nowsql.close()
            ##update timeout
            sign, besthint, sqlLatency = postgres.GetLatencyFromPg(sqls[i], None, verbose=False, check_hint_used=False,
                                                                   timeout=0, gethint=True, dropbuffer=dropbuffer,
                                                                   ml_clf_off=False)
            if sign == 0:
                raise ("pg error")
            bestplanslist[i].append([besthint])
            # logger.info("sqllatency:{}".format(sqlLatency))
            if sqlLatency < timeoutlist[i]:
                timeoutlist[i] = sqlLatency
        logger.info("dptime = {}".format(time.time() - stime))
        logger.info('now timeoutlist = {}'.format(timeoutlist))
        learning_rate = 1e-3
        # node.pkl path
        nodes = load_nodes('./node.pkl')
        pickstratTime = time.time()
        ##pick nodes
        newTrainnodes = pick_nodes(nodes, expnodes, FirstTrain, models, nodeFeaturizer, queryFeaturizer, timeoutlist,
                                   trainquery, epsilon=0.25)
        logger.info("PickNodestime = {}".format(time.time() - pickstratTime))
        getTrainpair(newTrainnodes, expnodes, trainpair)
        loss_fn = nn.CrossEntropyLoss()
        FirstTrain = False
        logger.info('Train start ,iter ={} '.format(iter))
        logger.info('trainpair num ={},new trainnodes num = {}'.format(getexpnum(trainpair), getexpnum(newTrainnodes)))
        ttime = time.time()
        for modelnum in range(2, maxlevel + 1):
            optimizer = torch.optim.AdamW(models[modelnum].parameters(), lr=learning_rate)
            if len(trainpair[modelnum]) < 1:
                continue
            for epoch in range(0, 10000):
                shuffled_indices = np.random.permutation(len(trainpair[modelnum]))
                # train
                current_idx = 0
                while current_idx <= len(shuffled_indices):
                    currentTrainPair = [trainpair[modelnum][idx] for idx in
                                        shuffled_indices[current_idx: current_idx + batchsize]]
                    query_feats = []
                    nodes = []
                    latencies = []
                    costs = []
                    for i in currentTrainPair:
                        query_feats.append(i[0][0])
                        query_feats.append(i[3][0])
                        nodes.append(i[0][1])
                        nodes.append(i[3][1])
                        latencies.append(i[1])
                        latencies.append(i[4])
                        costs.append(i[2])
                        costs.append(i[5])
                    query_feats = (torch.cat(query_feats, dim=0)).to(DEVICE)
                    trees, indexes = encoding.TreeConvFeaturize(nodeFeaturizer, nodes)
                    if torch.cuda.is_available():
                        trees = trees.to(DEVICE)
                        indexes = indexes.to(DEVICE)
                    calibration = torch.tanh(models[modelnum](query_feats, trees, indexes).to(DEVICE)).add(1)
                    temloss = calculateLossForBatch(latencies, costs, calibration)
                    #  reg =torch.mean(((calibration.sub(1).mul(calibration.sub(1)))*gamma).squeeze(1), 0)
                    losslist = temloss.tolist()
                    loss = torch.mean(temloss, 0)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    current_idx += batchsize

                cout = len(shuffled_indices)
                acc = 0
                current_idx = 0
                #  test
                while current_idx <= len(shuffled_indices):
                    currentTrainPair = [trainpair[modelnum][idx] for idx in
                                        shuffled_indices[current_idx: current_idx + batchsize]]
                    query_feats = []
                    nodes = []
                    latencies = []
                    costs = []
                    for i in currentTrainPair:
                        query_feats.append(i[0][0])
                        query_feats.append(i[3][0])
                        nodes.append(i[0][1])
                        nodes.append(i[3][1])
                        latencies.append(i[1])
                        latencies.append(i[4])
                        costs.append(i[2])
                        costs.append(i[5])
                    query_feats = (torch.cat(query_feats, dim=0)).to(DEVICE)
                    trees, indexes = encoding.TreeConvFeaturize(nodeFeaturizer, nodes)
                    if torch.cuda.is_available():
                        trees = trees.to(DEVICE)
                        indexes = indexes.to(DEVICE)
                    calibration = torch.tanh(models[modelnum](query_feats, trees, indexes).to(DEVICE)).add(1)
                    calibration = calibration.view(-1, 2)
                    costs = torch.tensor(costs, device=DEVICE).view(-1, 2)
                    calibratedCost = calibration * costs
                    softm = nn.functional.softmax(calibratedCost, dim=1)
                    prediction = torch.max(softm, dim=1)[1]
                    res = []
                    for i in range(0, len(latencies), 2):
                        if latencies[i] > latencies[i + 1]:
                            res.append(0)
                        else:
                            res.append(1)
                    res = torch.tensor(res, device=DEVICE)
                    current_idx += batchsize
                    acc += torch.sum(res == prediction).data.cpu().numpy().squeeze()
                logger.info("iter:{},model:{},train iters：{}，acc:{} ".format(iter, modelnum, epoch + 1, acc / cout))
                if acc / cout > 0.96 or epoch > 10:
                    modelname = log_dir + '/model_' + str(modelnum) + '.pth'
                    torch.save(models[modelnum], modelname)
                    break
        logger.info('train time ={}'.format(time.time() - ttime))
        a_file = open(log_dir + '/Bestplans_' + logs_name + '.pkl', 'wb')
        pickle.dump(bestplanslist, a_file)
        a_file.close()
    logger.info('all time = {} '.format(time.time() - allstime))
