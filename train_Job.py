import os
import torch
import pickle
from torch import nn
import numpy as np
from util import postgres, envs, treeconv_dropout, DP
import copy
import gc
import time
import datetime
import logging
from util.encoding import TreeConvFeaturize

import random
import math


def getexpnum(exp):
    num = 0
    for i in exp:
        num = num + len(i)
    return num


def seed_torch(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def getTrainPair(output1, output2, trainpair):
    for i in range(0, len(output1)):
        if len(output1[i]) == 0 or len(output2[i]) == 0:
            continue
        for j in output1[i]:
            for k in output2[i]:
                if (j[2] == k[2]) and (j[1] == k[1]):
                    #  print('equal')
                    continue
                if (j[3] == k[3]):
                    #  print('equal')
                    continue
                if(j[6] != k[6]):
                    continue
                tem = []
                # encoding
                tem.append(j[4])
                # latency
                tem.append(j[3])
                # cost
                tem.append(j[0])
                tem.append(k[4])
                tem.append(k[3])
                tem.append(k[0])
                trainpair[i].append(tem)


def getBestTrainPair(output1, output2):
    trainpair = [[] for _ in range(20)]
    for i in range(0, len(output1)):
        if len(output1[i]) == 0 or len(output2[i]) == 0:
            continue
        for j in output1[i]:
            for k in output2[i]:
                if (j[2] == k[2]) and (j[1] == k[1]):
                    #  print('equal')
                    continue
                if (j[3] == k[3]):
                    #  print('equal')
                    continue
                tem = []
                # encoding
                tem.append(j[4])
                # latency
                tem.append(j[3])
                # cost
                tem.append(j[0])
                tem.append(k[4])
                tem.append(k[3])
                tem.append(k[0])
                trainpair[i].append(tem)
    return trainpair


def getPartTrainPair(alltrainpair, rate=0.2):
    if len(alltrainpair) == 0:
        return []
    sortlist = sorted(alltrainpair, key=lambda x: x[6], reverse=True)
    alltrainpair = sortlist[int(len(alltrainpair) * rate):]
    return sortlist[0:int(len(alltrainpair) * rate)]


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

def calculateLossForBatch_MR_0(latencies: list, costs: list, calibration: torch.Tensor):
    """
    :param latencies: real latency
    :param costs: PG estimated cost
    :param calibration: ML model's calibration for cost
    :return: loss to learn the ML model
    """
    loss_fn = nn.MarginRankingLoss()
    calibration = calibration.view(-1, 2)
    costs = torch.tensor(costs, device=DEVICE).view(-1, 2)
    calibratedCost = calibration * costs
    c1,c2 = torch.chunk(calibratedCost,2,dim=1)
    c1 =torch.squeeze(c1)
    c2 =torch.squeeze(c2)
    assert (2 * len(costs) == len(latencies)) and (len(latencies) % 2 == 0)
    res = []
    for i in range(0, len(latencies), 2):
        if latencies[i] > latencies[i + 1]:
            res.append(1)
        else:
            res.append(-1)
    res = torch.tensor(res, device=DEVICE)
    return loss_fn(c1,c2,res)


def geometric_mean(data):  # 计算几何平均数
    total = 1
    for i in data:
        total *= i  # 等同于total=total*i
    return pow(total, 1 / len(data))


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


def load_sql_Files(sql_list: list):
    """
    :param sql_list: list of sql template name
    :return: list of path of sql query file path
    """
    sqllist = []
    for i in range(0, len(sql_list)):
        sqlFiles = 'join-order-benchmark/' + sql_list[i] + '.sql'
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


def collects(finnode, workload, exp, timeout):
    allPlans = [finnode]
    # print('collect')
    currentChild = finnode
    temlevel = currentChild.info.get("currentLevel")
    if (not temlevel == None) and temlevel > 1:

        temsql = currentChild.to_sql(currentChild.info["join_conds"],
                                     with_select_exprs=True)

        temhint = currentChild.hint_str()
        found = False
        for i in exp[temlevel]:
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
                nodelatency = postgres.GetLatencyFromPg(temsql, temhint, verbose=False, check_hint_used=False,
                                                        timeout=timeout, dropbuffer=False)
                tem.append(nodelatency)
                tem.append([currentChild.info["encoding"], currentChild.info["node"]])
                tem.append(currentChild)
                tem.append(currentChild.info["join_ids"])
                exp[temlevel].append(copy.deepcopy(tem))
            else:
                tem.append(nodelatency)
                tem.append([currentChild.info["encoding"], currentChild.info["node"]])
                tem.appedn(currentChild)
                tem.append(currentChild.info["join_ids"])
                exp[temlevel].append(copy.deepcopy(tem))
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
                for i in exp[temlevel]:
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
                        nodelatency = postgres.GetLatencyFromPg(temsql, temhint, verbose=False, check_hint_used=False,
                                                                timeout=timeout, dropbuffer=False)
                        tem.append(nodelatency)
                        tem.append([currentChild.info["encoding"], currentChild.info["node"]])
                        tem.append(currentChild)
                        tem.append(currentChild.info["join_ids"])
                        exp[temlevel].append(copy.deepcopy(tem))
                    else:
                        tem.append(nodelatency)
                        tem.append([currentChild.info["encoding"], currentChild.info["node"]])
                        tem.append(currentChild)
                        tem.append(currentChild.info["join_ids"])
                        exp[temlevel].append(copy.deepcopy(tem))


def getGMRL(sqls, modellist, pg_latency, nodeFeaturizer, costCache, workload, exp=None, old=None):
    sql_ = load_sql(load_sql_Files(sqls))
    hints = []
    alllatency = []
    nodes = []
    for i in sqls:
        join_graph, all_join_conds, query_leaves, origin_dp_tables = DP.getPreCondition(
            'join-order-benchmark/' + i + '.sql')
        # TEST_left_prune_bayes
        bestplanhint, finnode = DP.dp.TEST_left_prune_bayes(join_graph, all_join_conds, query_leaves, origin_dp_tables,
                                                            workload,
                                                            modellist, nodeFeaturizer, costCache)
        hints.append(bestplanhint)
        nodes.append(finnode)
    for i in range(0, len(sql_)):
        tem = 0
        for j in range(0, 3):
            latency = postgres.GetLatencyFromPg(sql_[i], hints[i], verbose=False, check_hint_used=False,
                                                timeout=90000, dropbuffer=dropbuffer)
            tem = tem + latency
        print(sqls[i], tem / 3.0, pg_latency[i], (tem / 3.0) / pg_latency[i])
        alllatency.append((tem / 3.0) / pg_latency[i])
    if old != None:
       for i in range(len(sqls)):
           if alllatency[i] > 1.4:
               print('degradation collect')
               collects(finnode, workload, exp, old[i])
    return geometric_mean(alllatency)


def getModels(maxLevel):
    modellist = ['blank', 'blank']
    optlist = ['blank', 'blank']
    for i in range(maxLevel - 1):
        modellist.append(treeconv_dropout.TreeConvolution(820, 123, 1).to(DEVICE))
        optlist.append(torch.optim.AdamW(modellist[i + 2].parameters(),  lr=0.001))
    return modellist, optlist


def getModelsFromFile(maxLevel, modelpath):
    modellist = ['blank', 'blank']
    optlist = ['blank', 'blank']
    for i in range(2, maxLevel + 1):
        if not os.path.exists(modelpath + str(i) + '.pth'):
            model = treeconv_dropout.TreeConvolution(820, 123, 1).to(DEVICE)
            optlist.append(torch.optim.AdamW(model.parameters(), lr=0.001))
        else:
            model = torch.load(modelpath + str(i) + '.pth').to(DEVICE if torch.cuda.is_available() else 'cpu')
            optlist.append(torch.optim.AdamW(model.parameters(), lr=0.001))
            print('load succssful ', i)
        modellist.append(model)
    return modellist, optlist


def setInitialTimeout(sqls: list, dropbuffer, testtime=3):
    """
    :param sqls: list of sql string
    :return: timeout list
    """
    timeoutlist = []
    for i in sqls:
        tem1 = 0
        for j in range(0, testtime):
            tem1 = tem1 + postgres.GetLatencyFromPg(i, None, verbose=False, check_hint_used=False, timeout=90000,
                                                    dropbuffer=dropbuffer)
        timeout = tem1 / float(testtime)
        timeoutlist.append(round(timeout, 3))
    return timeoutlist


def getPG_latency(sqls):
    pg_latency = []
    for i in sqls:
        tem1 = 0
        for j in range(0, 3):
            tem1 = tem1 + postgres.GetLatencyFromPg(i, None, verbose=False, check_hint_used=False, timeout=90000,
                                                    dropbuffer=False)
        latency = tem1 / 3.0
        pg_latency.append(latency)
    return pg_latency


def getdistribution(data, model):
    # [dp_costs,dp_query_encodings,dp_nodes]
    dp_costs = data[0]
    dp_query_encodings = data[1]
    dp_nodes = data[2]
    query_feats = (torch.cat(dp_query_encodings, dim=0)).to(DEVICE)
    trees, indexes = TreeConvFeaturize(nodeFeaturizer, dp_nodes)
    if torch.cuda.is_available():
        trees = trees.to(DEVICE)
        indexes = indexes.to(DEVICE)
        torch_dpcosts = (torch.tensor(dp_costs)).to(DEVICE)
    costbais = []
    for i in range(10):
        costbais.append(torch.tanh(model(query_feats, trees, indexes).to(DEVICE)).add(1))
    costbais = torch.cat(costbais, 1)
    costbais_mean = torch.mean(costbais, dim=1)
    costlist = torch.mul(costbais_mean, torch_dpcosts)
    distri = costlist / (torch.tensor(0) - torch.sum(costlist))
    return distri


def getOriDistribution(leveldict, model, oridistribution):
    for k, v in leveldict.items():
        oridistribution[k] = getdistribution(v, model).detach()


def getKL(ori_dis, newdis_tir):
    # KL = scipy.stats.entropy(ori_dis, newdis_tir)
    KL = torch.nn.functional.kl_div(newdis_tir.softmax(dim=-1).log(), ori_dis.softmax(dim=-1), reduction='sum')
    return KL


def getKLreg(leveldict, model, oridistribution):
    reg = 0
    for k, v in leveldict.items():
        newdis_part = getdistribution(v, model)
        reg_part = getKL(oridistribution[k], newdis_part)
        reg = reg + reg_part
    return reg


if __name__ == '__main__':
    logs_name = ''
    ISOTIMEFORMAT = '%m%d-%H%M%S'
    config = {'log_path': ''}
    timestamp = str(datetime.datetime.now().strftime(ISOTIMEFORMAT))
    loglogs = '_'.join((logs_name, timestamp))
    log_dir = os.path.join(config['log_path'], loglogs)
    os.makedirs(log_dir)
    log_file_name = os.path.join(log_dir, "running_log")
    logger = get_logger(log_file_name)
    logger.info(config)
    trainpair = [[] for _ in range(20)]
    ########################################################
    FirstTrain = True
    ########################################################
    seed_torch()
    if FirstTrain:
        exp = [[] for _ in range(20)]
        finexp = [[] for _ in range(20)]
        costCache = {}
    else:
        b_file = open('', 'rb')
        exp = pickle.load(b_file)
        modelpath = ''
        b_file.close()
        c_file = open('', 'rb')
        costCache = pickle.load(c_file)
        c_file.close()
        d_file = open('', 'rb')
        finexp = pickle.load(d_file)
        d_file.close()
        getTrainPair(exp, exp, trainpair)
        print('load exp bestsubplans costcache success !!')
    allstime = time.time()
    workload = envs.JoinOrderBenchmark(envs.JoinOrderBenchmark.Params())
    workload.workload_info.table_num_rows = postgres.GetAllTableNumRows(workload.workload_info.rel_names)
    # need to change parms
    gamma = 0.25
    learning_rate = 1e-3
    dropbuffer = False
# queries for train
    trainquery = ['1a','2a','3a','4a','5a','6a','7a','8a','9a','10a','11a','12a','13a','14a','15a','16a','17a','18a','19a','20a',
                  '21a','22a','23a','24a','25a','26a','27a','28a','29a','30a','31a','32a','33a']
# queries for test
    Ttrainquery = ['1a','2a','3a','4a','5a','6a','7a','8a','9a','10a','11a','12a','13a','14a','15a','16a','17a','18a','19a','20a','21a',
                   '22a','23a','24a','25a','26a','27a','28a','29a','30a','31a','32a','33a']
    testquery = ['1b','2b','3b','4b','5b','6b','7b','8b','9b','10b','11b','12b','13b','14b','15b','16b','17b','18b','19b','20b','21b','22b',
                 '23b','24b','25b','26b','27b','28b','29b','30b','31b','32b','33b']
    dp_Signs = [True for i in range(len(trainquery))]
    sqllist = load_sql_Files(trainquery)
    testsqllist = load_sql_Files(testquery)
    Ttrainsqllist = load_sql_Files(Ttrainquery)
    logger.info("Train SQL List {}".format(sqllist))
    sqls = load_sql(sqllist)
    testsqls = load_sql(testsqllist)
    trainsqls = load_sql(Ttrainsqllist)
    bestplandata = [[[] for _ in range(20)] for _ in range(len(trainquery))]
    bestplanslist = [[] for _ in range(len(sqls))]
    iteration_num = 30

    # initial timeout and it will update in dp
    timeoutlist = setInitialTimeout(sqls, dropbuffer, testtime=3)
    pg_latency_train = getPG_latency(trainsqls)
    print('pg_base_latency_train', pg_latency_train)
    pg_latency_test = getPG_latency(testsqls)
    print('pg_base_latency_test', pg_latency_test)
    train_gmrl = []
    test_gmrl = []
    logger.info("timeoutList:{}".format(timeoutlist))
    batchsize = 256
    DEVICE = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    maxLevel = 0
    greedy = -1.0
    bestTrainGmrl = 20
    bestTestGmrl = 20
    decay = greedy / (iteration_num * 2)
    # ```````````````
    model_levels = []
    loss_fn = ''
    from util import plans_lib
    nodeFeaturizer = plans_lib.PhysicalTreeNodeFeaturizer(workload.workload_info)
    dpsign = True
    for i in range(0, len(sqls)):
        join_graph, all_join_conds, query_leaves, origin_dp_tables = DP.getPreCondition(sqllist[i])
        dp_tables1 = copy.deepcopy(origin_dp_tables)
        maxLevel = maxLevel if maxLevel > len(query_leaves) else len(query_leaves)
    if not FirstTrain:
        model_levels, optlist = getModelsFromFile(maxLevel, modelpath)
    else:
        model_levels, optlist = getModels(maxLevel)

    for iter in range(0, iteration_num):
        logger.info('iter {} start!'.format(str(iter)))
        stime = time.time()
        levelList = [{} for _ in range(20)]
        for i in range(0, len(sqls)):
            if dp_Signs[i]:
                join_graph, all_join_conds, query_leaves, origin_dp_tables = DP.getPreCondition(sqllist[i])
                dp_tables1 = copy.deepcopy(origin_dp_tables)
                output1, bestplanhint, num, timeout = DP.dp.UCB_left_prune_replay_fix_kl(join_graph, all_join_conds,
                                                                                         query_leaves,
                                                                                         dp_tables1, workload, exp,
                                                                                         FirstTrain, model_levels,
                                                                                         timeoutlist[i],
                                                                                         dropbuffer=dropbuffer,
                                                                                         nodeFeaturizer=nodeFeaturizer,
                                                                                         greedy=greedy,
                                                                                         subplans_fin=finexp,
                                                                                         finsql=sqls[i],
                                                                                         costCache=costCache,
                                                                                         dpsign=dpsign, levelList=levelList)
                greedy = greedy - decay
                timeoutlist[i] = round(timeout, 3)
                bestplanslist[i].append([bestplanhint, num])
                getTrainPair(output1, exp, trainpair)
                output1.clear()
            if timeoutlist[i] < pg_latency_train[i] * 0.68:
                dp_Signs[i] = False
        dpsign = not dpsign
        print(dp_Signs)
        logger.info("dptime = {}".format(time.time() - stime))
        logger.info('now timeoutlist = {}'.format(timeoutlist))

        loss_fn = nn.CrossEntropyLoss()

        logger.info('Train start ,iter ={} '.format(iter))
        logger.info(
            'trainpair num ={},now experience num = {},best exp num  = {} '.format(getexpnum(trainpair), getexpnum(exp),
getexpnum(finexp))
            )

        trainTimes = 0
        testTimes = 0
        FirstTrain = False
        for modelnum in range(2, len(model_levels)):
            optimizer = optlist[modelnum]
            temtrainpair = copy.deepcopy(trainpair[modelnum])
            if len(temtrainpair)<2:
                continue
            for epoch in range(0, 500):
                ttime = time.time()
                shuffled_indices = np.random.permutation(len(temtrainpair))
                # train
                current_idx = 0
                while current_idx < len(shuffled_indices):
                    currentTrainPair = [temtrainpair[idx] for idx in
                                        shuffled_indices[current_idx: current_idx + batchsize]]
                    query_feats = []
                    nodes = []
                    latencies = []
                    costs = []
                    torch.cuda.empty_cache()
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
                    trees, indexes = TreeConvFeaturize(nodeFeaturizer, nodes)
                    if torch.cuda.is_available():
                        trees = trees.to(DEVICE)
                        indexes = indexes.to(DEVICE)
                    calibration = []
                    for i in range(10):
                        calibration.append(
                            torch.tanh(model_levels[modelnum](query_feats, trees, indexes).to(DEVICE)).add(1))

                    calibration = torch.cat(calibration, 1)
                    calibration = torch.mean(calibration, dim=1)
                    temloss = calculateLossForBatch(latencies, costs, calibration)
                    loss = torch.mean(temloss, 0) 
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    current_idx += batchsize
                trainTimes = trainTimes + time.time() - ttime
                tetime = time.time()
                acc = 0
                cout = 0
                current_idx = 0
                while current_idx < len(shuffled_indices):

                    currentTrainPair = [temtrainpair[idx] for idx in
                                        shuffled_indices[current_idx: current_idx + batchsize]]
                    query_feats = []
                    nodes = []
                    latencies = []
                    costs = []
                    for i in currentTrainPair:
                        cout = cout + 1
                        query_feats.append(i[0][0])
                        query_feats.append(i[3][0])
                        nodes.append(i[0][1])
                        nodes.append(i[3][1])
                        latencies.append(i[1])
                        latencies.append(i[4])
                        costs.append(i[2])
                        costs.append(i[5])
                    query_feats = (torch.cat(query_feats, dim=0)).to(DEVICE)
                    trees, indexes = TreeConvFeaturize(nodeFeaturizer, nodes)
                    if torch.cuda.is_available():
                        trees = trees.to(DEVICE)
                        indexes = indexes.to(DEVICE)
                    calibration = []
                    for m in range(10):
                        with torch.no_grad():
                            calibration.append(
                                torch.tanh(model_levels[modelnum](query_feats, trees, indexes)).add(1))
                    calibration = torch.cat(calibration, 1)
                    calibration = torch.mean(calibration, dim=1)
                    calibration = calibration.unsqueeze(1)
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
                testTimes = testTimes + time.time() - tetime
                logger.info("iter:{},model:{},train iters：{}，acc:{} ".format(iter, modelnum, epoch + 1, acc / cout))
                if acc / cout > 0.96 or epoch > 13:
                    break

        logger.info('train time ={} test time = {}'.format(trainTimes, testTimes))
        testtime = time.time()

        nowtraingmrl = getGMRL(Ttrainquery, model_levels, pg_latency_train, nodeFeaturizer, costCache, workload,
                               exp=exp, old=pg_latency_train)
        if nowtraingmrl < bestTrainGmrl:
            bestTrainGmrl = nowtraingmrl
            for modelnum in range(2, len(model_levels)):
                modelname = log_dir + '/BestTrainModel_' + logs_name + '_' + str(modelnum) + '.pth'
                torch.save(model_levels[modelnum], modelname)
        train_gmrl.append(nowtraingmrl)
        nowtestgmrl = getGMRL(testquery, model_levels, pg_latency_test, nodeFeaturizer, costCache, workload)
        if nowtestgmrl < bestTestGmrl:
            bestTestGmrl = nowtestgmrl
            for modelnum in range(2, len(model_levels)):
                modelname = log_dir + '/BestTestModel_' + logs_name + '_' + str(modelnum) + '.pth'
                torch.save(model_levels[modelnum], modelname)
        test_gmrl.append(nowtestgmrl)

        logger.info('GMRL test  time ={}'.format(time.time() - testtime))
        logger.info('train_gmrl ={}'.format(train_gmrl))
        logger.info('test_gmrl ={}'.format(test_gmrl))
        levelList.clear()
        gc.collect()
        a_file = open(log_dir + '/Bestplans_' + logs_name + '.pkl', 'wb')
        b_file = open(log_dir + '/Exp_' + logs_name + '.pkl', 'wb')
        c_file = open(log_dir + '/cost_' + logs_name + '.pkl', 'wb')
        d_file = open(log_dir + '/finexp_' + logs_name + '.pkl', 'wb')
        pickle.dump(exp, b_file)
        pickle.dump(bestplanslist, a_file)
        pickle.dump(costCache, c_file)
        pickle.dump(finexp, d_file)
        a_file.close()
        b_file.close()
        c_file.close()
        d_file.close()
    logger.info('all time = {} '.format(time.time() - allstime))
