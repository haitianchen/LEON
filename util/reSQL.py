import re


def _FormatJoinCond(tup):
    t1, c1, t2, c2 = tup
    return f"{t1}.{c1} = {t2}.{c2}"


def _GetJoinConds(sql):
    """Returns a list of join conditions in the form of (t1, c1, t2, c2)."""
    join_cond_pat = re.compile(
        r"""
        (\w+)  # 1st table
        \.     # the dot "."
        (\w+)  # 1st table column
        \s*    # optional whitespace
        =      # the equal sign "="
        \s*    # optional whitespace
        (\w+)  # 2nd table
        \.     # the dot "."
        (\w+)  # 2nd table column
        """, re.VERBOSE)
    join_conds = join_cond_pat.findall(sql)
    result = [_FormatJoinCond(c) for c in join_conds]
    return result


def deleteUnless(oldlist):
    newlist = []
    for i in oldlist:
        if i == '' or i == ';':
            continue
        else:
            if i[-1] == ';':
                i = i[:len(i) - 1]
            newlist.append(i)
    return newlist


def dealwithBetween(andlist):
    newlist = []
    temFlag = -1
    for i in range(0, len(andlist)):
        if 'BETWEEN' in andlist[i]:
            newlist.append(andlist[i] + ' AND ' + andlist[i + 1])
            temFlag = i + 1
        if i > temFlag:
            newlist.append(andlist[i])
    return newlist


def getFliters(sqlfile):
    with open(sqlfile, 'r') as f:
        data = f.read().splitlines()
        sql = ''.join(data)
    joins = _GetJoinConds(sql)
    begin = sql.index('WHERE')
    tem = sql[begin:].replace('WHERE', '')
    for i in joins:
        tem = tem.replace(i, '')
    temandList = []
    for i in tem.split('AND'):
        temandList.append(i.strip())
    andList = deleteUnless(temandList)
    andList = dealwithBetween(andList)
    return andList


def getSelectExp(sqlfile):
    with open(sqlfile, 'r') as f:
        data = f.read().splitlines()
        sql = ''.join(data)
    begin = sql.index('SELECT')
    end = sql.index('FROM')
    tem = sql[begin:end].replace('SELECT', '')
    selectExplist = []
    for i in tem.split(','):
        selectExplist.append(i.strip())
    return selectExplist


if __name__ == '__main__':
    sqlFiles = './join-order-benchmark/1b.sql'
    print(getFliters(sqlFiles))
    s = getSelectExp(sqlFiles)
    for i in s:
        print(i[i.index('(') + 1:i.index(')')].split('.')[0])
