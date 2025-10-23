# -*- coding:utf-8 -*-
"""
Author: KittenCN
"""

import pandas as pd
from .config import *

datacnt = [0] * 81
dataori = [i for i in range(81)]
ori_data = []

def BasicAnalysis(oridata):
    # Basic analysis of the data
    # ori_data: original data
    # Return: None
    # Author: KittenCN
    global datacnt, dataori
    datacnt = [0] * 81
    dataori = [i for i in range(81)]
    for row in oridata:
        for item in row:
            datacnt[int(item)] += 1
    datacnt, dataori = sortcnt(datacnt, dataori, 81)
    lastcnt = -1
    for i in range(81):
        if dataori[i] == 0:
            continue
        if lastcnt != datacnt[i]:
            print()
            print("{}: {}".format(datacnt[i], dataori[i]), end = " ")
            lastcnt = datacnt[i]
        elif lastcnt == datacnt[i]:
            print(dataori[i], end = " ")
    return datacnt, dataori
    
def sortcnt(datacnt, dataori, rangenum=81):
    for i in range(rangenum):
        for j in range(i + 1, rangenum):
            if datacnt[i] < datacnt[j]:
                datacnt[i], datacnt[j] = datacnt[j], datacnt[i]
                dataori[i], dataori[j] = dataori[j], dataori[i]
            elif datacnt[i] == datacnt[j]:
                if dataori[i] < dataori[j]:
                    datacnt[i], datacnt[j] = datacnt[j], datacnt[i]
                    dataori[i], dataori[j] = dataori[j], dataori[i]
    return datacnt, dataori

def getdata():
    strdata = input("Enter the numbers to count occurrences, comma-separated, -1 to end: ").split(',')
    if strdata[0] == "-1":
        return None, None
    data = [int(i) for i in strdata]
    oridata = []
    for i in range(81):
        if dataori[i] == 0:
            continue
        if datacnt[i] in data:
            oridata.append(dataori[i])
    booldata = [False] * len(oridata)
    return oridata, booldata

def dfs(oridata, booldata, getnums, dep, ans, cur):
    if dep == getnums:
        ans.append(cur.copy())
        return
    for i in oridata:
        if booldata[oridata.index(i)] or i <= cur[dep - 1]:
            continue
        booldata[oridata.index(i)] = True
        cur[dep] = i
        dfs(oridata,booldata, getnums, dep + 1, ans, cur)
        booldata[oridata.index(i)] = False
    return ans

def shrink(oridata, booldata):
    getnums = int(input("How many numbers to reduce to? (-1 to end) "))
    while getnums != -1:
        ans = dfs(oridata,booldata, getnums, 0, [], [0] * getnums)
        print("Total of {} results, can be reduced to {} numbers.".format(len(ans), getnums))
        strSumMinMax = input("Enter min and max sum values, comma-separated").split(',')
        SumMinMax = [int(i) for i in strSumMinMax]
        SumMin = SumMinMax[0]
        SumMax = SumMinMax[1]
        for i in range(len(ans)):
            if sum(ans[i]) < SumMin or sum(ans[i]) > SumMax:
                continue
            print(ans[i])
        getnums = int(input("How many numbers to reduce to? (-1 to end) "))

def sumanalyusis(limit=-1):
    oridata = pd.read_csv("{}{}".format(name_path["kl8"]["path"], data_file_name))
    data = oridata.iloc[:, 2:].values
    sumori = [i for i in range(1401)]
    sumcnt = [0] * 1401
    linenum = 0
    for row in data:
        if limit != -1 and linenum >= limit:
            break
        linenum += 1
        sum = 0
        for item in row:
            sum += item
        sumcnt[sum] += 1
    sumcnt, sumori = sortcnt(sumcnt, sumori, 1401)
    lastcnt = -1
    for i in range(1401):
        if sumori[i] == 0 or sumcnt[i] == 0:
            continue
        if lastcnt != sumcnt[i]:
            print()
            print("{}: {}".format(sumcnt[i], sumori[i]), end = " ")
            lastcnt = sumcnt[i]
        elif lastcnt == sumcnt[i]:
            print(sumori[i], end = " ")
    print()
    sumtop = int(input("Enter how many top positions to calculate sum:"))
    lastsum = -1
    sumans = []
    sumanscnt = 0
    for i in range(1401):
        if sumcnt[i] == 0:
            continue
        if sumcnt[i] != lastsum:
            if sumanscnt == sumtop:
                break;
            else:
                lastsum = sumcnt[i]
                sumanscnt += 1
        sumans.append(sumori[i])
    print(sumans)

if __name__ == "__main__":
    while True:
        print()
        print(" 1. Read prediction data and analyze\r\n 2. Reduce\r\n 3. Sum value analysis\r\n 0. Exit\r\n")
        choice = int(input("input your choice:"))
        if choice == 1:
            _datainrow = []
            n = int(input("Enter number of data groups, -1 to read from file:"))
            if n != -1:
                for i in range(n):
                    tmpdata = input("Enter data group #{}: ".format(i + 1)).strip().split(' ')
                    for item in tmpdata:
                        _datainrow.append(int(item))
                    _datainrow.append(int(item) for item in tmpdata)
                    ori_data.append(tmpdata)
            else:
                filename = input("Enter filename: ")
                fileadd = "{}{}{}{}".format(predict_path, "kl8/", filename, ".csv")
                ori_data = pd.read_csv(fileadd).values
                limit = int(input("Total of {} data groups, enter how many groups to analyze:".format(len(ori_data))))
                ori_data = ori_data[:limit]
                for row in ori_data:
                    for item in row:
                        _datainrow.append(item)           
            datacnt, dataori = BasicAnalysis(ori_data)
            print()
            currentnums = input("Enter current winning data, -1 to end: ").split(' ')
            if currentnums[0] != "-1":
                curnums = [int(i) for i in currentnums]
                curcnt = 0
                tmp_cnt = [0] * len(ori_data)
                for item in curnums:
                    for i, row in enumerate(ori_data):
                        if item in row:
                            curcnt += 1
                            tmp_cnt[i] += 1
                            break
                totalnums = len(list(set(_datainrow)))
                for i in range(len(tmp_cnt)):
                    print("In data group {}, current winning data appears {} times, probability: {:.2f}%".format(i + 1, tmp_cnt[i], tmp_cnt[i] / totalnums * 100))
                print("Hits / Total predictions: {} / {}".format(curcnt, totalnums))
                lastcnt = -1
                for i in range(81):
                    if dataori[i] == 0:
                        continue
                    elif dataori[i] in curnums:
                        if lastcnt != datacnt[i]:
                            print()
                            print("{}: {}".format(datacnt[i], dataori[i]), end = " ")
                            lastcnt = datacnt[i]
                        elif lastcnt == datacnt[i]:
                            print(dataori[i], end = " ")
                print()
            oridata, booldata = getdata()
            print(oridata)

        elif choice == 2:
            oridata, booldata = getdata()
            shrink(oridata, booldata)
        elif choice == 3:
            limit = int(input("Enter number of data groups to analyze, -1 for all:"))
            sumanalyusis(limit)
        if choice == 0:
            break
