import os
import sys
import re
from fastdtw import fastdtw
import matplotlib.pyplot as plt


TESTFILEPATH = '/home/anand/part1/PRML_MP/Task1/U04S37.txt'

def getXY(xAll, yAll, TESTFILEPATH):

    # READ THE TEST FILE X and Y COORDINATES

    fid = open(TESTFILEPATH, 'r')
    lines=fid.readlines(); lines = [l.strip('\n\r') for l in lines]
    rows = lines[0]
    lines = lines[1:len(lines)]
    xVal_test = []; yVal_test = []
    for x in lines:
        xVal_test.append(x.split(' ')[0])
        yVal_test.append(x.split(' ')[1])
    x0 = int(xVal_test[0])
    y0 = int(yVal_test[0])
    for i in range(int(rows)):
        xVal_test[i] = int(xVal_test[i])
        yVal_test[i] = int(yVal_test[i])
        xVal_test[i] = xVal_test[i]-x0
        yVal_test[i] = yVal_test[i]-y0

    # COMPUTE DTW

    distx = []
    for iterUser in range(0, 800, 40):
        xRef = xAll[iterUser:iterUser+5]
        dminx = fastdtw(xVal_test, xRef[0], dist=None)[0]
        for i in range(1,len(xRef)):
            d1 = fastdtw(xVal_test, xRef[i], dist=None)[0]
            if d1<dminx:
                dminx = d1
        distx.append(dminx)
    x_dist = min(distx)          # X value
    userID = distx.index(x_dist)
    u = userID*40
    yRef = yAll[u:u+5]
    dminy = fastdtw(yVal_test, yRef[0], dist=None)[0]
    for i in range(1,len(yRef)):
            d1 = fastdtw(yVal_test, yRef[i], dist=None)[0]
            if d1<dminy:
                dminy = d1
    y_dist = dminy      # Y value

    xRef_fake = xAll[u+20:u+25]
    yRef_fake = yAll[u+20:u+25]
    dminx_fake = fastdtw(xVal_test, xRef_fake[0], dist=None)[0]
    for i in range(1,len(xRef_fake)):
            d1 = fastdtw(xVal_test, xRef_fake[i], dist=None)[0]
            if d1<dminx_fake:
                dminx_fake = d1
    x_dist_fake = dminx_fake      # X value: test fake
    dminy_fake = fastdtw(yVal_test, yRef_fake[0], dist=None)[0]
    for i in range(1,len(yRef_fake)):
            d1 = fastdtw(yVal_test, yRef_fake[i], dist=None)[0]
            if d1<dminy_fake:
                dminy_fake = d1
    y_dist_fake = dminx_fake      # X value: test fake

    return x_dist, y_dist, x_dist_fake, y_dist_fake, userID+1



if __name__ == "__main__":
    dir = './Task1/'
    files = os.listdir(dir)
    files = sorted(files, key=lambda x: (int(re.sub('\D','',x)),x)) #Natural sort (we want U1S1 < U10S1)

    xAll = []; yAll =[]

    for iterFile in range(0,len(files)):
        currFile = files[iterFile]
        fid = open(dir+currFile, "r")
        lines=fid.readlines(); lines = [l.strip('\n\r') for l in lines]
        rowsCount = lines[0]
        lines = lines[1:len(lines)]
        xVal = []; yVal = []

        for x in lines:
            xVal.append(x.split(' ')[0])
            yVal.append(x.split(' ')[1])
        fid.close()

        xFirst = int(xVal[0])
        yFirst = int(yVal[0])
        for i in range(int(rowsCount)):
            xVal[i] = int(xVal[i])
            yVal[i] = int(yVal[i])
            xVal[i] = xVal[i]-xFirst
            yVal[i] = yVal[i]-yFirst

        xAll.append(xVal)
        yAll.append(yVal)

    X, Y, Xfake, Yfake, userID = getXY(xAll, yAll, TESTFILEPATH)
    print X, Y, Xfake, Yfake
    print '\nUSER ID: ', userID, '\n'
