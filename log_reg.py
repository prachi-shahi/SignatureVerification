import numpy as np
import re
import os
from fastdtw import fastdtw
import random

alpha = 0.001
GRAD_DES_THRESH = 1e-5

def train():
    with open('gendistx.txt', 'r') as f:
        lines = f.readlines()
    x1 = [x.strip() for x in lines]
    x1 = np.array([x1]).astype(float)
    f.close()

    with open('gendisty.txt', 'r') as f:
        lines = f.readlines()
    y1 = [x.strip() for x in lines]
    y1 = np.array([y1]).astype(float)
    f.close()

    with open('fakedistx.txt', 'r') as f:
        lines = f.readlines()
    x2 = [x.strip() for x in lines]
    x2 = np.array([x2]).astype(float)
    f.close()

    with open('fakedisty.txt', 'r') as f:
        lines = f.readlines()
    y2 = [x.strip() for x in lines]
    y2 = np.array([y2]).astype(float)
    f.close()

    x1 = np.divide(x1, np.max(x1))
    x2 = np.divide(x2, np.max(x2))
    y1 = np.divide(y1, np.max(y1))
    y2 = np.divide(y2, np.max(y2))
    X1 = np.hstack((x1, x2))
    X2 = np.hstack((y1, y2))
    X=np.vstack((X1,X2)).T

    length=X.shape[0]

    y = np.zeros((length,1))

    for i in range(length):
        if (i<2000):
            y[i] = 1
        else:
            y[i] = 0

    new_theta = grad_descent(X, y)
    return new_theta

def sigmoid(X, theta):

    inner_prod = np.dot(X,theta)
    h_theta = 1/(1 + np.e**(-inner_prod))
    return (h_theta)

def deriv(X,y,theta,l):

    gradJi = np.multiply((sigmoid(X,theta) - y).T, np.array([X[:,l]]))
    gradJ = np.sum(gradJi)
    return gradJ


def grad_descent(X, y):

    global alpha
    global GRAD_DES_THRESH

    length = X.shape[0]
    X=X.T
    O=np.ones((1,length))
    X=np.vstack((O,X))
    X=X.T

    no_of_dim = X.shape[1]
    theta = np.zeros((no_of_dim, 1))
    new_theta = np.zeros((no_of_dim, 1))
    iter = 0

    while True:
        theta = np.copy(new_theta)
        for l in range(0,len(theta)):
            new_theta[l] = theta[l] - alpha*deriv(X, y, theta, l)
        iter = iter+1
        maxDiff = np.absolute(np.max(new_theta - theta))
        print maxDiff
        if maxDiff<GRAD_DES_THRESH:
            print iter
            break

    return new_theta

def getXY(xAll, yAll, TESTFILEPATH):

    # READ THE TEST FILE X and Y COORDINATES

    fid = open(TESTFILEPATH, 'r')
    lines=fid.readlines(); lines = [l.strip('\n\r') for l in lines]
    lines = lines[1:len(lines)]
    xVal_test = []; yVal_test = []
    for x in lines:
        xVal_test.append(float(x.split(', ')[0]))
        yVal_test.append(float(x.split(', ')[1]))


    # COMPUTE DTW

    distx = []
    for iterUser in range(0, 1600, 40):
        xRef = xAll[iterUser:iterUser+5]
        dminx = fastdtw(xVal_test, xRef[0], dist=None)[0]
        for i in range(1,len(xRef)):
            d1 = fastdtw(xVal_test, xRef[i], dist=None)[0]
            if d1<dminx:
                dminx = d1
        distx.append(dminx)
    distx = distx/max(distx)
    x_dist = min(distx)          # X value
    #userIDx = distx.index(x_dist)

    disty = []
    for iterUser in range(0, 1600, 40):
        yRef = yAll[iterUser:iterUser+5]
        dminy = fastdtw(yVal_test, yRef[0], dist=None)[0]
        for i in range(1,len(yRef)):
            d1 = fastdtw(yVal_test, yRef[i], dist=None)[0]
            if d1<dminy:
                dminy = d1
        disty.append(dminy)
    disty = disty/max(disty)
    y_dist = min(disty)
    #userIDy = disty.index(y_dist)    # Y value

    return x_dist, y_dist

def test(new_theta):

    dir = './Task1/'
    files = os.listdir(dir)
    files = sorted(files, key=lambda x: (int(re.sub('\D','',x)),x)) #Natural sort (we want U1S1 < U10S1)

    testDir = './testFiles/'
    testfiles = os.listdir(testDir)
    #random.shuffle(testfiles)

    xAll = []; yAll =[]

    for iterFile in range(0,len(files)):
        currFile = files[iterFile]
        fid = open(dir+currFile, "r")
        lines=fid.readlines(); lines = [l.strip('\n\r') for l in lines]
        lines = lines[1:len(lines)]
        xVal = []; yVal = []

        for x in lines:
            xVal.append(float(x.split(', ')[0]))
            yVal.append(float(x.split(', ')[1]))
        fid.close()

        xAll.append(xVal)
        yAll.append(yVal)

    result = []
    count = 0;
    for testIdx in range(0, len(testfiles)):
        name = testfiles[testIdx]
        TESTFILEPATH = dir+name
        X, Y = getXY(xAll, yAll, TESTFILEPATH)
        testMatrix = np.array([[1, X, Y]])
        print name, "\t", testIdx
        if sigmoid(testMatrix,new_theta)>=0.5:
            print "Genuine"
        else:
            print "Fake"
        if ((int(name[4:6])<=20) and (sigmoid(testMatrix,new_theta)>=0.5)) or ((int(name[4:6])>20) and (sigmoid(testMatrix,new_theta)<0.5)):
            count += 1;
    print "Accuracy: ", count*100/len(testfiles)

test(new_theta=train())
