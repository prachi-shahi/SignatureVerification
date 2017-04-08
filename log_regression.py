import sys, argparse
import os
import re
import random
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw

args=[]
alpha = 0.1
def getXY(xAll, yAll, TESTFILEPATH):

    # READ THE TEST FILE X and Y COORDINATES

    fid = open(TESTFILEPATH, 'r')
    lines=fid.readlines(); lines = [l.strip('\n\r') for l in lines]
    lines = lines[1:len(lines)]
    xVal_test = []; yVal_test = []
    for x in lines:
        xVal_test.append(float(x.split()[0]))
        yVal_test.append(float(x.split()[1]))


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

    return (x_dist, y_dist, userID+1)






def sigmoid(z) :

    h_theta = 1/(1 + np.e**(-z))
    return (h_theta)

def derivative(y, h_theta, x, n):

    return ((1/n)*np.dot((h_theta - y).T,x))

def compute_cost(h_theta, y):

    return (-y*np.log(h_theta) - (1-y)*np.log(1-h_theta))


def grad_descent(X, y, length):

    global alpha


    no_of_dim = X.shape[1]
    theta = np.ones((no_of_dim+1, 1))*1
    d = np.zeros((no_of_dim+1, 1))
    done = np.zeros((no_of_dim+1, 1))
    cnt = np.zeros((no_of_dim+1, 1))
    h_theta = np.zeros((length, 1))
    cost = np.zeros((length, 1))
    flag = 1
    X=X.T
    O=np.ones((1,length))
    X=np.vstack((O,X))
    X=X.T
    count = 0


    while flag==1:
        for i in range(length):
            inner_prod = np.dot(X[i,:],theta)
            h_theta[i] = sigmoid(inner_prod)
            cost[i] = compute_cost(h_theta[i],y[i])

        total_cost = (1/length)*np.sum(cost)

        for i in range(no_of_dim+1):
            temp_flag = 0
            for j in range(no_of_dim+1):
                if(done[j]==i+1):
                    temp_flag = 1
                    break
            if (temp_flag == 0):
                d[i] = derivative(y, h_theta, X[:,i],length)
            if ((d[i]<=0.001) and (d[i]>=(-0.001))):
                done[i] = i+1;
                cnt[i] = 1
                flag = 0
            if ((d[i]>0.001) or (d[i] < (-0.001))):
                theta[i] = theta[i] - alpha*d[i]


        count += 1
        print("Ite:",count,'\n')
    return (theta)


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
    no_of_dim=X.shape[1]

    for i in range(length):
        if (i<1000):
            y[i] = 1
        else:

            y[i] = 0

    theta = np.zeros((no_of_dim+1, 1))
    theta = grad_descent(X,y,length)
    print(theta)
    return(theta)


def predict(theta):
    print(theta)
    dir = 'Task1/'
    files = os.listdir(dir)
    files = sorted(files, key=lambda x: (int(re.sub('\D', '', x)), x))  # Natural sort (we want U1S1 < U10S1)

    testDir = 'testfiles/'
    testfiles = os.listdir(testDir)
    random.shuffle(testfiles)

    xAll = []

    yAll = []

    for iterFile in range(0, len(files)):
        currFile = files[iterFile]
        fid = open(dir + currFile, "r")
        lines = fid.readlines();
        lines = [l.strip('\n\r') for l in lines]
        lines = lines[1:len(lines)]
        xVal = [];
        yVal = []

        for x in lines:
            xVal.append(float(x.split()[0]))
            yVal.append(float(x.split()[1]))
        fid.close()

        xAll.append(xVal)
        yAll.append(yVal)

    result = []

    for testIdx in range(0, len(testfiles)):
        name = testfiles[testIdx]
        print(name)
        TESTFILEPATH =testDir + name
        (X, Y, userID) = getXY(xAll, yAll, TESTFILEPATH)
        print('\nUSER ID: ', userID, '\n')
        if (int(name[4:6]) < 20):
            gen = 1;

        else:
            gen = 0;
        data=np.array((1,X,Y))

        inner_prod = np.dot(data, theta)
        h_theta = sigmoid(inner_prod)
        if (h_theta >= 0.5):
            gentes=1;
        else:
            gentes=0;
        if (gen==gentes):
            result.append(1);
        else:
            result.append(0);
    countOnes = 0
    print(result)
    for i in range(0, len(result)):
        if result[i] == 1:
            countOnes += 1
    accuracy = (float(countOnes) / len(result)) * 100
    print(accuracy)



def main():

    theta=[]
    theta = train()
    predict(theta)




if __name__ == '__main__':
    main()
