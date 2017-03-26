import os
import re
from fastdtw import fastdtw
import numpy as np


def gaussianND(X, mu, sigma):

    n = X.shape[1]
    meanDiff = np.subtract(X, mu)
    num = np.exp(-0.5 * np.array([np.sum((np.multiply(np.matmul(meanDiff,np.linalg.inv(sigma)),meanDiff)),axis=1)]))
    pdf = float(1)/np.sqrt(np.abs(np.power(2*np.pi,n)*np.linalg.det(sigma))) * num
    return pdf

def weightedAverage(weights, values):

    weights = np.array([weights])
    val = np.matmul(weights , values)
    val = val / np.sum(weights)
    return val

def expectationMaximization(X, k):

    m = X.shape[0]
    n = 2

    indices = np.random.permutation(m)
    mu = X[indices[0:k],:]

    sigma = []

    # Variance of each cluster
    for j in range(0,k):
        sigma.append(np.cov(X.T))

    # Weights for each cluster
    phi = np.ones((1,k)) * (float(1)/k)

    # Matrix that holds the probability that each data point belongs to each of the cluster
    # No of rows = no of data points and no of coloumns = no of clusters
    W = np.zeros((m,k))

    for iter in range(0,1000):

        pdf = np.zeros((m,k))

        # Evaluate the Gaussian for all data points for cluster 'j'
        for j in range(0,k):
            pdf[:,j] = gaussianND(X, mu[j,:], sigma[j])


        pdf_w = np.multiply(pdf, phi)

        W = np.divide(pdf_w.astype(float), np.array([np.sum(pdf_w,axis=1)]).T)

        prevMu = np.copy(mu)

        for j in range(0,k):
            phi[0][j] = np.mean(W[:,j], axis=0)
            mu[j, :] = weightedAverage(W[:,j], X)
            sigma_k = np.zeros((n,n))
            Xm = np.subtract(X,mu[j,:])
            for i in range(0,m):
                sigma_k = sigma_k + W[i,j]*np.matmul(np.array([Xm[i,:]]).T, np.array([Xm[i,:]]))
            sigma[j] = np.divide(sigma_k, sum(W[:,j]))
        if np.array_equal(mu, prevMu):
            break

    print mu
    return (mu, sigma, phi)

def getXY(xAll, yAll, TESTFILEPATH):

    # READ THE TEST FILE X and Y COORDINATES

    fid = open(TESTFILEPATH, 'r')
    lines=fid.readlines(); lines = [l.strip('\n\r') for l in lines]
    lines = lines[0:len(lines)]
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

    return x_dist, y_dist, userID+1



#   READ THE TRAINING DATA
FILEPATH_xGenGen = './gendistx.txt'
FILEPATH_yGenGen = './gendisty.txt'
FILEPATH_xGenFake = './fakedistx.txt'
FILEPATH_yGenFake = './fakedisty.txt'

with open(FILEPATH_xGenGen, 'r') as f:
    lines = f.readlines()
x1 = [x.strip() for x in lines]
x1 = np.array([x1]).astype(float)
f.close()

with open(FILEPATH_yGenGen, 'r') as f:
    lines = f.readlines()
y1 = [x.strip() for x in lines]
y1 = np.array([y1]).astype(float)
f.close()

with open(FILEPATH_xGenFake, 'r') as f:
    lines = f.readlines()
x2 = [x.strip() for x in lines]
x2 = np.array([x2]).astype(float)
f.close()

with open(FILEPATH_yGenFake, 'r') as f:
    lines = f.readlines()
y2 = [x.strip() for x in lines]
y2 = np.array([y2]).astype(float)
f.close()


#   NORMALIZE THE DTW VALUES
# x1 = np.divide(x1, np.max(x1))
# x2 = np.divide(x2, np.max(x2))
# y1 = np.divide(y1, np.max(y1))
# y2 = np.divide(y2, np.max(y2))

X1 = np.vstack((x1, y1)).T
X2 = np.vstack((x2, y2)).T
# So now X1 is the 2D matrix containing the genuine genuine DTWs and X2 contains genuine fake dtws


#   TRAINING
k = 2                       # k is the number of clusters in each GMM
(muGen, sigmaGen, phiGen) = expectationMaximization(X1, k)
(muFake, sigmaFake, phiFake) = expectationMaximization(X2, k)

#   TESTING
# if len(sys.argv)!=2:
#     print 'Parse error\nCorrect usage:$ python '+sys.argv[0]+' <data Directory path>'
#     exit()
#
# TESTFILEPATH = sys.argv[1]
TESTFILEPATH = './Task1/U19S16.txt'

if __name__ == "__main__":
    dir = './Task1/'
    files = os.listdir(dir)
    files = sorted(files, key=lambda x: (int(re.sub('\D','',x)),x)) #Natural sort (we want U1S1 < U10S1)

    xAll = []; yAll =[]

    for iterFile in range(0,len(files)):
        currFile = files[iterFile]
        fid = open(dir+currFile, "r")
        lines=fid.readlines(); lines = [l.strip('\n\r') for l in lines]
        lines = lines[0:len(lines)]
        xVal = []; yVal = []

        for x in lines:
            xVal.append(float(x.split(', ')[0]))
            yVal.append(float(x.split(', ')[1]))
        fid.close()

        xAll.append(xVal)
        yAll.append(yVal)


    X, Y, userID = getXY(xAll, yAll, TESTFILEPATH)
    print '\nUSER ID: ', userID, '\n'

    dtw = np.array([[X, Y]])


#   FINAL CLASSIFICATION BASED ON POSTERIOR PROBABILITIES
posteriorGen = 0
posteriorFake = 0
for i in range(0,k):
    posteriorGen = posteriorGen + phiGen[0][i]*gaussianND(dtw, muGen[i], sigmaGen[i])
    posteriorFake = posteriorFake + phiFake[0][i]*gaussianND(dtw, muFake[i], sigmaGen[i])

if posteriorGen > posteriorFake:
    print "Genuine"
else:
    print "Forged"

