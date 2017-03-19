import numpy as np
import matplotlib.pyplot as plt
import math

def gaussianND(X, mu, sigma):
    n = X.shape[1]
    meanDiff = np.subtract(X, mu)
    num = np.exp(-0.5 * np.array([np.sum((np.multiply(np.matmul(meanDiff,np.linalg.inv(sigma)),meanDiff)),axis=1)]))
    pdf = float(1)/math.sqrt(np.abs(math.pow(2*np.pi,n)*np.linalg.det(sigma))) * num
    return pdf

def weightedAverage(weights, values):
    weights = np.array([weights])
    val = np.matmul(weights , values)
    val = val / np.sum(weights)
    return val

FILEPATH_xGenGen = '/home/anand/part1/PRML_MP/data/gendist.txt'
FILEPATH_yGenGen = '/home/anand/part1/PRML_MP/data/gendisty.txt'
FILEPATH_xGenFake = '/home/anand/part1/PRML_MP/data/fakedist.txt'
FILEPATH_yGenFake = '/home/anand/part1/PRML_MP/data/fakedisty.txt'

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

x1 = np.divide(x1, np.max(x1))
x2 = np.divide(x2, np.max(x2))
y1 = np.divide(y1, np.max(y1))
y2 = np.divide(y2, np.max(y2))

X1 = np.vstack((x1, y1)).T
X2 = np.vstack((x2, y2)).T

# So now X1 is the 2D matrix containing the genuine genuine dtws
# X2 contains genuine fake dtws

X = np.vstack((X1,X2))

# plt.plot(x1, y1, 'x')
# plt.show()
#
# plt.plot(x2, y2, 'x')
# plt.show()

m = X.shape[0]

# No of clusters and data points
k = 2
n = 2

indices = np.random.permutation(m)
#indices = np.arange(m)
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
    #flag = 0
    print ' EM interation ', iter, '\n'

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
print mu, '\n\n', sigma, '\n\n'



