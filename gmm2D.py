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

mu1  = [1, 2]
sigma1 = [[3, 0.2], [0.2, 2]]
m1 = 200

mu2 = [-1, -2]
sigma2 = [[2, 0], [0, 1]]
m2 = 100

x1, y1 = np.random.multivariate_normal(mu1, sigma1, m1).T
x2, y2 = np.random.multivariate_normal(mu2, sigma2, m2).T

x1 = np.divide(x1, np.max(x1))
x2 = np.divide(x2, np.max(x2))
y1 = np.divide(y1, np.max(y1))
y2 = np.divide(y2, np.max(y2))

X1 = np.vstack((x1, y1)).T
X2 = np.vstack((x2, y2)).T

X = np.vstack((X1,X2))

# plt.plot(x1, y1, 'x')
# plt.show()
#
# plt.plot(x2, y2, 'x')
# plt.show()

m = X.shape[0]

k = 2
n = 2

indices = np.random.permutation(m)
mu = X[indices[0:k],:]

sigma = []

for j in range(0,k):
    sigma.append(np.cov(X.T))

phi = np.ones((1,k)) * (float(1)/k)

W = np.zeros((m,k))


for iter in range(0,1000):

    flag = 0
    print ' EM interation ', iter, '\n'

    pdf = np.zeros((m,k))


    for j in range(0,k):
        if np.linalg.det(sigma[j]) < 10e-10:
            flag = 1
        pdf[:,j] = gaussianND(X, mu[j,:], sigma[j])

    if flag is 1:
        print 'iteration ', iter, ' failed\n'
        continue

    pdf_w = np.multiply(pdf, phi)

    W = np.divide(pdf_w.astype(float), np.array([np.sum(pdf_w,axis=1)]).T)

    prevMu = np.copy(mu)

    for j in range(0,k):
        phi[0][j] = np.mean(W[:,j], axis=0)

        mu[j, :] = weightedAverage(W[:,j], X)

        sigma_k = np.zeros((n,n))

        Xm = np.subtract(X,mu[j,:])

        for i in range(0,m):
            sigma_k = sigma_k + W[i,j]*np.matmul(np.array([Xm[0,:]]).T, np.array([Xm[0,:]]))

        sigma[j] = np.divide(sigma_k, sum(W[:,j]))
    if np.array_equal(mu, prevMu):
        break
print mu, '\n\n', sigma, '\n\n'



