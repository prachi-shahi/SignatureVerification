import numpy as np
import matplotlib.pyplot as plt

alpha = 0.001
GRAD_DES_THRESH = 1e-5
THR = 0.6

def train(num_data):

    p = []
    with open('gendistx.txt', 'r') as f:
        lines = f.readlines()
    x1 = [x.strip() for x in lines]
    x1 = np.array([x1]).astype(float)
    for j in range(40):
        p.append(np.ndarray.tolist(x1[0][j*50:j*50+num_data]))
    pp = np.asarray(p)
    x1 = pp.reshape((1,40*num_data))
    f.close()

    p = []
    with open('gendisty.txt', 'r') as f:
        lines = f.readlines()
    y1 = [x.strip() for x in lines]
    y1 = np.array([y1]).astype(float)
    for j in range(40):
        p.append(np.ndarray.tolist(y1[0][j*50:j*50+num_data]))
    pp = np.asarray(p)
    y1 = pp.reshape((1,40*num_data))
    f.close()

    p = []
    with open('fakedistx.txt', 'r') as f:
        lines = f.readlines()
    x2 = [x.strip() for x in lines]
    x2 = np.array([x2]).astype(float)
    for j in range(40):
        p.append(np.ndarray.tolist(x2[0][j*50:j*50+num_data]))
    pp = np.asarray(p)
    x2 = pp.reshape((1,40*num_data))
    f.close()

    p = []
    with open('fakedisty.txt', 'r') as f:
        lines = f.readlines()
    y2 = [x.strip() for x in lines]
    y2 = np.array([y2]).astype(float)
    for j in range(40):
        p.append(np.ndarray.tolist(y2[0][j*50:j*50+num_data]))
    pp = np.asarray(p)
    y2 = pp.reshape((1,40*num_data))
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
    if (i<length/2):
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

    gradJi = np.multiply(np.array(sigmoid(X,theta) - y).T, np.array([X[:,l]]))
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
        maxDiff = np.max(np.absolute(new_theta - theta))
        if maxDiff<GRAD_DES_THRESH:
            break

    return new_theta

def test(new_theta):

    global THR
    count = 0
    X, Y = np.loadtxt('./testDTW.txt',unpack=True)
    for i in range(len(X)):
        testMatrix = np.array([[1, X[i], Y[i]]])
        if (i%4<2 and (sigmoid(testMatrix,new_theta)>=THR)) or (i%4>=2 and (sigmoid(testMatrix,new_theta)<THR)):
            count += 1;

    acc = (float(count)/len(X))*100
    return acc


def validation(new_theta):

    global THR
    count = 0
    X, Y = np.loadtxt('./validateDTW.txt',unpack=True)
    for i in range(len(X)):
        testMatrix = np.array([[1, X[i], Y[i]]])
        if (i%6<3 and (sigmoid(testMatrix,new_theta)>=THR)) or (i%6>=3 and (sigmoid(testMatrix,new_theta)<THR)):
            count += 1;
    acc = (float(count)/len(X))*100
    return acc


def learningCurves():

    err_val = [0]
    err_test = [58]
    xcoord = [0]
    for i in range(5,55,5):
        new_theta = train(i)
        err_val.append(100-validation(new_theta))
        err_test.append(100-test(new_theta))
        xcoord.append(i/5)

    print err_val
    print err_test
    plt.plot(xcoord,err_val,label="Validation Error")
    plt.plot(xcoord,err_test,label="Test Error")
    plt.legend()
    plt.xlabel("Number of Train Signatures")
    plt.show()

learningCurves()