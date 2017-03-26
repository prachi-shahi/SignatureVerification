import numpy as np
import os
import re
import math
import matplotlib.pyplot as plt
import time

dir = '../PRML_MP/Task1/'

files = os.listdir(dir)
files = sorted(files, key=lambda x: (int(re.sub('\D','',x)),x)) #Natural sort (we want U1S1 < U10S1)


def pca(X):
    cov_mat = np.cov(X)
    eig_val, eig_vec = np.linalg.eig(cov_mat)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    matrix_w = np.hstack((eig_pairs[0][1]))
    transformed = matrix_w.T.dot(X)
    return eig_pairs[0][1]


for iterFile in range(0, len(files)):
    currFile = files[iterFile]
    print currFile
    fid = open(dir+currFile, "r")
    lines=fid.readlines(); lines = [l.strip('\n\r') for l in lines]
    rowsCount = lines[0]
    lines = lines[1:len(lines)]
    xVal = []; yVal = []

    for x in lines:
        xVal.append(x.split(' ')[0])
        yVal.append(x.split(' ')[1])
    fid.close()

        # xFirst = int(xVal[0])
        # yFirst = int(yVal[0])
        # for i in range(int(rowsCount)):
        # 	xVal[i] = int(xVal[i])
        # 	yVal[i] = int(yVal[i])
        # 	xVal[i] = xVal[i]-xFirst
        # 	yVal[i] = yVal[i]-yFirst

    xVal = np.array([xVal]); yVal = np.array([yVal])
    img = np.vstack((xVal, yVal)).astype(float)
    dirVec = pca(img)
    angle = math.atan(dirVec[1]/dirVec[0])
    rotationMatrix = np.array([[math.cos(angle), -math.sin(angle)],[math.sin(angle), math.cos(angle)]])
    imgRotated = rotationMatrix.T.dot(img)

    fig = plt.figure()
    ax = plt.subplot("211")
    ax.set_title(currFile)
    ax.plot(img[0][:], img[1][:])
    ax = plt.subplot("212")
    ax.plot(imgRotated[0][:], img[0][:])
    plt.show()




