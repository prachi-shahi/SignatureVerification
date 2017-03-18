# Usage: $python getData.py dataSetDirectory
# It does the following:
#	1. Opens the directory
#   2. Opens all the files in the directory sequentialy
#   3. Reads the features from the data files

import os
import sys
import re
import numpy as np
import time
start_time = time.time()

#Comment the following and specify the dir for debugging, for ex
# if len(sys.argv)!=2:
# 	print 'Parse error\nCorrect usage:$ python '+sys.argv[0]+' <data Directory path>'
# 	exit()
# dir = sys.argv[1]

def dist_DTW(A, B):
    n = len(A[0]); m = len(B[0])
    DTW = np.zeros((n,m))
    for i in range(1,n):
        DTW[i][0] = np.inf
    for i in range(1,m):
        DTW[0][i] = np.inf
    DTW[0][0] = 0
    for i in range(1,n):
        for j in range(1,m):
            cost = np.abs(A[0][i].astype(int)-B[0][j].astype(int))    #L1 norm
            DTW[i][j] = cost+min(DTW[i-1][j], DTW[i][j-1], DTW[i-1][j-1])
    return DTW[n-1,m-1]


dir = './Task1/'

files = os.listdir(dir)
#files = sorted(files, key=lambda x: (int(re.sub('\D','',x)),x)) #Natural sort (we want U1S1 < U10S1)

NO_OF_USERS = 40

xAll = []; yAll = []
xGen = []; xFake = []
yGen  = []; yFake = []
flag = 1
counter = 0
for iterFile in range(0,len(files)):
	currFile = files[iterFile]
	fid = open(dir+currFile, "r")
	lines=fid.readlines(); lines = [l.strip('\n\r') for l in lines]
	rowsCount = lines[0]
	lines = lines[1:len(lines)]
	xVal = []; yVal = []; timeStamp = []; buttonStatus = []; azimuth = []; alt = []; psi = []

	for x in lines:
		xVal.append(x.split(' ')[0])
        yVal.append(x.split(' ')[1])
        timeStamp.append(x.split(' ')[2])
        buttonStatus.append(x.split(' ')[3])
	# Uncomment the next three lines for Task2
        #azimuth.append(x.split(' ')[4])
        #alt.append(x.split(' ')[5])
        #psi.append(x.split(' ')[6])

	xAll.append(xVal)
	yAll.append(yVal)
	fid.close()

	if flag:
		xGen.append(xVal)
		yGen.append(yVal)
	else:
		xFake.append(xVal)
		yFake.append(yVal)
	counter+=1
	if counter % 20 == 0:
         flag = not flag

errorGen = []; errorFake = []
for i in range(0,NO_OF_USERS):
    xRef_gen = xGen[20*i:20*i+5]
    xTrain_gen = xGen[20*i+5:20*i+15]
    for j in range(0,len(xRef_gen)):
        for k in range(0,len(xTrain_gen)):
            e_gen = dist_DTW(np.array([xRef_gen[j]]), np.array([xTrain_gen[k]]))
            errorGen.append(e_gen)

    xRef_fake = xFake[20*i:20*i+5]
    xTrain_fake = xFake[20*i+5:20*i+15]
    for j in range(0,len(xRef_fake)):
        for k in range(0,len(xTrain_fake)):
            e_fake = dist_DTW(np.array([xRef_fake[j]]), np.array([xTrain_fake[k]]))
            errorFake.append(e_fake)
p=0
print (time.time()-start_time)
