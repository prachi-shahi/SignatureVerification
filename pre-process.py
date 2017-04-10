import numpy as np
import os
import re
from sklearn.decomposition import PCA

#   OPEN THE FILES SEQUENTIALLY
dir = "./testFiles/"

files = os.listdir(dir)
files = sorted(files, key=lambda x: (int(re.sub('\D','',x)),x)) #Natural sort (we want U1S1 < U10S1)

xGen = []; xFake = []
yGen = []; yFake = []
flag = 1
counter = 0

for iterFile in range(0,len(files)):
	currFile = files[iterFile]
	fid = open(dir+currFile, "r")
	lines=fid.readlines(); lines = [l.strip('\n\r') for l in lines]
	rowsCount = lines[0]
	lines = lines[1:len(lines)]
	xVal = []; yVal = []; timeStamp = []; buttonStatus = [];

	for x in lines:
		xVal.append(x.split(', ')[0])
		yVal.append(x.split(', ')[1])

	fid.close()

	#   ADJUST TRANSLATION
	xFirst = float(xVal[0])
	yFirst = float(yVal[0])
	for i in range(int(rowsCount)):
		xVal[i] = float(xVal[i])
		yVal[i] = float(yVal[i])
		xVal[i] = xVal[i]-xFirst
		yVal[i] = yVal[i]-yFirst


	#   ADJUST ROTATION
	feat = np.column_stack((xVal, yVal))
	pca = PCA(n_components=1)
	pca.fit(feat)
	print pca.components_
	theta = -np.arctan(pca.components_[0][1]/pca.components_[0][0])
	rot = [[np.cos(theta), np.sin(-theta)], [np.sin(theta), np.cos(theta)]]
	rotated_feat = np.matmul(rot,feat.T)
	xVal = rotated_feat[0]
	yVal = rotated_feat[1]

	fid = open(dir+currFile, "w")
	fid.writelines(rowsCount+'\n')
	for i in range(int(rowsCount)):
		fid.writelines(str(xVal[i])+', '+str(yVal[i])+'\n')
	fid.close()
