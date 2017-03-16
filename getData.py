# Usage: $python getData.py dataSetDirectory
# It does the following:
#	1. Opens the directory
#   2. Opens all the files in the directory sequentialy
#   3. Reads the features from the data files

import os
import sys
import re

#Comment the following and specify the dir for debugging, for ex
# if len(sys.argv)!=2:
# 	print 'Parse error\nCorrect usage:$ python '+sys.argv[0]+' <data Directory path>'
# 	exit()
# dir = sys.argv[1]

dir = './Task1/'

files = os.listdir(dir)
files = sorted(files, key=lambda x: (int(re.sub('\D','',x)),x)) #Natural sort (we want U1S1 < U10S1)

xAll = []; yAll = []
xGen = []; xFake = []
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
        #azimuth.append(x.split(' ')[4])
        #alt.append(x.split(' ')[5])
        #psi.append(x.split(' ')[6])

	xAll.append(xVal)
	yAll.append(yVal)
	fid.close()

	if flag:
		xGen.append(xVal)
	else:
		xFake.append(xVal)
	counter+=1
	if counter % 20 == 0:
		flag = not flag
p=0
