# Usage: $python getData.py dataSetDirectory
# It does the following:
# 	1. Opens the directory
#   2. Opens all the files in the directory sequentially
#   3. Reads the features from the data files

import os
import re
from fastdtw import fastdtw

# Comment the following and specify the dir for debugging, for ex
# if len(sys.argv)!=2:
# 	print 'Parse error\nCorrect usage:$ python '+sys.argv[0]+' <data Directory path>'
# 	exit()
# dir = sys.argv[1]

dir = './Task1/'

files = os.listdir(dir)
files = sorted(files, key=lambda x: (int(re.sub('\D', '', x)), x)) #Natural sort (we want U1S1 < U10S1)

xAll = []; yAll =[]

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
		# azimuth.append(x.split(' ')[4])
		# alt.append(x.split(' ')[5])
		# psi.append(x.split(' ')[6])

	fid.close()

	xFirst = int(xVal[0])
	yFirst = int(yVal[0])
	for i in range(int(rowsCount)):
		xVal[i] = int(xVal[i])
		yVal[i] = int(yVal[i])
		xVal[i] = xVal[i]-xFirst
		yVal[i] = yVal[i]-yFirst

	xAll.append(xVal)
	yAll.append(yVal)


#   DTW
xGenDTW = []
yGenDTW = []
xFakeDTW = []
yFakeDTW = []
xGenTest = []
xFakeTest = []
yGenTest = []
yFakeTest = []

for i in range(0, 800, 40):
    xRef = xAll[i:i+5]
    xGenTrain = xAll[i+5:i+15]
    xGenTest.append(xAll[i+15:i+20])
    xFakeTrain = xAll[i+20:i+35]
    xFakeTest.append(xAll[i+35:i+40])


    yRef = yAll[i:i+5]
    yGenTrain = yAll[i+5:i+15]
    yGenTest.append(yAll[i+15:i+20])
    yFakeTrain = yAll[i+20:i+35]
    yFakeTest.append(yAll[i+35:i+40])

    for j in range(0,5):
		for k in range(0,10):
			xDist, path = fastdtw(xRef[j], xGenTrain[k], dist=None)
			xGenDTW.append(xDist)
			yDist, path = fastdtw(yRef[j], yGenTrain[k], dist=None)
			yGenDTW.append(yDist)
		for k in range(0,15):
			xDist, path = fastdtw(xRef[j], xFakeTrain[k], dist=None)
			xFakeDTW.append(xDist)
			yDist, path = fastdtw(yRef[j], yFakeTrain[k], dist=None)
			yFakeDTW.append(yDist)


fGenTrain = open("genTrain.txt", "w")
fGenTrain.writelines("%s " % str(item) for item in xGenDTW)
fGenTrain.write("\n")
fGenTrain.writelines("%s " % str(item) for item in yGenDTW)
fGenTrain.close()

fFakeTrain = open("fakeTrain.txt", "w")
fFakeTrain.writelines("%s " % str(item) for item in xFakeDTW)
fFakeTrain.write("\n")
fFakeTrain.writelines("%s " % str(item) for item in yFakeDTW)
fFakeTrain.close()

fGenTest = open("genTest.txt", "w")
fGenTest.writelines("%s " % str(item) for item in xGenTest)
fGenTest.write("\n")
fGenTest.writelines("%s " % str(item) for item in yGenTest)
fGenTest.close()

fFakeTest = open("fakeTest.txt", "w")
fFakeTest.writelines("%s " % str(item) for item in xFakeTest)
fFakeTest.write("\n")
fFakeTest.writelines("%s " % str(item) for item in yFakeTest)
fFakeTest.close()

