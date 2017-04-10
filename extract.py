import numpy
import matplotlib.pyplot as plt

f = open("./Sample/USER1_1.txt", "r")
num = int(f.read(3))
f.seek(5)
lines = f.readlines()

x = []
y = []
btn = []
for i in lines:
	x.append(i.split(' ')[0])
	y.append(i.split(' ')[1])
	btn.append(i.split(' ')[3])
f.close()

print btn
print num	

xplt = []
yplt = []
for i in range(num-1):
	if int(btn[i])==1:
		xplt.append(x[i])
		yplt.append(y[i])

xplt = map(int, xplt)
yplt = map(int, yplt)

print xplt
print yplt


