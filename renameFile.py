import os
import sys
import re

dir = './Task1/'

files = os.listdir(dir)

for iterFile in range(len(files)):

	name = files[iterFile]
	user = ""
	sign = ""

	a = 1
	while name[a]!='S':
		user = user+name[a]
		a+=1

	a+=1
	while name[a]!='.':
		sign = sign+name[a]
		a+=1

	if len(user)==1:
		user = '0'+user

	if len(sign)==1:
		sign = '0'+sign

	rename = 'U'+user+'S'+sign+'.txt'

	path = os.path.join(dir, files[iterFile])
	target = os.path.join(dir, rename)
	os.rename(path, target)
