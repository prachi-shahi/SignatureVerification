import os
import sys

if len(sys.argv)!=2:
      print 'Parse error\nCorrect usage:$ python '+sys.argv[0]+' <directory path>'
      exit()
dir = sys.argv[1]
#dir = './Task1(copy)/'

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
print 'File names in '+sys.argv[1]+' standardised'
