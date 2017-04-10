import math
from collections import Counter
import random
import sys
def generate_samples():
	for i in range(1000):
		print (random.gauss(1,1))
	for i in range(100):
		print (random.gauss(8,1))
def get_density(x,m,s):
	return((1/(s*math.sqrt(2*math.pi)))*(math.pow(math.e,-(x-m)*(x-m)/(2*s*s))))
def get_posterior(x,priors,means,stdevs):
	posteriors=Counter()
	for k in priors.keys():
		posteriors[k]=get_density(x,means[k],stdevs[k])*priors[k]
	sum_posteriors=sum(posteriors.values())
	
	for k in posteriors.keys():
		posteriors[k]/=sum_posteriors
		
	return(posteriors)
	
def relearn(plines):
	priors=Counter()
	means=Counter()
	stdevs=Counter()
	for(posterior,line) in plines:
		for k in posterior.keys():
			priors[k]+=posterior[k]
			means[k]+=posterior[k]*float(line[0])
			stdevs[k]+=posterior[k]*float(line[0])*float(line[0])
	sum_priors=sum(priors.values())
	for k in priors.keys():
			means[k]/=priors[k]
			stdevs[k]/=priors[k]
			stdevs[k]=math.sqrt(stdevs[k]-means[k]*means[k])
			priors[k]/=sum_priors
		
	
	return(priors,means,stdevs)
def main():
	op=sys.argv[1]
	if op =='generate':
			generate_samples()
	else:
		filename=sys.argv[2]
		priors=Counter({"0":0.5,"1":0.5})
		means=Counter({"0":random.random(),"1":random.random()})
		stdevs=Counter({"0":random.random(),"1":random.random()})
		
	
		for i in range(100):
			lines=[l.strip().split()  for l in open(filename).readlines()]
			plines=[]
			for line in lines:
					plines.append((get_posterior(float(line[0]),priors,means,stdevs),line))	
					
			print("iteration", i)
			print("Priors ", priors)
			print("Means", means)
			print("Stdevs", stdevs)
			priors=relearn(plines)
			(priors,means,stdevs) = relearn(plines)
			
if __name__=='__main__':
		main()
					