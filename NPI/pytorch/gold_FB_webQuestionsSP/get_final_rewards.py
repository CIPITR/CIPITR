import os
import cPickle as pkl
import sys
output = {}
if len(sys.argv)<3:
	dir = '.'
else:
	dir = sys.argv[2]
for f in os.listdir(dir):
    if f.startswith(sys.argv[1]):
	output2 = pkl.load(open(dir+'/'+f))
	for k,v in output2.items():
		if k not in output:
			output[k] = v
		else:
			output[k].extend(v)

avg_reward = 0.0
count = 0.0
for k,v in output.items():
	top_prob = 0.0
	reward = 0.0	
	#if len(v)>1:
	#	print 'found length ', len(v)
	for vi in v:
		#reward = max(reward,vi['reward'])
		vi['reward'] = max(0, vi['reward'] )			
		reward = max(reward,vi['reward'])
		'''
		if top_prob<vi['probability']:
			top_prob = vi['probability']
			reward = vi['reward']
		elif abs(top_prob-vi['probability'])<1:#top_prob==vi['probability']:#abs(top_prob-vi['probability'])<1:
			top_prob = vi['probability']
                        reward = max(reward,vi['reward'])
		'''
	avg_reward += reward
	count += 1
print avg_reward/count, count
