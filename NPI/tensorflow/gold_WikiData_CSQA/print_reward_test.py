import sys
import numpy as np
batch_size = int(sys.argv[1])
beam_size = int(sys.argv[2])
if len(sys.argv)>5:
	reward_type = sys.argv[5]
else:
	reward_type = 'jaccard'
reward_per_beam_id = np.zeros([batch_size, beam_size])
beam_id = -1
batch_id = -1
printed_reward = False
avg_reward_over_epoch = 0.0
avg_reward_over_epoch_topbeam = 0.0
running_avg_reward = []
running_avg_reward_topbeam = []
num_batches = 0
def print_reward(reward):
    batch_size = len(reward)
    beam_size= len(reward[0])
    best_reward_till_beam = {i:0.0 for i in xrange(beam_size)}
    avg_reward_at_beam = {i:0.0 for i in xrange(beam_size)}
    for batch_id in xrange(batch_size):
        for beam_id in xrange(beam_size):
            best_reward_till_beam[beam_id] += float(max(reward[batch_id][:(beam_id+1)]))
            avg_reward_at_beam[beam_id] += float(reward[batch_id][beam_id])
    best_reward_till_beam = {k:v/float(batch_size) for k,v in best_reward_till_beam.items()}
    avg_reward_at_beam = {k:v/float(batch_size) for k,v in avg_reward_at_beam.items()}
    ''' 
    for k in xrange(beam_size):
        print '[[NEW]] for beam ', k, ' best reward till this beam', best_reward_till_beam[k], ' (avg reward at this beam =', avg_reward_at_beam[k], ')'
    print ''
    if any([x>1 for x in best_reward_till_beam.values()]):
	print best_reward_till_beam
	raise Exception('reward >1')
    '''
    return best_reward_till_beam[beam_size-1], best_reward_till_beam[int(sys.argv[4])]

#validation_going_on = False
for line in open(sys.argv[3]).readlines():
        '''
	if 'Going for validation' in line:
		#print 'validation start'
		validation_going_on = True
		continue
	if validation_going_on:
		if 'Validation over' in line:
			#print 'validation end'	
			validation_going_on = False
		else:
			continue	
	'''
	if not (line.startswith('batch id') and ':: Query ::' in line) and not line.startswith('beam id') and not line.startswith('In calculate') and not 'best reward till this beam' in line and not 'Validation over... overall avg. test reward' in line and not '(avg over batch) test reward' in line:
                continue
	if line.startswith('batch id') and ':: Query ::' in line:
		batch_id = int(line.split(':')[0].replace('batch id','').strip())	
		#if batch_id==0:
		#	print '\n'	
		#print line.strip()
		printed_reward = False
	if line.startswith('beam id'):
		beam_id = int(line.strip().replace('beam id',''))	
		#print line.strip(),
		printed_reward = False
	if line.startswith('In calculate'):
                try:
			line = line.split('reward_all =')[1].replace('[','').replace(']','').strip()
			rewards = [float(x.strip()) for x in line.split(' ')]
	                if reward_type=='recall':	
        	                reward = rewards[0]
                	elif reward_type=='precision':
                        	reward = rewards[1]
	                elif reward_type=='jaccard':
        	                reward = rewards[2]
                	elif reward_type=='f1':
                        	reward = rewards[3]
		except:
			reward = 0.0
		#line = line.split('flag')[0].strip()
		#reward = max(float(line.strip().split('=')[-1].strip()),0.)
		reward_per_beam_id[batch_id][beam_id] = reward
		#print reward
		printed_reward = False
	if 'best reward till this beam' in line:
		if not printed_reward:
			batch_reward, batch_reward_at_topbeam = print_reward(reward_per_beam_id)
			printed_reward = True
			reward_per_beam_id = np.zeros([batch_size, beam_size])
			if batch_reward==-100.0:
				batch_reward = -1.0
			avg_reward_over_epoch += batch_reward
			avg_reward_over_epoch_topbeam += batch_reward_at_topbeam
			running_avg_reward.append(batch_reward)
			running_avg_reward_topbeam.append(batch_reward_at_topbeam)
			num_batches += 1
		#print '[[OLD]] ',line.strip()
	'''	
	if '(avg over batch) ' in line:
                #print len(running_avg_reward)
		if len(running_avg_reward)==50:
			#print '[[OLD]] ', line.strip()
			print '[[NEW]] (avg over batch) train reward (',reward_type, ')',batch_reward, 'running avg over batches (best over all) ', float(sum(running_avg_reward))/50.0, '  and (top beam) ',  float(sum(running_avg_reward_topbeam))/50.0, ' running avg from top ', float(avg_reward_over_epoch_topbeam)/float(num_batches)
			running_avg_reward = []
			running_avg_reward_topbeam = []
	'''			
	if 'Validation over... overall avg. test reward' in line:
		if reward_type=='jaccard':
			print '[[OLD]] ', line.strip()
		epoch = line.split(' ')[2]
		print '[[NEW]]  epoch ', epoch, 'training is completed ... overall avg. train reward (',reward_type, ') (best over all) ', float(avg_reward_over_epoch)/float(num_batches), ' avg. train reward (top beam) ', float(avg_reward_over_epoch_topbeam)/float(num_batches)
		avg_reward_over_epoch = 0.0
		avg_reward_over_epoch_topbeam = 0.0
		num_batches = 0
		print ''
