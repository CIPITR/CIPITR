from model_vanilla_stock import NPI
from read_data import ReadBatchData
import itertools
from interpreter import Interpreter
import datetime
import numpy as np
import json
import random
import sys
import re
import string
import cPickle as pkl
import os
import math
from pipeproxy import proxy
from multiprocessing import Process, Lock, freeze_support
import msgpack
import time
import torch



class TrainModel():
    def __init__(self, param):
        np.random.seed(1)
        torch.manual_seed(999)
        #if torch.cuda.is_available(): torch.cuda.manual_seed_all(999)
        self.param = param
        self.run_interpreter = True
        self.run_validation = False
        self.generate_data = False
        self.param = param
        if not self.generate_data and os.path.exists(self.param['model_dir']+'/model_data.pkl'):
            self.pickled_train_data = pkl.load(open(self.param['model_dir']+'/model_data.pkl'))
        else:
            self.pickled_train_data = {}
        if 'use_kb_emb' not in self.param:
            self.param['use_kb_emb'] = True
        #self.param['use_kb_emb'] = False
        self.starting_epoch = 0
        self.starting_overall_step_count = 0
        self.starting_validation_reward_overall = 0
        self.starting_validation_reward_topbeam = 0
        if 'dont_look_back_attention' not in self.param:
            self.param['dont_look_back_attention'] = False
        if 'concat_query_npistate' not in self.param:
            self.param['concat_query_npistate'] = False
        if 'query_attention' not in self.param:
            self.param['query_attention'] = False
        if self.param['dont_look_back_attention']:
            self.param['query_attention'] = True
        if 'single_reward_function' not in self.param:
            self.param['single_reward_function'] = False
        if 'terminate_prog' not in self.param:
            self.param['terminate_prog'] = False
            terminate_prog = False
        else:
            terminate_prog = self.param['terminate_prog']
        if 'none_decay' not in self.param:
            self.param['none_decay'] = 0

        if 'train_mode' not in self.param:
            self.param['train_mode'] = 'reinforce'
        self.qtype_wise_batching = self.param['questype_wise_batching']
        self.read_data = ReadBatchData(param)
	if self.param['question_type']=='all':
                self.param['question_type']=','.join(self.read_data.all_questypes_inv.values())
        print "initialized read data"
        if 'relaxed_reward_till_epoch' in self.param:
            relaxed_reward_till_epoch = self.param['relaxed_reward_till_epoch']
        else:
            self.param['relaxed_reward_till_epoch'] = [-1,-1]
            relaxed_reward_till_epoch = [-1,-1]
        if 'params_turn_on_after' not in self.param:
            self.param['params_turn_on_after'] = 'epoch'
        if self.param['params_turn_on_after']!='epoch' and self.param['params_turn_on_after']!='batch':
            raise Exception('params_turn_on_after should be epoch or batch')
        if 'print' in self.param:
            self.printing = self.param['print']
        else:
            self.param['print'] = False
            self.printing = True
        if 'prune_beam_type_mismatch' not in self.param:
            self.param['prune_beam_type_mismatch'] = 0
        if 'prune_after_epoch_no.' not in self.param:
            self.param['prune_after_epoch_no.'] = [self.param['max_epochs'],1000000]
        if self.param['question_type']=='verify':
            boolean_reward_multiplier = 1
        else:
            boolean_reward_multiplier = 0.1
        if 'print_valid_freq' not in self.param:
            self.param['print_valid_freq'] = self.param['print_train_freq']
        if 'valid_freq' not in self.param:
            self.param['valid_freq'] = 100
        if 'unused_var_penalize_after_epoch' not in self.param:
            self.param['unused_var_penalize_after_epoch'] =[self.param['max_epochs'],1000000]
        unused_var_penalize_after_epoch = self.param['unused_var_penalize_after_epoch']
        if 'epoch_for_feasible_program_at_last_step' not in self.param:
            self.param['epoch_for_feasible_program_at_last_step']=[self.param['max_epochs'],1000000]
        if 'epoch_for_biasing_program_sample_with_target' not in self.param:
            self.param['epoch_for_biasing_program_sample_with_target'] = [self.param['max_epochs'],1000000]
        if 'epoch_for_biasing_program_sample_with_last_variable' not in self.param:
            self.param['epoch_for_biasing_program_sample_with_last_variable'] = [self.param['max_epochs'],100000]
        if 'use_var_key_as_onehot' not in self.param:
            self.param['use_var_key_as_onehot'] = False
        reward_func = "f1"
        self.param['reward_function'] = "f1"
        if 'relaxed_reward_strict' not in self.param:
            relaxed_reward_strict = False
            self.param['relaxed_reward_strict'] = relaxed_reward_strict
        else:
            relaxed_reward_strict = self.param['relaxed_reward_strict']
        if param['parallel']==1:
            raise Exception('Need to fix the intermediate rewards for parallelly executing interpreter')
        for k,v in param.items():
            print 'PARAM: ', k , ':: ', v
        print 'loaded params '
        self.train_data = []
        if os.path.isdir(param['train_data_file']):
            self.training_files = [param['train_data_file']+'/'+x for x in os.listdir(param['train_data_file']) if x.endswith('.pkl')]
        elif not isinstance(param['train_data_file'], list):
            self.training_files = [param['train_data_file']]
        else:
            self.training_files = param['train_data_file']
            random.shuffle(self.training_files)
        self.valid_data = []
        if os.path.isdir(param['test_data_file']):
            self.valid_files = [param['test_data_file']+'/'+x for x in os.listdir(param['test_data_file']) if x.endswith('.pkl')]
        elif not isinstance(param['test_data_file'], list):
            self.valid_files = [param['test_data_file']]
        else:
            self.valid_files = param['test_data_file']
        for file in self.valid_files:
            temp = pkl.load(open(file))
	    #temp = self.rectify_ques_type(temp)
            #temp = self.remove_bad_data(temp)
            temp = self.add_data_id(temp)
            self.valid_data.extend(temp)
        if self.qtype_wise_batching:
            self.valid_data_map = self.read_data.get_data_per_questype(self.valid_data)
            self.valid_batch_size_types = self.get_batch_size_per_type(self.valid_data_map)
            self.n_valid_batches = int(math.ceil(float(sum([len(x) for x in self.valid_data_map.values()])))/float(self.param['batch_size']))
        else:
            self.n_valid_batches = int(math.ceil(float(len(self.valid_data))/float(self.param['batch_size'])))

        if not os.path.exists(param['model_dir']):
            os.mkdir(param['model_dir'])
        self.model_file = os.path.join(param['model_dir'],param['model_file'])
        learning_rate = param['learning_rate']
        start = time.time()
        self.model = NPI(param, self.read_data.none_argtype_index, self.read_data.num_argtypes, \
                         self.read_data.num_progs, self.read_data.max_arguments,
                         self.read_data.wikidata_rel_embed, self.read_data.wikidata_rel_date_embed, \
                         self.read_data.vocab_init_embed, self.read_data.program_to_argtype, \
                         self.read_data.program_to_targettype)
        self.checkpoint_prefix = os.path.join(param['model_dir'], param['model_file'])
        if os.path.exists(self.checkpoint_prefix):
            self.model.load_state_dict(torch.load(self.checkpoint_prefix))
            fr = open(self.param['model_dir']+'/metadata.txt').readlines()
            self.starting_epoch = int(fr[0].split(' ')[1].strip())
            self.starting_overall_step_count = int(fr[1].split(' ')[1].strip())
            self.starting_validation_reward_overall = float(fr[2].split(' ')[1].strip())
            self.starting_validation_reward_topbeam = float(fr[3].split(' ')[1].strip())
            print 'restored model'
        end = time.time()
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas=[0.9,0.999], weight_decay=1e-5)
        print self.model
	#self.printing = False
        print 'model created in ', (end-start), 'seconds'

        self.interpreter = Interpreter(self.param['freebase_dir'], self.param['num_timesteps'], \
                                       self.read_data.program_type_vocab, self.read_data.argument_type_vocab, False, terminate_prog, relaxed_reward_strict, reward_function = reward_func, boolean_reward_multiplier = boolean_reward_multiplier, relaxed_reward_till_epoch=relaxed_reward_till_epoch, unused_var_penalize_after_epoch=unused_var_penalize_after_epoch)
        if self.param['parallel'] == 1:
            self.InterpreterProxy, self.InterpreterProxyListener = proxy.createProxy(self.interpreter)
            self.interpreter.parallel = 1
            self.lock = Lock()
	self.rule_based_logs = json.load(open(self.param['freebase_dir']+'/rulebased_program_jaccard_date_operation.json'))
	self.aggregated_results = {}
        print "initialized interpreter"

    def perform_full_validation(self, epoch, overall_step_count):
        self.model = self.model.eval()
        valid_reward = 0
        valid_reward_at0 = 0
	print 'number of batches ', self.n_valid_batches, 
        for i in xrange(self.n_valid_batches):
            train_batch_dict = self.get_batch(i, self.valid_data, self.valid_data_map, self.valid_batch_size_types)
            avg_batch_reward_at0, avg_batch_reward, _ = self.perform_validation(train_batch_dict, epoch, overall_step_count)
            if i%self.param['print_valid_freq']==0 and i>0:
                valid_reward += avg_batch_reward
                valid_reward_at0 += avg_batch_reward_at0
                avg_valid_reward = float(valid_reward)/float(i+1)
                avg_valid_reward_at0 = float(valid_reward_at0)/float(i+1)
                print ('Valid Results in Epoch  %d Step %d (avg over batch) valid reward (over all) =%.6f valid reward (at top beam)=%.6f running avg valid reward (over all)=%.6f running avg valid reward (at top beam)=%.6f' %(epoch, i, avg_batch_reward, avg_batch_reward_at0, avg_valid_reward, avg_valid_reward_at0))
		#print 'aggregated results ',self.aggregated_results
		pkl.dump(self.aggregated_results, open(self.param['output_file'],'w'))
        overall_avg_valid_reward = valid_reward/float(self.n_valid_batches)
        overall_avg_valid_reward_at0 = valid_reward_at0/float(self.n_valid_batches)
        return overall_avg_valid_reward_at0, overall_avg_valid_reward

    def feeding_dict2(self, reward, ProgramProb, logProgramProb, per_step_prob, entropy, \
                        IfPosIntermediateReward, mask_IntermediateReward, IntermediateReward, relaxed_rewards, overall_step_count):
        feed_dict = {}
        feed_dict['reward'] = reward.astype(np.float32)
        feed_dict['ProgramProb'] = ProgramProb.data.cpu().numpy().astype(np.float32)
        feed_dict['logProgramProb'] = logProgramProb.data.cpu().numpy().astype(np.float32)
        feed_dict['per_step_prob'] = per_step_prob.astype(np.float32)
        feed_dict['entropy'] = entropy.data.cpu().numpy().astype(np.float32)
        feed_dict['IfPosIntermediateReward'] = IfPosIntermediateReward.astype(np.float32)
        feed_dict['mask_IntermediateReward'] = mask_IntermediateReward.astype(np.float32)
        feed_dict['IntermediateReward'] = IntermediateReward.astype(np.float32)
        feed_dict['Relaxed_reward'] = relaxed_rewards.astype(np.float32)
        feed_dict['overall_step_count'] = overall_step_count
        return feed_dict

    def feeding_dict1(self, encoder_inputs_w2v, encoder_inputs_kb_emb, variable_mask, \
                      variable_embed, variable_atten, kb_attention, batch_response_type, \
                      batch_required_argtypes, feasible_program_at_last_step, bias_prog_sampling_with_target,\
                      bias_prog_sampling_with_last_variable, epoch_inv, epsilon, PruneNow):
        feed_dict = {}
        feed_dict['encoder_text_inputs_w2v'] = np.transpose(encoder_inputs_w2v)
        feed_dict['encoder_text_inputs_kb_emb'] = encoder_inputs_kb_emb
        feed_dict['preprocessed_var_mask_table'] = variable_mask
        feed_dict['preprocessed_var_emb_table'] = variable_embed
        feed_dict['kb_attention'] = kb_attention
        # in phase 1 we should sample only generative programs
        temp = np.zeros([self.param['batch_size'], self.read_data.num_progs], dtype = np.int64)
        for i in self.read_data.program_variable_declaration_phase:
            temp[:,i] = 1
        feed_dict['progs_phase_1'] = temp
        # in phase 2 we should not sample generative programs and  we can sample all other programs
        temp = np.zeros([self.param['batch_size'], self.read_data.num_progs], dtype = np.int64)
        for i in self.read_data.program_algorithm_phase:
            temp[:,i] = 1
        feed_dict['progs_phase_2'] = temp
        feed_dict['gold_target_type'] = batch_response_type.astype(np.int64)
        feed_dict['randomness_threshold_beam_search'] = epsilon
        feed_dict['DoPruning'] = PruneNow
        feed_dict['last_step_feasible_program'] = feasible_program_at_last_step*np.ones((1,1))
        feed_dict['bias_prog_sampling_with_last_variable'] = bias_prog_sampling_with_last_variable*np.ones((1,1))
        feed_dict['bias_prog_sampling_with_target'] = bias_prog_sampling_with_target*np.ones((1,1))
        feed_dict['required_argtypes'] = batch_required_argtypes
        feed_dict['relaxed_reward_multipler'] = epoch_inv*np.ones((1,1), dtype=np.float32)
        return feed_dict

    def map_multiply(self, arg):
        orig_shape = arg[0].shape
        arg0 = np.reshape(arg[0], (self.param['batch_size']*self.param['beam_size'], -1))
        arg1 = np.reshape(arg[1], (self.param['batch_size']*self.param['beam_size'],1))
        mul = np.reshape(np.multiply(arg0, arg1), orig_shape)
        return np.sum(mul,axis=(0,1))

    # Disable
    def blockPrint(self):
        sys.stdout = open(os.devnull, 'w')

    # Restore
    def enablePrint(self):
        sys.stdout = sys.__stdout__

    def forward_pass_interpreter(self, batch_orig_context, a_seq, per_step_probs, \
                                 program_probabilities, variable_value_table, batch_response_entities, \
                                 batch_response_dates, epoch_number, overall_step_count, kb_attention, data_ids):
        program_probabilities = program_probabilities.data.cpu().numpy()

        #Reward_From_Model = np.transpose(np.array(a_seq['Model_Reward_Flag']))

        keys = ['program_type','argument_type','target_type','target_table_index','argument_table_index','variable_value_table']

        new_a_seq = [[dict.fromkeys(keys) for beam_id in xrange(self.param['beam_size'])] \
                      for batch_id in xrange(self.param['batch_size'])]
        def asine(batch_id,beam_id,key):
            new_a_seq[batch_id][beam_id][key] = ['phi' for _ in xrange(self.param['num_timesteps'])]
        [[[asine(batch_id,beam_id,key) for key in keys] for beam_id in xrange(self.param['beam_size'])] \
                              for batch_id in xrange(self.param['batch_size'])]


        def handle_variable_value_table(key,beam_id,timestep,batch_id):
            if key is not 'variable_value_table':
                new_a_seq[batch_id][beam_id][key][timestep] = a_seq[key][beam_id][timestep][batch_id].data.cpu().numpy()
            else:
                new_a_seq[batch_id][beam_id][key] = variable_value_table[batch_id]


        [handle_variable_value_table(key,beam_id,timestep,batch_id) for (key,beam_id,timestep,batch_id) in list(itertools.product\
         (keys,xrange(self.param['beam_size']),xrange(self.param['num_timesteps']),xrange(self.param['batch_size'])))]

        for batch_id in xrange(self.param['batch_size']):
            for beam_id in xrange(self.param['beam_size']):
                new_a_seq[batch_id][beam_id]['program_probability'] = program_probabilities[batch_id][beam_id]
	rewards = []
        intermediate_rewards_flag = []
        mask_intermediate_rewards = []
        intermediate_rewards = []
        relaxed_rewards = []
	
        for batch_id in xrange(self.param['batch_size']):
            if self.printing:
                print 'batch id ', batch_id, ':: Query :: ', batch_orig_context[batch_id]
		#try:
		if data_ids[batch_id] not in self.rule_based_logs:
			new_id = data_ids[batch_id].split('.')[0]+'.P0'
		else:
			new_id = data_ids[batch_id]
		if new_id in self.rule_based_logs:
			print 'Rule Based\'s Query ',self.rule_based_logs[new_id]['query']
			print 'Rule Based Program ',self.rule_based_logs[new_id]['program']
			print 'Rule Based Program\'s Jaccard ', self.rule_based_logs[new_id]['jaccard'] 
		#except:
		#	print ''
		variable_value_table = new_a_seq[batch_id][0]['variable_value_table']
		print 'number of entities: ', len(variable_value_table[self.read_data.argument_type_vocab['entity']]) - (variable_value_table[self.read_data.argument_type_vocab['entity']]==None).sum(),'(', ','.join([v for v in variable_value_table[self.read_data.argument_type_vocab['entity']] if v is not None]),')'
            	print 'number of relations: ', len(variable_value_table[self.read_data.argument_type_vocab['relation']]) - (variable_value_table[self.read_data.argument_type_vocab['relation']]==None).sum(),'(', ','.join([v for v in variable_value_table[self.read_data.argument_type_vocab['relation']] if v is not None and len(v)>1]), ')'
            	print 'number of relation_dates: ', len(variable_value_table[self.read_data.argument_type_vocab['relation_date']]) - (variable_value_table[self.read_data.argument_type_vocab['relation_date']]==None).sum(),'(', ','.join([v for v in variable_value_table[self.read_data.argument_type_vocab['relation_date']] if v is not None and len(v)>1]),')'
            	print 'number of dates: ', len(variable_value_table[self.read_data.argument_type_vocab['date']]) - (variable_value_table[self.read_data.argument_type_vocab['date']]==None).sum(),'(', ','.join([str(v) for v in variable_value_table[self.read_data.argument_type_vocab['date']] if v is not None]),')'
	    data_id_aggregated = data_ids[batch_id].split('.')[0]
            rewards_batch = []
	    likelihood_batch = []
            intermediate_rewards_flag_batch = []
            relaxed_rewards_batch = []
            mask_intermediate_rewards_batch = []
            intermediate_rewards_batch = []
	    beam_count = 0
            for beam_id in xrange(self.param['beam_size']):
                new_a_seq_i = new_a_seq[batch_id][beam_id]
		for timestep in range(len(new_a_seq_i['program_type'])):
                    prog = int(new_a_seq_i['program_type'][timestep])
                    new_a_seq[batch_id][beam_id]['program_type'][timestep] = int(new_a_seq[batch_id][beam_id]['program_type'][timestep])
                    new_a_seq[batch_id][beam_id]['target_type'][timestep] = int(new_a_seq[batch_id][beam_id]['target_type'][timestep])
                    args = new_a_seq_i['argument_table_index'][timestep]
		args = (new_a_seq[batch_id][beam_id], \
                       batch_response_entities[batch_id], \
                       batch_response_dates[batch_id])
                target_value, reward, max_intermediate_reward, relaxed_reward, intermediate_mask, intermediate_reward_flag = self.interpreter.calculate_program_reward(args, epoch_number, overall_step_count)
                if target_value is None:
                       continue	
		variable_value_table = new_a_seq[batch_id][beam_id]['variable_value_table']
		if self.printing:
                    print 'beam id', beam_count
                    print 'per_step_probs',per_step_probs[batch_id,beam_id]
                    print 'product_per_step_prob', np.product(per_step_probs[batch_id,beam_id])
                    print 'per_step_programs [',	
                for timestep in range(len(new_a_seq_i['program_type'])):
                    if self.printing:
			prog = int(new_a_seq_i['program_type'][timestep])
			if self.read_data.program_type_vocab_inv[prog]=='none' or self.read_data.program_type_vocab_inv[prog]=='terminate':
				continue	
			args = new_a_seq_i['argument_table_index'][timestep]
			arg_strs = []
			for arg in range(len(args)):
				argtype = self.read_data.program_to_argtype[prog][arg]
				arg = args[arg]
				v = variable_value_table[argtype][arg]
				if type(v)==list or type(v)==set:
					arg_str = self.read_data.argument_type_vocab_inv[argtype]+'('+str(arg)+')'
				else:
					arg_str = str(v)
				arg_strs.append(arg_str)
			arg_strs = ','.join(arg_strs)
			print self.read_data.program_type_vocab_inv[prog]+'( '+arg_strs+' )',
                        #print self.read_data.program_type_vocab_inv[prog]+'( '+','.join([str(\
                        #           self.read_data.argument_type_vocab_inv[self.read_data.program_to_argtype[prog][arg]])+\
                        #           '('+str(args[arg])+')' for arg in range(len(args))])+' )',
                if self.printing:
                    print ']'
		beam_count += 1
                rewards_batch.append(reward)
                intermediate_rewards_flag_batch.append(intermediate_reward_flag)
                relaxed_rewards_batch.append(relaxed_reward)
                mask_intermediate_rewards_batch.append(intermediate_mask)
                intermediate_rewards_batch.append(max_intermediate_reward)
		likelihood_batch.append(np.product(per_step_probs[batch_id,beam_id]))
	    while len(rewards_batch)<self.param['beam_size']:
		beam_id = beam_id-1
		if self.printing:
                    print 'beam id', beam_count
                    print 'per_step_probs',per_step_probs[batch_id,beam_id]
                    print 'product_per_step_prob', np.product(per_step_probs[batch_id,beam_id])
                    print 'per_step_programs [',
                for timestep in range(len(new_a_seq_i['program_type'])):
                    if self.printing:
			prog = int(new_a_seq_i['program_type'][timestep])
                        args = new_a_seq_i['argument_table_index'][timestep]
			arg_strs = []
                        for arg in range(len(args)):
                                argtype = self.read_data.program_to_argtype[prog][arg]
                                arg = args[arg]
                                v = variable_value_table[argtype][arg]
                                if type(v)==list or type(v)==set:
                                        arg_str = self.read_data.argument_type_vocab_inv[argtype]+'('+str(arg)+')'
                                else:
                                        arg_str = str(v)
                                arg_strs.append(arg_str)
                        arg_strs = ','.join(arg_strs)
                        print self.read_data.program_type_vocab_inv[prog]+'( '+arg_strs+' )',	
                        #print self.read_data.program_type_vocab_inv[prog]+'( '+','.join([str(variable_value_table[self.read_data.program_to_argtype[prog][arg]][args[arg]]) for arg in range(len(args))])+' )',
			#print self.read_data.program_type_vocab_inv[prog]+'( '+','.join([str(\
                        #           self.read_data.argument_type_vocab_inv[self.read_data.program_to_argtype[prog][arg]])+\
                        #           '('+str(args[arg])+')' for arg in range(len(args))])+' )',
                if self.printing:
                    print ']'
		beam_count += 1
		rewards_batch.append(reward)
                intermediate_rewards_flag_batch.append(intermediate_reward_flag)
                relaxed_rewards_batch.append(relaxed_reward)
                mask_intermediate_rewards_batch.append(intermediate_mask)
                intermediate_rewards_batch.append(max_intermediate_reward)
		likelihood_batch.append(np.product(per_step_probs[batch_id,beam_id]))	
	    top_prob = likelihood_batch[0]
	    for i in range(len(rewards_batch)):
		if likelihood_batch[i]==top_prob:
			if data_id_aggregated not in self.aggregated_results:
				self.aggregated_results[data_id_aggregated] = []
			self.aggregated_results[data_id_aggregated].append({'probability':likelihood_batch[i], 'reward':rewards_batch[i]})	
            rewards.append(rewards_batch)
            intermediate_rewards_flag.append(intermediate_rewards_flag_batch)
            mask_intermediate_rewards.append(mask_intermediate_rewards_batch)
            intermediate_rewards.append(intermediate_rewards_batch)
            relaxed_rewards.append(relaxed_rewards_batch)
        rewards = np.array(rewards)
        #if self.param['reward_from_model']:
        #    rewards = np.where(Reward_From_Model == 0, rewards, -1*np.ones_like(rewards))
        intermediate_rewards = np.array(intermediate_rewards)
        intermediate_rewards_flag = np.array(intermediate_rewards_flag)
        mask_intermediate_rewards = np.array(mask_intermediate_rewards)
        relaxed_rewards = np.array(relaxed_rewards)
        return rewards, intermediate_rewards, relaxed_rewards, intermediate_rewards_flag, mask_intermediate_rewards



    def get_ml_rewards(self,rewards):
       max_rewards = rewards.max(axis=1)[:,None]
       ml_rewards = (rewards == max_rewards).astype(float)
       return ml_rewards

    def get_data_and_feed_dict(self, batch_dict, epoch, overall_step_count):
        batch_orig_context, batch_context_nonkb_words, batch_context_kb_words, \
        batch_context_entities, batch_context_rel, batch_context_rel_dates, batch_context_dates, \
        batch_orig_response, batch_response_entities, batch_response_dates, batch_response_type, batch_required_argtypes, \
        variable_mask, variable_embed, variable_atten, kb_attention, variable_value_table, data_ids = self.read_data.get_batch_data(batch_dict)

        if (self.param['params_turn_on_after']=='epoch' and epoch >= self.param['epoch_for_feasible_program_at_last_step'][0]) or (self.param['params_turn_on_after']=='batch' and overall_step_count >= self.param['epoch_for_feasible_program_at_last_step'][1]):
            feasible_program_at_last_step = 1.
            print 'Using feasible_program_at_last_step'
        else:
            feasible_program_at_last_step = 0.
        if (self.param['params_turn_on_after']=='epoch' and epoch >= self.param['epoch_for_biasing_program_sample_with_target'][0]) or (self.param['params_turn_on_after']=='batch' and overall_step_count >= self.param['epoch_for_biasing_program_sample_with_target'][1]):
            print 'Using program biasing with target'
            bias_prog_sampling_with_target = 1.
        else:
            bias_prog_sampling_with_target = 0.
        if (self.param['params_turn_on_after']=='epoch' and epoch >= self.param['epoch_for_biasing_program_sample_with_last_variable'][0]) or (self.param['params_turn_on_after']=='batch' and overall_step_count >= self.param['epoch_for_biasing_program_sample_with_last_variable'][1]):
            bias_prog_sampling_with_last_variable = 1.
        else:
            bias_prog_sampling_with_last_variable = 0.
        if (self.param['params_turn_on_after']=='epoch' and epoch >= self.param['relaxed_reward_till_epoch'][0]) or (self.param['params_turn_on_after']=='batch' and overall_step_count >= self.param['relaxed_reward_till_epoch'][1]):
            relaxed_reward_multipler = 0.
        else:
            if self.param['params_turn_on_after']=='epoch':
                relaxed_reward_multipler = (self.param['relaxed_reward_till_epoch'][0]-epoch)/float(self.param['relaxed_reward_till_epoch'][0])
                relaxed_reward_multipler = np.clip(relaxed_reward_multipler, 0, 1)
            elif self.param['params_turn_on_after']=='batch':
                relaxed_reward_multipler = (self.param['relaxed_reward_till_epoch'][1]-overall_step_count)/float(self.param['relaxed_reward_till_epoch'][1])
                relaxed_reward_multipler = np.clip(relaxed_reward_multipler, 0, 1)
        epsilon = 0
        if self.param['params_turn_on_after']=='epoch' and self.param['explore'][0] > 0:
            epsilon = self.param["initial_epsilon"]*np.clip(1.0-(epoch/self.param['explore'][0]),0,1)
        elif self.param['params_turn_on_after']=='batch' and self.param['explore'][1] > 0:
            epsilon = self.param["initial_epsilon"]*np.clip(1.0-(overall_step_count/self.param['explore'][1]),0,1)
        PruneNow = 0
        if (self.param['params_turn_on_after']=='epoch' and epoch >= self.param['prune_after_epoch_no.'][0]) or (self.param['params_turn_on_after']=='batch' and overall_step_count >= self.param['prune_after_epoch_no.'][1]):
            PruneNow = 1
        feed_dict1 = self.feeding_dict1(batch_context_nonkb_words, batch_context_kb_words, variable_mask, \
                                variable_embed, variable_atten, kb_attention, batch_response_type, batch_required_argtypes,\
                                 feasible_program_at_last_step, bias_prog_sampling_with_target, bias_prog_sampling_with_last_variable,\
                                  relaxed_reward_multipler, epsilon, PruneNow)
        return feed_dict1, batch_orig_context, batch_response_entities, batch_response_dates, variable_value_table, data_ids

    def perform_validation(self, batch_dict, epoch, overall_step_count):
        self.model = self.model.eval()
        with torch.no_grad():
            feed_dict1, batch_orig_context, batch_response_entities, batch_response_dates, variable_value_table, data_ids = self.get_data_and_feed_dict(batch_dict, epoch, overall_step_count)
            #a_seq, program_probabilities, per_step_probs = self.sess.run([self.action_sequence, self.program_probs, self.per_step_probs], feed_dict=feed_dict1)
            a_seq, program_probabilities, logprogram_probabilities, beam_props, per_step_probs, entropy = self.model(feed_dict1)
            # reshaping per_step_probs for printability
            per_step_probs = per_step_probs.data.cpu().numpy()
            [reward, intermediate_rewards, relaxed_rewards, intermediate_rewards_flag, \
             mask_intermediate_rewards] = self.forward_pass_interpreter(batch_orig_context, a_seq, per_step_probs, \
                                             program_probabilities, variable_value_table, batch_response_entities, \
                                             batch_response_dates, epoch, overall_step_count, feed_dict1['kb_attention'], data_ids)
            reward = np.array(reward)
            relaxed_rewards = np.array(relaxed_rewards)
            reward[reward<0.] = 0.
            self.print_reward(reward)
	    	
            return sum(reward[:,0])/float(self.param['batch_size']), sum(np.max(reward,axis=1))/float(self.param['batch_size']), 0


        #def apply_gradients(self, model, optimizer, gradients):
        #    optimizer.apply_gradients(zip(gradients, self.model.variables))

    def perform_training(self, batch_dict, epoch, overall_step_count):
        self.model = self.model.train()
        print 'in perform_training'
        feed_dict1, batch_orig_context, batch_response_entities, batch_response_dates, variable_value_table, data_ids = self.get_data_and_feed_dict(batch_dict, epoch, overall_step_count)
        # =============================================================================
        # For Proper Run Use this
        # =============================================================================
        if self.param['Debug'] == 0:
            self.optimizer.zero_grad()
            a_seq, program_probabilities, logprogram_probabilities, beam_props, per_step_probs, entropy = self.model(feed_dict1)
            # reshaping per_step_probs for printability
            per_step_probs = per_step_probs.data.cpu().numpy()

            if self.param['parallel'] is not 1:
                [reward, intermediate_rewards, relaxed_rewards, intermediate_rewards_flag, \
                 mask_intermediate_rewards] = self.forward_pass_interpreter(batch_orig_context, a_seq, per_step_probs, \
                                                 program_probabilities, variable_value_table, batch_response_entities, \
                                                 batch_response_dates, epoch, overall_step_count, feed_dict1['kb_attention'], data_ids)
            reward = np.array(reward)
            relaxed_rewards = np.array(relaxed_rewards)
            reward_copy = np.array(reward)
            if self.param['train_mode'] == 'ml':
                reward = self.get_ml_rewards(reward)
                rescaling_term_grad = reward
            else:
                rescaling_term_grad = reward
            feed_dict2 = self.feeding_dict2(reward, program_probabilities, logprogram_probabilities, per_step_probs, entropy, intermediate_rewards_flag, mask_intermediate_rewards, intermediate_rewards, relaxed_rewards, epoch)
            loss = -self.model.backprop(feed_dict2)
            self.optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)


            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()

        # -----------------------------------------------------------------------------
        # =============================================================================
        # For Debugging Use This
        # =============================================================================
        else:
            [a_seq, program_probabilities, logprogram_probabilities, \
             beam_props, per_step_probs] = self.sess.run([self.action_sequence, self.program_probs, self.logProgramProb, \
                                           self.beam_props, self.per_step_probs], feed_dict=feed_dict1)
            per_step_probs = np.array(per_step_probs)
            reward = np.zeros([self.param['batch_size'], self.param['beam_size']])
            loss = 0
        # -----------------------------------------------------------------------------
        reward = reward_copy
        reward[reward<0.] = 0.
        self.print_reward(reward)
        output_loss = loss.detach().cpu().numpy()
        del loss,  feed_dict2, feed_dict1, rescaling_term_grad, intermediate_rewards, relaxed_rewards, intermediate_rewards_flag, \
                 mask_intermediate_rewards, a_seq, program_probabilities, logprogram_probabilities, beam_props, per_step_probs, entropy
        return sum(reward[:,0])/float(self.param['batch_size']), sum(np.max(reward,axis=1))/float(self.param['batch_size']),  output_loss/float(self.param['batch_size'])



    def print_reward(self, reward):
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
        for k in xrange(beam_size):
            print 'for beam ', k, ' best reward till this beam', best_reward_till_beam[k], ' (avg reward at this beam =', avg_reward_at_beam[k], ')'

    def add_data_id(self, data):
        if len(data[0])>=16:
            return data
        for i in xrange(len(data)):
            data[i].append('')
        return data

   
    def is_date(self, x):
        x_orig=x
        if x.startswith('m.'):
                return False
        if len(x.split('-'))==1:
                x= x+'-01-01'
        elif len(x.split('-'))==2:
                x= x+'-01'
        try:
                yyyy = int(x.split('-')[0])
                mm = int(x.split('-')[1])
                dd = int(x.split('-')[2])
                d = datetime.datetime(yyyy, mm, dd)
                return True
        except Exception as e:
                #traceback.print_exc(e)
                return False

    def get_date(self, x):
        x_orig=x
        if x.startswith('m.'):
                return False
        if len(x.split('-'))==1:
                x= x+'-01-01'
        elif len(x.split('-'))==2:
                x= x+'-01'
        try:
                yyyy = int(x.split('-')[0])
                mm = int(x.split('-')[1])
                dd = int(x.split('-')[2])
                d = datetime.datetime(yyyy, mm, dd)
                return d
        except:
                return None

    def date_in(self, q):
        w = q.split(' ')
        dates = []
        for wi in w:
                if self.is_date(wi):
                        d = self.get_date(wi)
                        dates.append(d)
        if len(dates)>0:
                dates = list(set(dates))
        return dates

    def rectify_ques_type(self, data):
        for index in range(len(data)):
	    dates_in_q = self.date_in(data[index][0])	
	    if len(dates_in_q)==0:
	        date_in_q = False
    	    else:
        	date_in_q = True
	   	dates = dates_in_q
		if len(dates)>4:
	            dates=dates[:4]
	    	if len(dates)<4:
	            dates=dates+[0]*(4-len(dates))
	        data[index][6] = dates
	    	if date_in_q and not data[index][10].endswith('_date'):
		    data[index][10] = data[index][10]+'_date'
	return data
    
    def remove_bad_data(self, data):
	for index, d in enumerate(data[:]):		
	    response_ents = set(d[8])
            response_dates = set(d[9])
            if len(response_ents)==0 and len(response_dates)==0:
                data.remove(d)
                continue
            ents = set(d[3])
            if 0 in ents:
                ents.remove(0)
            if len(ents)==0:
                data.remove(d)
                continue
            rels = set(d[4])
            if 0 in rels:
                rels.remove(0)
            rel_dates = set(d[5])
            if 0 in rel_dates:
                rel_dates.remove(0)
	    dates = set(d[6])
	    if 0 in dates:
		dates.remove(0)
            if len(rels)==0 and len(rel_dates)==0:
                data.remove(d)
                continue
	   
            ent_rel_kb_subgraph = sum(d[11])
            ent_rel_date_kb_subgraph = sum(d[12])
	    ent_rel_rel_kb_subgraph = sum(d[13])
	    ent_rel_rel_date_kb_subgraph = sum(d[14])
	    ent_rel_rel_dc_kb_subgraph = sum(d[15])
	    ent_rel_rel_date_dc_kb_subgraph = sum(d[16])
	    if 'infchain_2' in d[10] and ent_rel_rel_kb_subgraph==0 and ent_rel_rel_date_kb_subgraph==0:
		data.remove(d)
		continue
            if 'infchain_1' in d[10] and ent_rel_kb_subgraph==0 and ent_rel_date_kb_subgraph==0:
                data.remove(d)
		continue
	    '''if '_date' in d[10] and (len(dates)==0 or (ent_rel_rel_dc_kb_subgraph==0 and ent_rel_rel_date_dc_kb_subgraph==0)):
		if '_date' in d[10] and (len(dates)==0):
			print 'removed data because of no date ', d[0]
		if ent_rel_rel_dc_kb_subgraph==0 and ent_rel_rel_date_dc_kb_subgraph==0:
			print 'ent_rel_rel_dc_kb_subgraph==0 and ent_rel_rel_date_dc_kb_subgraph==0'
		data.remove(d)
	    '''
        return data

    def get_batch_size_per_type(self, data_map):
        num_data_types = len(data_map)
        batch_size_types = {qtype:int(float(self.param['batch_size'])/float(num_data_types)) for qtype in data_map}
        diff = self.param['batch_size'] - sum(batch_size_types.values())
        qtypes = data_map.keys()
        count = 0
        while diff>0 and count<len(qtypes):
            batch_size_types[qtypes[count]]+=1
            count += 1
            if count == len(qtypes):
                count = 0
            diff -= 1
        if sum(batch_size_types.values())!=self.param['batch_size']:
            raise Exception("sum(batch_size_types.values())!=self.param['batch_size']")
        return batch_size_types

    def get_batch(self, i, data, data_map, batch_size_types):
        if not self.qtype_wise_batching:
            batch_dict = data[i*self.param['batch_size']:(i+1)*self.param['batch_size']]
            if len(batch_dict)<self.param['batch_size']:
                batch_dict.extend(data[:self.param['batch_size']-len(batch_dict)])
        else:
            batch_dict = []
            for qtype in data_map:
                data_map_qtype = data_map[qtype][i*batch_size_types[qtype]:(i+1)*batch_size_types[qtype]]
                if len(data_map_qtype)<batch_size_types[qtype]:
                    data_map_qtype.extend(data_map[qtype][:batch_size_types[qtype]-len(data_map_qtype)])
                batch_dict.extend(data_map_qtype)
            if len(batch_dict)!=self.param['batch_size']:
                raise Exception("len(batch_dict)!=self.param['batch_size']")
        return batch_dict

    def train(self):
        best_valid_loss = float("inf")
        best_valid_epoch = 0
        last_overall_avg_train_loss = 0

        last_avg_valid_reward = self.starting_validation_reward_overall
        last_avg_valid_reward_at0 = self.starting_validation_reward_topbeam
        overall_step_count = self.starting_overall_step_count
        epochwise_step_count = 0
        self.qtype_wise_batching = True
	epoch = self.starting_epoch
	avg_valid_reward_at0, avg_valid_reward = self.perform_full_validation(epoch, overall_step_count)
	pkl.dump(self.aggregated_results, open(self.param['output_file'],'w'))
	#json.dump(self.aggregated_results, open('aggregated_results.json','w'), indent=1)
	print 'Epoch ', epoch, ' Validation over... overall avg. valid reward (over all)', avg_valid_reward, ' valid reward (at top beam)', avg_valid_reward_at0

def main():
    #tf.enable_eager_execution(config=tf.ConfigProto(allow_soft_placement=True,
    #                                    log_device_placement=True), device_policy=tfe.DEVICE_PLACEMENT_SILENT)
    torch.backends.cudnn.benchmark = True
    params_file = sys.argv[1]
    timestamp = sys.argv[2]
    param = json.load(open(params_file))
    param['model_dir']= param['model_dir']+'/'+param['question_type']+'_'+timestamp
    param['output_file'] = 'aggregated_output_'+param['question_type']+'_'+timestamp+'.pkl'
    train_model = TrainModel(param)
    train_model.train()


if __name__=="__main__":
    freeze_support()
    main()




