#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@authors: G. Ahmed Ansari, Amrita Saha
"""
from model import NPI
from read_data import ReadBatchData
import itertools
from interpreter import Interpreter
import numpy as np
import json
import random
import sys
import re
import string
import cPickle as pkl
import os
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import math
import time

class TrainModel():
    def __init__(self, param):
        # fixing seeds
        np.random.seed(1)
        tf.set_random_seed(1)
        self.param = param # a shared dictionoary of parameters
        # in case of restoting the model from a certain point, the epoch numbers are stored
        self.starting_epoch = 0
        self.starting_overall_step_count = 0
        self.starting_validation_reward_overall = 0
        self.starting_validation_reward_topbeam = 0
        if 'npi_core_dim' not in self.param:
            self.param['npi_core_dim'] = self.param['hidden_dim']
        if 'cell_dim' not in self.param:
            self.param['cell_dim'] = self.param['hidden_dim']
        if 'dont_look_back_attention' not in self.param: #boolean flag indicating whether the never-look-back attention should be implemented over the query words, when sampling for actions
            self.param['dont_look_back_attention'] = False
        if 'concat_query_npistate' not in self.param: #boolean flag indicating whether the query embedding should be concatenated with the program state vector when sampling actions
            self.param['concat_query_npistate'] = False
        if 'query_attention' not in self.param: #boolean flag indicating whether the attention over query words should be applied when sampling for actions
            self.param['query_attention'] = False
        if self.param['dont_look_back_attention']:
            self.param['query_attention'] = True
        if 'single_reward_function' not in self.param:
            self.param['single_reward_function'] = False
        if 'terminate_prog' not in self.param: # param['terminate_prog'] controls whether terminate operator is a part of the operator vocabulary
            self.param['terminate_prog'] = False
            terminate_prog = False
        else:
            terminate_prog = self.param['terminate_prog']
        if 'none_decay' not in self.param:
            self.param['none_decay'] = 0 # controls whether or not to penalize for sampling none/no-op operator
            # none or no-op terms are used interchangeably

        if 'train_mode' not in self.param: # controls whether to just use reinforce or alternatively do an ML(supervised) pretraining prior to reinforce
            self.param['train_mode'] = 'reinforce'
        self.qtype_wise_batching = self.param['questype_wise_batching'] # @todo : @amrita 1 line abt this
        self.read_data = ReadBatchData(param) # initializes the ReadBatchData object for preprocessing and reading the training/test data batch by batch
        print "initialized read data"
        # param['relaxed_reward_till_epoch'] controls whether or not to supply auxillary rewards for atleast predicting the correct answer type
        if 'quantitative' in self.param['question_type'] or 'comparative' in self.param['question_type']:
            if 'relaxed_reward_till_epoch' in self.param:
                relaxed_reward_till_epoch = self.param['relaxed_reward_till_epoch']
            else:
                self.param['relaxed_reward_till_epoch'] = [-1,-1]
                relaxed_reward_till_epoch = [-1,-1]
        else:
            self.param['relaxed_reward_till_epoch'] = [-1,-1]
            relaxed_reward_till_epoch = [-1,-1]

        if 'params_turn_on_after' not in self.param: # this parameter controls whether to turn on features after units in epochs or in no. of batches
            self.param['params_turn_on_after'] = 'epoch'
        if self.param['params_turn_on_after']!='epoch' and self.param['params_turn_on_after']!='batch':
            raise Exception('params_turn_on_after should be epoch or batch')
        if 'print' in self.param: # controls print verbosity
            self.printing = self.param['print']
        else:
            self.param['print'] = False
            self.printing = True
        if 'prune_beam_type_mismatch' not in self.param: # controls whether or not to do beam pruning
            self.param['prune_beam_type_mismatch'] = 0
        if 'prune_after_epoch_no.' not in self.param:
            self.param['prune_after_epoch_no.'] = [self.param['max_epochs'],1000000] # controls when to do beam pruning
        if self.param['question_type']=='verify':
            boolean_reward_multiplier = 1 #an internal weight factor for the reward, to ensure that the `Verififcation' question type which has dense reward structure does not bias the model or the model does not get biased towards generating only boolean answers 
        else:
            boolean_reward_multiplier = 0.1
        # param['print_train_freq'] print frequency during training
        # param['print_valid_freq'] print frequency during validation
        if 'print_valid_freq' not in self.param:
            self.param['print_valid_freq'] = self.param['print_train_freq']
        if 'valid_freq' not in self.param:
            self.param['valid_freq'] = 100
        if 'unused_var_penalize_after_epoch' not in self.param: #boolean flag indicating whether the model should be penalized for not using declared variables
            self.param['unused_var_penalize_after_epoch'] =[self.param['max_epochs'],1000000]
        unused_var_penalize_after_epoch = self.param['unused_var_penalize_after_epoch']
        if 'epoch_for_feasible_program_at_last_step' not in self.param: # controls when to enable feasible_samoling feature explained in NPI class
            self.param['epoch_for_feasible_program_at_last_step']=[self.param['max_epochs'],1000000]
        if 'epoch_for_biasing_program_sample_with_target' not in self.param: #hyperparameter deciding the number of epochs after which the biasing of the action sampling towards the desired answer type gets activated
            self.param['epoch_for_biasing_program_sample_with_target'] = [self.param['max_epochs'],1000000]
        if 'epoch_for_biasing_program_sample_with_last_variable' not in self.param: #hyperparameter deciding the number of epochs after which biasing of the action sampling towards consuming the last created variable gets activated
            self.param['epoch_for_biasing_program_sample_with_last_variable'] = [self.param['max_epochs'],100000]
        if 'use_var_key_as_onehot' not in self.param: #boolean flag indicating whether the key embedding should be simply a one-hot embedding of the variable location in the memory
            self.param['use_var_key_as_onehot'] = False
        if 'reward_function' not in self.param: # controls which reward metric to choose between Jaccard Score or F1 Score
            reward_func = "jaccard"
            self.param['reward_function'] = "jaccard"
        else:
            reward_func = self.param['reward_function']
        if 'relaxed_reward_strict' not in self.param: #boolean flag indicating whether the relaxed reward 
            relaxed_reward_strict = False
            self.param['relaxed_reward_strict'] = relaxed_reward_strict
        else:
            relaxed_reward_strict = self.param['relaxed_reward_strict']

        for k,v in param.items():
            print 'PARAM: ', k , ':: ', v
        print 'loaded params '
        self.train_data = []
        # loading of train data =============================================================================================
        if os.path.isdir(param['train_data_file']):
            self.training_files = [param['train_data_file']+'/'+x for x in os.listdir(param['train_data_file']) if x.endswith('.pkl')]
        elif not isinstance(param['train_data_file'], list):
            self.training_files = [param['train_data_file']]
        else:
            self.training_files = param['train_data_file']
            random.shuffle(self.training_files)
        sys.stdout.flush()
        # -------------------------------------------------------------------------------------------------------------------
        # loading of validation data ========================================================================================
        self.valid_data = []
        if os.path.isdir(param['valid_data_file']):
            self.valid_files = [param['valid_data_file']+'/'+x for x in os.listdir(param['valid_data_file']) if x.endswith('.pkl')]
        elif not isinstance(param['valid_data_file'], list):
            self.valid_files = [param['valid_data_file']]
        else:
            self.valid_files = param['valid_data_file']
        for file in self.valid_files:
            self.valid_data.extend(pkl.load(open(file)))
        if self.qtype_wise_batching:
            self.valid_data_map = self.read_data.get_data_per_questype(self.valid_data)
            self.valid_batch_size_types = self.get_batch_size_per_type(self.valid_data_map)
            self.n_valid_batches = int(math.ceil(float(sum([len(x) for x in self.valid_data_map.values()])))/float(self.param['batch_size']))
        else:
            self.n_valid_batches = int(math.ceil(float(len(self.valid_data))/float(self.param['batch_size'])))
        # -------------------------------------------------------------------------------------------------------------------

        #provisions form saving/loading model ===============================================================================
        if not os.path.exists(param['model_dir']):
            os.mkdir(param['model_dir'])
        self.model_file = os.path.join(param['model_dir'],param['model_file'])
        # -------------------------------------------------------------------------------------------------------------------
        with tf.Graph().as_default():
            start = time.time()
            # initializing the model from the NPI (model) class
            self.model = NPI(self.param, self.read_data.none_argtype_index, self.read_data.num_argtypes, \
                             self.read_data.num_progs, self.read_data.max_arguments, \
                             self.read_data.rel_index, self.read_data.type_index, \
                             self.read_data.wikidata_rel_embed, self.read_data.wikidata_type_embed, \
                             self.read_data.vocab_init_embed, self.read_data.program_to_argtype, \
                             self.read_data.program_to_targettype)
            self.model.create_placeholder() # creating place-holders for different model inputs

            [self.action_sequence, self.program_probs, self.logProgramProb, self.Reward_placeholder, self.Relaxed_rewards_placeholder, \
             self.train_op, self.loss, self.beam_props, self.per_step_probs, self.IfPosIntermediateReward, \
             self.mask_IntermediateReward, self.IntermediateReward] = self.model.reinforce() # the reinforce function is where the loss calculation happens

            # to be used for debugging
            if param['Debug'] == 0:
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.Session(config=config)
                self.sess = tf.Session()
            else:
                self.sess = tf_debug.LocalCLIDebugWrapperSession(tf.Session())
            self.saver = tf.train.Saver()

        #provisions form saving/loading model ===============================================================================
            ckpt = tf.train.get_checkpoint_state(param['model_dir'])
            if ckpt and ckpt.model_checkpoint_path:
                print "best model exists in ", self.model_file, "... restoring from there "
                self.saver = tf.train.Saver()
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                fr = open(self.param['model_dir']+'/metadata.txt').readlines()
                self.starting_epoch = int(fr[0].split(' ')[1].strip())
                self.starting_overall_step_count = int(fr[1].split(' ')[1].strip())
                self.starting_validation_reward_overall = float(fr[2].split(' ')[1].strip())
                self.starting_validation_reward_topbeam = float(fr[3].split(' ')[1].strip())
                print 'restored model'
        # -------------------------------------------------------------------------------------------------------------------
            else:
                init = tf.global_variables_initializer()
                self.sess.run(init)
                print 'initialized model'
            end = time.time()
            print 'model created in ', (end-start), 'seconds'
            sys.stdout.flush()
        # the interpreter module processes the programs generated and returns rewards
        self.interpreter = Interpreter(self.param['wikidata_dir'], self.param['num_timesteps'], \
                                       self.read_data.program_type_vocab, self.read_data.argument_type_vocab, self.printing, \
                                       relaxed_reward_strict, reward_function = reward_func, \
                                       boolean_reward_multiplier = boolean_reward_multiplier, \
                                       relaxed_reward_till_epoch=relaxed_reward_till_epoch, \
                                       unused_var_penalize_after_epoch=unused_var_penalize_after_epoch)

        print "initialized interpreter"

    def perform_full_validation(self, epoch, overall_step_count):
        """
        This method invokes the inference (forward pass) on the validation set and prints the batch-wise average reward
        """
        valid_reward = 0
        valid_reward_at0 = 0
        for i in xrange(self.n_valid_batches):
            train_batch_dict = self.get_batch(i, self.valid_data, self.valid_data_map, self.valid_batch_size_types)
            avg_batch_reward_at0, avg_batch_reward, _ = self.perform_validation(train_batch_dict, epoch, overall_step_count)
            if i%self.param['print_valid_freq']==0 and i>0:
                valid_reward += avg_batch_reward
                valid_reward_at0 += avg_batch_reward_at0
                avg_valid_reward = float(valid_reward)/float(i+1)
                avg_valid_reward_at0 = float(valid_reward_at0)/float(i+1)
                print ('Valid Results in Epoch  %d Step %d (avg over batch) valid reward (over all) =%.6f valid reward (at top beam)=%.6f running avg valid reward (over all)=%.6f running avg valid reward (at top beam)=%.6f' %(epoch, i, avg_batch_reward, avg_batch_reward_at0, avg_valid_reward, avg_valid_reward_at0))
                sys.stdout.flush()
        overall_avg_valid_reward = valid_reward/float(self.n_valid_batches)
        overall_avg_valid_reward_at0 = valid_reward_at0/float(self.n_valid_batches)
        return overall_avg_valid_reward_at0, overall_avg_valid_reward

    def feeding_dict1(self, encoder_inputs_w2v, encoder_inputs_kb_emb, variable_mask, \
                      variable_embed, variable_atten, kb_attention, batch_response_type, \
                      batch_required_argtypes, feasible_program_at_last_step, bias_prog_sampling_with_target,\
                      bias_prog_sampling_with_last_variable, epoch_inv, epsilon, PruneNow):
        """
        This method prepares the feed_dict used for feeding values into the placeholder variables
        encoder_inputs_w2v: array of binarized vocab indices of the query words 
        encoder_inputs_kb_emb: array of TransE embeddings of the KB entities in the query   
        variable_mask: boolean weight vector indicating which variable locations in the scratch memory contains a valid variable and which are empty positions
        variable_embed: numpy matrix containing the embedding vector of the variables prepopulated in the memory
        variable_atten: numpy matrix containing the attention distribution over the variables of each type in the memory
        kb_attention: numpy matrix indicating for each operator, the argument variable instantiations that are consistent wrt KB
        batch_response_type: numpy matrix containing the desired (predicted) variable type of the answer variable
        batch_required_argtypes: numpy matrix indicating the variable types required to be created in the program in order to generate an answer of the desired (predicted) variable type
        feasible_program_at_last_step: boolean flag indicating whether the action sampling at last timestep is biased towards creating an answer variable of the desired (predicted) variable type
        bias_prog_sampling_with_target: boolean flag indicating whether the action sampling is biased w.r.t the desired (predicted) variable type
        bias_prog_sampling_with_last_variable: boolean flag indicating whether the action sampling is biased wrt the variable type of the last generated variable
        epoch_inv: 1/epoch
        PruneNow: boolean flag indicating whether the beam pruning is activated 
        """
        feed_dict = {}
        for model_enc_inputs_w2v, enc_inputs_w2v in zip(self.model.encoder_text_inputs_w2v, encoder_inputs_w2v):
            feed_dict[model_enc_inputs_w2v] = enc_inputs_w2v
        feed_dict[self.model.encoder_text_inputs_kb_emb] = encoder_inputs_kb_emb
        #print 'preprocessed variable mask for None (',variable_mask.shape, ')',  variable_mask[0]
        for i in xrange(variable_mask.shape[0]):
            for j in xrange(variable_mask.shape[1]):
                feed_dict[self.model.preprocessed_var_mask_table[i][j]] = variable_mask[i][j]
        for i in xrange(variable_embed.shape[0]):
            for j in xrange(variable_embed.shape[1]):
                feed_dict[self.model.preprocessed_var_emb_table[i][j]] = variable_embed[i][j]
        feed_dict[self.model.kb_attention] = kb_attention

        # in phase 1 we should sample only generative programs
        temp = np.zeros([self.param['batch_size'], self.read_data.num_progs], dtype = np.int32)
        for i in self.read_data.program_variable_declaration_phase:
            temp[:,i] = 1
        feed_dict[self.model.progs_phase_1] = temp

        # in phase 2 we should not sample generative programs and  we can sample all other programs
        temp = np.zeros([self.param['batch_size'], self.read_data.num_progs], dtype = np.int32)
        for i in self.read_data.program_algorithm_phase:
            temp[:,i] = 1
        feed_dict[self.model.progs_phase_2] = temp
        feed_dict[self.model.gold_target_type] = batch_response_type
        feed_dict[self.model.required_argtypes] = batch_required_argtypes
        feed_dict[self.model.randomness_threshold_beam_search] = epsilon
        feed_dict[self.model.DoPruning] = PruneNow
        feed_dict[self.model.relaxed_reward_multipler] = epoch_inv*np.ones((1,1), dtype=np.float32)
        feed_dict[self.model.last_step_feasible_program] = feasible_program_at_last_step*np.ones((1,1))
        feed_dict[self.model.bias_prog_sampling_with_target] = bias_prog_sampling_with_target*np.ones((1,1))
        feed_dict[self.model.bias_prog_sampling_with_last_variable] = bias_prog_sampling_with_last_variable*np.ones((1,1))
        return feed_dict


    def forward_pass_interpreter(self, batch_orig_context, a_seq, per_step_probs, \
                                 program_probabilities, variable_value_table, batch_response_entities, \
                                 batch_response_ints, batch_response_bools, epoch_number, overall_step_count):
        """
        This function parses the action sequences that are generated by the NPI module into a format that can be understood by the interpreter. It also prints teh current progress.
        Keyword arguments:
        batch_orig_context: @amrita
        a_seq: The input action sequence recieved from the model. Dictionary object with keys like 'program_type','argument_type', ....
                Each element of the dictionary is a list of length beam_size containing a list of length num_timesteps
                containing tensors of shape batch_size x Dim (Dim depends upon type of dictionary key)
        per_step_probs: tensor containing probability of beams with shape batch_size x beam_size x num_timesteps
        variable_value_table: @amrita
        batch_response_entities: @amrita
        batch_response_ints: @amrita
        batch_response_bools: @amrita
        epoch_number: Current epoch no.
        overall_step_count: Overall no.  of batches elapsed
        """
        Reward_From_Model = np.transpose(np.array(a_seq['Overflow_Penalize_Flag']))

        keys = ['program_type','argument_type','target_type','target_table_index','argument_table_index','variable_value_table']

        new_a_seq = [[dict.fromkeys(keys) for beam_id in xrange(self.param['beam_size'])] \
                      for batch_id in xrange(self.param['batch_size'])]
        def asine(batch_id,beam_id,key):
            new_a_seq[batch_id][beam_id][key] = ['phi' for _ in xrange(self.param['num_timesteps'])]
        [[[asine(batch_id,beam_id,key) for key in keys] for beam_id in xrange(self.param['beam_size'])] \
                              for batch_id in xrange(self.param['batch_size'])]


        def handle_variable_value_table(key,beam_id,timestep,batch_id):
            if key is not 'variable_value_table':
                new_a_seq[batch_id][beam_id][key][timestep] = a_seq[key][beam_id][timestep][batch_id]
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
            rewards_batch = []
            intermediate_rewards_flag_batch = []
            relaxed_rewards_batch = []
            mask_intermediate_rewards_batch = []
            intermediate_rewards_batch = []
            for beam_id in xrange(self.param['beam_size']):
                if self.printing:
                    print 'beam id', beam_id
                    print 'per_step_probs',per_step_probs[batch_id,beam_id]
                    print 'product_per_step_prob', np.product(per_step_probs[batch_id,beam_id])
                    print 'per_step_programs [',
                new_a_seq_i = new_a_seq[batch_id][beam_id]
                for timestep in range(len(new_a_seq_i['program_type'])):
                    prog = new_a_seq_i['program_type'][timestep]
                    args = new_a_seq_i['argument_table_index'][timestep]
                    if self.printing:
                        print self.read_data.program_type_vocab_inv[prog]+'( '+','.join([str(\
                                   self.read_data.argument_type_vocab_inv[self.read_data.program_to_argtype[prog][arg]])+\
                                   '('+str(args[arg])+')' for arg in range(len(args))])+' )',
                if self.printing:
                    print ']'
                args = (new_a_seq[batch_id][beam_id], \
                       batch_response_entities[batch_id], \
                       batch_response_ints[batch_id], \
                       batch_response_bools[batch_id])
                reward, max_intermediate_reward, relaxed_reward, intermediate_mask, intermediate_reward_flag = self.interpreter.calculate_program_reward(args, epoch_number, overall_step_count)
                rewards_batch.append(reward)
                intermediate_rewards_flag_batch.append(intermediate_reward_flag)
                relaxed_rewards_batch.append(relaxed_reward)
                mask_intermediate_rewards_batch.append(intermediate_mask)
                intermediate_rewards_batch.append(max_intermediate_reward)

            rewards.append(rewards_batch)
            intermediate_rewards_flag.append(intermediate_rewards_flag_batch)
            mask_intermediate_rewards.append(mask_intermediate_rewards_batch)
            intermediate_rewards.append(intermediate_rewards_batch)
            relaxed_rewards.append(relaxed_rewards_batch)
        rewards = np.array(rewards)
        if self.param['reward_from_model']:
            rewards = np.where(Reward_From_Model == 0, rewards, -1*np.ones_like(rewards))
        intermediate_rewards = np.array(intermediate_rewards)
        intermediate_rewards_flag = np.array(intermediate_rewards_flag)
        mask_intermediate_rewards = np.array(mask_intermediate_rewards)
        relaxed_rewards = np.array(relaxed_rewards)
        return rewards, intermediate_rewards, relaxed_rewards, intermediate_rewards_flag, mask_intermediate_rewards



    def get_ml_rewards(self,rewards):
        """
        For Maximum Likelihood pretraining
        Keyword Arguments:
        rewards: input reward tensor of shape batch_size x beam_size

        returns reward for Maximum Likelihood Training
        """
        ml_rewards = np.zeros((self.param['batch_size'], self.param['beam_size']))
        for i in xrange(self.param['batch_size']):
            max_reward = -100.0
            max_index = -1
            for j in xrange(self.param['beam_size']):
                if rewards[i][j] > max_reward:
                    max_reward = rewards[i][j]
                    max_index = j
            if max_index != -1 and max_reward > 0:
                ml_rewards[i][max_index] = 1.0
            if max_index != -1 and max_reward < 0:
                ml_rewards[i][max_index] = -1.0
        return ml_rewards

    def get_data_and_feed_dict(self, batch_dict, epoch, overall_step_count):
        """
        The method fetches the preprocessed data for the current batch and populates the feed_dict with the preprocessed data
        """
        batch_orig_context, batch_context_nonkb_words, batch_context_kb_words, \
        batch_context_entities, batch_context_types, batch_context_rel, batch_context_ints, \
        batch_orig_response, batch_response_entities, batch_response_ints, batch_response_bools, batch_response_type, batch_required_argtypes, \
        variable_mask, variable_embed, variable_atten, kb_attention, variable_value_table = self.read_data.get_batch_data(batch_dict)

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
        return feed_dict1, batch_orig_context, batch_response_entities, batch_response_ints, batch_response_bools, variable_value_table

    def perform_validation(self, batch_dict, epoch, overall_step_count):
        """
        This method invokes the inferencing on the current batch of the validation data and returns the reward for that batch
        """
        feed_dict1, batch_orig_context, batch_response_entities, batch_response_ints, batch_response_bools, variable_value_table = self.get_data_and_feed_dict(batch_dict, epoch, overall_step_count)
        a_seq, program_probabilities, per_step_probs = self.sess.run([self.action_sequence, self.program_probs, self.per_step_probs], feed_dict=feed_dict1)

        # reshaping per_step_probs for printability
        per_step_probs = np.array(per_step_probs)

        [reward, intermediate_rewards, relaxed_rewards, intermediate_rewards_flag, \
         mask_intermediate_rewards] = self.forward_pass_interpreter(batch_orig_context, a_seq, per_step_probs, \
                                         program_probabilities, variable_value_table, batch_response_entities, \
                                         batch_response_ints, batch_response_bools, epoch, overall_step_count)
        reward = np.array(reward)
        relaxed_rewards = np.array(relaxed_rewards)
        reward[reward<0.] = 0.
        self.print_reward(reward)
        return sum(reward[:,0])/float(self.param['batch_size']), sum(np.max(reward,axis=1))/float(self.param['batch_size']), 0


    def perform_training(self, batch_dict, epoch, overall_step_count):
        """
        This method invokes the inferencing on the current batch of the train data and returns the reward for the batch
        """
        feed_dict1, batch_orig_context, batch_response_entities, batch_response_ints, \
        batch_response_bools, variable_value_table = self.get_data_and_feed_dict(batch_dict, epoch, overall_step_count)
        if self.param['Debug'] == 0:# for proper run
            # setting up bench for partial run
            partial_run_op = self.sess.partial_run_setup([self.action_sequence, self.program_probs, self.logProgramProb, \
                                                          self.train_op, self.loss, self.beam_props, self.per_step_probs], \
                                                        feed_dict1.keys()+[self.Reward_placeholder, self.Relaxed_rewards_placeholder, \
                                                                       self.IfPosIntermediateReward, \
                                                                       self.mask_IntermediateReward, \
                                                                       self.IntermediateReward])

            a_seq, program_probabilities, logprogram_probabilities, \
            beam_props, per_step_probs = self.sess.partial_run(partial_run_op, \
                                                           [self.action_sequence, self.program_probs, self.logProgramProb, \
                                                            self.beam_props, self.per_step_probs], feed_dict=feed_dict1)
            # reshaping per_step_probs for printability
            per_step_probs = np.array(per_step_probs)

            if self.param['parallel'] is not 1:
                [reward, intermediate_rewards, relaxed_rewards, intermediate_rewards_flag, \
                 mask_intermediate_rewards] = self.forward_pass_interpreter(batch_orig_context, a_seq, per_step_probs, \
                                                 program_probabilities, variable_value_table, batch_response_entities, \
                                                 batch_response_ints, batch_response_bools, epoch, overall_step_count)
            else:
                reward = self.parallel_forward_pass_interpreter(batch_orig_context, a_seq, per_step_probs, program_probabilities, \
                                                   variable_value_table, batch_response_entities, batch_response_ints, \
                                                   batch_response_bools)
            reward = np.array(reward)
            relaxed_rewards = np.array(relaxed_rewards)

            if self.param['train_mode'] == 'ml':
                reward = self.get_ml_rewards(reward)
                rescaling_term_grad = reward
            else:
                rescaling_term_grad = reward

            a,loss = self.sess.partial_run(partial_run_op, [self.train_op, self.loss], \
                                           feed_dict = {self.Reward_placeholder:rescaling_term_grad, \
                                                        self.Relaxed_rewards_placeholder:relaxed_rewards, \
                                                        self.IfPosIntermediateReward:intermediate_rewards_flag,\
                                                        self.mask_IntermediateReward:mask_intermediate_rewards, \
                                                        self.IntermediateReward:intermediate_rewards,})
        else:# for debugging
            [a_seq, program_probabilities, logprogram_probabilities, \
             beam_props, per_step_probs] = self.sess.run([self.action_sequence, self.program_probs, self.logProgramProb, \
                                           self.beam_props, self.per_step_probs], feed_dict=feed_dict1)
            per_step_probs = np.array(per_step_probs)
            reward = np.zeros([self.param['batch_size'], self.param['beam_size']])
            loss = 0
        reward[reward<0.] = 0. # for printing purposes
        self.print_reward(reward)
        return sum(reward[:,0])/float(self.param['batch_size']), sum(np.max(reward,axis=1))/float(self.param['batch_size']),  loss/float(self.param['batch_size'])



    def print_reward(self, reward):
        """
        This method prints the reward
        """
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

    def remove_bad_data(self, data):
        """
        This method removes the bad data from the train, valid and test data
        """
        for index, d in enumerate(data[:]):
            utter = d[0].lower()
            utter_yes_no_removed = utter.replace('yes','').replace('no','')
            utter_yes_no_removed = re.sub(' +',' ',utter_yes_no_removed)
            utter_yes_no_removed = utter_yes_no_removed.translate(string.maketrans("",""), string.punctuation).strip()
            if 'no, i meant' in utter or 'could you tell me the answer for that?' in utter or len(utter_yes_no_removed)<=1:
                data.remove(d)
        return data

    def get_batch_size_per_type(self, data_map):
        """
        If every batch is constructed out of uniform number of instances from each of the question category (so that the model 
        does not get biased towards any one question category), this method computes the batch size corresponding to each question category
        """
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
        """
        This method creates a batch of data
        """
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
        """
        This method performs the end-to-end training
        """
        best_valid_loss = float("inf")
        best_valid_epoch = 0
        last_overall_avg_train_loss = 0

        last_avg_valid_reward = self.starting_validation_reward_overall
        last_avg_valid_reward_at0 = self.starting_validation_reward_topbeam
        overall_step_count = self.starting_overall_step_count
        epochwise_step_count = 0
        self.qtype_wise_batching = True
        for e in xrange(self.param['max_epochs']):
            epoch = self.starting_epoch+e
            epochwise_step_count =0

            #self.epsilon = self.param["initial_epsilon"]*(1.0-(epoch)/(self.param['max_epochs']))
            if self.param['train_mode']=='ml' and 'change_train_mode_after_epoch' in self.param and epoch >= self.param['change_train_mode_after_epoch']:
                self.param['train_mode']='reinforce'
            num_train_batches = 0.
            train_loss = 0.
            train_reward = 0.
            train_reward_at0 = 0.
            for file in self.training_files:
                train_data = pkl.load(open(file))
                train_data = self.remove_bad_data(train_data)
                random.shuffle(train_data)
                if self.qtype_wise_batching:
                    train_data_map = self.read_data.get_data_per_questype(train_data)
                    if len(train_data_map)==0:
                        continue
                    batch_size_types = self.get_batch_size_per_type(train_data_map)
                    n_batches = int(math.ceil(float(sum([len(x) for x in train_data_map.values()])))/float(self.param['batch_size']))
                else:
                    train_data_map = None
                    n_batches = int(math.ceil(float(len(train_data))/float(self.param['batch_size'])))
                print 'Number of batches ', n_batches, 'len train data ', len(train_data), 'batch size' , self.param['batch_size']
                sys.stdout.flush()
                for i in xrange(n_batches):
                    num_train_batches+=1.
                    train_batch_dict = self.get_batch(i, train_data, train_data_map, batch_size_types)
                    avg_batch_reward_at0, avg_batch_reward, sum_batch_loss = self.perform_training(train_batch_dict, epoch, overall_step_count)
                    avg_batch_loss = sum_batch_loss / float(self.param['batch_size'])
                    if overall_step_count%self.param['print_train_freq']==0:
                        train_loss = train_loss + sum_batch_loss
                        train_reward += avg_batch_reward
                        train_reward_at0 += avg_batch_reward_at0
                        avg_train_reward = float(train_reward)/float(num_train_batches)
                        avg_train_reward_at0 = float(train_reward_at0)/float(num_train_batches)
                        print ('Epoch  %d Step %d (avg over batch) train loss =%.6f  train reward (over all) =%.6f train reward (at top beam)=%.6f running avg train reward (over all)=%.6f running avg train reward (at top beam)=%.6f' %(epoch, epochwise_step_count, avg_batch_loss, avg_batch_reward, avg_batch_reward_at0, avg_train_reward, avg_train_reward_at0))
                        sys.stdout.flush()
                    if overall_step_count%self.param['valid_freq']==0 and overall_step_count>self.starting_overall_step_count:
                        print 'Going for validation'
                        avg_valid_reward_at0, avg_valid_reward = self.perform_full_validation(epoch, overall_step_count)
                        print 'Epoch ', epoch, ' Validation over... overall avg. valid reward (over all)', avg_valid_reward, ' valid reward (at top beam)', avg_valid_reward_at0
                        if avg_valid_reward_at0>last_avg_valid_reward_at0:
                            with open(self.param['model_dir']+'/metadata.txt','w') as fw:
                                fw.write('Epoch_number '+str(epoch)+'\n')
                                fw.write('overall_step_count '+str(overall_step_count)+'\n')
                                fw.write('Avg_valid_reward '+str(avg_valid_reward)+'\n')
                                fw.write('avg_valid_reward_at0 '+str(avg_valid_reward_at0)+'\n')
                            self.saver.save(self.sess, self.model_file, write_meta_graph=False)
                            last_avg_valid_reward_at0 = avg_valid_reward_at0
                            print 'Saving Model in ', self.model_file
                    overall_step_count += 1
                    epochwise_step_count += 1
            overall_avg_train_loss = train_loss/float(num_train_batches)
            if overall_avg_train_loss>last_overall_avg_train_loss:
                print 'Avg train loss increased by ', (overall_avg_train_loss-last_overall_avg_train_loss), ' from ', last_overall_avg_train_loss, 'to', overall_avg_train_loss
            overall_avg_train_reward = train_reward/float(num_train_batches)
            overall_avg_train_reward_at0 = train_reward_at0/float(num_train_batches)
            print 'Epoch ',epoch,' of training is completed ... overall avg. train loss ', overall_avg_train_loss, ' train reward (over all)', overall_avg_train_reward, ' train reward (top beam)', overall_avg_train_reward_at0

def main():
    params_file = sys.argv[1]
    timestamp = sys.argv[2]
    param = json.load(open(params_file))
    param['model_dir']= param['model_dir']+'/'+param['question_type']+'_'+timestamp
    train_model = TrainModel(param)
    train_model.train()

if __name__=="__main__":
    main()

