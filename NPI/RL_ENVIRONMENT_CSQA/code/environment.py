#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 10:14:37 2018

@author: Ahmed Ansari
@email: ansarighulamahmed@gmail.com
"""

from dataops import Datastore
import itertools
import numpy as np
import json
import random
import sys
import cPickle as pkl
import math
from words2number import text2int



class Environment(object):
    def __init__(self, param, mode = 'Train'):
        '''
        This is a simple environment object to perform question answering on the CSQA dataset.
        Keyword arguments:
        param: shared dictionary of parameters
        mode: specifies whether doing training or validation. 'Train' for training mode, 'Valid' for validation mode.
        '''
        self.mode = mode
        self.data_obj = Datastore(param)
        self.cur_file_id = 0
        self.cur_batch_id = 0
        self.epoch_id = 0
        self.param = param
        self.reset_file()


    def reset_file(self):
        if self.mode is 'Train':
            source = self.data_obj.training_files
        elif self.mode is 'Valid':
            source = self.data_obj.valid_files
        else:
            raise  Exception(' mode should be either "Train" or "Valid"')
        self.data_obj.qtype_wise_batching = True
        if self.cur_file_id == len(source):
            self.cur_file_id = 0
            self.epoch_id += 1
        file = source[self.cur_file_id]
        train_data = pkl.load(open(file))
        if self.mode is 'Train':
            train_data = self.data_obj.remove_bad_data(train_data)
            random.shuffle(train_data)
        if self.data_obj.qtype_wise_batching:
            train_data_map = self.data_obj.read_data.get_data_per_questype(train_data)
            if len(train_data_map)==0:
                self.cur_file_id += 1
                self.reset_epoch()
            batch_size_types = self.data_obj.get_batch_size_per_type(train_data_map)
            n_batches = int(math.ceil(float(sum([len(x) for x in train_data_map.values()])))/float(self.param['batch_size']))
        else:
            train_data_map = None
            n_batches = int(math.ceil(float(len(train_data))/float(self.param['batch_size'])))

        self.data_obj.train_data = train_data
        self.data_obj.train_data_map = train_data_map
        self.data_obj.batch_size_types = batch_size_types
        self.data_obj.n_batches = n_batches

    def fetch_batch(self):
        '''
        This functions fetches a fresh batch from the CSQA dataset. It returns, the input questions, the Wikidata entitities, relations and types associated with each of the questions
        Returns:
        input_questions: Batch of questions or the queries sampled from the CSQA dataset. shape is 'batch_size'
        entities_in_questions: The Linked Wikidata Entities present in the Question/Query. shape is 'batch_size x max_num_var' where max_num_var is the maximum number of variables of any variable type. It is set in the param file.
        types_in_questions: The Linked Wikidata Types present in the Question/Query. shape is 'batch_size x max_num_var'
        relations_in_questions: The Linked Wikidata Relations present in the Question/Query. shape is 'batch_size x max_num_var'
        ints_in_questions: The integers present in the query if any. shape is 'batch_size x max_num_var'
        '''
        if self.cur_batch_id == self.data_obj.n_batches:
            self.reset_file()
        train_batch_dict = self.data_obj.get_batch(self.cur_batch_id, self.data_obj.train_data, self.data_obj.train_data_map, self.data_obj.batch_size_types)
        self.cur_batch_id += 1
        self.current_batch = train_batch_dict

        num_data = len(train_batch_dict)
        # retrieving the string of questions
        input_questions = [train_batch_dict[i][0] for i in range(num_data)]
        # retrieving the Wikidata Entities linked in the questions
        batch_context_entities = [train_batch_dict[i][3] for i in range(num_data)]
        entities_in_questions = [[self.data_obj.read_data.wikidata_ent_vocab_inv[e] if e!=self.data_obj.read_data.pad_kb_symbol_index else \
                                        None for e in batch_context_entities[i]] for i in range(num_data)]
        # retrieving the Wikidata Types linked in the questions
        batch_context_types = [train_batch_dict[i][4] for i in range(num_data)]
        types_in_questions = [[self.data_obj.read_data.wikidata_type_vocab_inv[t] if t!=self.data_obj.read_data.pad_kb_symbol_index else None \
                                      for t in batch_context_types[i]] for i in range(num_data)]
        # retrieving the Wikidata Relations linked in the questions
        batch_context_rel = [train_batch_dict[i][5] for i in range(num_data)]
        relations_in_questions= [[self.data_obj.read_data.wikidata_rel_vocab_inv[r] if r!=self.data_obj.read_data.pad_kb_symbol_index else None \
                                     for r in batch_context_rel[i]] for i in range(num_data)]
        # retrieving integers present in the question
        batch_context_ints = [train_batch_dict[i][6] for i in range(num_data)]
        batch_context_ints = np.asarray([[i if i==self.data_obj.read_data.pad_kb_symbol_index else text2int(i) for i in context_int] for context_int in batch_context_ints])
        ints_in_questions = [[i if i!=self.data_obj.read_data.pad_kb_symbol_index else None for i in ints] for ints in batch_context_ints]
        return input_questions, entities_in_questions, types_in_questions, relations_in_questions, ints_in_questions

    def argument_vocabulary(self):
        '''
        This functions returns the vocabulary of argument types employed.
        '''
        return self.data_obj.read_data.argument_type_vocab

    def operator_type_vocabulary(self):
        '''
        This function returns the vocabulary of operators employed.
        '''
        return self.data_obj.read_data.program_type_vocab

    def operator_to_argument_type_mapping(self):
        '''
        This functions returns the mappings of operator to its compatible input argument types.
        '''
        return self.data_obj.read_data.prog_to_argtypes_map, self.data_obj.read_data.program_to_argtype

    def output_variable_type_mapping(self):
        '''
        This functions returns the mapping of operator to its output variable types.
        '''
        return self.data_obj.read_data.targettype_prog_map, self.data_obj.read_data.program_to_targettype

    def step(self,programs_batch):
        '''
        This function takes in as input candidate programs for every question in the batch and returns a reward for each program.
        The number of candidate programs per question is referred to as beam_size here, beam_size Need not be hard-set for this function to work.
        A program is a sequence of actions with maximum length as num_timesteps which is set in the param file.
        Each action is an instantiation of an operator(see program_types.txt) and corresponding input variables (see program_to_arguments.txt)

        Guideline for designing programs_batch.
        programs_batch should be a dictionary with keys in "'program_type','argument_type','target_type','target_table_index','argument_table_index','variable_value_table'"
        Each element of the dictionary is a num_timestep length sequence of corresponding objects defined in subsequently.

        A program , say P, is a num_timesteps length sequence of actions , say [A_1, A_2, A_3, ...], where each A_i is of the form O_i(V_i)
        where O_i is operator (see operator_type_vocabulary()), V_i is input argument variables(see argument_vocabulary, operator_to_argument_type_mapping) for O_i
        each V_i is a sequence of max_arguments length say, [vi_1,vi_2,...]

            The programs_batch is a batch_size x beam_size array of program objects. Any programs_batch[i][j] item is of the form:-

                {
                    'argument_table_index':
                        [
                                array([0, 0, 0], dtype=int32), # comment: @timestep 0. select 0th index for each v0_j
                                array([0, 0, 0], dtype=int32),
                                array([0, 0, 0], dtype=int32),
                                array([0, 0, 0], dtype=int32)
                        ],
                     'argument_type':
                         [
                              array([1, 2, 3], dtype=int32), # comment: @timestep 0. v0_0 is of type 1 (entity), v0_1 is of type 2 (relation), v0_2 is of type 3 (type)
                              array([0, 0, 0], dtype=int32), # comment: @timestep 1. v1_0 is of type 0 (none), v1_1 is of type 2 (none), v1_2 is of type 3 (none)
                              array([0, 0, 0], dtype=int32),
                              array([0, 0, 0], dtype=int32)
                        ],
                     'program_type': [1, 0, 20, 20], # comment: O_0 is 1 (gen_set), O_1 is 0 (none), O_2 is 20 (terminate), O_3 is 20 (terminate)
                     'target_table_index': [0, 0, 0, 0], # comment: Output of A_0 is placed at 0th position in corresponding table, similarly A_1 is placed at 0th posti....
                     'target_type': [6, 0, 0, 0], # comment: Output variable generated by A_0 is of type 6 (set), Output variable generated by A_1 is of type 0 (None), ....
                }
        '''
        self.fetch_batch()
        batch_orig_context, batch_response_entities, batch_response_ints, batch_response_bools, \
        variable_value_table = self.data_obj.get_data_and_feed_dict(self.current_batch, self.epoch_id, self.cur_batch_id)
        [reward, intermediate_rewards, relaxed_rewards, intermediate_rewards_flag, \
         mask_intermediate_rewards] = self.Interpret_Program_batch(batch_orig_context, programs_batch, \
                                         variable_value_table, batch_response_entities, \
                                         batch_response_ints, batch_response_bools, self.epoch_id, self.cur_batch_id)
        return reward

    def Interpret_Program_batch(self, batch_orig_context, a_seq, \
                                 variable_value_table, batch_response_entities, \
                                 batch_response_ints, batch_response_bools, epoch_number, overall_step_count):
        """
        This function parses the action sequences that are generated by the NPI module into a format that can be understood by the interpreter. It also prints teh current progress.
        Keyword arguments:
        batch_orig_context: batch of queries
        a_seq: The input action sequence recieved from the model. Dictionary object with keys like 'program_type','argument_type', ....
                Each element of the dictionary is a list of length beam_size containing a list of length num_timesteps
                containing tensors of shape batch_size x Dim (Dim depends upon type of dictionary key)
        per_step_probs: tensor containing probability of beams with shape batch_size x beam_size x num_timesteps
        variable_value_table: table of preprocessed input variables extracted directly form query. types of variables here would be entity, relation, type and int
        batch_response_entities: The batch of entities if present in gold answer
        batch_response_ints: The batch of ints if present in gold answer
        batch_response_bools: The batch of bools if present in gold answer
        epoch_number: Current epoch no.
        overall_step_count: Overall no.  of batches elapsed
        """
        beam_size = self.param['beam_size']
        keys = ['program_type','argument_type','target_type','target_table_index','argument_table_index','variable_value_table']


        new_a_seq = a_seq

        def handle_variable_value_table(key,beam_id,timestep,batch_id):
            if key is 'variable_value_table':
                new_a_seq[batch_id][beam_id][key] = variable_value_table[batch_id]

        [handle_variable_value_table(key,beam_id,timestep,batch_id) for (key,beam_id,timestep,batch_id) in list(itertools.product\
         (keys,xrange(beam_size),xrange(self.param['num_timesteps']),xrange(self.param['batch_size'])))]

        rewards = []
        intermediate_rewards_flag = []
        mask_intermediate_rewards = []
        intermediate_rewards = []
        relaxed_rewards = []
        for batch_id in xrange(self.param['batch_size']):
            if self.data_obj.printing:
                print 'batch id ', batch_id, ':: Query :: ', batch_orig_context[batch_id]
            rewards_batch = []
            intermediate_rewards_flag_batch = []
            relaxed_rewards_batch = []
            mask_intermediate_rewards_batch = []
            intermediate_rewards_batch = []
            for beam_id in xrange(beam_size):
                if self.data_obj.printing:
                    print 'beam id', beam_id
                    print 'per_step_programs [',
                new_a_seq_i = new_a_seq[batch_id][beam_id]
                for timestep in range(len(new_a_seq_i['program_type'])):
                    prog = new_a_seq_i['program_type'][timestep]
                    args = new_a_seq_i['argument_table_index'][timestep]
                    if self.data_obj.printing:
                        print self.data_obj.read_data.program_type_vocab_inv[prog]+'( '+','.join([str(\
                                   self.data_obj.read_data.argument_type_vocab_inv[self.data_obj.read_data.program_to_argtype[prog][arg]])+\
                                   '('+str(args[arg])+')' for arg in range(len(args))])+' )',
                if self.data_obj.printing:
                    print ']'
                args = (new_a_seq[batch_id][beam_id], \
                       batch_response_entities[batch_id], \
                       batch_response_ints[batch_id], \
                       batch_response_bools[batch_id])
                reward, max_intermediate_reward, relaxed_reward, intermediate_mask, intermediate_reward_flag = self.data_obj.interpreter.calculate_program_reward(args, epoch_number, overall_step_count)
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
        intermediate_rewards = np.array(intermediate_rewards)
        intermediate_rewards_flag = np.array(intermediate_rewards_flag)
        mask_intermediate_rewards = np.array(mask_intermediate_rewards)
        relaxed_rewards = np.array(relaxed_rewards)
        return rewards, intermediate_rewards, relaxed_rewards, intermediate_rewards_flag, mask_intermediate_rewards