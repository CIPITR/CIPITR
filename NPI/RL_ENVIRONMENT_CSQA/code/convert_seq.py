#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 11:20:44 2019

@author: Ahmed Ansari
@email: ansarighulamahmed@gmail.com
"""


import json
import cPickle as pkl
import itertools

params_file = '../gold_WikiData_CSQA/parameters/parameters_simple_small.json'
param = json.load(open(params_file))
with open('aseq.pkl') as fp:
    a_seq = pkl.load(fp)

keys = ['program_type','argument_type','target_type','target_table_index','argument_table_index']

new_a_seq = [[dict.fromkeys(keys) for beam_id in xrange(param['beam_size'])] \
              for batch_id in xrange(param['batch_size'])]
def asine(batch_id,beam_id,key):
    new_a_seq[batch_id][beam_id][key] = ['phi' for _ in xrange(param['num_timesteps'])]

[[[asine(batch_id,beam_id,key) for key in keys] for beam_id in xrange(param['beam_size'])] \
                      for batch_id in xrange(param['batch_size'])]

def handle_variable_value_table(key,beam_id,timestep,batch_id):
    if key is not 'variable_value_table':
        new_a_seq[batch_id][beam_id][key][timestep] = a_seq[key][beam_id][timestep][batch_id]

[handle_variable_value_table(key,beam_id,timestep,batch_id) for (key,beam_id,timestep,batch_id) in list(itertools.product\
 (keys,xrange(param['beam_size']),xrange(param['num_timesteps']),xrange(param['batch_size'])))]