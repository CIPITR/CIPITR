from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import math

class NPI(nn.Module):
    def __init__(self, params, none_argtype_index, num_argtypes, num_programs,
                 max_arguments, rel_index, type_index,
                 rel_embedding, type_embedding, vocab_embed,
                 program_to_argtype_table, program_to_targettype_table):
        super(NPI, self).__init__()
        self.seed = 1
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.params = params
        self.num_timesteps = params['num_timesteps']
        self.max_num_phase_1_steps = self.num_timesteps / 2
        self.state_dim = params['state_dim']
        self.batch_size = params['batch_size']
        self.prog_embed_dim = params['prog_embed_dim']
        self.argtype_embed_dim = params['argtype_embed_dim']
        self.var_embed_dim = params['var_embed_dim']
        self.npi_core_dim = params['npi_core_dim']
        self.env_dim = params['env_dim']
        self.hidden_dim = params['hidden_dim']
        self.empty_argtype_id = none_argtype_index
        self.sample_with = params["sample_with"]
        self.num_argtypes = num_argtypes
        self.num_progs = num_programs
        self.max_arguments = max_arguments
        self.max_num_var = params['max_num_var']
        self.prog_key_dim = params['prog_key_dim']
        self.var_key_dim = params['var_key_dim']
        if params['use_key_as_onehot']:
            self.use_key_as_onehot = True
            self.var_key_dim = self.num_argtypes + self.max_num_var
            self.prog_key_dim = self.num_progs
        else:
            self.use_key_as_onehot = False
        self.max_len = params['max_len']
        self.wikidata_embed_dim = params['wikidata_embed_dim']
        self.text_embed_dim = params['text_embed_dim']
        self.cell_dim = params['cell_dim']
        self.eps = 1e-20
        self.learning_rate = params['learning_rate']
        self.beam_size = params['beam_size']
        self.num_programs_to_sample = params['num_programs_to_sample']
        self.num_variables_to_sample = params['num_variables_to_sample']
        self.num_actions = self.num_variables_to_sample * self.num_programs_to_sample
        self.temperature = 0.1
        self.terminate_threshold = 0.7
        self.phase_change_threshold = 0.2
        self.bias_to_gold_type_threshold = 0.3
        self.rel_embedding = None
        self.query_rel_atten = None
        # program_table contains two parameters "Program_Embedding" and "Program_Keys".
        # Program_Keys is of dimension num_programs x prog_key_dim.
        # Program_Embedding is of dimension num_programs x prog_embed_dim
        self.program_embedding = None
        self.program_keys = None
        # program_to_argtype_table contains a list of list of integers of dimension num_programs x max_arguments
        self.program_to_argtype_table = None
        # self.argument_type_table contains a parameter "ArgumentType_Embedding".
        # No "Argument_Keys" is needed because argument types are determined by the program itself by looking up the self.
        # program_to_argtype_table. ArgumentType_Embedding is of dimension num_argtypes x argtype_embed_dim
        self.argumenttype_embedding = None
        # self.variable_table contains a parameter "Variable_Embedding" and "Variable_Keys".
        # Variable tables are 2-way table i.e. for every argument type, there is a list (of maximum upto) N variables of that type.
        # So Variable_Keys is of dimension number_of_argtypes x batch_size x max_num_var x var_key_dim (var_key_dim being used to Id the variable)
        # and "Variable_Embedding" being of dimension num_argtypes x batch_size x max_num_var x var_embed_dim
        self.variable_embedding = None
        self.variable_keys = None
        # self.variable_mask is of dimension num_argtypes x batch_size x max_num_var
        self.variable_mask = None
        # self.variable_atten_table contains the attention over all variables
        # declared till now. is of dimension num_argtypes x max_num_var
        self.variable_atten_table = None
        self.kb_attention = None
        assert self.beam_size <= self.num_programs_to_sample * self.num_variables_to_sample
        self.keep_prob = params['dropout_keep_prob']
        self.global_program_indices_matrix = None
        self.dont_look_back_attention = params['dont_look_back_attention']
        self.concat_query_npistate = params['concat_query_npistate']
        self.query_attention = params['query_attention']
        self.forced_normalize_ir = bool(1-params['normalize_length'])
        self.dtype_float = torch.float
        self.dtype_int64 = torch.long
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device("cpu")
        print 100*"$"
        print self.device
        print 100*"$"
        #if self.device=="cuda":
        #torch.int32 = torch.cuda.int32
        #torch.float32 = torch.cuda.float32
        #torch.int64 = torch.cuda.int64
        #print 'changed dtypes to cuda'
        def create_cell_scopes():
            self.dropout = nn.Dropout(1-self.keep_prob)
            self.elu = nn.ELU()
            self.batch_normalizer = nn.BatchNorm1d(self.npi_core_dim+self.var_embed_dim)
            self.npi_scope = "npi_scope"
            self.npi_rnn = nn.GRU(input_size=self.npi_core_dim+self.var_embed_dim, \
                                  hidden_size=self.npi_core_dim, batch_first=False)
            self.env_scope = "env_scope"
            self.env_rnn = nn.GRU(input_size=self.var_embed_dim, hidden_size=self.env_dim, batch_first=False)
            self.sentence_scope = "sentence_scope"
            self.sentence_rnn = nn.GRU(input_size=self.wikidata_embed_dim+self.text_embed_dim, \
                                       hidden_size=self.cell_dim, batch_first=True)
            self.reset_scope = 'reset_scope'
            self.reset_layer = nn.Linear(self.npi_core_dim, self.npi_core_dim)
            self.state_encoding_scope_1 = 'state_encoding_scope_layer1'
            self.state_encoding_layer1 = nn.Linear(self.npi_core_dim+self.var_embed_dim, self.hidden_dim)
            self.state_encoding_scope_2 = 'state_encoding_scope_layer2'
            self.state_encoding_layer2 = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.state_encoding_scope_3 = 'state_encoding_scope_layer3'
            self.state_encoding_layer3 = nn.Linear(self.hidden_dim, self.state_dim)
            self.phase_change_scope = 'phase_change_scope'
            self.phase_change_layer = nn.Linear(self.npi_core_dim, 1)
            self.prog_key_scope1 = 'prog_net_fcc1'
            if self.concat_query_npistate:
                self.prog_key_layer1 = nn.Linear(self.npi_core_dim+self.var_embed_dim+self.cell_dim, self.prog_key_dim)
            else:
                self.prog_key_layer1 = nn.Linear(self.npi_core_dim+self.var_embed_dim, self.prog_key_dim)
            self.prog_key_scope2 = 'prog_net_fcc2'
            self.prog_key_layer2 = nn.Linear(self.prog_key_dim, self.prog_key_dim)
            self.inp_var_key_scope = 'inp_var_key_scope'
            self.inp_var_key_layer = nn.Linear(self.max_num_var, self.var_key_dim)
            self.get_target_var_key_and_embedding_arg_scope = 'get_target_var_key_and_embedding_arg_scope'
            self.target_var_key_and_embedding_arg_layer = nn.ModuleList([nn.Linear(2*self.argtype_embed_dim, self.argtype_embed_dim) \
                                                           for i in xrange(self.max_arguments)])
            self.get_target_var_key_and_embedding_var_scope = 'get_target_var_key_and_embedding_var_scope'
            self.target_var_key_and_embedding_var_layer = nn.ModuleList([nn.Linear(2*self.var_embed_dim, self.var_embed_dim) \
                                                           for i in xrange(self.max_arguments)])
            self.get_target_var_key_and_embedding_targetembed_scope = 'get_target_var_key_and_embedding_targetembed_scope'
            self.target_var_key_and_embedding_targetembed_layer = nn.Linear(self.var_embed_dim+self.argtype_embed_dim+\
                                                                                   self.prog_embed_dim, self.var_embed_dim)

            self.get_target_var_key_and_embedding_targetkey_scope = 'get_target_var_key_and_embedding_targetkey_scope'
            self.target_var_key_and_embedding_targetkey_layer = nn.Linear(self.var_embed_dim+self.argtype_embed_dim+\
                                                                                   self.prog_embed_dim, self.var_key_dim)

            self.update_attention_scope = 'update_attention_scope'
            self.update_attention_layer = nn.ModuleList([nn.Linear(self.max_num_var+self.npi_core_dim,self.max_num_var) \
                                           for i in xrange(self.num_argtypes)])

            self.batch_ids = torch.arange(0,self.batch_size,dtype = self.dtype_int64, device=self.device)
            # tensor is of dimension batch_size x 1 i.e. [0, 1, 2, ... batch_size]

            rel_embedding_mat = torch.tensor(rel_embedding, device=self.device, dtype=self.dtype_float)
            self.rel_embedding = nn.Embedding(rel_embedding.shape[0], rel_embedding.shape[1], _weight=rel_embedding_mat)
            type_embedding_mat = torch.tensor(type_embedding, device=self.device)
            self.type_embedding = nn.Embedding(type_embedding.shape[0], type_embedding.shape[1], _weight=type_embedding_mat)
            max_val = 6. / np.sqrt(self.cell_dim + self.wikidata_embed_dim)
            self.query_rel_atten = nn.Parameter(torch.tensor(np.random.normal(-max_val, max_val, [self.cell_dim, self.wikidata_embed_dim]), \
                                                requires_grad=True, device=self.device))
            word_embeddings_mat = torch.tensor(vocab_embed, device=self.device, dtype=self.dtype_float)
            self.word_embeddings = nn.Embedding(vocab_embed.shape[0], vocab_embed.shape[1], _weight=word_embeddings_mat)
            self.enc_scope_text = "encoder_text"
            max_val = 6. / np.sqrt(self.num_progs + self.prog_embed_dim)
            program_embedding_mat = torch.tensor(np.random.normal(-max_val, max_val, [self.num_progs, self.prog_embed_dim]), \
                                                 device=self.device, dtype=self.dtype_float)
            self.program_embedding = nn.Embedding(self.num_progs, self.prog_embed_dim, _weight=program_embedding_mat)
            max_val = 6. / np.sqrt(1 + self.hidden_dim)
            self.query_attention_h_mat = nn.Parameter(torch.tensor(np.random.normal(-max_val, max_val, [1, self.hidden_dim]), \
                                                      requires_grad=True, device=self.device, dtype=self.dtype_float))
            max_val = 6. / np.sqrt(self.wikidata_embed_dim + self.var_embed_dim)
            self.preprocessed_var_emb_mat = nn.Parameter(torch.tensor(np.random.normal(-max_val, max_val, \
                                                                          [self.wikidata_embed_dim, self.var_embed_dim]), \
                                                                            requires_grad=True, device=self.device, \
                                                                            dtype=self.dtype_float))
            max_val = 6. / np.sqrt(self.num_progs + self.prog_key_dim)
            self.init_state = nn.Parameter(torch.zeros([1, self.cell_dim],requires_grad=True,device=self.device, dtype=self.dtype_float))
            self.program_keys = nn.Parameter(torch.tensor(np.random.normal(-max_val, max_val, [self.num_progs, self.prog_key_dim]), \
                                             requires_grad=True, device=self.device, dtype=self.dtype_float))
            self.program_to_argtype_table = torch.tensor(program_to_argtype_table, device=self.device, \
                                                         dtype=self.dtype_int64)
            self.program_to_targettype_table = torch.tensor(program_to_targettype_table,
                                                            device=self.device, dtype=self.dtype_int64)
            max_val = 6. /np.sqrt(self.num_argtypes + self.argtype_embed_dim)
            argtype_embedding_mat = torch.tensor(np.random.normal(-max_val, max_val, \
                                                                  [self.num_argtypes, self.argtype_embed_dim]), \
                                                                            device=self.device, dtype=self.dtype_float)
            self.argtype_embedding = nn.Embedding(self.num_argtypes, self.argtype_embed_dim, _weight=argtype_embedding_mat)
            self.program_to_num_arguments = torch.max(input=self.one_hot(self.program_to_argtype_table,depth=self.num_argtypes),dim=1)[0]
            #program_to_num_arguments is of dimension num_progs x num_argtypes
            # accomodating for the beam_size

        create_cell_scopes()

    def get_parameters(self):
        return (self.program_keys, self.program_embedding, self.word_embeddings, \
                self.argtype_embedding, self.query_attention_h_mat)

    def create_placeholder(self):
        self.encoder_text_inputs_w2v = None
        self.encoder_text_inputs_kb_emb = None
        self.preprocessed_var_mask_table = [[None]*self.max_num_var]*self.num_argtypes
        self.preprocessed_var_emb_table = [[None]*self.max_num_var]*self.num_argtypes
        self.kb_attention = None
        self.progs_phase_1 = None
        self.progs_phase_2 = None
        self.gold_target_type = None
        self.randomness_threshold_beam_search = None
        self.DoPruning = None
        self.last_step_feasible_program = None
        self.bias_prog_sampling_with_target = None
        self.bias_prog_sampling_with_last_variable = None
        self.required_argtypes = None
        self.relaxed_reward_multipler = None
        self.IfPosIntermediateReward = None
        self.mask_IntermediateReward = None
        self.IntermediateReward = None

    def manual_gather_nd(self, params,indices):
        param_shape = list(params.shape)
        #print param_shape
        number_of_axes = indices.shape[-1]
        f_indices  = self.map_index_to_flattened(indices,param_shape[0:number_of_axes])
        f_params = params.contiguous().view([-1]+param_shape[number_of_axes:]) #this reshaping cannot be avoided
        return torch.index_select(f_params,0,f_indices)


    def get_final_feasible_progs_for_last_timestep(self, feasible_progs, beam_properties, beam_id, feasible_progs_for_last_timestep, t):
        if t == self.num_timesteps-1:
            feasible_progs_for_last_timestep = feasible_progs_for_last_timestep.type(feasible_progs[beam_id].dtype)
            #feasible_progs[beam_id] = tf.add(feasible_progs[beam_id], tf.zeros_like(feasible_progs[beam_id]))
            temp = torch.where((self.gold_target_type==beam_properties['target_type'][beam_id]), \
                               torch.ones_like(self.gold_target_type, device=self.device), \
                               torch.zeros_like(self.gold_target_type, device=self.device))
            current_equal_to_gold_target_type = torch.unsqueeze(temp, dim=1).repeat([1, self.num_progs]).type(feasible_progs[beam_id].dtype)
            #current_equal_to_gold_target_type is of size batch_size x num_progs
            t1 = self.one_hot(torch.zeros([self.batch_size], device=self.device), depth=self.num_progs)
            t2 = self.one_hot((self.num_progs-1)*torch.ones([self.batch_size], device=self.device), depth=self.num_progs)
            temp = (t1 + t2).type(feasible_progs[beam_id].dtype)
            #temp is of size batch_size x num_progs
            feasible_progs_for_last_timestep = current_equal_to_gold_target_type*temp + (1-temp)*feasible_progs_for_last_timestep
            temp2 = (1-self.last_step_feasible_program)*feasible_progs[beam_id] + \
                    self.last_step_feasible_program*torch.mul(feasible_progs[beam_id], feasible_progs_for_last_timestep)
            temp3 = torch.unsqueeze(torch.sum((1-temp)*temp2, dim=1),dim=1).repeat([1,self.num_progs])
            #temp3 is of dimension batch_size x num_progs
            feasible_progs[beam_id] = torch.where((temp3==0), feasible_progs[beam_id], temp2)
        return feasible_progs[beam_id]

    def forward(self, feed_dict):
        #with tf.device(tf.test.gpu_device_name()):
            with torch.no_grad():
                self.variable_embedding = []
                self.variable_keys = []
                self.variable_atten_table = []
                self.variable_mask = []
                max_val = 6. / np.sqrt(self.max_num_var)
                for beam_id in xrange(self.beam_size):
                    self.variable_embedding.append(torch.zeros([self.num_argtypes, self.batch_size, \
                                                                self.max_num_var, self.var_embed_dim], \
                                                                device=self.device))
                    self.variable_keys.append(torch.zeros([self.num_argtypes, self.batch_size, \
                                                           self.max_num_var, self.var_key_dim], \
                                                            device=self.device))
                    temp = torch.zeros([self.num_argtypes, self.batch_size, self.max_num_var], device=self.device)

                    self.variable_atten_table.append(list(torch.unbind(temp, dim=0)))
                    self.variable_mask.append(torch.zeros([self.num_argtypes, self.batch_size, self.max_num_var], \
                                                          device=self.device))
                self.encoder_text_inputs_w2v = torch.tensor(feed_dict['encoder_text_inputs_w2v'], \
                                                            device=self.device, dtype=self.dtype_int64)
                self.preprocessed_var_mask_table = [[torch.tensor(feed_dict['preprocessed_var_mask_table'][i][j], \
                                                                  device=self.device, dtype=self.dtype_float) \
                                                        for j in range(self.max_num_var)] for i in range(self.num_argtypes)]
                self.preprocessed_var_emb_table = [[torch.tensor(feed_dict['preprocessed_var_emb_table'][i][j], \
                                                                 device=self.device, dtype=self.dtype_float) \
                                                                for j in range(self.max_num_var)] for i in range(self.num_argtypes)]
                self.encoder_text_inputs_kb_emb = torch.tensor(feed_dict['encoder_text_inputs_kb_emb'], \
                                                               device=self.device, dtype=self.dtype_float)
                self.kb_attention = torch.tensor(feed_dict['kb_attention'], device=self.device, dtype=self.dtype_float)
                self.progs_phase_1 = torch.tensor(feed_dict['progs_phase_1'], device=self.device, dtype=self.dtype_int64)
                self.progs_phase_2 = torch.tensor(feed_dict['progs_phase_2'], device=self.device, dtype=self.dtype_int64)
                self.gold_target_type = torch.tensor(feed_dict['gold_target_type'], device=self.device, dtype=self.dtype_int64)
                self.randomness_threshold_beam_search = torch.tensor(feed_dict['randomness_threshold_beam_search'], \
                                                                     device=self.device, dtype=self.dtype_float)
                self.DoPruning = torch.tensor(feed_dict['DoPruning'], device=self.device, dtype=self.dtype_float)
                self.last_step_feasible_program = torch.tensor(feed_dict['last_step_feasible_program'], \
                                                               device=self.device, dtype=self.dtype_float)
                self.bias_prog_sampling_with_last_variable = torch.tensor(feed_dict['bias_prog_sampling_with_last_variable'], \
                                                                          device=self.device, dtype=self.dtype_float)
                self.bias_prog_sampling_with_target = torch.tensor(feed_dict['bias_prog_sampling_with_target'], \
                                                                   device=self.device, dtype=self.dtype_float)
                self.required_argtypes = torch.tensor(feed_dict['required_argtypes'], device=self.device, dtype=self.dtype_int64)
                self.relaxed_reward_multipler = torch.tensor(feed_dict['relaxed_reward_multipler'], device=self.device, \
                                                             dtype=self.dtype_float)
            sentence_state, attention_states = self.sentence_encoder()
            beam_properties = defaultdict(list)
            beam_properties['Model_Reward_Flag'] = [torch.zeros([self.batch_size], device=self.device) \
                           for beam_id in xrange(self.beam_size)]
            for beam_id in xrange(self.beam_size):
                beam_properties['Model_Reward_Flag'][beam_id] = self.add_preprocessed_output_to_variable_table(beam_id)
            init_h_states, init_e_state, init_target_var_embedding = self.reset_state(sentence_state)
            unswitched_beam_properties = defaultdict(list)
            beam_properties['h_states'] = [init_h_states for beam_id in xrange(self.beam_size)]
            beam_properties['h'] = [None for beam_id in xrange(self.beam_size)]
            beam_properties['e_state'] = [init_e_state for beam_id in xrange(self.beam_size)]
            beam_properties['target_var_embedding'] = [init_target_var_embedding for beam_id in xrange(self.beam_size)]
            beam_properties['prog_sampled_indices'] = [None for beam_id in xrange(self.beam_size)]
            beam_properties['input_var_sampled_indices'] = [None for beam_id in xrange(self.beam_size)]
            unswitched_beam_properties['total_beam_score'] = [torch.zeros([self.batch_size], device=self.device)] + \
                                              [-30*torch.ones([self.batch_size], device=self.device) \
                                               for beam_id in xrange(self.beam_size-1)]
            #beam_properties['total_beam_score'] = [tf.zeros([self.batch_size]) for beam_id in xrange(self.beam_size)]
            beam_properties['terminate'] = [torch.zeros([self.batch_size,1], device=self.device) \
                                           for beam_id in xrange(self.beam_size)]
            beam_properties['length'] = [torch.zeros([self.batch_size,1], device=self.device) \
                                           for beam_id in xrange(self.beam_size)]
            beam_properties['target_type'] = [torch.zeros([self.batch_size], device=self.device, dtype=self.dtype_int64) \
                                               for beam_id in xrange(self.beam_size)]
            beam_properties['phase_elasticity'] = [torch.ones([self.batch_size,1], device=self.device) for beam_id in xrange(self.beam_size)]
            beam_properties['program_argument_table_index'] = [torch.ones([self.batch_size, self.num_progs,
                                  int(math.pow(self.max_num_var,self.max_arguments))], device=self.device) for beam_id in xrange(self.beam_size)]
            beam_properties['query_attentions_till_now'] = [torch.zeros([self.batch_size,self.max_len], device=self.device) for beam_id in xrange(self.beam_size)]
            self.debug_beam_terminate = defaultdict(list)
            beam_properties['none_count'] = [torch.zeros([self.batch_size,1], device=self.device) for beam_id in xrange(self.beam_size)]
#            beam_properties['check_penalization'] = [torch.zeros([self.batch_size,1], device=self.device) for beam_id in xrange(self.beam_size)]

            to_return_per_step_prob = -1*torch.ones([self.batch_size,self.beam_size,self.num_timesteps], device=self.device)
            #[-1*tf.ones([self.batch_size, self.beam_size]) for time_step in xrange(self.num_timesteps)]

            to_return_sequence_logprob = torch.zeros([self.batch_size, self.beam_size], device=self.device)
            # this should finally contain a tensor of batch_size x beam_size
            to_return_action_sequence = dict.fromkeys(['program_type','argument_type','target_type',\
                                                       'target_table_index','argument_table_index'])
            for key in ['program_type','argument_type','target_type','target_table_index','argument_table_index']:
                to_return_action_sequence[key] = [[] for beam_id in xrange(self.beam_size)]

            self.entropy = torch.tensor(0, device=self.device, dtype=self.dtype_float)
            feasible_progs_for_last_timestep = self.get_feasible_progs_for_last_timestep()

            for t in xrange(self.num_timesteps):
                entropy = torch.tensor(0, device=self.device, dtype=self.dtype_float)
                # =============================================================================
                current_beam_score = [score+0 for score in unswitched_beam_properties['total_beam_score']]

                if t > 0:
                    beam_properties['phase_elasticity'] = [self.phase_change_net(h.view([self.batch_size, -1]),t, old_p_el) \
                                   for h,old_p_el in zip(beam_properties['h'], beam_properties['phase_elasticity'])]
                feasible_progs = self.get_feasible_progs(t, beam_properties['phase_elasticity'])

                to_penalize_beams = [torch.zeros([self.batch_size,self.num_actions], device=self.device) for beam_id in xrange(self.beam_size)]
                for beam_id in xrange(self.beam_size):
                    beam_properties['e_state'][beam_id] = self.env_encoding(beam_properties['e_state'][beam_id], \
                                                               beam_properties['target_var_embedding'][beam_id])[1]

                    [beam_properties['h'][beam_id],
                     beam_properties['h_states'][beam_id]] = self.npi_core(beam_properties['h_states'][beam_id], \
                                                               beam_properties['e_state'][beam_id], \
                                                               beam_properties['target_var_embedding'][beam_id])


                    feasible_progs[beam_id] = self.get_final_feasible_progs_for_last_timestep(feasible_progs, \
                                                  beam_properties, beam_id, feasible_progs_for_last_timestep, t)

                    [prog_sampled_probs, prog_sampled_indices, \
                     prog_sampled_embeddings, kb_attention_for_sampled_progs, \
                     beam_properties['query_attentions_till_now'][beam_id]] = self.prog_net(beam_properties['h'][beam_id],
                                                                                 sentence_state, attention_states,
                                                                                 beam_properties['query_attentions_till_now'][beam_id], \
                                                                                 feasible_progs[beam_id], \
                                                                                 self.num_programs_to_sample, \
                                                                                 beam_properties['terminate'][beam_id], \
                                                                                 beam_properties['target_type'][beam_id])
                    # prog_sampled_probs batch_size x num_programs_to_sample
                    # prog_sampled_indices batch_size x num_programs_to_sample
                    # prog_sampled_embeddings is a tensor of shape batch_size x num_programs_to_sample x prog_embedding_dim
                    # kb_attention_for_sampled_progs is a num_programs_to_sample length list a flat tensor of size max_var * max_var * max_var

                    beam_properties['prog_sampled_indices'][beam_id] = prog_sampled_indices

                    complete_action_probs = []
                    # for every sampled program will contain the probability of action obtained by sampling every possible var
                    per_program_input_var_sampled_indices = []
                    #for every sampled program will contain the possible variable samples

                    for _prog_sample_, _prog_embedding_, \
                        _kb_attention_for_sampled_progs_ , \
                        _program_prob_ in zip(list(torch.unbind(prog_sampled_indices, dim = 1)),\
                                              list(torch.unbind(prog_sampled_embeddings, dim = 1)),\
                                              list(torch.unbind(kb_attention_for_sampled_progs, dim = 0)),\
                                              list(torch.unbind(prog_sampled_probs, dim = 1))):

                        arg_types = self.argument_type_net(_prog_sample_)[0]
                        past_program_variables = self.manual_gather_nd(beam_properties['program_argument_table_index'][beam_id], \
                                          torch.cat([torch.unsqueeze(self.batch_ids, dim=1), torch.unsqueeze(_prog_sample_, dim=1)], dim=1))

                        #past_program_variables is of dimension batch_size x (max_arguments * max_num_var)
                        input_var_sampled_probs, input_var_sampled_indices = self.input_var_net(beam_properties['h'][beam_id],\
                                                               arg_types, _prog_sample_, _prog_embedding_,\
                                                               _kb_attention_for_sampled_progs_, \
                                                               beam_id, self.num_variables_to_sample, \
                                                               beam_properties['terminate'][beam_id], \
                                                               past_program_variables)[0:-1]
                        # input_var_sampled_indices has shape batch_size x num_variables_to_sample
                        # input_var_sampled_probs has shape batch_size x num_variables_to_sample

                        per_program_input_var_sampled_indices.append(input_var_sampled_indices)
                        complete_action_probs.append(torch.mul(input_var_sampled_probs, _program_prob_.view([-1,1])))

                    beam_properties['input_var_sampled_indices'][beam_id] = torch.stack(per_program_input_var_sampled_indices, dim=1)
                    # beam_properties['input_var_sampled_indices'] is beam_sized list containing tensors of
                    # shape batch_size x num_programs_to_sample x num_variables_to_sample

                    complete_action_probs = torch.stack(complete_action_probs, dim=1)
                    #complete_action_probs is a tensor of shape batch_size x num_progs_to_sample x num_vars_to_sample
                    complete_action_probs =  complete_action_probs.view([self.batch_size,-1])
                    #complete_action_probs is a tensor of shape batch_size x num_actions.
                    # each program and joint_variables selectiont becomes an action

                    complete_action_probs = torch.clamp(complete_action_probs,self.eps,0.9)
                    log_complete_action_probs = torch.log(complete_action_probs)
                    entropy = entropy+ (-1*torch.sum(complete_action_probs*log_complete_action_probs))

                    if self.params['normalize_length'] is 1:
                        if t is 0:
                            current_beam_score[beam_id] = log_complete_action_probs + torch.unsqueeze(current_beam_score[beam_id],dim=1)
                        else:
                            score_if_terminated = log_complete_action_probs + torch.unsqueeze(current_beam_score[beam_id],dim=1)
                            power = 0.4
                            n1 = torch.pow(beam_properties['length'][beam_id], power)/torch.pow(beam_properties['length'][beam_id]+1.0, power)
                            n2 = 1.0/torch.pow(beam_properties['length'][beam_id]+1.0, power)
                            score_if_not_terminated = n2*log_complete_action_probs + n1*torch.unsqueeze(current_beam_score[beam_id],dim=1)
                            old_cbs = torch.unsqueeze(current_beam_score[beam_id],dim=1)*torch.ones_like(log_complete_action_probs, device=self.device)
                            current_beam_score[beam_id] = beam_properties['terminate'][beam_id]*score_if_terminated + \
                                                            (1-beam_properties['terminate'][beam_id])*score_if_not_terminated
                            current_beam_score[beam_id] = torch.where(current_beam_score[beam_id]>old_cbs, old_cbs,current_beam_score[beam_id])
                    else:
                        current_beam_score[beam_id] = log_complete_action_probs + torch.unsqueeze(current_beam_score[beam_id],dim=1)


                    if self.params['none_decay'] is 1:
                        power_decay = 0.2
                        penalize_factor = torch.mul(beam_properties['none_count'][beam_id].type(self.dtype_float),\
                                                                  -1*torch.log(torch.tensor(math.pow(t+1,power_decay), device=self.device)*
                                                                   torch.ones_like(beam_properties['none_count'][beam_id], device=self.device, dtype=self.dtype_float)))
                        current_beam_score[beam_id] = current_beam_score[beam_id]+penalize_factor




                    beam_target_type = beam_properties['target_type'][beam_id].view([self.batch_size, 1])
                    beam_gold_type = self.gold_target_type.view([self.batch_size, 1])
                    beam_if_terminated = beam_properties['terminate'][beam_id]

                    if self.params['prune_beam_type_mismatch'] is 1:
#                        print self.DoPruning, "$___DoPruning___$"
                        toadd = self.DoPruning*self.check_if_gold_target(beam_target_type, beam_gold_type, beam_if_terminated, t)
#                        toadd = self.check_if_gold_target(beam_target_type, beam_gold_type, beam_if_terminated, t)
                        to_penalize_beams[beam_id] = toadd+to_penalize_beams[beam_id]
#                        print 100*"#"
#                        print to_penalize_beams[0]
#                        print 100*"#"
#                    beam_properties['check_penalization'][beam_id] = to_penalize_beams[beam_id]
                    if t > 0:
                        penalize_none_start = torch.where(beam_target_type==0,\
                                                       torch.ones_like(beam_target_type, device=self.device),torch.zeros_like(beam_target_type, device=self.device)).type(self.dtype_float)

                        to_penalize_beams[beam_id] = penalize_none_start + to_penalize_beams[beam_id]
                    to_penalize_beams[beam_id] = torch.clamp(to_penalize_beams[beam_id],0,1)
                    current_beam_score[beam_id] = torch.clamp(current_beam_score[beam_id],2*math.log(self.eps),0)

                self.entropy = self.entropy+entropy
                current_score = torch.stack(current_beam_score,dim = 1)
                #current_score is a tensor of shape batch_size x beam_size x num_actions
                to_penalize_score = torch.stack(to_penalize_beams,dim = 1)
#                flag_penalize = torch.min(to_penalize_score,dim=1,keepdim=True)[0]
                flag_penalize = torch.prod(to_penalize_score,dim=1,keepdim=True)
                to_penalize_score = to_penalize_score * (1-flag_penalize)
                to_penalize_score = math.log(self.eps)*to_penalize_score
                current_score = current_score+to_penalize_score
                current_score = torch.clamp(current_score,2*math.log(self.eps),0)

                self.debug_beam_terminate['current_score'].append(current_score)
                current_score = current_score.view([self.batch_size,-1])
                top_scores, indices_top_scores = torch.topk(current_score, k = self.beam_size)
                # top_scores has shape batch_size x beam_size
                # indices_top_scores has shape batch_size x beam_size

                to_return_sequence_logprob = top_scores+0
                #to_return_sequence_logprob  has shape batch_size x beam_size

                old_score = torch.stack(unswitched_beam_properties['total_beam_score'])
                #need to transform this old_score w.r.t changes in beam_id
                #old_score has shape beam_size x batch_size

                #updating the score list
                unswitched_beam_properties['total_beam_score'] = list(torch.unbind(top_scores,dim = 1))


                new_beam_ids, action_ids = self.map_index_to_unflattened(indices_top_scores, [self.beam_size, self.num_actions])
                #new_beam_ids has shape  batch_size x beam_size
                # action_ids has shape batch_size x beam_size

                action_ids = torch.transpose(action_ids, 1,0)
                # action_ids has shape beam_size x batch_size
                # updating the memory w.r.t beams
                new_beam_ids = torch.transpose(new_beam_ids,1,0)
                #new_beam_ids has shape beam_size x batch_size
                self.debug_beam_terminate['new_beam_ids'].append(new_beam_ids)

                #updating old_score w.r.t change in beam_ids
                old_score = self.beam_switch(old_score, new_beam_ids)

                # =============================================================================
                # updating the to_return_per_step_prob w.r.t beam_id changes
                if t > 0:
                    old_prop_val = to_return_per_step_prob+0
                    old_prop_val = torch.transpose(old_prop_val, 1,0)
                    to_return_per_step_prob = self.beam_switch(old_prop_val, new_beam_ids)
                    to_return_per_step_prob = torch.transpose(to_return_per_step_prob, 1,0)
                # ______________________________________________________________________________
                ################################################################################

                # =============================================================================
                # For Printing Per Step Prob
                delta_score  =  to_return_sequence_logprob-torch.transpose(old_score, 1,0)
                current_probs = torch.exp(delta_score)
                multiplier = self.one_hot(t*torch.ones([self.batch_size, self.beam_size], device=self.device), depth = self.num_timesteps, dtype=self.dtype_float)
                additand = torch.mul(multiplier, current_probs.view([self.batch_size, self.beam_size,1]).repeat(1,1,self.num_timesteps))
                additand2 = torch.mul(to_return_per_step_prob,1-multiplier)
                to_return_per_step_prob = additand2+additand
                # ______________________________________________________________________________
                ################################################################################
                self.debug_beam_terminate['to_return_per_step_prob'].append(to_return_per_step_prob)
                self.debug_beam_terminate['to_return_sequence_logprob'].append(torch.exp(to_return_sequence_logprob))

                # =============================================================================
                # updating the beam_properties w.r.t beam_id changes
                for prop in beam_properties.keys():
                    old_prop_val = torch.stack(beam_properties[prop],dim=0)
                    # each beam_prop will be of shape beam_size x batch_size x Tensor_shape
                    new_prop_val = self.beam_switch(old_prop_val, new_beam_ids)
                    beam_properties[prop] = list(torch.unbind(new_prop_val, dim = 0))
                # ______________________________________________________________________________
                ###############################################################################

                # =============================================================================
                # updating the variable properties corresponding to beams w.r.t beam_id changes
                    # variable_properties are :-
                        #variable_embedding - beam_size x [num_argtypes, batch_size, max_num_var, var_embed_dim]
                        #variable_keys - beam_size x [num_argtypes, batch_size, max_num_var, var_key_dim]
                        #variable_mask - beam_size x [num_argtypes, batch_size, max_num_var]
                        #variable_atten_table - beam_size x num_argtypes x [batch_size, max_num_var]
                # keeping in mind beam_size

                #1)variable_embedding
                old_prop_val = torch.stack(self.variable_embedding, dim=0)
                old_prop_val = old_prop_val.permute([0,2,1,3,4])
                # now old_prop_val has shape beam_size x batch_size x (tensor_shape = num_argtypes x max_num_var x var_embed_dim)
                new_prop_val = self.beam_switch(old_prop_val, new_beam_ids)
                self.variable_embedding = list(torch.unbind(new_prop_val.permute([0,2,1,3,4]), dim = 0))
                # variable_embedding beam_size x [num_argtypes, batch_size, max_num_var, var_embed_dim]

                #2)variable_keys
                old_prop_val = torch.stack(self.variable_keys, dim=0)
                # old_prop_val [beam_size, num_argtypes, batch_size, max_num_var, var_key_dim]
                old_prop_val = old_prop_val.permute([0,2,1,3,4])
                new_prop_val = self.beam_switch(old_prop_val, new_beam_ids)
                self.variable_keys = list(torch.unbind(new_prop_val.permute([0,2,1,3,4]), dim = 0))
                # variable_keys beam_size x [num_argtypes, batch_size, max_num_var, var_key_dim]

                #3)variable_mask
                old_prop_val = torch.stack(self.variable_mask, dim=0)
                # old_prop_val [beam_size, num_argtypes, batch_size, max_num_var]
                old_prop_val = old_prop_val.permute([0,2,1,3])
                new_prop_val = self.beam_switch(old_prop_val, new_beam_ids)
                self.variable_mask = list(torch.unbind(new_prop_val.permute([0,2,1,3]), dim = 0))
                # variable_mask beam_size x [num_argtypes, batch_size, max_num_var]

                #4)variable attention table
                #variable_atten_table - beam_size x num_argtypes x [batch_size, max_num_var]
                old_prop_val = []
                for beam_id in xrange(self.beam_size):
                    old_prop_val.append(torch.stack(self.variable_atten_table[beam_id], dim=1))
                old_prop_val = torch.stack(old_prop_val, dim = 0)
                # old_prop_val [beam_size, batch_size, num_argtypes, max_num_var]
                new_prop_val = self.beam_switch(old_prop_val, new_beam_ids)
                temp = list(torch.unbind(new_prop_val.permute([0,2,1,3]), dim = 0))
                self.variable_atten_table = [list(torch.unbind(_temp_, dim = 0)) for (beam_id, _temp_) in \
                                             zip(xrange(self.beam_size), temp)]
                # variable_atten_table beam_size x num_argtypes x [batch_size, max_num_var]
                # done updating beam_memory
                # done updating variable_memeory
                # ______________________________________________________________________________
                ###############################################################################

                # =============================================================================
                # have to update to_return_action_* w.r.t to change in beam_id
                if t > 0:
                    for key in ['program_type','argument_type','target_type','target_table_index','argument_table_index']:

                        # to_return_action_sequence beam_size x seq_length x [tensor_shape]
                        old_prop_val = []
                        for beam_id in xrange(self.beam_size):
                            temp = torch.stack(to_return_action_sequence[key][beam_id], dim=1)
                            # temp [seq_length x tensor_shape]
                            old_prop_val.append(temp)
                        old_prop_val = torch.stack(old_prop_val, dim = 0)
                        # beam_size x batch_size x seq_length x tensor_shape
                        new_prop_val = self.beam_switch(old_prop_val, new_beam_ids)
                        temp = torch.unbind(new_prop_val, dim = 0)
                        to_return_action_sequence[key] = [list(torch.unbind(_temp_, dim = 1)) for (beam_id, _temp_) in \
                                                 zip(xrange(self.beam_size), temp)]
                        #print key, ':: to_return_action_sequence[',key,']', to_return_action_sequence[key]
                        # done updating to_return_action_* w.r.t to change in beam_id
                # _____________________________________________________________________________
                ###############################################################################

                # =============================================================================
                #getting the pointer to program sample and pointer to variable sample from action_id
                [pointer_to_prog_sample, \
                 pointer_to_variable_sample] = self.map_index_to_unflattened(action_ids,[self.num_programs_to_sample,\
                                                                                           self.num_variables_to_sample])
                # pointer_to_prog_sample has shape beam_size x batch_size
                # pointer_to_variable_sample has shape beam_size x batch_size

                # getting the actual program samples
                # pointer_to_prog_sample beam_size x batch_size
                multiplicand_2 = torch.stack(beam_properties['prog_sampled_indices'], dim = 0)
                #multiplicand_2 beam_size x batch_size x num_programs_to_sample
                multiplicand_1 = self.one_hot(pointer_to_prog_sample, depth=self.num_programs_to_sample, dtype=multiplicand_2.dtype)
                #multiplicand_1 beam_size x batch_size x num_programs_to_sample
                true_program_sample = torch.sum(torch.mul(multiplicand_1, multiplicand_2), dim = 2)
                #true_program_sample is a tensor of shape beam_size x batch_size
                # _____________________________________________________________________________
                ###############################################################################

                # =============================================================================
                # checking if any beam has terminated
                for prog_samples, beam_id in zip(list(torch.unbind(true_program_sample, dim = 0)), xrange(self.beam_size)):
                    beam_properties['terminate'][beam_id] = self.terminate_net(prog_samples, beam_properties['terminate'][beam_id])
                    #update the length
                    beam_properties['length'][beam_id] = beam_properties['length'][beam_id] + (1.0-beam_properties['terminate'][beam_id])
                    beam_properties['none_count'][beam_id] = self.none_finder_net(prog_samples)
                # _____________________________________________________________________________
                ###############################################################################

                # =============================================================================
                # getting the actual variable samples
                # beam_properties['input_var_sampled_indices'] is a list of length beam_size with
                # tensor of shape [batch_size x num_programs_to_sample x num_variables_to_sample]
                multiplicand_1 = torch.stack(beam_properties['input_var_sampled_indices'], dim = 0)
                multiplicand_2 = torch.unsqueeze(self.one_hot(pointer_to_prog_sample, depth = self.num_programs_to_sample, dtype=multiplicand_1.dtype), dim = 3)
                flattened_input_var_sample = torch.sum(torch.mul(multiplicand_1, multiplicand_2),dim = 2)
                # flattened_input_var_sample  has shape [beam_size x batch_size x num_variables_to_sample]
                multiplicand_1 = flattened_input_var_sample
                multiplicand_2 = self.one_hot(pointer_to_variable_sample, depth = self.num_variables_to_sample, dtype=multiplicand_1.dtype)
                flattened_input_var_sample = torch.sum(torch.mul(multiplicand_1, multiplicand_2), dim = 2)
                # flattened_input_var_sample  has shape [beam_size x batch_size]
                actual_var_samples_list = self.map_index_to_unflattened(flattened_input_var_sample, \
                                                                         [self.max_num_var for _ in xrange(self.max_arguments)])
                # is a max_arguments sized list containing tensors of shape [beam_size x batch_size]
                # this contains the actual variable samples
                #print 'actual_var_samples_list ', actual_var_samples_list
                actual_var_samples_list = list(torch.unbind(torch.stack(actual_var_samples_list,dim = 2).type(self.dtype_int64), dim = 0))
                # actual_var_samples_list is a list of beam_size length containing tensors of shape [batch_size x max_arguments]
                # _____________________________________________________________________________
                ###############################################################################

                # =============================================================================
                # Code For Preventing step repitition in generated trajectories
                # need to do a scatter update on 'program_argument_table_index' and set used steps to 0
                index_0 = torch.range(0,self.beam_size*self.batch_size-1,device = self.device, dtype=self.dtype_int64)
                # index_0 has shape [(beam_size*batch_size)]
                index_1 = true_program_sample.view([-1])
                # index_1 has shape [(beam_size*batch_size)]
                index_2 = flattened_input_var_sample.view([-1])
                # index_2 has shape [(beam_size*batch_size)]
                new_value = torch.ones_like(index_1, device=self.device)
                b1 = (index_1==torch.zeros_like(index_1, device=self.device)).type(self.dtype_float)
                b2 = (index_1==self.num_progs-1*torch.ones_like(index_1,device=self.device)).type(self.dtype_float)
                new_value = torch.where(torch.max(b1,b2)>0, new_value, 0*new_value)
                old_property_value = torch.stack(beam_properties['program_argument_table_index'], dim = 0)
                # old_property_value has shape beam_size x batch_size x num_progs x (max_arguments*max_num_var)
                old_property_value = old_property_value.view([-1, self.num_progs, int(math.pow(self.max_num_var,self.max_arguments))])
                index_for_scatter = torch.stack([index_0, index_1, index_2], dim = 1)
                old_property_value = self.immutable_scatter_nd_constant_update(old_property_value, index_for_scatter, new_value)
                old_property_value = old_property_value.view([self.beam_size, self.batch_size, self.num_progs,\
                                                                     int(math.pow(self.max_num_var,self.max_arguments))])
                beam_properties['program_argument_table_index'] = torch.unbind(old_property_value)
                true_program_sample = true_program_sample.type(self.dtype_int64)
                # _____________________________________________________________________________
                ###############################################################################

                # =============================================================================
                # returning the program samples and similar stuff
                for beam_id, true_prog_samples, true_var_samples in zip(xrange(self.beam_size),\
                                                                         list(torch.unbind(true_program_sample)), actual_var_samples_list):

                    to_return_action_sequence['program_type'][beam_id].append(true_prog_samples)

                    arg_types, argtype_embedding = self.argument_type_net(true_prog_samples)
                    to_return_action_sequence['argument_type'][beam_id].append(torch.transpose(arg_types, 1,0))

                    to_return_action_sequence['argument_table_index'][beam_id].append(true_var_samples)

                    #with tf.device('/cpu:0'):
                    target_types = torch.index_select(self.program_to_targettype_table, 0, true_prog_samples)
                    to_return_action_sequence['target_type'][beam_id].append(target_types)

                    # =============================================================================
                    # need to track current target program type so that we can terminate if gold type occurs
                    condition = torch.max((target_types==torch.zeros_like(target_types, device=self.device)).type(self.dtype_float), (target_types==(self.num_progs-1*torch.ones_like(target_types,device=self.device))).type(self.dtype_float))
                    beam_properties['target_type'][beam_id] = torch.where(condition>0., beam_properties['target_type'][beam_id].type(self.dtype_int64), target_types)
                    # _____________________________________________________________________________

                    prog_sampled_embeddings = self.program_embedding(true_prog_samples)
                    argtypes = list(torch.unbind(arg_types, dim=0))
                    var_embed = [self.manual_gather_nd(self.variable_embedding[beam_id], torch.stack([argtypes[i], self.batch_ids], dim=1)) \
                                 for i in xrange(self.max_arguments)]
                    #var_embed is a max_arguments sized list of batch_size x max_num_var x var_embed_dim
                    var_sample = list(torch.unbind(true_var_samples, dim = 1))
                    # var_sample  is a max_arguments sized list of tensors of shape batch_size
                    var_sample_index = [torch.stack([self.batch_ids, var_sample[i]], dim=1) for i in range(self.max_arguments)]
                    input_var_embedding = [self.manual_gather_nd(var_embed[i], var_sample_index[i]) for i in xrange(self.max_arguments)]
                    num_variables_till_now, R_Flag = self.get_num_variables_till_now(beam_id, target_types)
                    [target_var_key, \
                     beam_properties['target_var_embedding'][beam_id]] = self.target_var_net(input_var_embedding, \
                                                                                             argtype_embedding, \
                                                                                             prog_sampled_embeddings, num_variables_till_now, \
                                                                                             target_types)
                    self.add_to_variable_table(target_types, target_var_key,\
                                                                        beam_properties['target_var_embedding'][beam_id], \
                                                                        num_variables_till_now, beam_id = beam_id)




                    # =============================================================================
                    # whenever any variable table overflows we need to give negative reward for that
                    beam_properties['Model_Reward_Flag'][beam_id] = beam_properties['Model_Reward_Flag'][beam_id]+R_Flag
                    # _____________________________________________________________________________

                    to_return_action_sequence['target_table_index'][beam_id].append(num_variables_till_now)
                # _____________________________________________________________________________
                ###############################################################################

            # reshaping stuff so that it can be handled by main function
            for beam_id in xrange(self.beam_size):
                for i in xrange(self.num_timesteps):
                    to_return_action_sequence['argument_table_index'][beam_id][i] = list(torch.unbind(\
                                             to_return_action_sequence['argument_table_index'][beam_id][i],dim = 0))

            # =============================================================================
            #setting the Model Reward FLAG
            to_return_action_sequence['Model_Reward_Flag'] = beam_properties['Model_Reward_Flag']
            # _____________________________________________________________________________
            ###############################################################################

            # to_return_action_sequence['argument_table_index'] is a list of length beam_size containing a list of
            # length num_timesteps containing a list of max argument length with tensors of shape batch size
            self.ProgramProb = torch.exp(to_return_sequence_logprob)
            self.logProgramProb = to_return_sequence_logprob
            self.per_step_prob = to_return_per_step_prob
            self.entropy = self.entropy/self.num_timesteps
#            print 100*'%'
#            print beam_properties['target_type']
#            print beam_properties['terminate']
#            print self.gold_target_type
#            print beam_properties['check_penalization']
#            print 100*'%'
            return to_return_action_sequence, torch.exp(to_return_sequence_logprob), \
                to_return_sequence_logprob, self.debug_beam_terminate, to_return_per_step_prob, self.entropy/self.num_timesteps


    def get_feasible_progs(self, timestep, phase_elasticity):
        num_variables = [torch.transpose(torch.sum(self.variable_mask[i], dim=2, dtype=self.dtype_int64), 1,0) for i in range(len(self.variable_mask))]
        #num_variables is a beam_size sized list of dimension batch_size x num_argtypes
        num_variables_remaining  = [self.required_argtypes - num_variables[i]  for i in range(len(self.variable_mask))]
        num_variables_remaining = [torch.where(num_variables_remaining[i]>0, num_variables_remaining[i], torch.zeros_like(num_variables_remaining[i], device=self.device)) for i in range(len(self.variable_mask))]
        num_variables_remaining = [torch.unsqueeze(num_variables_remaining[i], 1).repeat([1, self.num_progs, 1]) for i in range(len(self.variable_mask))]
        program_to_targettype_onehot = self.one_hot(self.program_to_targettype_table, depth=self.num_argtypes)
        #program_to_targettype_onehot is of dimension num_progs x num_argtypes
#        print num_variables_remaining[0].dtype,program_to_targettype_onehot.dtype
        reqd_programs = [torch.max(torch.mul(num_variables_remaining[i], program_to_targettype_onehot), dim=2)[0].type(self.dtype_float) for i in range(len(self.variable_mask))]
        #reqd_programs is a beam_size sized list of dimension batch_size x num_progs
        #self.program_to_num_arguments is of dimension num_progs x num_argtypes
        num_variable_types = [torch.max(self.variable_mask[i], dim=2)[0].type(self.dtype_int64) for i in range(len(self.variable_mask))]
        #num_variable_types is a beam_size sized list of dimension  num_argtypes x  batch_size
        num_variable_types = [torch.unsqueeze(num_variable_types[i],dim=0).repeat([self.num_progs,1,1]).permute([2,0,1]) for i in \
                         range(len(self.variable_mask))]
        #num_variable_types is a beam_size sized list of dimension batch_size x num_progs x num_argtypes
        feasible_progs = [torch.where(num_variable_types[i]>=self.program_to_num_arguments, \
                                   torch.ones_like(num_variable_types[i], device=self.device), torch.zeros_like(num_variable_types[i], device=self.device)) \
                                    for i in range(len(self.variable_mask))]
        #feasible_progs is of dimension batch_size x num_progs x num_argtypes
        feasible_progs = [torch.prod(feasible_progs[i], dim=2).type(self.dtype_float) for i in range(len(self.variable_mask))]
        #feasible_progs is of dimension batch_size x num_progs

        program_to_kb_attention = torch.max(self.kb_attention, dim=2)[0]
        feasible_progs = [torch.mul(program_to_kb_attention, feasible_progs[i]) for i in range(len(self.variable_mask))]


        def separate_phases(arg):
            feasible_prog = arg[0]
            phase_elasticity = arg[1]
            if timestep < self.max_num_phase_1_steps:
                temp = phase_elasticity.repeat([1,self.num_progs])
                multiplicand1 = self.progs_phase_1.type(self.dtype_float)
            else:
                temp = (1-phase_elasticity).repeat([1,self.num_progs])
                multiplicand1 = self.progs_phase_2.type(self.dtype_float)
            multiplicand2 = 1 - multiplicand1
            multiplicand = torch.mul(temp, multiplicand1) + torch.mul(1-temp, multiplicand2)
            feasible_prog = torch.mul(feasible_prog, multiplicand)
            return feasible_prog

        feasible_progs = map(separate_phases, zip(feasible_progs,phase_elasticity))


        # =============================================================================
        # Hard Rules
        temp = self.one_hot(torch.zeros([self.batch_size], device=self.device), depth=self.num_progs, dtype=self.dtype_float)
        feasible_progs = [temp + (1-temp)*feasible_progs[i] for i in range(len(self.variable_mask))]
        if timestep == 0:
            def make_none_impossible(prog_mask):
                temp = self.one_hot(torch.zeros([self.batch_size], device=self.device), depth = self.num_progs, dtype=self.dtype_float)
                new_mask = -1*temp + (1-temp)
                prog_mask = torch.mul(new_mask, prog_mask)
                return prog_mask
            feasible_progs = map(make_none_impossible,feasible_progs)
        # _____________________________________________________________________________
        #print 'feasible progs ', [feasible_progs[i] for i in range(len(self.variable_mask))]
        #print 'feasible_progs[i]+reqd_programs[i] ', [feasible_progs[i]+reqd_programs[i] for i in range(len(self.variable_mask))]
        feasible_progs_new = [torch.where(feasible_progs[i]>0, feasible_progs[i]+reqd_programs[i], feasible_progs[i])  for i in range(len(self.variable_mask))]
        feasible_progs = [torch.mul(self.bias_prog_sampling_with_target, feasible_progs_new[i]) + torch.mul((1.0-self.bias_prog_sampling_with_target), feasible_progs[i]) for i in range(len(self.variable_mask))]
        return feasible_progs

    def add_preprocessed_output_to_variable_table(self, beam_id):
        R_Flag = torch.zeros([self.batch_size], device=self.device)
        for i in xrange(self.num_argtypes):
            if i==self.empty_argtype_id:
                continue
            for j in xrange(self.max_num_var):
                ones = i*torch.ones([1, self.batch_size], dtype=self.dtype_int64, device=self.device)
                empties = self.empty_argtype_id*torch.ones([self.max_arguments-1, self.batch_size], dtype=self.dtype_int64, device=self.device)
                argtype = torch.cat([ones, empties], dim=0)
                #argtype is of dimension max_arguments x batch_size
                argtype_embed = self.argtype_embedding(argtype)
                input_var_embedding = torch.unsqueeze(torch.matmul(self.preprocessed_var_emb_table[i][j], self.preprocessed_var_emb_mat), dim=0)
                #input_var_embedding is of dimension 1 x batch_size x var_embed_dim
                zeros_embedding = torch.zeros([self.max_arguments-1, self.batch_size, self.var_embed_dim], device=self.device)
                input_var_embedding = torch.cat([input_var_embedding, zeros_embedding], dim=0)
                #input_var_embedding is of dimension max_arguments x batch_size x var_embed_dim
                target_types = i*torch.ones([self.batch_size], device=self.device, dtype=self.dtype_int64)
                num_variables_till_now, cur_r_flag = self.get_num_variables_till_now(beam_id, target_types)
                [target_var_key, \
                 target_var_embedding] = self.target_var_net_for_preprocessed_output(input_var_embedding, argtype_embed, num_variables_till_now, target_types)
                #target_types is of dimension batch_size
                self.add_to_variable_table(target_types, target_var_key, target_var_embedding, num_variables_till_now, beam_id = beam_id)
                R_Flag = R_Flag + cur_r_flag
        # once variable props from preprocessing are copied to main variable table
        # update main variable mask. Initialize main variable mask with the masks in preprocessed variable mask table
        self.variable_mask[beam_id] = torch.stack([torch.stack(temp, dim = 1) for temp in \
                                                  self.preprocessed_var_mask_table], dim = 0)
        self.variable_atten_table[beam_id] = list(torch.unbind(self.variable_mask[beam_id]+0))
        return R_Flag


    def sentence_encoder(self):
        sentence_outputs = None
        rnn_inputs_w2v = self.word_embeddings(self.encoder_text_inputs_w2v)
        rnn_inputs_kb_emb = self.encoder_text_inputs_kb_emb
        rnn_inputs = torch.cat([rnn_inputs_w2v, rnn_inputs_kb_emb], dim = 2)
        init_state = torch.unsqueeze(self.init_state.repeat([self.batch_size, 1]), dim=0)
        print rnn_inputs.shape, init_state.shape
        sentence_outputs, states = self.sentence_rnn(rnn_inputs, init_state)
        attention_states = torch.transpose(sentence_outputs.view([self.batch_size,self.max_len,-1]), 1,0)
        #attention_states is of dimension max_len x batch_size x cell_dim
        return states, attention_states

    def get_feasible_progs_for_last_timestep(self):
        gold_type = self.gold_target_type
        #gold_type is of dimension batch_size
        gold_type = torch.unsqueeze(gold_type, dim=1).repeat([1, self.num_progs])
        #gold_type is of dimension batch_size x num_progs
        feasible_progs_for_last_timestep = torch.where((gold_type==self.program_to_targettype_table), torch.ones_like(gold_type, device=self.device), torch.zeros_like(gold_type, device=self.device))
        #feasible_progs_for_last_timestep is of dimension batch_size x num_progs
        return feasible_progs_for_last_timestep


    def attention_on_relations(self, attention_states):
        attention_states = torch.matmul(torch.matmul(attention_states, self.query_rel_atten), torch.transpose(self.rel_embedding,1,0))
        attention_states = torch.sum(attention_states, dim=0)
        attention_states = nn.functional.softmax(attention_states)
        #attention_states is of dimension batch_size x num_rel
        return attention_states

    def attention_on_types(self, attention_states):
        attention_states = torch.matmul(torch.matmul(attention_states, self.query_type_atten), torch.transpose(self.type_embedding,1,0))
        attention_states = torch.sum(attention_states, dim=0)
        attention_states = nn.functional.softmax(attention_states)
        return attention_states

    def reset_state(self, sentence_state):
        zero_state = torch.zeros([self.batch_size, self.npi_core_dim],device=self.device)
        h_states = zero_state
        e_state = self.dropout(self.elu(self.reset_layer(sentence_state)))
        target_var_embedding = torch.zeros([self.batch_size, self.var_embed_dim],device=self.device)
        h_states = torch.unsqueeze(h_states, dim=0)
        return h_states, e_state, target_var_embedding

    def npi_core(self, h_state, e_state, target_var_embedding):
        s_in = torch.unsqueeze(self.state_encoding(e_state, target_var_embedding), dim=0)
        #s_in is of dimension 1 x batch_size x state_dim
        target_var_embedding = torch.unsqueeze(target_var_embedding, dim=0)
        c = torch.cat([s_in, target_var_embedding], dim=2)
        #c is of dimension 1 x batch_size x (state_dim + var_embed_dim)
        #c = torch.transpose(c, 1,0)
        h_state, c = self.npi_rnn(c, h_state)
        #h_state is of dimension batch_size x npi_core_dim
        return c, h_state

    def env_encoding(self, e_state, target_var_embedding):
        c = torch.unsqueeze(target_var_embedding,dim=0)
        #c is of dimension 1 x batch_size x var_embed_dim
        #c = torch.transpose(c, 1,0)
        c, e_state = self.env_rnn(c, e_state)
        return c, e_state

    def state_encoding(self, e_state, target_var_embedding):
        merge = torch.cat([e_state.view([self.batch_size, -1]), target_var_embedding], dim=1)
        #merge is of dimension batch_size x (self.npi_core_dim+var_embed_dim)
        elu = self.dropout(self.elu(self.state_encoding_layer1(merge)))
        #elu is of dimension batch_size x hidden_dim
        elu = self.dropout(self.elu(self.state_encoding_layer2(elu)))
        #elu is of dimension batch_size x hidden_dim
        out = self.dropout(self.elu(self.state_encoding_layer3(elu)))
        #out is of dimension batch_size x state_dim
        return out

    def terminate_net(self, progs_taken, old_terminate):
        temp1 = torch.ones_like(progs_taken, device=self.device, dtype=self.dtype_int64)
        temp2 = torch.zeros_like(progs_taken, device=self.device, dtype=self.dtype_int64)
        # 0 is the None action
        terminate = torch.where((progs_taken==self.num_progs-1), temp1, temp2)
        terminate = terminate.view([self.batch_size, 1]).type(old_terminate.dtype)
        terminate = torch.where(terminate>=old_terminate, terminate, old_terminate)
        return terminate
        # this will return tensor of shape batch_size x 1

    def none_finder_net(self, progs_taken):
        temp1 = torch.ones_like(progs_taken, device=self.device)
        temp2 = torch.zeros_like(progs_taken, device=self.device)
        # 0 is the None action
        out = torch.where((progs_taken==0), temp1, temp2)
        out = out.view([self.batch_size, 1])
        return out
        # this will return tensor of shape batch_size x 1

    def check_if_gold_target(self, beam_target_type, beam_gold_type, if_terminated, t):
        mask_same_type = torch.where((beam_target_type==beam_gold_type), torch.zeros_like(beam_target_type, device=self.device), \
                                  torch.ones_like(beam_target_type, device=self.device)).type(self.dtype_float)

        if t < self.num_timesteps-1:
            return torch.mul(mask_same_type,if_terminated)
        else:
            return mask_same_type

    def phase_change_net(self, h, timestep, old_p_el):
        if timestep < self.max_num_phase_1_steps:
            p_el = self.dropout(self.phase_change_layer(h))
            p_el = nn.functional.sigmoid(p_el)
            p_el = torch.where(p_el>old_p_el, old_p_el, p_el)
            temp = torch.ones_like(p_el, device=self.device)
            p_el = torch.where(p_el>self.phase_change_threshold, temp, p_el)
            return p_el
        else:
            temp = torch.zeros_like(old_p_el, device=self.device)
            return temp

    def prog_net(self, h, sentence_state, attention_states, query_attentions_till_now, feasible_progs, num_samples, terminate, last_target_type):
        #print 'feasible progs', feasible_progs
        #feasible_progs is of shape batch_size x num_progs
        # variable_mask beam_size x [num_argtypes, batch_size, max_num_var]
        #self.program_to_argtype_table is of dimension num_progs x max_arguments
        #last_target_type is of dimension batch_size
        last_target_type = last_target_type.view([-1,1,1]).repeat(1,self.num_progs, self.max_arguments)
        programs_consuming_last_targettype = torch.max(torch.where((self.program_to_argtype_table==last_target_type),\
                        torch.ones_like(last_target_type, dtype=self.dtype_int64, device=self.device), torch.zeros_like(last_target_type, device=self.device)), dim=2)[0].type(self.dtype_float)
        feasible_progs_new = torch.where(feasible_progs>0, feasible_progs+ programs_consuming_last_targettype, feasible_progs)
        feasible_progs = torch.mul(self.bias_prog_sampling_with_last_variable, feasible_progs_new) + torch.mul((1.0-self.bias_prog_sampling_with_target), feasible_progs)
        #programs_consuming_last_targettype is of dimension batch_size x num_progs
        #feasible_progs is of dimension batch_size x num_progs
        if self.concat_query_npistate:
            concat_hq = torch.cat([h, sentence_state], dim=1)
        else:
            concat_hq = h
        concat_hq = concat_hq.view([self.batch_size, -1])
        if self.query_attention:
            query_attention = torch.mul(attention_states, torch.mul(h, self.query_attention_h_mat))
            #temp is of dimension max_len x batch_size x cell_dim
            query_attention = nn.functional.softmax(torch.sum(query_attention, dim=2), dim=0)
            #query_attention is of dimension max_len x batch_size
            if self.dont_look_back_attention:
                query_attentions_till_now = torch.transpose(query_attentions_till_now, 1,0)
                query_attention = nn.functional.softmax(torch.mul(1.-query_attentions_till_now, query_attention), dim=0)
                query_attentions_till_now = nn.functional.softmax(query_attentions_till_now+query_attention, dim=0)
                query_attentions_till_now = torch.transpose(query_attentions_till_now, 1,0)

            query_attention = torch.unsqueeze(query_attention, dim=2)
            query_attention = torch.sum(torch.mul(query_attention, attention_states), dim=0)
            concat_hq = torch.cat([concat_hq, query_attention], dim=1)
        hidden = self.dropout(self.prog_key_layer1(concat_hq))
        key = self.dropout(self.prog_key_layer2(hidden))
        key = key.view([-1, 1, self.prog_key_dim])
        prog_sim = torch.mul(key, self.program_keys)
        prog_dist = torch.sum(prog_sim, 2)
        prog_dist = nn.functional.softmax(prog_dist, dim=1)
        if self.params['terminate_prog'] is True:
            temp = self.one_hot((self.num_progs-1)*torch.ones([self.batch_size], device=self.device), depth=self.num_progs, dtype=self.dtype_float)
            feasible_progs = terminate*temp + (1-terminate)*feasible_progs
        prog_dist = torch.mul(prog_dist, feasible_progs)
        #prog_dist is of dimension batch_size x num_progs

        prog_sampled_probs, prog_sampled_indices = self.bernoulli_program_sampling(prog_dist, num_samples)
        prog_sampled_probs = torch.div(prog_sampled_probs,torch.sum(torch.clamp(prog_dist,0,1), dim=1, keepdim=True))
        # prog_sampled_probs is a tensor of shape batch_size x num_samples
        # prog_sampled_indices is a tensor of shape batch_size x num_samples

        prog_sampled_embeddings = self.program_embedding(prog_sampled_indices)
        # prog_sampled_embeddings is a tensor of shape batch_size x num_samples x prog_embed_dim
        list_program_sample_index = list(torch.unbind(prog_sampled_indices,dim=1))
        # list_program_sample_index is a num_samples length list composed of batch_size sized tensors
        kb_attention_for_sampled_progs = []
        for prog_sample_index in list_program_sample_index:
            prog_sample_index = torch.stack([self.batch_ids, prog_sample_index], dim=1)
            kb_attention_for_sampled_progs.append(self.manual_gather_nd(self.kb_attention, prog_sample_index))
        # kb_attention_for_sampled_progs is a num_samples length list composed of batch_size x max_var x max_var x max_var sized tensors
        return prog_sampled_probs, prog_sampled_indices, prog_sampled_embeddings, \
                torch.stack(kb_attention_for_sampled_progs, dim = 0), query_attentions_till_now


    def argument_type_net(self, prog_sample):
        #with tf.device('/cpu:0'):
        arg_types = torch.index_select(self.program_to_argtype_table, 0, prog_sample)
        # argtypes is of dimension batch_size x max_arguments
        # argtypes is a list of argument types for that sampled program
        # in order to handle different length argtypes in a batch,
        # consider that for every program there is max upto N arguments only (with padding whenever necessary)
        argtype_embedding = self.argtype_embedding(arg_types)
        #argtype_embeddign is of dimension batch_size x max_arguments x argtype_embed_dim
        arg_types = torch.transpose(arg_types, 1,0)
        argtype_embedding = torch.transpose(argtype_embedding, 1,0)
        #argtype_embeddign is of dimension max_arguments  x batch_size x argtype_embed_dim
        return arg_types, argtype_embedding

    def input_var_net(self, h, arg_types, prog_sample, prog_embedding, kb_attention, beam_id, num_samples, terminate, past_program_variables):
        #prog_sample is of batch_size
        target_types = torch.index_select(self.program_to_targettype_table, 0, prog_sample)
        # targettypes is of dimension batch_size
        argtypes = list(torch.unbind(arg_types, dim=0))
        # argtypes is a max_arguments sized list of dimension batch_size each
        local_var_atten = torch.stack(self.variable_atten_table[beam_id], dim=0)
        #with tf.device('/cpu:0'):
        var_atten = [self.manual_gather_nd(local_var_atten,torch.stack([argtypes[i], self.batch_ids], dim=1)) \
                     for i in xrange(self.max_arguments)]
        # var_atten is a max_arguments sized list of batch_size x max_num_var
        #with tf.device('/cpu:0'):
        var_mask = [self.manual_gather_nd(self.variable_mask[beam_id],torch.stack([argtypes[i], self.batch_ids], dim=1)) \
                    for i in xrange(self.max_arguments)]
        # var_mask is a max_arguments sized list of batch_size x max_num_var
        var_atten = [self.update_attention(var_atten[i], h, i) for i in range(self.max_arguments)]
        var_atten = [self.mask_attention(var_atten[i], var_mask[i]) for i in xrange(self.max_arguments)]
        # var_atten is a max_arguments sized list of batch_size x max_num_var
        #with tf.device('/cpu:0'):
        var_keys = [self.manual_gather_nd(self.variable_keys[beam_id], torch.stack([argtypes[i], self.batch_ids], dim=1)) \
                    for i in xrange(self.max_arguments)]
        # var_keys is a max_arguments sized list of batch_size x max_num_var x var_key_dim
        # var_atten is a max_arguments sized list of batch_size x max_num_var
        key = [self.dropout(self.elu(self.inp_var_key_layer(var_atten[i]))) for i in xrange(self.max_arguments)]
        key = [key[i].view([-1, 1, self.var_key_dim]) for i in xrange(self.max_arguments)]
        var_sim = [torch.mul(key[i], var_keys[i]) for i in xrange(self.max_arguments)]
        # var_sim is of dimension batch_size x max_num_var x var_key_dim
        var_dist = [torch.sum(var_sim[i], 2) for i in xrange(self.max_arguments)]
        var_dist = [nn.functional.softmax(var_dist[i], dim=1) for i in xrange(self.max_arguments)]
        var_dist = [torch.mul(var_dist[i],var_mask[i]) for i in xrange(self.max_arguments)]
        # var_dist is a max_arguments sized list of dimension batch_size x max_num_var

        # we have to get the joint distribution over the different arguments.
        var_dist = torch.stack(var_dist,dim=1)

        #var_mask is of dimension batch_size x max_arguments x max_num_var
        split_var_dist = list(torch.unbind(var_dist, dim=0))
        # split_var_dist is a batch_size sized list of dimension max_arguments x max_num_var
        joint_var_dist = []
        for _var_dist_ in split_var_dist:
            list_vectors_dist = list(torch.unbind(_var_dist_,dim=0))
            joint_var_dist.append(self.recursive_joint_prob_generator(list_vectors_dist))
        joint_var_dist = torch.stack(joint_var_dist,dim=0)
        flattened_joint_var_dist = joint_var_dist.view([self.batch_size,-1])
        flattened_joint_var_dist = torch.mul(flattened_joint_var_dist, kb_attention)
        flattened_joint_var_dist = torch.mul(flattened_joint_var_dist, past_program_variables)
        # =============================================================================
        # ensuring all 0 variable probability vector is handled appropriately
        marker = torch.mean(flattened_joint_var_dist,dim = 1, keepdim=True)
        marker = torch.where((marker==0), 0*torch.ones_like(marker, device=self.device), torch.ones_like(marker, device=self.device))
        flattened_joint_var_dist = self.mask_attention(flattened_joint_var_dist, torch.ones_like(flattened_joint_var_dist, device=self.device))
        flattened_joint_var_dist = torch.mul(flattened_joint_var_dist, marker)
        # ______________________________________________________________________________

        var_sampled_probs, var_sampled_indices = torch.topk(flattened_joint_var_dist, k = num_samples)
        # var_sampled_probs is a tensor of shape batch_size x num_samples
        # var_sampled_indices is a tensor of shape batch_size x num_samples
        return var_sampled_probs, var_sampled_indices, target_types

    def get_num_variables_till_now(self, beam_id, targettypes):
        t = torch.stack([targettypes.type(self.dtype_int64), self.batch_ids], dim=1)
        var_mask = self.manual_gather_nd(self.variable_mask[beam_id], t)
        # var_mask is of dimension batch_size x max_num_var
        num_variables_till_now = torch.sum(var_mask, dim=1).type(self.dtype_int64)
#        num_variables_till_now = num_variables_till_now.type(self.dtype_int64)
        # num_variables_till_now is of dimension batch_size
        # =================================================================================================================
        # for None arg_type we should always ensure there is only one element in table to have consistent probabilities
        # 0 is none type
        num_variables_till_now = torch.where(targettypes==0, torch.zeros_like(num_variables_till_now, device=self.device), num_variables_till_now)
        # =================================================================================================================

        # =============================================================================
        # Return a negative reward if table overpopulates
        temp = (self.max_num_var-1) * torch.ones_like(num_variables_till_now, device=self.device)
        R_Flag = torch.zeros_like(num_variables_till_now, device=self.device, dtype=self.dtype_float)
        R_Flag = torch.where(num_variables_till_now > temp, 1+R_Flag, R_Flag)
        # Overpopulation - Rewrite last entry in table
        num_variables_till_now = torch.where(num_variables_till_now > temp, temp, num_variables_till_now)
        return num_variables_till_now, R_Flag

    def target_var_net(self, input_var_embedding, argtype_embedding, prog_embedding, num_variables_till_now, target_type):
        var_embedding = torch.stack(input_var_embedding, dim=0)
        #var_embedding is of dimension max_arguments x batch_size x var_embed_dim
        argument_type_embedding = argtype_embedding
        #argument_type_embedding is of dimension max_arguments x batch_size x argtype_embed_dim
        target_var_key, target_var_embedding = self.get_target_var_key_and_embedding(var_embedding, \
                                                                                     prog_embedding, \
                                                                                     argument_type_embedding, \
                                                                                     num_variables_till_now, target_type)
        #prog_embedding is of dimension batch_size x prog_embed_dim
        #target_var_embedding is of dimension batch_size x var_embed_dim
        #target_var_key is of dimension batch_size x var_key_dim
        return target_var_key, target_var_embedding

    def target_var_net_for_preprocessed_output(self, input_var_embedding, argtype_embedding, num_variables_till_now, target_type):
        [target_var_key, \
         target_var_embedding] = self.get_target_var_key_and_embedding(input_var_embedding, None, argtype_embedding, num_variables_till_now, target_type)
        return target_var_key, target_var_embedding

    def add_to_variable_table(self, targettypes, target_var_key, target_var_embedding, num_variables_till_now, beam_id = None):
        # =============================================================================

        indices_to_update = torch.stack([targettypes, self.batch_ids, num_variables_till_now], dim=1)
        # indices_to_update is of dimension batch_size x 3
        # variable_mask is of dimension num_argtypes x batch_size x max_num_var
        mask_value_to_update = torch.ones([self.batch_size], device=self.device)
        self.variable_mask[beam_id] = self.immutable_scatter_nd_constant_update(self.variable_mask[beam_id], \
                                                                       indices_to_update, mask_value_to_update)
        # variable_mask is of dimension num_argtypes x batch_size x max_num_var
        self.variable_keys[beam_id] = self.immutable_scatter_nd_1d_update(self.variable_keys[beam_id], \
                                                                       indices_to_update, target_var_key)
        # self.variable_keys is of dimension num_argtypes x batch_size x max_num_var x var_key_dim
        self.variable_embedding[beam_id] = self.immutable_scatter_nd_1d_update(self.variable_embedding[beam_id], \
                                                                      indices_to_update, target_var_embedding)
        # self.variable_embedding is of dimension num_argtypes x batch_size x max_num_var x var_embed_dim
        #### VARIABLE_ATTENTION TABLE ALSO NEEDED TO BE UPDATED (SO THAT THE NEWLY ADDED ROW DOES NOT GET 0 ATTENTION)
        local_var_atten = torch.stack(self.variable_atten_table[beam_id], dim=0)
        # local_var_atten has shape = [self.num_argtypes, self.batch_size, self.max_num_var]
        local_var_atten = self.immutable_scatter_nd_constant_update(local_var_atten, indices_to_update, mask_value_to_update)
        self.variable_atten_table[beam_id] = list(torch.unbind(torch.nn.functional.normalize(local_var_atten,p=1, dim = 2), dim = 0))


    def get_target_var_key_and_embedding(self, var_embedding, prog_embedding, argtype_embedding, num_variables_till_now, target_type):
        #var_embedding is of dimension max_arguments x batch_size x var_embed_dim
        #prog_embedding is of dimension batch_size x prog_embed_dim
        #argtype_embedding is of dimension max_arguments x batch_size x argtype_embed_dim
        #target_var_embedding is batch_size x var_embed_dim
        #var_embedding and prog_embedding may be None
        if prog_embedding is None:
            prog_embedding = torch.zeros([self.batch_size, self.prog_embed_dim], device=self.device)

        list_argtype_embedding = list(torch.unbind(argtype_embedding, dim = 0))
        input_1 = list_argtype_embedding[0]
        input_2 = list_argtype_embedding[1]
        for current_argtype_id in range(len(list_argtype_embedding)):
            input_1 = self.dropout(self.elu(self.target_var_key_and_embedding_arg_layer[current_argtype_id](torch.cat([input_1,input_2],dim=1))))
#            temp = torch.cat([input_1,input_2],dim=1)
#            input_1 = self.target_var_key_and_embedding_arg_layer[current_argtype_id](temp)
            if current_argtype_id + 2 > len(list_argtype_embedding)-1:
                break
            input_2 = list_argtype_embedding[current_argtype_id+2]
        l2_input_1 = input_1
        list_var_embedding = list(torch.unbind(var_embedding, dim = 0))
        input_1 = list_var_embedding[0]
        input_2 = list_var_embedding[1]
        for current_var_id in range(len(list_var_embedding)):
            input_1 = self.dropout(self.elu(self.target_var_key_and_embedding_var_layer[current_var_id](torch.cat([input_1, input_2],dim=1))))
            if current_var_id + 2 > len(list_var_embedding)-1:
                break
            input_2 = list_var_embedding[current_var_id+2]
        l2_input_2 = input_1
        l2_input_3 = prog_embedding
        l2_input = torch.cat([l2_input_1,l2_input_2,l2_input_3],dim=1)
        target_var_embedding = self.dropout(self.elu(self.target_var_key_and_embedding_targetembed_layer(l2_input)))
        if self.use_key_as_onehot:
            target_type_onehot = self.one_hot(target_type, depth=self.num_argtypes)
            num_variables_till_now_onehot = self.one_hot(num_variables_till_now, depth=self.max_num_var)
            #target_type_onehot is  batch_size x num_argtypes
            #num_variables_till_now_onehot is batch_size x max_num_var
            target_var_key = torch.cat([target_type_onehot, num_variables_till_now_onehot], dim=1)
        else:
            target_var_key = self.dropout(self.elu(self.target_var_key_and_embedding_targetkey_layer(l2_input)))
        return target_var_key, target_var_embedding

    def update_attention(self, static_atten, h, i):
        #static_atten is of dimension batch_size x num_var
        #h is of dimension batch_size x cell_dim
        inputs = torch.cat([static_atten,h.view([self.batch_size,-1])], dim = 1)
        new_static_atten = nn.functional.softmax(self.elu(self.update_attention_layer[i](inputs)), dim=-1)
        return new_static_atten

    def mask_attention(self, static_atten, mask):
        #static_atten is of dimension batch_size x num_var
        #mask is of dimension batch_size x num_var
        masked_atten = torch.mul(static_atten, mask)
        num = len(masked_atten.shape)
        l1norm = torch.sum(masked_atten, dim=1)
        stacked_norm = torch.mul(torch.ones_like(masked_atten, device=self.device),torch.unsqueeze(l1norm,num-1))
        masked_atten = torch.where(stacked_norm==0, torch.ones_like(masked_atten, device=self.device), masked_atten)
        new_l1_norm = torch.sum(masked_atten, dim=1)
        masked_atten = masked_atten/new_l1_norm.view([-1,1])
        return masked_atten

    def train(self, feed_dict2):
        with torch.no_grad():
            self.Reward = torch.tensor(feed_dict2['reward'], dtype=self.dtype_float, device=self.device)
            #self.ProgramProb = feed_dict['ProgramProb']
            #self.logProgramProb = feed_dict['logProgramProb']
            #self.per_step_prob = feed_dict['per_step_prob']
            #self.entropy = feed_dict['entropy']
            self.IfPosIntermediateReward = torch.tensor(feed_dict2['IfPosIntermediateReward'], dtype=self.dtype_float, device=self.device)
            self.mask_IntermediateReward = torch.tensor(feed_dict2['mask_IntermediateReward'], dtype=self.dtype_float, device=self.device)
            self.IntermediateReward = torch.tensor(feed_dict2['IntermediateReward'], dtype=self.dtype_float, device=self.device)
            self.Relaxed_reward = torch.tensor(feed_dict2['Relaxed_reward'], dtype=self.dtype_float, device=self.device)
            overall_step_count = feed_dict2['overall_step_count']

        def reinforce():
            #mask_cnf = torch.where(self.Reward>0,torch.ones_like(self.Reward),torch.zeros_like(self.Reward))
            current_baseline = torch.sum(torch.mul(self.Reward,self.ProgramProb),dim=1,keepdim=True).detach()

            #self.Relaxed_reward = tf.placeholder(tf.float32, [self.batch_size, self.beam_size])
            current_baseline_relaxed = torch.div(torch.sum(torch.mul(self.Relaxed_reward, self.ProgramProb), dim=1, keepdim=True), torch.sum(self.ProgramProb,dim=1,keepdim=True)).detach()

            # from intermediate rewards
            #self.IfPosIntermediateReward = tf.placeholder(tf.float32, [self.batch_size, self.beam_size])
            self.rate_intermediate_reward = self.params['lr_intermideate_reward']
            #self.mask_IntermediateReward = tf.placeholder(tf.float32, [self.batch_size, self.beam_size, self.num_timesteps])
            #self.IntermediateReward = tf.placeholder(tf.float32, [self.batch_size, self.beam_size])
            int_reward = torch.mul(self.IntermediateReward,self.IfPosIntermediateReward)
            prob_intermediate = torch.mul(self.mask_IntermediateReward, self.per_step_prob)
            prob_intermediate = torch.where(self.mask_IntermediateReward==0,torch.ones_like(self.mask_IntermediateReward, device=self.device),prob_intermediate)
            prob_intermediate = torch.prod(prob_intermediate, dim = 2)
            if self.forced_normalize_ir:
                len_IntermediateReward = torch.sum(self.mask_IntermediateReward, dim=2)
                fraclen_IntermediateReward = (1.0/float(self.num_timesteps))*len_IntermediateReward
                prob_intermediate = torch.mul(fraclen_IntermediateReward, prob_intermediate)
            log_prob_intermediate = torch.log(prob_intermediate)
            unbackpropable_intermediate_prob = prob_intermediate.detach()#requires_grad_(False)
            baseline_ir = torch.sum(torch.mul(unbackpropable_intermediate_prob,int_reward), dim = 1, keepdim=True)

            #combining stuff
            new_baseline = current_baseline + baseline_ir
            new_baseline = torch.div(new_baseline,torch.sum(self.ProgramProb,dim=1,keepdim=True)+torch.sum(unbackpropable_intermediate_prob,dim=1,keepdim=True))
            #self.OldBaseline = tf.placeholder(tf.float32,[self.batch_size,1])
            final_baseline = new_baseline.detach()
            #final_baseline = (final_baseline + 0.5*self.OldBaseline)/1.5

            #coming back to reinforce_main
            scaling_term_1 = torch.mul(self.ProgramProb,self.Reward-final_baseline).detach()
            loss_reinforce = torch.mul(self.logProgramProb, scaling_term_1)
#            if overall_step_count<=15:
            #loss_reinforce = torch.mul(loss_reinforce,mask_cnf)
            loss_reinforce = torch.where(torch.isnan(loss_reinforce), torch.zeros_like(loss_reinforce, device=self.device), loss_reinforce)
            loss_reinforce = torch.sum(torch.mean(loss_reinforce, dim = 0))

            #coming back to intermediate reward part
            scaling_term_2 = torch.mul(self.IfPosIntermediateReward,torch.mul((int_reward - final_baseline) ,unbackpropable_intermediate_prob)).detach()
            loss_ir = torch.mul(scaling_term_2, log_prob_intermediate)
            loss_ir = torch.where(torch.isnan(loss_ir), torch.zeros_like(loss_ir, device=self.device), loss_ir)
            loss_ir = torch.sum(torch.mean(loss_ir,dim=0))

            relaxed_scaling_term_1 = torch.mul(self.ProgramProb, self.Relaxed_reward-current_baseline_relaxed).detach()
            loss_relaxed_reinforce = torch.mul(self.logProgramProb, relaxed_scaling_term_1)
            loss_relaxed_reinforce = torch.where(torch.isnan(loss_relaxed_reinforce), torch.zeros_like(loss_relaxed_reinforce, device=self.device),\
                                              loss_relaxed_reinforce)
            loss_relaxed_reinforce = torch.sum(torch.mean(loss_relaxed_reinforce,dim=0))

            self.entropy = torch.where(torch.isnan(self.entropy), torch.zeros_like(self.entropy, device=self.device), self.entropy)
            self.entropy = self.entropy/self.batch_size

            loss = loss_reinforce + self.params['Rate_Entropy']*self.entropy + self.rate_intermediate_reward*loss_ir +\
                        torch.mul(self.relaxed_reward_multipler, loss_relaxed_reinforce)
            self.loss = loss
            return loss
        #val_grad_fn = tfe.value_and_gradients_function(reinforce)#tfe.implicit_gradients(self.reinforce)
        #value, grads_and_vars = val_grad_fn()
        #print grads_and_vars
        #self.optimizer.apply_gradients(grads_and_vars)#feed_dict2))
        #return value
        return reinforce()

    def recursive_joint_prob_generator(self,list_dists):
        if len(list_dists) == 2:
            dist_1 = list_dists[0].view([-1,1])
            dist_2 = list_dists[1].view([-1,1])
            out = torch.matmul(dist_1,torch.transpose(dist_2,1,0))
            return out
        else:
            current_dist = list_dists[-1]
            #has shape batch_size x max_num_var
            new_list_dists = list_dists[0:-1]
            probs_list = list(torch.unbind(current_dist, dim = 0))
            penultimate_output = self.recursive_joint_prob_generator(new_list_dists)
            #has shape batch_size x max_num_var x max_num_var ....
            out = []
            for prob in probs_list:
                #prob is tensor of shape batch_size
                out.append(torch.mul(penultimate_output,prob))
            return torch.stack(out,dim = len(list_dists)-1)

    def map_index_to_unflattened(self,number,shape):
        out = []
        for divisor in shape[::-1]:
            remainder = torch.remainder(number,divisor).type(self.dtype_int64)#number // divisor
            number = torch.div(number.type(self.dtype_float),float(divisor)).floor_()#number % divisor
            out.append(remainder)
            #print 'remainder ', remainder
        return out[::-1]

    def map_index_to_flattened(self,number, dimensions):
        number = number.type(self.dtype_int64)
        one = torch.tensor(1,dtype=self.dtype_int64,device=self.device)
        dimensions = list(torch.unbind(torch.tensor(dimensions, dtype=self.dtype_int64, device=self.device), dim = 0))
        dimensions.append(one)
        out = []
        for i in range(0,len(dimensions)-1):
            out.append(torch.prod(torch.stack(dimensions[i+1:] , dim = 0) ,dim = 0))
        out = torch.stack(out)
        out = torch.mul(number,out)
        out = torch.sum(out, len(number.shape)-1)
        return out

    def immutable_scatter_nd_constant_update(self, inp1, inp2, inp3):
        shape = inp1.shape
#        inp1 = tf.to_float(inp1)
        inp1 = inp1.contiguous().view([-1]) #this reshaping cannot be avoided
        inp2 = self.map_index_to_flattened(inp2, shape)
        z1 = self.one_hot(inp2, list(inp1.shape)[0], dtype=inp3.dtype)
        z2 = inp3.view([-1,1])
        z3 = torch.mul(z1,z2)
        update_input = torch.sum(z3+torch.zeros_like(inp1, device=self.device, dtype=inp3.dtype),dim = 0)
        m1 = torch.sum(z1, dim = 0).type(inp1.dtype)
        m1 = 1-m1
        new_inp1 = torch.mul(inp1,m1)
        out = new_inp1 + update_input.type(new_inp1.dtype)
        return out.view(shape)

    def immutable_scatter_nd_1d_update(self, inp1, inp2, inp3):
        shape = inp1.shape
        dim = shape[-1]
        index_shape  = shape[0:-1]
#        inp1 = tf.to_float(inp1)
        inp1 = inp1.contiguous().view([dim, -1]) #this reshaping cannot be avoided
        inp2 = self.map_index_to_flattened(inp2, index_shape)
        z1 = self.one_hot(inp2, inp1.shape[1], dtype=inp3.dtype)
        z1 = torch.unsqueeze(torch.transpose(z1,1,0),dim = 2)
        z2 = inp3.view([-1,dim])
        z3 = torch.mul(z2,z1)
        update_input = torch.sum(z3,dim = 1)

        m1 = torch.sum(z1, dim = 1)
        m1 = 1-m1
        inp1 = inp1.view([-1, dim])
        new_inp1 = torch.mul(inp1,m1)
        out = new_inp1+update_input
        return out.view(shape)


    def beam_switch(self, old_prop_val, new_beam_ids):
        # the matrix should be input in the shape beam_size x batch_size x Tensor_shape
        old_shape = old_prop_val.shape
        old_prop_val = old_prop_val.contiguous().view([self.beam_size, self.batch_size, -1]) #this reshaping cannot be avoided
        new_prop_val = []

        expanded_beam_ids = self.one_hot(new_beam_ids, depth = self.beam_size, dtype=old_prop_val.dtype)
        #expanded_beam_ids has shape beam_size x batch_size x beam_size
        expanded_beam_ids = torch.transpose(expanded_beam_ids,2,1)
        for multiplier in list(torch.unbind(expanded_beam_ids,dim=0)):
            multiplier = torch.unsqueeze(multiplier,dim=-1)
            new_prop_val.append(torch.sum(torch.mul(multiplier,old_prop_val), dim = 0))
        new_prop_val = torch.stack(new_prop_val,dim = 0)
        new_prop_val = new_prop_val.view(old_shape)
        return new_prop_val

    def bernoulli_program_sampling(self,distribution, k):
        out1_vals, out1_ind = torch.topk(distribution, k)
        if self.params["explore"][0] is -1:
            return out1_vals, out1_ind
        p = torch.randn([])>self.randomness_threshold_beam_search
        p = p.type(self.dtype_float)
#        temp = torch.stack([torch.randperm(self.num_progs) for _ in xrange(self.batch_size)],dim=0)
        out2_ind = torch.randint_like(out1_ind,0,self.num_progs, device=self.device)#temp[:,0:k]
        multiplicand_1 = self.one_hot(out2_ind, depth=self.num_progs)
        multiplicand_1 = multiplicand_1.permute([1,0,2])
        out2_vals = torch.sum(torch.mul(multiplicand_1.type(distribution.dtype), distribution), dim=2)
        out2_vals = out2_vals.permute([1,0])
        out_ind = p*out1_ind.type(p.dtype)+(1-p)*out2_ind.type(p.dtype)
        out_vals = p*out1_vals+(1-p)*out2_vals
        return out_vals, out_ind.type(self.dtype_int64)

    def one_hot(self,batch,depth,dtype=None):
        n_dims = depth
        y_tensor = batch
        y_tensor = y_tensor.contiguous().view(-1, 1)
        y_one_hot = torch.zeros(y_tensor.size()[0], n_dims, dtype = self.dtype_int64, device=self.device).scatter_(1, y_tensor.type(self.dtype_int64), 1)
        y_one_hot = y_one_hot.view(*(list(batch.shape)+[-1]))
        if dtype is None:
            return y_one_hot
        return y_one_hot.type(dtype)


