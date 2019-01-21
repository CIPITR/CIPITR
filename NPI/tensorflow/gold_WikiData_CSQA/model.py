import tensorflow as tf
from collections import defaultdict
import numpy as np
import math



class NPI():
    def __init__(self, params, none_argtype_index, num_argtypes, num_programs, max_arguments, rel_index, type_index, \
                 rel_embedding, type_embedding, vocab_embed, program_to_argtype_table, program_to_targettype_table):
        np.random.seed(1)
        tf.set_random_seed(1)
        # following are hyperparameters used by the model
        self.params = params # a shared dictionoary of parameters
        self.num_timesteps = params['num_timesteps'] # maximum number of steps in a program
        self.max_num_phase_1_steps = self.num_timesteps / 2 # maximum length of the variable declaration phase
        self.state_dim = params['state_dim'] # CIPTR state dimension
        self.batch_size = params['batch_size']
        self.prog_embed_dim = params['prog_embed_dim'] # operator embedding dimension
        self.argtype_embed_dim = params['argtype_embed_dim']# argument type embedding dimension
        self.var_embed_dim = params['var_embed_dim'] # variable embedding dimension
        self.npi_core_dim = params['npi_core_dim'] # NPI Core RNN hidden state dimension
        self.env_dim = params['env_dim'] # environment encoding dimension # dimension of env rnn hidden state
        self.hidden_dim = params['hidden_dim'] # one of the hidden dimensions of FCC layers
        self.empty_argtype_id = none_argtype_index # to accomodate operators that require less than maximum no. of arguments
        self.num_argtypes = num_argtypes # total no. of argument types. eg. 'int', 'bool', 'entity', 'type' ,...
        self.num_progs = num_programs # total no. of available operators
        self.max_arguments = max_arguments # maximum no. of arguments required by any operator
        self.max_num_var = params['max_num_var'] # maximum variable table size for each variable type
        self.prog_key_dim = params['prog_key_dim'] # dimension of operator key
        self.var_key_dim = params['var_key_dim'] # dimension of variable key
        if params['use_key_as_onehot']:
            self.use_key_as_onehot = True
            self.var_key_dim = self.num_argtypes + self.max_num_var
            self.prog_key_dim = self.num_progs
        else:
            self.use_key_as_onehot = False
        self.max_len = params['max_len'] # maximum query length(length of question)
        self.wikidata_embed_dim = params['wikidata_embed_dim'] # dimension of embedding coming from KB
        self.text_embed_dim = params['text_embed_dim']  # dimension of embedding coming from Glove
        self.cell_dim = params['cell_dim']
        self.eps = 1e-20 # epsilon for random perturbations in sampling
        self.learning_rate = params['learning_rate']
        self.beam_size = params['beam_size'] # no. of beams
        self.num_programs_to_sample = params['num_programs_to_sample'] # initial operator set size for beam search
        self.num_variables_to_sample = params['num_variables_to_sample'] # initial variable set size for beam search
        self.num_actions = self.num_variables_to_sample * self.num_programs_to_sample # No. A = OxV
        self.phase_change_threshold = 0.2 # threshold to switch output of phase change network
        self.bias_to_gold_type_threshold = 0.3
        self.rel_embedding = None
        # operator table contains two parameters "program_Embedding" and "program_Keys".
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
        #self.variable_mask is of dimension num_argtypes x batch_size x max_num_var
        self.variable_mask = None
        #self.variable_atten_table contains the attention over all variables declared till now. is of dimension num_argtypes x max_num_var
        self.variable_atten_table = None
        self.kb_attention = None # this is used to obtain some signal from the KB to determine if a certain action is feasible w.r.t KB
        assert self.beam_size <= self.num_programs_to_sample * self.num_variables_to_sample
        self.keep_prob = params['dropout_keep_prob'] # for dropout layers
        self.global_program_indices_matrix = None
        self.dont_look_back_attention = params['dont_look_back_attention'] #boolean flag indicating whether the attention over query words is dynamically updated, providing maximum attention on distinct words at each time-step
        self.concat_query_npistate = params['concat_query_npistate'] #boolean flag indicating whether the query embedding should be concatenated with the program state, when sampling actions, conditional to it
        self.query_attention = params['query_attention'] # @boolean flag indicating whether the attention over query words is used when sampling actions


        def create_cell_scopes():
            """
            This method creates the scope of the different building blocks of the NPI model and initializes the tensorflow variables corresponding to them
            """
            self.npi_cell = tf.nn.rnn_cell.GRUCell(self.npi_core_dim) # Core NPI GRU
            self.npi_scope = "npi_scope"
            self.env_cell = tf.nn.rnn_cell.GRUCell(self.env_dim) # Environment GRU
            self.env_scope = "env_scope"
            self.batch_ids = tf.constant(np.asarray(xrange(self.batch_size)), dtype=tf.int32)
            # tensor is of dimension batch_size x 1 i.e. [0, 1, 2, ... batch_size]
            self.rel_embedding = tf.get_variable('rel_embedding_matrix', initializer=rel_embedding, dtype=tf.float32)
            # embedding matrix of relations
            self.type_embedding = tf.get_variable('type_embedding_matrix', initializer=type_embedding, dtype=tf.float32)
            # embedding matrix of KB types
            self.word_embeddings = tf.get_variable('embedding_matrix', initializer=vocab_embed, dtype=tf.float32)  # for glove embedding
            self.enc_scope_text = "encoder_text"
            max_val = 6. / np.sqrt(self.num_progs + self.prog_embed_dim) # for initialization of wt.'s
            self.program_embedding = tf.get_variable('program_embedding', \
                                                     shape = [self.num_progs, self.prog_embed_dim], \
                                                     initializer = tf.random_normal_initializer(-max_val, max_val))
            # the operator embedding tensor
            max_val = 6. / np.sqrt(1 + self.hidden_dim)
            self.query_attention_h_mat = tf.get_variable('query_attention_h_mat', shape = [1, self.hidden_dim], \
                                                     initializer = tf.random_normal_initializer(-max_val, max_val)) # it is an additional parameter used during combining the query attention vector and the program state vector

            max_val = 6. / np.sqrt(self.wikidata_embed_dim + self.var_embed_dim)
            self.preprocessed_var_emb_mat = tf.get_variable('preprocessed_var_emb_mat', shape = [self.wikidata_embed_dim, self.var_embed_dim], \
                                                     initializer = tf.random_normal_initializer(-max_val, max_val))
            # tensor for storing embeddings of prepopulated variables(linked KB-artifacts) that are captured directly from query
            max_val = 6. / np.sqrt(self.num_progs + self.prog_key_dim)
            self.program_keys = tf.get_variable('program_keys', \
                                                shape = [self.num_progs, self.prog_key_dim], \
                                                initializer = tf.random_normal_initializer(-max_val, max_val))
            # this tensor contains operator keys for lookup
            self.program_to_argtype_table = tf.constant(program_to_argtype_table, dtype=tf.int32, \
                                                        shape = [self.num_progs, self.max_arguments],
                                                        name = 'program_to_argtype_table')
            # mapping of operators to their respective input argument types
            self.program_to_targettype_table = tf.constant(program_to_targettype_table, dtype=tf.int32, \
                                                           shape = [self.num_progs], \
                                                           name = 'program_to_argtype_table')
            # mapping of operators to their respective output variable types
            max_val = 6. /np.sqrt(self.num_argtypes + self.argtype_embed_dim)
            self.argtype_embedding = tf.get_variable('argtype_embedding', \
                                                     shape = [self.num_argtypes, self.argtype_embed_dim], \
                                                     initializer = tf.random_normal_initializer(-max_val, max_val))
            # argument type embedding
            self.program_to_num_arguments = tf.reduce_max(tf.one_hot(self.program_to_argtype_table, depth=self.num_argtypes), axis=1)
            # program_to_num_arguments is of dimension num_progs x num_argtypes.

            # accomodating for the beam_size.
            # Exploring different beams will generate a dynamic set of variable tables, one for each beam
            self.variable_embedding = []
            self.variable_keys = []
            self.variable_atten_table = []
            self.variable_mask = []
            # embedding, key, attention vector and mask are the salient properties required for the variable tables
            max_val = 6. / np.sqrt(self.max_num_var)
            # for every beam we initialzie the variable tables
            for beam_id in xrange(self.beam_size):
                self.variable_embedding.append(tf.get_variable('variable_embedding'+'_beam_'+str(beam_id), \
                                               shape = [self.num_argtypes, self.batch_size, self.max_num_var, self.var_embed_dim], \
                                               initializer = tf.zeros_initializer, trainable = False))
                # to store the variable embeddings
                self.variable_keys.append(tf.get_variable('variable_keys'+'_beam_'+str(beam_id), \
                                          shape = [self.num_argtypes, self.batch_size, self.max_num_var, self.var_key_dim], \
                                          initializer = tf.zeros_initializer, trainable = False))
                # to store the variable keys
                temp = tf.get_variable('variable_atten_table'+'_beam_'+str(beam_id), \
                                       shape = [self.num_argtypes, self.batch_size, self.max_num_var], \
                                       dtype = tf.float32, initializer = tf.zeros_initializer, trainable = False)
                self.variable_atten_table.append(tf.unstack(temp, axis=0))
                # to store the attention vector over the variables
                self.variable_mask.append(tf.get_variable('variable_mask'+'_beam_'+str(beam_id), \
                                                          shape = [self.num_argtypes, self.batch_size, self.max_num_var], \
                                                          initializer = tf.zeros_initializer, dtype = tf.float32, \
                                                          trainable = False))
                # a boolean mask corresponding to whether a location is filled or not in the variable tables

            # declaring scopes for all the future layers here
            self.reset_scope = 'reset_scope'
            self.state_encoding_scope_1 = 'state_encoding_scope_layer1'
            self.state_encoding_scope_2 = 'state_encoding_scope_layer2'
            self.state_encoding_scope_3 = 'state_encoding_scope_layer3'
            self.bias_to_gold_scope_1 = 'bias_to_gold_scope_fcc1'
            self.bias_to_gold_scope_2 = 'bias_to_gold_scope_fcc2'
            self.phase_change_scope =  'phase_change_scope'
            self.prog_hidden_scope = 'prog_net_fcc1'
            self.prog_key_scope = 'prog_net_fcc2'
            self.inp_var_key_scope = 'inp_var_key_scope'
            self.get_target_var_key_and_embedding_arg_scope = 'get_target_var_key_and_embedding_arg_scope'
            self.get_target_var_key_and_embedding_var_scope = 'get_target_var_key_and_embedding_var_scope'
            self.get_target_var_key_and_embedding_targetembed_scope = 'get_target_var_key_and_embedding_targetembed_scope'
            self.get_target_var_key_and_embedding_targetkey_scope = 'get_target_var_key_and_embedding_targetkey_scope'
            self.update_attention_scope = 'update_attention_scope'

        create_cell_scopes()

    def create_placeholder(self):
        """
        This method creates the tensorflow placeholder variables for taking input to the NPI model
        """
        self.encoder_text_inputs_w2v = [tf.placeholder(tf.int32, [None], name='encoder_text_inputs_w2v') \
                                        for i in range(self.max_len)]
        self.encoder_text_inputs_kb_emb = tf.placeholder(tf.float32, [None, self.max_len, self.wikidata_embed_dim], \
                                                         name='encoder_text_inputs_kb')
        self.preprocessed_var_mask_table = [[tf.placeholder(tf.float32, [self.batch_size]) \
                                            for i in xrange(self.max_num_var)] for j in xrange(self.num_argtypes)]
        # tensor for storing mask for prepopulated variables
        self.preprocessed_var_emb_table = [[tf.placeholder(tf.float32, [None, self.wikidata_embed_dim]) \
                                            for i in xrange(self.max_num_var)] for j in xrange(self.num_argtypes)]
        # tensor for storing variable embeddings of prepopulated variables
        self.kb_attention = tf.placeholder(tf.float32, [None, self.num_progs, pow(self.max_num_var, self.max_arguments)])
        # this is used to obtain signal from the KB to determine if a certain action is feasible w.r.t KB
        self.progs_phase_1 = tf.placeholder(shape = [self.batch_size, self.num_progs], dtype = tf.int32)
        # operators allowed in variable declaration phase
        self.progs_phase_2 = tf.placeholder(shape = [self.batch_size, self.num_progs], dtype = tf.int32)
        # operators allowed in algorithm phase
        self.gold_target_type = tf.placeholder(shape = [self.batch_size], dtype = tf.int32)
        # the gold target type is an input recieved from another network that predicts the gold target type
        self.randomness_threshold_beam_search = tf.placeholder(dtype = tf.float32)
        # used for exploration in beam search
        self.DoPruning = tf.placeholder(dtype = tf.float32)
        # flag to enable/disable pruning of beams
        self.last_step_feasible_program = tf.placeholder(shape = [1,1], dtype = tf.float32)
        # flag to enable/disable instantiation of optimum programs in last time step
        self.bias_prog_sampling_with_target = tf.placeholder(shape = [1,1], dtype = tf.float32)
        self.bias_prog_sampling_with_last_variable = tf.placeholder(shape = [1,1], dtype = tf.float32)
        # maintains a list of variable types required to be generated in a program in order to create an answer variable of the desired (predicted) variable type
        self.required_argtypes = tf.placeholder(shape = [self.batch_size, self.num_argtypes], dtype = tf.float32)
        # scale to control feedback from auxiliary rewards
        self.relaxed_reward_multipler = tf.placeholder(shape = [1,1], dtype = tf.float32)
        
    def get_final_feasible_progs_for_last_timestep(self, feasible_progs, beam_properties, beam_id, feasible_progs_for_last_timestep, t):
        """ This function returns those programs which can output variables that have the same type as the predicted gold target type
        It is intended to be applied only at the last time-step
        If this feasibility check results in any operator (other than none or terminate) and the variable type of the variable last 
          created is NOT the same as the desired type then in that case, none and terminate is removed from the feasibility list
        If this feasibility check does not result in any operator (other than none or terminate), this feasibility check is not applied 
        """
        if t == self.num_timesteps-1:
            #feasible_progs[beam_id] = tf.add(feasible_progs[beam_id], tf.zeros_like(feasible_progs[beam_id]))
            current_equal_to_gold_target_type = tf.tile(tf.expand_dims(tf.cast(tf.where(tf.equal(self.gold_target_type, beam_properties['target_type'][beam_id]), tf.ones_like(self.gold_target_type), tf.zeros_like(self.gold_target_type)), dtype=tf.float32), axis=1), [1, self.num_progs])
            #current_equal_to_gold_target_type is of size batch_size x num_progs
            t1 = tf.one_hot(tf.zeros(shape=[self.batch_size], dtype=tf.int32), depth=self.num_progs, dtype=feasible_progs_for_last_timestep.dtype)
            t2 = tf.one_hot((self.num_progs-1)*tf.ones(shape=[self.batch_size], dtype=tf.int32), depth=self.num_progs, dtype=feasible_progs_for_last_timestep.dtype)
            temp = t1 + t2
            #temp is of size batch_size x num_progs
            feasible_progs_for_last_timestep = current_equal_to_gold_target_type*temp + (1-temp)*feasible_progs_for_last_timestep
            temp2 = (1-self.last_step_feasible_program)*feasible_progs[beam_id] + self.last_step_feasible_program*tf.multiply(feasible_progs[beam_id], feasible_progs_for_last_timestep)
            temp3 = tf.tile(tf.expand_dims(tf.reduce_sum((1-temp)*temp2, axis=1),axis=1),[1,self.num_progs])
            #temp3 is of dimension batch_size x num_progs
            feasible_progs[beam_id] = tf.where(tf.equal(temp3,0), feasible_progs[beam_id], temp2)
        return feasible_progs[beam_id]

    def inference(self):
        """Main method for the forward pass, which generates the program by sampling actions at each time step, based on the policy parameters
        """
        sentence_state, attention_states = self.sentence_encoder()
        # sentence_state will store a represenation of the query sentence
        # attention_states will store a context representation of each word in the query, to be later used to attend upon those words
        # we have defined or own  highly customizable beam search
        # every beam is defined by a set of properties which get updated as search progresses
        beam_properties = defaultdict(list)
        beam_properties['Overflow_Penalize_Flag'] = [tf.zeros([self.batch_size]) for beam_id in xrange(self.beam_size)]
        # Overflow_Penalize_Flag can be used to enable penalization if the variable tables have overflowed
        # it is set to one when oveflow happens. In final runs, we do not use this feature as it hinders exploration.
        for beam_id in xrange(self.beam_size):
            # add_preprocessed_output_to_variable_table initializes the variable tables based upon identified KB-artifacts and other indentified variables in the query sentence
            beam_properties['Overflow_Penalize_Flag'][beam_id] = self.add_preprocessed_output_to_variable_table(beam_id)

        # reset all RNN before starting
        init_h_states, init_e_state, init_target_var_embedding = self.reset_state(sentence_state)
        # following are the properties required to be logged for every beam and switched as the top beams switch themselves
            # rnn related states
        beam_properties['h_states'] = [init_h_states for beam_id in xrange(self.beam_size)]
        beam_properties['h'] = [None for beam_id in xrange(self.beam_size)]
        beam_properties['e_state'] = [init_e_state for beam_id in xrange(self.beam_size)]
            # embedding of the output target variable
        beam_properties['target_var_embedding'] = [init_target_var_embedding for beam_id in xrange(self.beam_size)]
            # index of the operator selected by this beam
        beam_properties['prog_sampled_indices'] = [None for beam_id in xrange(self.beam_size)]
            # indices of the corresponding variables selected for this beam
        beam_properties['input_var_sampled_indices'] = [None for beam_id in xrange(self.beam_size)]
            # flag indicating whether the terminate operator has been sampled
        beam_properties['terminate'] = [tf.zeros([self.batch_size,1]) for beam_id in xrange(self.beam_size)]
            # stores the current length of the beam. to be used for penalizing beams based upon length
        beam_properties['length'] = [tf.zeros([self.batch_size,1]) for beam_id in xrange(self.beam_size)]
        beam_properties['target_type'] = [tf.zeros([self.batch_size], dtype = tf.int32) for beam_id in xrange(self.beam_size)]
            # phase_elasticity is the current tendency to change from variable declaration phase to algorithm face
        beam_properties['phase_elasticity'] = [tf.ones([self.batch_size,1]) for beam_id in xrange(self.beam_size)]
            # for every operator, program_argument_table_index stores a record of what variable combinations have been
            # sampled already. Used to prevent sampling repetitive programs(OxV) or actions.
        beam_properties['program_argument_table_index'] = [tf.ones([self.batch_size, self.num_progs,
                                                                    int(math.pow(self.max_num_var,self.max_arguments))], dtype=tf.float32) \
                                                                    for beam_id in xrange(self.beam_size)]
            # for every query in the batch, it maintains the cumulative attention distribution over the query words, till this time step
            # this is used for the never-look-back attention, which makes the attention over query words at current time step inversely
            # proportional to the cumulative attention till this time step
        beam_properties['query_attentions_till_now'] = [tf.zeros([self.batch_size,self.max_len]) for beam_id in xrange(self.beam_size)]
            # none_count stores a count of no. of none operators sampled till now in the beam.
        beam_properties['none_count'] = [tf.zeros([self.batch_size,1]) for beam_id in xrange(self.beam_size)]

        # Other than the switchable properties. we have to track the top scores in unswitched_beam_properties
        unswitched_beam_properties = defaultdict(list)
        # initialize total_beam_score carefully to prevent starting from redundant K beams.
        unswitched_beam_properties['total_beam_score'] = [tf.zeros([self.batch_size])] + \
        [-30*tf.ones([self.batch_size]) for beam_id in xrange(self.beam_size-1)]

        # returnable objects. {to_return_per_step_prob,to_return_sequence_logprob, to_return_action_sequence}
            # stores the per step probability
        to_return_per_step_prob = -1*tf.ones([self.batch_size,self.beam_size,self.num_timesteps])\
            # stores the log probability for entire beams
        to_return_sequence_logprob = tf.zeros([self.batch_size, self.beam_size])
            # stores properties of entire action sequence for each beam
        to_return_action_sequence = dict.fromkeys(['program_type','argument_type','target_type',\
                                                   'target_table_index','argument_table_index'])
        for key in ['program_type','argument_type','target_type','target_table_index','argument_table_index']:
            to_return_action_sequence[key] = [[] for beam_id in xrange(self.beam_size)]
        # self.entropy is used to track entropy in action selection which is maximized
        self.entropy = tf.constant(0,dtype = tf.float32)
        # get_feasible_progs_for_last_timestep is a feature for enabling of sampling those operators that produce 
        #  the correct predicted type at the last time step
        feasible_progs_for_last_timestep = self.get_feasible_progs_for_last_timestep()

        for t in xrange(self.num_timesteps):
            entropy = tf.constant(0,dtype = tf.float32)
            # retrieve current top beam scores
            current_beam_score = [tf.add(score,0) for score in unswitched_beam_properties['total_beam_score']]
            if t > 0:
                # monitor phase_elasticity and change phase accordingly based upon output of phase_change_net
                beam_properties['phase_elasticity'] = [self.phase_change_net(h,t, old_p_el) \
                               for h,old_p_el in zip(beam_properties['h'], beam_properties['phase_elasticity'])]
            # retrieve the current list of feasible programs
            feasible_progs = self.get_feasible_progs(t, beam_properties['phase_elasticity'])
            # initialize beam penalization vector
            to_penalize_beams = [tf.zeros([self.batch_size,self.num_actions]) for beam_id in xrange(self.beam_size)]
            for beam_id in xrange(self.beam_size):
                # update GRU states ================================================================================
                beam_properties['e_state'][beam_id] = self.env_encoding(beam_properties['e_state'][beam_id], \
                                                           beam_properties['target_var_embedding'][beam_id])[1]

                [beam_properties['h'][beam_id], \
                 beam_properties['h_states'][beam_id]] = self.npi_core(beam_properties['h_states'][beam_id], \
                                                           beam_properties['e_state'][beam_id], \
                                                           beam_properties['target_var_embedding'][beam_id])
                # --------------------------------------------------------------------------------------------------
                # get update feasible programs list for every beam
                feasible_progs[beam_id] = self.get_final_feasible_progs_for_last_timestep(feasible_progs, \
                                              beam_properties, beam_id, feasible_progs_for_last_timestep, t)
                # Operator sampling  ================================================================================
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

                # for every sampled program will contain the probability of action obtained by sampling every possible var
                complete_action_probs = []

                # for every sampled program will contain the possible variable samples
                per_program_input_var_sampled_indices = []

                # variable sampling corresponding to sampled operators=================================================
                for _prog_sample_, _prog_embedding_, \
                    _kb_attention_for_sampled_progs_ , \
                    _program_prob_ in zip(tf.unstack(prog_sampled_indices, axis = 1),\
                                          tf.unstack(prog_sampled_embeddings, axis = 1),\
                                          tf.unstack(kb_attention_for_sampled_progs, axis = 0),\
                                          tf.unstack(prog_sampled_probs,axis = 1)):

                    arg_types = self.argument_type_net(_prog_sample_)[0]
                    past_program_variables = tf.gather_nd(beam_properties['program_argument_table_index'][beam_id], \
                                                          tf.concat([tf.expand_dims(self.batch_ids, axis=1), \
                                                          tf.expand_dims(_prog_sample_, axis=1)], axis=1))

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
                    # tracking the sampled program or action probabilites for sampled operator and variables in every beam
                    complete_action_probs.append(tf.multiply(input_var_sampled_probs, tf.reshape(_program_prob_,[-1,1])))
                # --------------------------------------------------------------------------------------------------
                # beam_properties['input_var_sampled_indices'] is beam_sized list containing tensors of
                # shape batch_size x num_programs_to_sample x num_variables_to_sample
                beam_properties['input_var_sampled_indices'][beam_id] = tf.stack(per_program_input_var_sampled_indices, axis=1)
                # complete_action_probs is a tensor of shape batch_size x num_progs_to_sample x num_vars_to_sample
                complete_action_probs = tf.stack(complete_action_probs, axis=1)
                # complete_action_probs is a tensor of shape batch_size x num_actions.
                complete_action_probs =  tf.reshape(complete_action_probs,[self.batch_size,-1])

                # each program and joint_variables selected becomes an action
                complete_action_probs = tf.clip_by_value(complete_action_probs,self.eps,0.9)
                # need log of probability to update scores and get next top -K
                log_complete_action_probs = tf.log(complete_action_probs)
                # updating entropy
                entropy = tf.add(entropy,-1*tf.reduce_sum(complete_action_probs*log_complete_action_probs))
                # penalizing beams =================================================================================
                    # length based normalization
                if self.params['normalize_length'] is 1:
                    if t is 0:
                        current_beam_score[beam_id] = tf.add(log_complete_action_probs, \
                                          tf.expand_dims(current_beam_score[beam_id],axis=1))
                    else:
                        score_if_terminated = tf.add(log_complete_action_probs, \
                                          tf.expand_dims(current_beam_score[beam_id],axis=1))
                        power = 0.4 # hyperparameter
                        n1 = tf.pow(beam_properties['length'][beam_id], power)/tf.pow(beam_properties['length'][beam_id]+1.0, power)
                        n2 = 1.0/tf.pow(beam_properties['length'][beam_id]+1.0, power)
                        score_if_not_terminated = tf.add(n2*log_complete_action_probs, \
                                          n1*tf.expand_dims(current_beam_score[beam_id],axis=1))
                        old_cbs = tf.expand_dims(current_beam_score[beam_id],axis=1)*tf.ones_like(log_complete_action_probs)
                        current_beam_score[beam_id] = beam_properties['terminate'][beam_id]*score_if_terminated + \
                                                        (1-beam_properties['terminate'][beam_id])*score_if_not_terminated
                        current_beam_score[beam_id] = tf.where(tf.greater(current_beam_score[beam_id],\
                                          old_cbs),old_cbs,current_beam_score[beam_id])
                else:
                    current_beam_score[beam_id] = tf.add(log_complete_action_probs, \
                                      tf.expand_dims(current_beam_score[beam_id],axis=1))

                    # penalizing probability of sampling 'none' operator
                if self.params['none_decay'] is 1:
                    beam_properties['none_count'][beam_id] = tf.to_float(beam_properties['none_count'][beam_id])
                    power_decay = 0.2 # hyperparameter
                    penalize_factor = tf.multiply(beam_properties['none_count'][beam_id],\
                                                  -1*tf.log(tf.constant(math.pow(t+1,power_decay),dtype=tf.float32)*tf.ones_like(beam_properties['none_count'][beam_id])))
                    current_beam_score[beam_id] = tf.add(current_beam_score[beam_id], penalize_factor)

                    # syntactic penalization. penalizing beams with first operator as none.
                beam_target_type = tf.reshape(beam_properties['target_type'][beam_id],[self.batch_size, 1])
                beam_gold_type = tf.reshape(self.gold_target_type, [self.batch_size, 1])
                beam_if_terminated = beam_properties['terminate'][beam_id]

                if self.params['prune_beam_type_mismatch'] is 1:
                    toadd = self.DoPruning*self.check_if_gold_target(beam_target_type, beam_gold_type, beam_if_terminated)
                    to_penalize_beams[beam_id] = tf.add( toadd, to_penalize_beams[beam_id])

                    # syntactic penalization. penalizing beams with first operator as none.
                if t > 0:
                    penalize_none_start = tf.where(tf.equal(beam_target_type,0),\
                                                   tf.ones_like(beam_target_type),tf.zeros_like(beam_target_type))
#                    penalize_none_start = math.log(self.eps)*tf.to_float(penalize_none_start)
                    penalize_none_start = tf.to_float(penalize_none_start)
                    to_penalize_beams[beam_id] = tf.add(penalize_none_start, to_penalize_beams[beam_id])
                to_penalize_beams[beam_id] = tf.clip_by_value(to_penalize_beams[beam_id],0,1)
                current_beam_score[beam_id] = tf.clip_by_value(current_beam_score[beam_id],2*math.log(self.eps),0)
                # ----------------------------------------------------------------------------------------------------------
            # updating entropy
            self.entropy = tf.add(self.entropy,entropy)
            # calculating next top scores ==========================================================================
            current_score = tf.stack(current_beam_score,axis = 1)
            # current_score is a tensor of shape batch_size x beam_size x num_actions
            to_penalize_score = tf.stack(to_penalize_beams,axis = 1)
            flag_penalize = tf.reduce_prod(to_penalize_score,axis=1,keep_dims=True)
            to_penalize_score = to_penalize_score * (1-flag_penalize)
            to_penalize_score = tf.log(self.eps)*to_penalize_score
            current_score = tf.add(current_score, to_penalize_score)
            current_score = tf.clip_by_value(current_score,2*math.log(self.eps),0)
            current_score = tf.reshape(current_score, [self.batch_size,-1])
            top_scores, indices_top_scores = tf.nn.top_k(current_score, k = self.beam_size)
            # top_scores has shape batch_size x beam_size
            # indices_top_scores has shape batch_size x beam_size
            # ======================================================================================================
            # update return object to_return_sequence_logprob
            to_return_sequence_logprob = tf.add(top_scores,0)
            #to_return_sequence_logprob  has shape batch_size x beam_size

            old_score = tf.stack(unswitched_beam_properties['total_beam_score'])
            # need to transform this old_score w.r.t changes in beam_id
            # this is needed to calculate per step probabilities
            # old_score has shape beam_size x batch_size

            #updating the score list
            unswitched_beam_properties['total_beam_score'] = tf.unstack(top_scores,axis = 1)

            # getting the index of selected beam and actions from the index of the top scores
            # using function map_index_to_unflattened
            new_beam_ids, action_ids = self.map_index_to_unflattened(indices_top_scores, [self.beam_size, self.num_actions])
            #new_beam_ids has shape  batch_size x beam_size
            # action_ids has shape batch_size x beam_size
            # some reshaping for faster future operations
            action_ids = tf.transpose(action_ids, [1,0])
            # action_ids has shape beam_size x batch_size
            # for updating the memory w.r.t beams
            new_beam_ids = tf.transpose(new_beam_ids,[1,0])
            #new_beam_ids has shape beam_size x batch_size
            # for updating old_score w.r.t change in beam_ids

            # updating the to_return_per_step_prob w.r.t beam_id changes ==========================================
                # first updating old_score w.r.t change in beam_ids
            old_score = self.beam_switch(old_score, new_beam_ids)
                # calculating and appending current step probs
            if t > 0:
                old_prop_val = tf.add(to_return_per_step_prob, 0)
                old_prop_val = tf.transpose(old_prop_val, [1,0,2])
                to_return_per_step_prob = self.beam_switch(old_prop_val, new_beam_ids)
                to_return_per_step_prob = tf.transpose(to_return_per_step_prob, [1,0,2])
                # For Printing Per Step Prob
            delta_score  =  tf.subtract(to_return_sequence_logprob,tf.transpose(old_score, [1,0]))
            current_probs = tf.exp(delta_score)
            multiplier = tf.one_hot(tf.to_int32(t*tf.ones([self.batch_size, self.beam_size])), depth = self.num_timesteps)
            multiplier = tf.cast(multiplier, dtype = current_probs.dtype)
            additand = tf.multiply(multiplier, tf.tile(tf.reshape(current_probs, [self.batch_size, self.beam_size,1]), [1,1,self.num_timesteps]))
            to_return_per_step_prob = tf.cast(to_return_per_step_prob, multiplier.dtype)
            additand2 = tf.multiply(to_return_per_step_prob,1-multiplier)
            to_return_per_step_prob = tf.add(additand2, additand)
            # ------------------------------------------------------------------------------------------------------------
            # until this point the forward pass of the beam search is done. now need to switch beam properties
            # corresponding changes in new top beam ordering

            # updating the beam_properties w.r.t beam_id changes ==================================================
            for prop in beam_properties.keys():
                old_prop_val = tf.stack(beam_properties[prop],axis=0)
                # each beam_prop will be of shape beam_size x batch_size x Tensor_shape
                new_prop_val = self.beam_switch(old_prop_val, new_beam_ids)
                beam_properties[prop] = tf.unstack(new_prop_val, axis = 0)
            # -----------------------------------------------------------------------------------------------------


            # updating the variable properties corresponding to beams w.r.t beam_id changes
                # variable_properties are :-
                    #variable_embedding - beam_size x [num_argtypes, batch_size, max_num_var, var_embed_dim]
                    #variable_keys - beam_size x [num_argtypes, batch_size, max_num_var, var_key_dim]
                    #variable_mask - beam_size x [num_argtypes, batch_size, max_num_var]
                    #variable_atten_table - beam_size x num_argtypes x [batch_size, max_num_var]
            # keeping in mind beam_size

            #1)variable_embedding
            old_prop_val = tf.stack(self.variable_embedding, axis=0)
            old_prop_val = tf.transpose(old_prop_val, [0,2,1,3,4])
            # now old_prop_val has shape beam_size x batch_size x (tensor_shape = num_argtypes x max_num_var x var_embed_dim)
            new_prop_val = self.beam_switch(old_prop_val, new_beam_ids)
            self.variable_embedding = tf.unstack(tf.transpose(new_prop_val, [0,2,1,3,4]), axis = 0)
            # variable_embedding beam_size x [num_argtypes, batch_size, max_num_var, var_embed_dim]

            #2)variable_keys
            old_prop_val = tf.stack(self.variable_keys, axis=0)
            # old_prop_val [beam_size, num_argtypes, batch_size, max_num_var, var_key_dim]
            old_prop_val = tf.transpose(old_prop_val, [0,2,1,3,4])
            new_prop_val = self.beam_switch(old_prop_val, new_beam_ids)
            self.variable_keys = tf.unstack(tf.transpose(new_prop_val, [0,2,1,3,4]), axis = 0)
            # variable_keys beam_size x [num_argtypes, batch_size, max_num_var, var_key_dim]

            #3)variable_mask
            old_prop_val = tf.stack(self.variable_mask, axis=0)
            # old_prop_val [beam_size, num_argtypes, batch_size, max_num_var]
            old_prop_val = tf.transpose(old_prop_val, [0,2,1,3])
            new_prop_val = self.beam_switch(old_prop_val, new_beam_ids)
            self.variable_mask = tf.unstack(tf.transpose(new_prop_val, [0,2,1,3]), axis = 0)
            # variable_mask beam_size x [num_argtypes, batch_size, max_num_var]

            #4)variable attention table
            #variable_atten_table - beam_size x num_argtypes x [batch_size, max_num_var]
            old_prop_val = []
            for beam_id in xrange(self.beam_size):
                old_prop_val.append(tf.stack(self.variable_atten_table[beam_id], axis=1))
            old_prop_val = tf.stack(old_prop_val, axis = 0)
            # old_prop_val [beam_size, batch_size, num_argtypes, max_num_var]
            new_prop_val = self.beam_switch(old_prop_val, new_beam_ids)
            temp = tf.unstack(tf.transpose(new_prop_val, [0,2,1,3]), axis = 0)
            self.variable_atten_table = [tf.unstack(_temp_, axis = 0) for (beam_id, _temp_) in \
                                         zip(xrange(self.beam_size), temp)]
            # variable_atten_table beam_size x num_argtypes x [batch_size, max_num_var]
            # done updating beam_memory
            # done updating variable_memeory
            # ---------------------------------------------------------------------------------------------------------

            # have to update to_return_action_* w.r.t to change in beam_id=============================================
            if t > 0:
                for key in ['program_type','argument_type','target_type','target_table_index','argument_table_index']:

                    # to_return_action_sequence beam_size x seq_length x [tensor_shape]
                    old_prop_val = []
                    for beam_id in xrange(self.beam_size):
                        temp = tf.stack(to_return_action_sequence[key][beam_id], axis=1)
                        # temp [seq_length x tensor_shape]
                        old_prop_val.append(temp)
                    old_prop_val = tf.stack(old_prop_val, axis = 0)
                    # beam_size x batch_size x seq_length x tensor_shape
                    new_prop_val = self.beam_switch(old_prop_val, new_beam_ids)
                    temp = tf.unstack(new_prop_val, axis = 0)
                    to_return_action_sequence[key] = [tf.unstack(_temp_, axis = 1) for (beam_id, _temp_) in \
                                             zip(xrange(self.beam_size), temp)]
                    # done updating to_return_action_* w.r.t to change in beam_id
            # ----------------------------------------------------------------------------------------------------------

            #getting the pointer to program sample and pointer to variable sample from action_id========================
            [pointer_to_prog_sample, \
             pointer_to_variable_sample] = self.map_index_to_unflattened(action_ids,[self.num_programs_to_sample,\
                                                                                       self.num_variables_to_sample])
            # pointer_to_prog_sample has shape beam_size x batch_size
            # pointer_to_variable_sample has shape beam_size x batch_size

            # getting the actual program samples
            # pointer_to_prog_sample beam_size x batch_size
            multiplicand_2 = tf.stack(beam_properties['prog_sampled_indices'], axis = 0)
            #multiplicand_2 beam_size x batch_size x num_programs_to_sample
            multiplicand_1 = tf.one_hot(pointer_to_prog_sample, depth=self.num_programs_to_sample, dtype=multiplicand_2.dtype)
            #multiplicand_1 beam_size x batch_size x num_programs_to_sample
            true_program_sample = tf.reduce_sum(tf.multiply(multiplicand_1, multiplicand_2),axis = 2)
            #true_program_sample is a tensor of shape beam_size x batch_size
            # ----------------------------------------------------------------------------------------------------------

            # checking if any beam has terminated=======================================================================
            for prog_samples, beam_id in zip(tf.unstack(true_program_sample, axis = 0), xrange(self.beam_size)):
                beam_properties['terminate'][beam_id] = self.terminate_net(prog_samples, beam_properties['terminate'][beam_id])
                #update the length
                beam_properties['length'][beam_id] = tf.add(beam_properties['length'][beam_id],\
                                           1.0-beam_properties['terminate'][beam_id])
                beam_properties['none_count'][beam_id] = self.none_finder_net(prog_samples)
            # ----------------------------------------------------------------------------------------------------------

            # getting the actual variable samples=======================================================================
            # beam_properties['input_var_sampled_indices'] is a list of length beam_size with
            # tensor of shape [batch_size x num_programs_to_sample x num_variables_to_sample]
            multiplicand_1 = tf.stack(beam_properties['input_var_sampled_indices'], axis = 0)
            multiplicand_2 = tf.expand_dims(tf.one_hot(pointer_to_prog_sample, depth = self.num_programs_to_sample, \
                                                       dtype = multiplicand_1.dtype), axis = 3)
            flattened_input_var_sample = tf.reduce_sum(tf.multiply(multiplicand_1, multiplicand_2),axis = 2)
            # flattened_input_var_sample  has shape [beam_size x batch_size x num_variables_to_sample]
            multiplicand_1 = flattened_input_var_sample
            multiplicand_2 = tf.one_hot(pointer_to_variable_sample, depth = self.num_variables_to_sample, \
                                        dtype = flattened_input_var_sample.dtype)
            flattened_input_var_sample = tf.reduce_sum(tf.multiply(multiplicand_1, multiplicand_2), axis = 2)
            # flattened_input_var_sample  has shape [beam_size x batch_size]
            actual_var_samples_list = self.map_index_to_unflattened(flattened_input_var_sample, \
                                                                     [self.max_num_var for _ in xrange(self.max_arguments)])
            # is a max_arguments sized list containing tensors of shape [beam_size x batch_size]
            # this contains the actual variable samples
            actual_var_samples_list = tf.unstack(tf.stack(actual_var_samples_list,axis = 2), axis = 0)
            # actual_var_samples_list is a list of beam_size length containing tensors of shape [batch_size x max_arguments]
            # ---------------------------------------------------------------------------------------------------------

            # Code For Preventing action repitition in generated trajectories============================================
            # need to do a scatter update on 'program_argument_table_index' and set used steps to 0
            index_0 = tf.range(0,self.beam_size*self.batch_size)
            # index_0 has shape [(beam_size*batch_size)]
            index_1 = tf.reshape(true_program_sample, [-1])
            # index_1 has shape [(beam_size*batch_size)]
            index_2 = tf.reshape(flattened_input_var_sample, [-1])
            # index_2 has shape [(beam_size*batch_size)]
            new_value = tf.ones_like(index_1)

            new_value = tf.where(tf.logical_or(tf.equal(index_1,0), tf.equal(index_1,self.num_progs-1)), new_value, 0*new_value)
            old_property_value = tf.stack(beam_properties['program_argument_table_index'], axis = 0)
            # old_property_value has shape beam_size x batch_size x num_progs x (max_arguments*max_num_var)
            old_property_value = tf.reshape(old_property_value, [-1, self.num_progs, int(math.pow(self.max_num_var,self.max_arguments))])
            index_for_scatter = tf.stack([index_0, index_1, index_2], axis = 1)
            old_property_value = self.immutable_scatter_nd_constant_update(old_property_value, index_for_scatter, new_value)
            old_property_value = tf.reshape(old_property_value, [self.beam_size, self.batch_size, self.num_progs,\
                                                                 int(math.pow(self.max_num_var,self.max_arguments))])
            beam_properties['program_argument_table_index'] = tf.unstack(old_property_value)
            # ------------------------------------------------------------------------------------------------------------

            # updating the beam artifacts like opertor sampled, variables sampled .... ============================
            for beam_id, true_prog_samples, true_var_samples in zip(xrange(self.beam_size),\
                                                                     tf.unstack(true_program_sample), actual_var_samples_list):


                to_return_action_sequence['program_type'][beam_id].append(true_prog_samples)

                arg_types, argtype_embedding = self.argument_type_net(true_prog_samples)
                to_return_action_sequence['argument_type'][beam_id].append(tf.transpose(arg_types, [1,0]))

                to_return_action_sequence['argument_table_index'][beam_id].append(true_var_samples)


                target_types = tf.gather(self.program_to_targettype_table, true_prog_samples)
                to_return_action_sequence['target_type'][beam_id].append(target_types)

                # =============================================================================
                # need to track current target program type so that we can terminate if gold type occurs
                condition = tf.logical_or(tf.equal(target_types, 0), tf.equal(target_types, self.num_progs-1))
                beam_properties['target_type'][beam_id] = tf.where(condition, beam_properties['target_type'][beam_id], target_types)
                # _____________________________________________________________________________

                prog_sampled_embeddings = tf.nn.embedding_lookup(self.program_embedding, true_prog_samples)
                argtypes = tf.unstack(arg_types, axis=0)
                var_embed = [tf.gather_nd(self.variable_embedding[beam_id], tf.stack([argtypes[i], self.batch_ids], axis=1)) \
                             for i in xrange(self.max_arguments)]
                # var_embed is a max_arguments sized list of batch_size x max_num_var x var_embed_dim
                var_sample = tf.unstack(true_var_samples, axis = 1)
                # var_sample  is a max_arguments sized list of tensors of shape batch_size
                var_sample_index = [tf.stack([self.batch_ids, var_sample[i]], axis=1) for i in range(self.max_arguments)]
                input_var_embedding = [tf.gather_nd(var_embed[i], var_sample_index[i]) for i in xrange(self.max_arguments)]
                num_variables_till_now, R_Flag = self.get_num_variables_till_now(beam_id, target_types)
                [target_var_key, \
                 beam_properties['target_var_embedding'][beam_id]] = self.target_var_net(input_var_embedding, \
                                                                                         argtype_embedding, \
                                                                                         prog_sampled_embeddings, num_variables_till_now, \
                                                                                         target_types)
                self.add_to_variable_table(target_types, target_var_key,\
                                                                    beam_properties['target_var_embedding'][beam_id], \
                                                                    num_variables_till_now, beam_id = beam_id)

                # whenever any variable table overflows we need to give negative reward for that
                beam_properties['Overflow_Penalize_Flag'][beam_id] = tf.add(beam_properties['Overflow_Penalize_Flag'][beam_id], R_Flag)
                to_return_action_sequence['target_table_index'][beam_id].append(num_variables_till_now)
            # --------------------------------------------------------------------------------------------------------


        # reshaping stuff so that it can be handled by main function
        for beam_id in xrange(self.beam_size):
            for i in xrange(self.num_timesteps):
                to_return_action_sequence['argument_table_index'][beam_id][i] = tf.unstack(\
                                         to_return_action_sequence['argument_table_index'][beam_id][i],axis = 0)

        # setting the Model Reward FLAG
        to_return_action_sequence['Overflow_Penalize_Flag'] = beam_properties['Overflow_Penalize_Flag']

        # to_return_action_sequence['argument_table_index'] is a list of length beam_size containing a list of
        # length num_timesteps containing a list of max argument length with tensors of shape batch size
        self.debug_beam = tf.constant(0) # can use for debugging
        return to_return_action_sequence, tf.exp(to_return_sequence_logprob), \
                to_return_sequence_logprob, self.debug_beam, to_return_per_step_prob, self.entropy/self.num_timesteps



    def get_feasible_progs(self, timestep, phase_elasticity):
        """Calculate the feasible operators possible based upon history till this timestep and whether phase has changed or not.

        Keyword arguments:
        timestep -- current time-step
        phase_elasticity -- output corresponding to phase_change_net indicating probability of shifting phases
        """
        # sampling only those operators whose required input variable tables are non empty ==================================
        num_variables = [tf.transpose(tf.reduce_sum(self.variable_mask[i], axis=2), [1,0]) \
                         for i in xrange(len(self.variable_mask))]
        #num_variables is a beam_size sized list of dimension batch_size x num_argtypes
        #self.required_argtypes is a list of variable types that need to be generated in order to create an answer of the desired (predicted) variable type
        #num_variables_remaining is the list of variable types that are needed (in order to create an answer of the desired variable type) but has not been generated till now
        num_variables_remaining  = [self.required_argtypes - num_variables[i]  for i in range(len(self.variable_mask))]
        num_variables_remaining = [tf.where(tf.greater(num_variables_remaining[i],0), \
                                            num_variables_remaining[i], \
                                            tf.zeros_like(num_variables_remaining[i])) for i in range(len(self.variable_mask))]
        num_variables_remaining = [tf.tile(tf.expand_dims(num_variables_remaining[i], 1),[1, self.num_progs, 1]) \
                                   for i in xrange(len(self.variable_mask))]
        
        program_to_targettype_onehot = tf.one_hot(self.program_to_targettype_table, depth=self.num_argtypes, \
                                                  dtype=num_variables_remaining[0].dtype)
        #program_to_targettype_onehot is of dimension num_progs x num_argtypes
        reqd_programs = [tf.reduce_max(tf.multiply(num_variables_remaining[i], program_to_targettype_onehot), axis=2) \
                         for i in xrange(len(self.variable_mask))]
        # reqd_programs is a beam_size sized list of dimension batch_size x num_progs
        # reqd_programs is a boolean score for each operator, indicating whether it can create a variable of the required type (as obtained from num_variables_remaining)
        # self.program_to_num_arguments is of dimension num_progs x num_argtypes
        num_variable_types = [tf.reduce_max(self.variable_mask[i], axis=2) for i in range(len(self.variable_mask))]
        # num_variable_types is a beam_size sized list of dimension batch_size x num_argtypes
        num_variable_types = [tf.transpose(tf.tile(tf.expand_dims(num_variable_types[i],0), [self.num_progs,1,1]), [2,0,1]) \
                              for i in xrange(len(self.variable_mask))]
        #num_variable_types is a beam_size sized list of dimension batch_size x num_progs x num_argtypes
        feasible_progs = [tf.where(tf.greater_equal(num_variable_types[i], self.program_to_num_arguments), \
                                   tf.ones_like(num_variable_types[i]), tf.zeros_like(num_variable_types[i])) \
                                    for i in xrange(len(self.variable_mask))]
        #feasible_progs is of dimension batch_size x num_progs x num_argtypes
        feasible_progs = [tf.reduce_prod(feasible_progs[i], axis=2) for i in xrange(len(self.variable_mask))]
        #feasible_progs is of dimension batch_size x num_progs

        program_to_kb_attention = tf.reduce_max(self.kb_attention, axis=2)
        feasible_progs = [tf.multiply(program_to_kb_attention, feasible_progs[i]) for i in xrange(len(self.variable_mask))]

        # changing feasible progrmas depending upon if phase has changed or not =============================================
        def separate_phases(arg):
            feasible_prog = arg[0]
            phase_elasticity = arg[1]
            if timestep < self.max_num_phase_1_steps:
                temp = tf.cast(tf.tile(phase_elasticity, [1,self.num_progs]), dtype = feasible_prog.dtype)
                multiplicand1 = tf.cast(self.progs_phase_1, dtype=feasible_prog.dtype)
            else:
                temp = tf.cast(tf.tile(1-phase_elasticity, [1,self.num_progs]), dtype = feasible_prog.dtype)
                multiplicand1 = tf.cast(self.progs_phase_2, dtype=feasible_prog.dtype)

            multiplicand2 = 1 - multiplicand1
            multiplicand = tf.add(tf.multiply(temp, multiplicand1), tf.multiply(1-temp, multiplicand2))
            feasible_prog = tf.multiply(feasible_prog, multiplicand)
            return feasible_prog

        feasible_progs = map(separate_phases, zip(feasible_progs,phase_elasticity))
        # -------------------------------------------------------------------------------------------------------------------
        # making 1st operator none impossible ===============================================================================
        temp = tf.one_hot(tf.zeros([self.batch_size], dtype=tf.int32), depth=self.num_progs, dtype=feasible_progs[0].dtype)
        feasible_progs = [temp + (1-temp)*feasible_progs[i] for i in xrange(len(self.variable_mask))]
        if timestep == 0:
            def make_none_impossible(prog_mask):
                temp = tf.cast(tf.one_hot(tf.zeros([self.batch_size], dtype = tf.int32), depth = self.num_progs), \
                                dtype = prog_mask.dtype)
                new_mask = -1*temp + (1-temp)
                prog_mask = tf.multiply(new_mask, prog_mask)
                return prog_mask
            feasible_progs = map(make_none_impossible,feasible_progs)
        # -------------------------------------------------------------------------------------------------------------------
        # the following block weighs the feasibility scores proportionately w.r.t. whether the operator can generate a variable of the required variable type
        # this proportionality based scoring is only enabled if the boolean flag self.bias_prog_sampling_with_target is turned on
        feasible_progs_new = [tf.where(tf.greater(feasible_progs[i],0), tf.add(feasible_progs[i], reqd_programs[i]), \
                                       feasible_progs[i])  for i in xrange(len(self.variable_mask))]    
        feasible_progs = [tf.multiply(self.bias_prog_sampling_with_target, feasible_progs_new[i]) + \
                          tf.multiply((1.0-self.bias_prog_sampling_with_target), feasible_progs[i]) for i in xrange(len(self.variable_mask))]
        return feasible_progs

    def add_preprocessed_output_to_variable_table(self, beam_id):
        """Initially Populating the variable tables from the data obtained from the queries after preprocessing

        Keyword arguments:
        beam_id -- the id of the beam which you want to update
        """
        # R_Flag is used to track overflow of variable table
        R_Flag = tf.zeros([self.batch_size])
        for i in xrange(self.num_argtypes):
            if i==self.empty_argtype_id:
                continue
            for j in xrange(self.max_num_var):
                ones = i*tf.ones([1, self.batch_size], dtype=tf.int32)
                empties = self.empty_argtype_id*tf.ones([self.max_arguments-1, self.batch_size], dtype=tf.int32)
                argtype = tf.concat([ones, empties], axis=0)
                # argtype is of dimension max_arguments x batch_size
                argtype_embed = tf.nn.embedding_lookup(self.argtype_embedding, argtype)
                input_var_embedding = tf.expand_dims(tf.matmul(self.preprocessed_var_emb_table[i][j], self.preprocessed_var_emb_mat), axis=0)
                # input_var_embedding is of dimension 1 x batch_size x var_embed_dim
                zeros_embedding = tf.zeros([self.max_arguments-1, self.batch_size, self.var_embed_dim], dtype=tf.float32)
                input_var_embedding = tf.concat([input_var_embedding, zeros_embedding], axis=0)
                # input_var_embedding is of dimension max_arguments x batch_size x var_embed_dim
                target_type = i*tf.ones([self.batch_size], dtype=tf.int32)
                num_variables_till_now, cur_r_flag = self.get_num_variables_till_now(beam_id, target_type)
                # generating key and embedding vectors for each preprocessed input variable by instantiating the Program None(V_input)
                [target_var_key, \
                 target_var_embedding] = self.target_var_net(input_var_embedding, argtype_embed, None, \
                                                                         num_variables_till_now, target_type)
                # target_type is of dimension batch_size
                self.add_to_variable_table(target_type, target_var_key, target_var_embedding, num_variables_till_now, beam_id = beam_id)
                R_Flag = tf.add(R_Flag, cur_r_flag)
        # once variable props from preprocessing are copied to main variable table
        # update main variable mask. Initialize main variable mask with the masks in preprocessed variable mask table
        self.variable_mask[beam_id] = tf.stack([tf.stack(temp, axis = 1) for temp in \
                                                  self.preprocessed_var_mask_table], axis = 0)
        self.variable_atten_table[beam_id] = tf.unstack(tf.add(self.variable_mask[beam_id],0))
        return R_Flag


    def sentence_encoder(self):
        """Encodes the given query using rnn
        """
        sentence_outputs = None
        with tf.variable_scope(self.enc_scope_text, reuse=tf.AUTO_REUSE) as scope:
            rnn_inputs_w2v = tf.nn.embedding_lookup(self.word_embeddings, tf.stack(self.encoder_text_inputs_w2v, axis=1))
            rnn_inputs_kb_emb = self.encoder_text_inputs_kb_emb
            # combine embeddings by concatenating them
            rnn_inputs = tf.concat([rnn_inputs_w2v, rnn_inputs_kb_emb], axis = 2)
            cell = tf.nn.rnn_cell.GRUCell(self.cell_dim)
            init_state = tf.get_variable('init_state', [1, self.cell_dim], \
                                         initializer = tf.constant_initializer(0.0), dtype = tf.float32)
            init_state = tf.tile(init_state, [self.batch_size, 1])
            sentence_outputs, states = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state,scope=scope)
        attention_states = tf.reshape(sentence_outputs,[self.max_len,self.batch_size,-1])
        # attention_states is of dimension max_len x batch_size x cell_dim
        # attention_states stores the attention context vector for each of the query words, to be later used to attend upon query words during sampling actions
        return states, attention_states

    def get_feasible_progs_for_last_timestep(self):
        """
        Based upon the predicted type of solution for the given query. This module returns the possible operators that are
        of the same type as the predicted output type.
        """
        gold_type = self.gold_target_type
        #gold_type is of dimension batch_size
        gold_type = tf.tile(tf.expand_dims(gold_type, axis=1), [1, self.num_progs])
        #gold_type is of dimension batch_size x num_progs
        feasible_progs_for_last_timestep = tf.where(tf.equal(gold_type, self.program_to_targettype_table), tf.ones_like(gold_type), tf.zeros_like(gold_type))
        #feasible_progs_for_last_timestep is of dimension batch_size x num_progs
        feasible_progs_for_last_timestep = tf.cast(feasible_progs_for_last_timestep, dtype=tf.float32)
        return feasible_progs_for_last_timestep

    def reset_state(self, sentence_state):
        """
        initializing the initial states for the different rnn's

        Keyword arguments:
        sentence_state -- the representation for the input query
        """
        zero_state = tf.zeros([self.batch_size, self.npi_core_dim], dtype=tf.float32)
        h_states = zero_state
        with tf.variable_scope(self.reset_scope, reuse=tf.AUTO_REUSE) as scope:
            e_state = tf.contrib.layers.fully_connected(sentence_state, self.npi_core_dim, activation_fn=tf.nn.elu, scope=scope)
            e_state = tf.nn.dropout(e_state, keep_prob= self.keep_prob)
        target_var_embedding = tf.zeros([self.batch_size, self.var_embed_dim], dtype=tf.float32)
        return h_states, e_state, target_var_embedding

    def npi_core(self, h_state, e_state, target_var_embedding):
        """
        The core NPI rnn. The rnn needs to be updated everytime after sampling taking an action

        Keyword arguments:
        h_state -- hidden state for the npi core
        e_state -- encoding of environment
        target_var_embedding -- encoding of the sampled variable
        """
        s_in = tf.expand_dims(self.state_encoding(e_state, target_var_embedding), axis=0)
        # s_in is of dimension 1 x batch_size x state_dim
        target_var_embedding = tf.expand_dims(target_var_embedding, axis=0)
        c = tf.unstack(tf.concat([s_in, target_var_embedding], axis=2), axis=0)
        # c is of dimension 1 x batch_size x (state_dim + var_embed_dim)
        with tf.variable_scope(self.npi_scope, reuse=tf.AUTO_REUSE) as scope:
            c, h_state = tf.nn.static_rnn(self.npi_cell, c, initial_state=h_state, dtype=tf.float32, scope=scope)
        # top_state is of dimension batch_size x npi_core_dim
        return c[0], h_state

    def env_encoding(self, e_state, target_var_embedding):
        """
        Generates an encoding for the environment

        Keyword arguments:
        e_state -- encoding of environment
        target_var_embedding -- encoding of the sampled variable
        """
        c = tf.unstack(tf.expand_dims(target_var_embedding,axis=0), axis=0)
        # c is a 1-length list of dimension batch_size x var_embed_dim
        with tf.variable_scope(self.env_scope, reuse=tf.AUTO_REUSE) as scope:
            c, e_state = tf.nn.static_rnn(self.env_cell, c, initial_state=e_state, dtype=tf.float32, scope=scope)
        # c, e_state = tflearn.lstm(c, self.npi_core_dim, return_seq=False, initial_state=e_state, return_states=True)
        return c[0], e_state

    def state_encoding(self, e_state, target_var_embedding):
        """
        Keeps track of the agent's state (internal) in RL sense. This state is what is used to base your actions
        upon by the NPI core and few other modules.

        Keyword arguments:
        e_state -- encoding of environment
        target_var_embedding -- encoding of the sampled variable
        """
        merge = tf.concat([e_state, target_var_embedding], axis=1)#tflearn.merge([e_state, target_var_embedding], 'concat')
        # merge is of dimension batch_size x (self.npi_core_dim+var_embed_dim)
        with tf.variable_scope(self.state_encoding_scope_1, reuse=tf.AUTO_REUSE) as scope:
            elu = tf.contrib.layers.fully_connected(merge, self.hidden_dim, activation_fn=tf.nn.elu, scope=scope)
            elu = tf.nn.dropout(elu, keep_prob = self.keep_prob)
        # elu is of dimension batch_size x hidden_dim
        with tf.variable_scope(self.state_encoding_scope_2, reuse=tf.AUTO_REUSE) as scope:
            elu = tf.contrib.layers.fully_connected(elu, self.hidden_dim, activation_fn=tf.nn.elu, scope=scope)
            elu = tf.nn.dropout(elu, keep_prob = self.keep_prob)
        # elu is of dimension batch_size x hidden_dim
        with tf.variable_scope(self.state_encoding_scope_3, reuse=tf.AUTO_REUSE) as scope:
            out = tf.contrib.layers.fully_connected(elu, self.state_dim, scope=scope)
            out = tf.nn.dropout(out, keep_prob = self.keep_prob)
        # out is of dimension batch_size x state_dim
        return out

    def terminate_net(self, progs_taken, old_terminate):
        """
        Just keeps a track of whether the beam has terminated

        Keyword arguments:
        progs_taken -- operators sampled in this timestep
        old_terminate -- whether the beam had already terminated. 1 or 0.
        """
        temp1 = tf.ones_like(progs_taken)
        temp2 = tf.zeros_like(progs_taken)
        # 0 is the None action
        terminate = tf.where(tf.equal(progs_taken, self.num_progs-1), temp1, temp2)
        terminate = tf.reshape(terminate, [self.batch_size, 1])
        terminate = tf.cast(terminate, dtype = old_terminate.dtype)
        terminate = tf.where(tf.greater_equal(terminate, old_terminate), terminate, old_terminate)
        return terminate
        # this will return tensor of shape batch_size x 1

    def none_finder_net(self, progs_taken):
        """
        To check if the operated sampled at the current timestep is the no-op/none operator

        Keyword arguments:
        progs_taken -- operators sampled in this timestep
        """
        temp1 = tf.ones_like(progs_taken)
        temp2 = tf.zeros_like(progs_taken)
        # 0 is the None action
        out = tf.where(tf.equal(progs_taken, 0), temp1, temp2)
        out = tf.reshape(out, [self.batch_size, 1])
        return out
        # this will return tensor of shape batch_size x 1

    def check_if_gold_target(self, beam_target_type, beam_gold_type, if_terminated):
        """
        This method checks if the last variable created by that beam is same as the desired (predicted) variable type
        """
        mask_same_type = tf.where(tf.equal(beam_target_type, beam_gold_type), tf.zeros_like(beam_target_type), \
                                  tf.ones_like(beam_target_type))
        add_flag = tf.multiply(tf.to_float(mask_same_type),if_terminated)
        score_penalization = add_flag
        return score_penalization

    def phase_change_net(self, h, timestep, old_p_el):
        """
        Network that predicts whether to change phase of the beam or not

        Keyword arguments:
        h -- hidden state of the npi core
        timestep -- current time-step
        old_p_el -- the probability of changing in the previous timestep
        """
        if timestep < self.max_num_phase_1_steps:
            with tf.variable_scope(self.phase_change_scope, reuse=tf.AUTO_REUSE) as scope:
                p_el = tf.contrib.layers.fully_connected(h, 1, activation_fn = tf.keras.activations.linear, \
                                                        normalizer_fn = tf.contrib.layers.batch_norm, \
                                                        weights_regularizer = tf.contrib.layers.l2_regularizer, scope=scope)
                p_el = tf.nn.dropout(p_el, keep_prob = self.keep_prob)
            p_el = tf.nn.sigmoid(p_el)
            p_el = tf.where(tf.greater(p_el,old_p_el), old_p_el, p_el)
            temp = tf.ones_like(p_el)
            p_el = tf.where(tf.greater(p_el, self.phase_change_threshold), temp, p_el)
            return p_el
        else:
            temp = tf.zeros_like(old_p_el)
            return temp

    def prog_net(self, h, sentence_state, attention_states, query_attentions_till_now, feasible_progs, num_samples, terminate, last_target_type):
        """
        Network to sample the set of operators

        Keyword arguments:
        h -- hidden state of the npi core
        sentence_state -- encoding of query
        attention_states -- attention vector over query words
        query_attentions_till_now -- query attentions till last step
        feasible_progs -- mask over feasible set of programs
        num_samples -- number of operators to sample
        terminate -- flag if the terminate operator has been sampled till now
        last_target_type -- target type of the previous sampled operator
        """
        # feasible_progs is of shape batch_size x num_progs
        # variable_mask beam_size x [num_argtypes, batch_size, max_num_var]
        # self.program_to_argtype_table is of dimension num_progs x max_arguments
        # last_target_type is of dimension batch_size
        last_target_type = tf.tile(tf.expand_dims(tf.expand_dims(last_target_type, axis=1), axis=2), \
                                   [1, self.num_progs, self.max_arguments])
        programs_consuming_last_targettype = tf.reduce_max(tf.where(tf.equal(self.program_to_argtype_table, last_target_type), \
                                                        tf.ones_like(last_target_type), tf.zeros_like(last_target_type)), axis=2)
        programs_consuming_last_targettype = tf.cast(programs_consuming_last_targettype, dtype=tf.float32)
        feasible_progs_new = tf.where(tf.greater(feasible_progs,0), tf.add(feasible_progs, programs_consuming_last_targettype), feasible_progs)
        feasible_progs = tf.multiply(self.bias_prog_sampling_with_last_variable, feasible_progs_new) + \
                            tf.multiply((1.0-self.bias_prog_sampling_with_target), feasible_progs)
        # programs_consuming_last_targettype is of dimension batch_size x num_progs
        # feasible_progs is of dimension batch_size x num_progs
        if self.concat_query_npistate:
            concat_hq = tf.concat([h, sentence_state], axis=1)
        else:
            concat_hq = h
        if self.query_attention:
            query_attention = tf.multiply(attention_states, tf.multiply(h, self.query_attention_h_mat))
            # temp is of dimension max_len x batch_size x cell_dim
            query_attention = tf.nn.softmax(tf.reduce_sum(query_attention, axis=2), dim=0)
            # query_attention is of dimension max_len x batch_size
            # the following block weighs the attention over query words inversely proportionately to the cumulative attention distribution over the query words till this time step
            if self.dont_look_back_attention:
                query_attentions_till_now = tf.transpose(query_attentions_till_now, [1,0])
                query_attention = tf.nn.softmax(tf.multiply(1.-query_attentions_till_now, query_attention), dim=0)
                query_attentions_till_now = tf.nn.softmax(tf.add(query_attentions_till_now, query_attention), dim=0)
                query_attentions_till_now = tf.transpose(query_attentions_till_now, [1,0])
            query_attention = tf.expand_dims(query_attention, axis=2)
            query_attention = tf.reduce_sum(tf.multiply(query_attention, attention_states), axis=0)
            concat_hq = tf.concat([concat_hq, query_attention], axis=1)
        
        
        with tf.variable_scope(self.prog_hidden_scope, reuse=tf.AUTO_REUSE) as scope:
            hidden = tf.contrib.layers.fully_connected(concat_hq, self.prog_key_dim, activation_fn = tf.nn.elu, \
                                               weights_regularizer = tf.contrib.layers.l2_regularizer, scope=scope)
            hidden = tf.nn.dropout(hidden, keep_prob=self.keep_prob)

        with tf.variable_scope(self.prog_key_scope, reuse=tf.AUTO_REUSE) as scope:
            key = tf.contrib.layers.fully_connected(hidden, self.prog_key_dim, scope=scope)
            key = tf.nn.dropout(key, keep_prob=self.keep_prob)

        # prog_dist is the attention distribution over the operators, conditional to the program state
        key = tf.reshape(key, [-1, 1, self.prog_key_dim])
        prog_sim = tf.multiply(key, self.program_keys)
        prog_dist = tf.reduce_sum(prog_sim, [2])
        prog_dist = tf.nn.softmax(prog_dist, dim=1)
        
        if self.params['terminate_prog'] is True:
            temp = tf.one_hot((self.num_progs-1)*tf.ones([self.batch_size], dtype=tf.int32), \
                              depth=self.num_progs, dtype=feasible_progs[0].dtype)
            feasible_progs = terminate*temp + (1-terminate)*feasible_progs
        # distribution over all the operators, multiplying element wise with the operator feasibility ensures that only feasible programs get non zero attention
        prog_dist = tf.multiply(prog_dist, feasible_progs)
        # prog_dist is of dimension batch_size x num_progs

        #stochastic operator sampling returns the num_sampled number of sampled operators and their corresponding probabilities
        prog_sampled_probs, prog_sampled_indices = self.stochastic_program_sampling(prog_dist, num_samples)
        prog_sampled_probs = tf.divide(prog_sampled_probs,tf.reduce_sum(tf.clip_by_value(prog_dist,0,1), axis=1, keep_dims=True))
        # prog_sampled_probs is a tensor of shape batch_size x num_samples
        # prog_sampled_indices is a tensor of shape batch_size x num_samples

        # the following block gets the kb based consistency score for the sampled operators
        prog_sampled_embeddings = tf.nn.embedding_lookup(self.program_embedding, prog_sampled_indices)
        # prog_sampled_embeddings is a tensor of shape batch_size x num_samples x prog_embed_dim
        list_program_sample_index = tf.unstack(prog_sampled_indices,axis=1)
        # list_program_sample_index is a num_samples length list composed of batch_size sized tensors
        kb_attention_for_sampled_progs = []
        for prog_sample_index in list_program_sample_index:
            prog_sample_index = tf.stack([self.batch_ids, prog_sample_index], axis=1)
            kb_attention_for_sampled_progs.append(tf.gather_nd(self.kb_attention, prog_sample_index))
        # kb_attention_for_sampled_progs is a num_samples length list composed of batch_size x max_var x max_var x max_var sized tensors
        return prog_sampled_probs, prog_sampled_indices, prog_sampled_embeddings, \
                tf.stack(kb_attention_for_sampled_progs, axis = 0), query_attentions_till_now


    def argument_type_net(self, prog_sample):
        """
        Mapping from operator to input argument types

        Keyword arguments:
        prog_sample: Sampled program whose argument types need to be found
        """
        arg_types = tf.gather(self.program_to_argtype_table, prog_sample)
        # argtypes is of dimension batch_size x max_arguments
        # argtypes is a list of argument types for that sampled program
        # in order to handle different length argtypes in a batch,
        # consider that for every program there is max upto N arguments only (with padding whenever necessary)
        argtype_embedding = tf.nn.embedding_lookup(self.argtype_embedding, arg_types)
        # argtype_embeddign is of dimension batch_size x max_arguments x argtype_embed_dim
        arg_types = tf.transpose(arg_types, [1,0])
        argtype_embedding = tf.transpose(argtype_embedding, [1,0,2])
        # argtype_embeddign is of dimension max_arguments  x batch_size x argtype_embed_dim
        return arg_types, argtype_embedding

    def input_var_net(self, h, arg_types, prog_sample, prog_embedding, kb_attention, beam_id, num_samples, terminate, past_program_variables):
        """
        Network to sample the set of input variables for the corresponding sampled operators

        Keyword arguments:
        h -- hidden state of the npi core
        arg_types -- variable type of arguments of the sampled operator
        prog_sample -- sampled operator for which input variables need to be sampled
        prog_embedding -- embedding of sampled operator for which input variables need to be sampled
        kb_attention -- mask coming from KB to allow or disallow certain variable combinations
        beam_id -- the beam for which the sampling is being done
        num_samples -- number of input variables to sample
        terminate -- Flag stating whether the terminate operator has been called till now or not
        past_program_variables -- the list of exhausted variables for every operator
        """
        # prog_sample is of batch_size
        target_types = tf.gather(self.program_to_targettype_table, prog_sample)
        # targettypes is of dimension batch_size
        argtypes = tf.unstack(arg_types, axis=0)
        # argtypes is a max_arguments sized list of dimension batch_size each
        var_atten = [tf.gather_nd(self.variable_atten_table[beam_id],tf.stack([argtypes[i], self.batch_ids], axis=1)) \
                     for i in xrange(self.max_arguments)]
        # var_atten is a max_arguments sized list of batch_size x max_num_var
        var_mask = [tf.gather_nd(self.variable_mask[beam_id],tf.stack([argtypes[i], self.batch_ids], axis=1)) \
                    for i in xrange(self.max_arguments)]
        # var_mask is a max_arguments sized list of batch_size x max_num_var
        var_atten = [self.update_attention(var_atten[i], h, i) for i in range(self.max_arguments)]
        var_atten = [self.mask_attention(var_atten[i], var_mask[i]) for i in xrange(self.max_arguments)]
        # var_atten is a max_arguments sized list of batch_size x max_num_var
        var_keys = [tf.gather_nd(self.variable_keys[beam_id], tf.stack([argtypes[i], self.batch_ids], axis=1)) \
                    for i in xrange(self.max_arguments)]
        # var_keys is a max_arguments sized list of batch_size x max_num_var x var_key_dim
        # var_atten is a max_arguments sized list of batch_size x max_num_var
        with tf.variable_scope(self.inp_var_key_scope, reuse=tf.AUTO_REUSE) as scope:
            key = [tf.contrib.layers.fully_connected(var_atten[i], self.var_key_dim, activation_fn = tf.nn.elu, \
                                                 weights_regularizer = tf.contrib.layers.l2_regularizer, scope=scope) \
                                                 for i in xrange(self.max_arguments)]
            key = [tf.nn.dropout(item, keep_prob=self.keep_prob) for item in key]
        key = [tf.reshape(key[i], [-1, 1, self.var_key_dim]) for i in xrange(self.max_arguments)]
        var_sim = [tf.multiply(key[i], var_keys[i]) for i in xrange(self.max_arguments)]
        # var_sim is of dimension batch_size x max_num_var x var_key_dim
        var_dist = [tf.reduce_sum(var_sim[i], [2]) for i in xrange(self.max_arguments)]
        var_dist = [tf.nn.softmax(var_dist[i], dim=1) for i in xrange(self.max_arguments)]
        var_dist = [tf.multiply(var_dist[i],var_mask[i]) for i in xrange(self.max_arguments)]
        # var_dist is a max_arguments sized list of dimension batch_size x max_num_var

        # we have to get the joint distribution over the multiple arguments.
        var_dist = tf.stack(var_dist,axis=1)

        #var_mask is of dimension batch_size x max_arguments x max_num_var
        split_var_dist = tf.unstack(var_dist, axis = 0)
        # split_var_dist is a batch_size sized list of dimension max_arguments x max_num_var
        joint_var_dist = []
        for _var_dist_ in split_var_dist:
            list_vectors_dist = tf.unstack(_var_dist_,axis = 0)
            joint_var_dist.append(self.recursive_joint_prob_generator(list_vectors_dist))
        joint_var_dist = tf.stack(joint_var_dist,axis=0)
        flattened_joint_var_dist = tf.reshape(joint_var_dist,shape = [self.batch_size,-1])
        flattened_joint_var_dist = tf.multiply(flattened_joint_var_dist, kb_attention)
        flattened_joint_var_dist = tf.multiply(flattened_joint_var_dist, past_program_variables)
        # ensuring all 0 variable probability vector is handled appropriately=======================================
        marker = tf.reduce_mean(flattened_joint_var_dist,axis = 1, keep_dims=True)
        marker = tf.where(tf.equal(marker,0), 0*tf.ones_like(marker), tf.ones_like(marker))
        flattened_joint_var_dist = self.mask_attention(flattened_joint_var_dist, tf.ones_like(flattened_joint_var_dist))
        flattened_joint_var_dist = tf.multiply(flattened_joint_var_dist, marker)
        # ----------------------------------------------------------------------------------------------------------
        var_sampled_probs, var_sampled_indices = tf.nn.top_k(flattened_joint_var_dist, k = num_samples)
        # var_sampled_probs is a tensor of shape batch_size x num_samples
        # var_sampled_indices is a tensor of shape batch_size x num_samples

        return var_sampled_probs, var_sampled_indices, target_types

    def get_num_variables_till_now(self, beam_id, targettypes):
        """
        For the given beam, gives the number of variables till now for the given target types.
        Also returns flag indicating overpopulation.

        Keyword arguments:
        beam_id -- the beam for which the sampling is being done
        targettypes -- the type of variable table whose count needs to be returned
        """
        var_mask = tf.gather_nd(self.variable_mask[beam_id], tf.stack([targettypes, self.batch_ids], axis=1))
        # var_mask is of dimension batch_size x max_num_var
        num_variables_till_now = tf.reduce_sum(var_mask, axis=1)
        num_variables_till_now = tf.cast(num_variables_till_now, dtype=tf.int32)
        # num_variables_till_now is of dimension batch_size
        # for None arg_type we should always ensure there is only one element in table to have consistent probabilities
        num_variables_till_now = tf.where(tf.equal(targettypes, 0), \
                                          tf.zeros_like(num_variables_till_now), num_variables_till_now) # 0 is none type
        # Return a negative reward if table overpopulates
        temp = (self.max_num_var-1) * tf.ones_like(num_variables_till_now)
        R_Flag = tf.zeros_like(num_variables_till_now)
        R_Flag = tf.where(tf.greater(num_variables_till_now, temp), 1+R_Flag, R_Flag)
        R_Flag = tf.cast(R_Flag, dtype=tf.float32)
        # Overpopulation - Rewrite last entry in table
        num_variables_till_now = tf.where(tf.greater(num_variables_till_now, temp), temp, num_variables_till_now)
        return num_variables_till_now, R_Flag

    def target_var_net(self, input_var_embedding, argtype_embedding, prog_embedding, num_variables_till_now, target_type):
        """
        Get the key and embedding vector of the sampled Program {P(OxV)}

        Keyword arguments:
        input_var_embedding -- embedding of input variables (V)
        argtype_embedding -- embedding of argument types of input variables(V=v1 x v2 x .... vn)
        prog_embedding -- operator embedding (O)
        num_variables_till_now -- number of variables already in the table
        target_type -- type of the program (P)
        """
        if type(input_var_embedding) is list:
            var_embedding = tf.stack(input_var_embedding, axis=0)
        else:
            var_embedding = input_var_embedding

        # var_embedding is of dimension max_arguments x batch_size x var_embed_dim
        # argument_type_embedding is of dimension max_arguments x batch_size x argtype_embed_dim
        # prog_embedding is of dimension batch_size x prog_embed_dim


        if prog_embedding is None:
            prog_embedding = tf.zeros([self.batch_size, self.prog_embed_dim], dtype=tf.float32)

        # generate one vector from list of argument embeddings ========================================
        list_argtype_embedding = tf.unstack(argtype_embedding, axis = 0)
        input_1 = list_argtype_embedding[0]
        input_2 = list_argtype_embedding[1]
        for current_argtype_id in range(len(list_argtype_embedding)):
            with tf.variable_scope(self.get_target_var_key_and_embedding_arg_scope+str(current_argtype_id), reuse=tf.AUTO_REUSE) as scope:
                input_1 = tf.contrib.layers.fully_connected(tf.concat([input_1,input_2],axis=1),self.argtype_embed_dim,\
                                                        activation_fn = tf.nn.elu, scope=scope)
                input_1 = tf.nn.dropout(input_1, keep_prob=self.keep_prob)
            if current_argtype_id + 2 > len(list_argtype_embedding)-1:
                break
            input_2 = list_argtype_embedding[current_argtype_id+2]
        l2_input_1 = input_1 # next layers input 1 is ready
        # ---------------------------------------------------------------------------------------------------
        # generate one vector from list of input variable embeddings ========================================
        list_var_embedding = tf.unstack(var_embedding, axis = 0)
        input_1 = list_var_embedding[0]
        input_2 = list_var_embedding[1]
        for current_var_id in range(len(list_var_embedding)):
            with tf.variable_scope(self.get_target_var_key_and_embedding_var_scope+str(current_var_id), reuse=tf.AUTO_REUSE) as scope:
                input_1 = tf.contrib.layers.fully_connected(tf.concat([input_1,input_2],axis=1),self.var_embed_dim,\
                                                        activation_fn = tf.nn.elu, scope=scope)
                input_1 = tf.nn.dropout(input_1, keep_prob=self.keep_prob)
            if current_var_id + 2 > len(list_var_embedding)-1:
                break
            input_2 = list_argtype_embedding[current_var_id+2]
        l2_input_2 = input_1 # next layers input 2 is ready
        # ---------------------------------------------------------------------------------------------------
        l2_input_3 = prog_embedding # next layers input 3 is directly the program embedding as it is only one unlike variable and argument which are max_argument in number
        # nnet layer over transformed Argument, variable embedding and program embedding ====================================
        l2_input = tf.concat([l2_input_1,l2_input_2,l2_input_3],axis=1)
        with tf.variable_scope(self.get_target_var_key_and_embedding_targetembed_scope, reuse=tf.AUTO_REUSE) as scope:
            target_var_embedding = tf.contrib.layers.fully_connected(l2_input,self.var_embed_dim, activation_fn=tf.nn.elu, scope=scope)
            target_var_embedding = tf.nn.dropout(target_var_embedding, keep_prob=self.keep_prob)
        if self.use_key_as_onehot: # not being used in final run, can ignore
            target_type_onehot = tf.one_hot(target_type, depth=self.num_argtypes, dtype=tf.float32)
            num_variables_till_now_onehot = tf.one_hot(num_variables_till_now, depth=self.max_num_var, dtype=tf.float32)
            # target_type_onehot is  batch_size x num_argtypes
            # num_variables_till_now_onehot is batch_size x max_num_var
            target_var_key = tf.concat([target_type_onehot, num_variables_till_now_onehot], axis=1)
        else:
            with tf.variable_scope(self.get_target_var_key_and_embedding_targetkey_scope, reuse=tf.AUTO_REUSE) as scope:
                target_var_key = tf.contrib.layers.fully_connected(l2_input,self.var_key_dim, activation_fn=tf.nn.elu, scope=scope)
                target_var_key = tf.nn.dropout(target_var_key, keep_prob=self.keep_prob)

        # target_var_embedding is of dimension batch_size x var_embed_dim
        # target_var_key is of dimension batch_size x var_key_dim
        return target_var_key, target_var_embedding

    def add_to_variable_table(self, targettype, target_var_key, target_var_embedding, num_variables_till_now, beam_id = None):
        """
        Function for Adding a Target Variable that was generated by an action P(OxV), to the variable tables

        Keyword arguments:
        beam_id -- used to identify the beam whose tables need to be update. every beam has separate tables.
        num_variables_till_now -- number of variables already in the table
        target_var_embedding -- embedding for the target variable generated by target_var_net
        target_var_key -- key for the target variable generated by target_var_net
        target_type -- type of the variable generated
        """
        indices_to_update = tf.stack([targettype, self.batch_ids, num_variables_till_now], axis=1)
        # indices_to_update is of dimension batch_size x 3
        # variable_mask is of dimension num_argtypes x batch_size x max_num_var
        mask_value_to_update = tf.ones([self.batch_size], tf.float32)
        self.variable_mask[beam_id] = self.immutable_scatter_nd_constant_update(self.variable_mask[beam_id], \
                                                                       indices_to_update, \
                                                                       mask_value_to_update)
        # variable_mask is of dimension num_argtypes x batch_size x max_num_var
        self.variable_keys[beam_id] = self.immutable_scatter_nd_1d_update(self.variable_keys[beam_id], \
                                                                 indices_to_update, \
                                                                 target_var_key)
        # self.variable_keys is of dimension num_argtypes x batch_size x max_num_var x var_key_dim
        self.variable_embedding[beam_id] = self.immutable_scatter_nd_1d_update(self.variable_embedding[beam_id], \
                                                                      indices_to_update, \
                                                                      target_var_embedding)
        # self.variable_embedding is of dimension num_argtypes x batch_size x max_num_var x var_embed_dim
        # VARIABLE_ATTENTION TABLE ALSO NEEDED TO BE UPDATED (SO THAT THE NEWLY ADDED ROW DOES NOT GET 0 ATTENTION)
        local_var_atten = tf.stack(self.variable_atten_table[beam_id], axis=0)
        # local_var_atten has shape = [self.num_argtypes, self.batch_size, self.max_num_var]
        local_var_atten = self.immutable_scatter_nd_constant_update(local_var_atten,indices_to_update, mask_value_to_update)
        self.variable_atten_table[beam_id] = tf.unstack(tf.nn.l2_normalize(local_var_atten, dim = 2), axis = 0)

    def update_attention(self, static_atten, h, i):
        """
        Nnet Used to dynamically update the variable attention in the variable attention tables

        Keyword arguments:
        static_atten -- uthe static atten value in the variabe attention table
        h -- the CIPTR state condtionial to which the attentions will be updated
        i -- used to retrieve the required update_attention_layer
        """
        # static_atten is of dimension batch_size x num_var
        # h is of dimension batch_size x cell_dim
        inputs = tf.concat([static_atten,h], axis = 1)
        with tf.variable_scope(self.update_attention_scope+"arg_"+str(i), reuse=tf.AUTO_REUSE) as scope:
            new_static_atten = tf.contrib.layers.fully_connected(inputs, self.max_num_var, activation_fn=tf.nn.elu, scope=scope)
            new_static_atten = tf.nn.softmax(new_static_atten)
        return new_static_atten

    def mask_attention(self, static_atten, mask):
        """
        Utility function to apply mask to a distribution
        """
        # static_atten is of dimension batch_size x num_var
        # mask is of dimension batch_size x num_var
        masked_atten = tf.multiply(static_atten, mask)
        num = len(masked_atten.get_shape())
        l1norm = tf.reduce_sum(masked_atten, axis=1)
        stacked_norm = tf.multiply(tf.ones_like(masked_atten),tf.expand_dims(l1norm,axis = num-1))
        masked_atten = tf.where(tf.equal(stacked_norm, 0.), tf.ones_like(masked_atten), masked_atten)
        new_l1_norm = tf.reduce_sum(masked_atten, axis=1)
        masked_atten = masked_atten/tf.reshape(new_l1_norm, (-1,1))
        return masked_atten

    def reinforce(self):
        """
        This method computes the objective function and the corresponding loss for backpropagation
        """
        self.action_seq, self.ProgramProb, self.logProgramProb, BEAM_DEBUG_DATA, per_step_prob, entropy = self.inference()

        #from actual rewards
        self.Reward = tf.placeholder(tf.float32, [self.batch_size, self.beam_size])
        current_baseline = tf.reduce_sum(tf.multiply(self.Reward,self.ProgramProb),axis=1,keep_dims=True)
        current_baseline = tf.stop_gradient(current_baseline)

        self.Relaxed_reward = tf.placeholder(tf.float32, [self.batch_size, self.beam_size])
        current_baseline_relaxed = tf.reduce_sum(tf.multiply(self.Relaxed_reward, self.ProgramProb), axis=1, keep_dims=True)
        current_baseline_relaxed = tf.divide(current_baseline_relaxed, tf.reduce_sum(self.ProgramProb,axis=1,keep_dims=True))
        current_baseline_relaxed = tf.stop_gradient(current_baseline_relaxed)

        # from intermediate rewards
        self.IfPosIntermediateReward = tf.placeholder(tf.float32, [self.batch_size, self.beam_size])
        self.rate_intermediate_reward = self.params['lr_intermideate_reward']
        self.mask_IntermediateReward = tf.placeholder(tf.float32, [self.batch_size, self.beam_size, self.num_timesteps])
        self.IntermediateReward = tf.placeholder(tf.float32, [self.batch_size, self.beam_size])
        int_reward = tf.multiply(self.IntermediateReward,self.IfPosIntermediateReward)
        prob_intermediate = tf.multiply(self.mask_IntermediateReward, per_step_prob)
        prob_intermediate = tf.where(tf.equal(self.mask_IntermediateReward,0),tf.ones_like(self.mask_IntermediateReward),prob_intermediate)
        prob_intermediate = tf.reduce_prod(prob_intermediate, axis = 2)
        log_prob_intermediate = tf.log(prob_intermediate)
        unbackpropable_intermediate_prob = tf.stop_gradient(prob_intermediate)
        baseline_ir = tf.reduce_sum(tf.multiply(unbackpropable_intermediate_prob,int_reward),axis = 1, keep_dims=True)

        #combining stuff
        new_baseline = tf.add(current_baseline, baseline_ir)
        new_baseline = tf.divide(new_baseline,tf.add(tf.reduce_sum(self.ProgramProb,axis=1,keep_dims=True),\
                                                     tf.reduce_sum(unbackpropable_intermediate_prob,axis=1,keep_dims=True)))
        self.OldBaseline = tf.placeholder(tf.float32,[self.batch_size,1])
        final_baseline = tf.stop_gradient(new_baseline)


        #coming back to reinforce_main
        scaling_term_1 = tf.multiply(self.ProgramProb,self.Reward-final_baseline)
        scaling_term_1 = tf.stop_gradient(scaling_term_1)
        loss_reinforce = tf.multiply(self.logProgramProb, scaling_term_1)
        loss_reinforce = tf.where(tf.is_nan(loss_reinforce), tf.zeros_like(loss_reinforce), loss_reinforce)
        loss_reinforce = tf.reduce_sum(tf.reduce_mean(loss_reinforce, axis = 0))

        #coming back to intermediate reward part
        scaling_term_2 = int_reward - final_baseline
        scaling_term_2 = tf.multiply(scaling_term_2 ,unbackpropable_intermediate_prob)
        scaling_term_2 = tf.multiply(self.IfPosIntermediateReward,scaling_term_2)
        scaling_term_2 = tf.stop_gradient(scaling_term_2)
        loss_ir = tf.multiply(scaling_term_2, log_prob_intermediate)
        loss_ir = tf.where(tf.is_nan(loss_ir), tf.zeros_like(loss_ir), loss_ir)
        loss_ir = tf.reduce_sum(tf.reduce_mean(loss_ir,axis=0))

        relaxed_scaling_term_1 = tf.multiply(self.ProgramProb, self.Relaxed_reward-current_baseline_relaxed)
        relaxed_scaling_term_1 = tf.stop_gradient(relaxed_scaling_term_1)
        loss_relaxed_reinforce = tf.multiply(self.logProgramProb, relaxed_scaling_term_1)
        loss_relaxed_reinforce = tf.where(tf.is_nan(loss_relaxed_reinforce), tf.zeros_like(loss_relaxed_reinforce),\
                                          loss_relaxed_reinforce)
        loss_relaxed_reinforce = tf.reduce_sum(tf.reduce_mean(loss_relaxed_reinforce,axis=0))

        entropy = tf.where(tf.is_nan(entropy), tf.zeros_like(entropy), entropy)
        entropy = entropy/self.batch_size


        loss = loss_reinforce + self.params['Rate_Entropy']*entropy + self.rate_intermediate_reward*loss_ir +\
                    tf.multiply(self.relaxed_reward_multipler, loss_relaxed_reinforce)
        self.loss = loss
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-08)
        gradients_n_variables = optimizer.compute_gradients(-1*self.loss)
        self.train_op = optimizer.apply_gradients(gradients_n_variables)
        with tf.control_dependencies([self.train_op]):
            self.dummy = tf.constant(0)
        return self.action_seq, self.ProgramProb, self.logProgramProb, self.Reward, self.Relaxed_reward,\
                            self.dummy, -self.loss, BEAM_DEBUG_DATA, per_step_prob, self.IfPosIntermediateReward, \
                            self.mask_IntermediateReward, self.IntermediateReward

    def recursive_joint_prob_generator(self,list_dists):
        """
        Utility function to generate a distribution of V as joint distribution of v1,v2 ,.... vn
        """
        if len(list_dists) == 2:
            dist_1 = tf.reshape(list_dists[0],shape = [-1,1])
            dist_2 = tf.reshape(list_dists[1],shape = [-1,1])
            out = tf.matmul(dist_1,dist_2,transpose_b = True)
            return out
        else:
            current_dist = list_dists[-1]
            #has shape batch_size x max_num_var
            new_list_dists = list_dists[0:-1]
            probs_list = tf.unstack(current_dist, axis = 0)
            penultimate_output = self.recursive_joint_prob_generator(new_list_dists)
            #has shape batch_size x max_num_var x max_num_var ....
            out = []
            for prob in probs_list:
                #prob is tensor of shape batch_size
                out.append(tf.multiply(penultimate_output,prob))
            return tf.stack(out,axis = len(list_dists)-1)

    def map_index_to_unflattened(self,number,shape):
        """
        Utility function to map an index in a flattened tensor to corresponding index in unflattened tensor
        """
        out = []
        for divisor in shape[::-1]:
            remainder = tf.truncatemod(number,divisor)#number // divisor
            number = tf.floordiv(number,divisor)#number % divisor
            out.append(remainder)
        return out[::-1]

    def map_index_to_flattened(self,number, dimensions):
        """
        Utility function to map an index in a unflattened tensor to corresponding index in flattened tensor
        """
        dimensions = tf.unstack(tf.to_int32(dimensions), axis = 0)
        dimensions.append(tf.constant(1, tf.int32))

        out = []
        for i in range(0,len(dimensions)-1):
            out.append(tf.reduce_prod(tf.stack(dimensions[i+1:] , axis = 0) ,axis = 0))
        out = tf.stack(out)
        out = tf.multiply(number,out)
        out = tf.reduce_sum(out, len(number.get_shape())-1)
        return out

    def immutable_scatter_nd_constant_update(self, inp1, inp2, inp3):
        """
        Utility function
        """
        shape = inp1.get_shape()
        inp1 = tf.to_float(inp1)
        inp1 = tf.reshape(inp1, [-1])
        inp2 = self.map_index_to_flattened(inp2, shape)
        z1 = tf.to_float(tf.one_hot(inp2, inp1.get_shape()[0]))
        z2 = tf.to_float(tf.reshape(inp3,[-1,1]))
        z3 = tf.multiply(z1,z2)
        update_input = tf.reduce_sum(tf.add(z3, tf.zeros_like(inp1)),axis = 0)

        m1 = tf.reduce_sum(z1, axis = 0)
        m1 = 1-m1
        new_inp1 = tf.multiply(inp1,m1)
        out = tf.add(new_inp1, update_input)
        return tf.reshape(out, shape)

    def immutable_scatter_nd_1d_update(self, inp1, inp2, inp3):
        """
        Utility function
        """
        shape = inp1.get_shape()
        dim = shape[-1]
        index_shape  = shape[0:-1]
        inp1 = tf.to_float(inp1)
        inp1 = tf.reshape(inp1, [dim, -1])
        inp2 = self.map_index_to_flattened(inp2, index_shape)
        z1 = tf.to_float(tf.one_hot(inp2, inp1.get_shape()[1]))
        z1 = tf.expand_dims(tf.transpose(z1,[1,0]),axis = 2)
        z2 = tf.to_float(tf.reshape(inp3,[-1,dim]))
        z3 = tf.multiply(z2,z1)
        update_input = tf.reduce_sum(z3,axis = 1)

        m1 = tf.reduce_sum(z1, axis = 1)
        m1 = 1-m1
        inp1 = tf.reshape(inp1, [-1, dim])
        new_inp1 = tf.multiply(inp1,m1)
        out = tf.add(new_inp1, update_input)
        return tf.reshape(out, shape)


    def beam_switch(self, old_prop_val, new_beam_ids):
        """
        Utility function for switching beam properties corresponding to new beam ids
        """
        # the matrix should be input in the shape beam_size x batch_size x Tensor_shape
        old_shape = old_prop_val.get_shape()
        old_prop_val = tf.reshape(old_prop_val, [self.beam_size, self.batch_size, -1])
        new_prop_val = []
        expanded_beam_ids = tf.one_hot(new_beam_ids, depth = self.beam_size, dtype = old_prop_val.dtype)
        #expanded_beam_ids has shape beam_size x batch_size x beam_size
        for multiplier in tf.split(expanded_beam_ids, num_or_size_splits=self.beam_size,axis = 0):
            multiplier = tf.transpose(multiplier,[2,1,0])
            new_prop_val.append(tf.reduce_sum(tf.multiply(multiplier,old_prop_val), axis = 0))
        new_prop_val = tf.stack(new_prop_val,axis = 0)
        new_prop_val = tf.reshape(new_prop_val, old_shape)
        return new_prop_val

    def stochastic_program_sampling(self,distribution, k):
        """
        Utility function for randomized operator sampling
        """
        out1_vals, out1_ind = tf.nn.top_k(distribution, k)
        if self.params["explore"][0] is -1:
            return out1_vals, out1_ind
        p = tf.where(tf.greater(tf.random_uniform([1]), self.randomness_threshold_beam_search), [1], [0])

        if self.global_program_indices_matrix is None:
            temp = tf.expand_dims(tf.range(1,self.num_progs), axis=0)
            self.global_program_indices_matrix = tf.tile(temp,[self.batch_size,1])

        curr_shuffled_ind_matrix = tf.random_shuffle(self.global_program_indices_matrix)
        out2_ind = curr_shuffled_ind_matrix[:,0:k]
        multiplicand_1 = tf.one_hot(out2_ind, depth=self.num_progs)
        multiplicand_1 = tf.transpose(multiplicand_1,[1,0,2])
        out2_vals = tf.reduce_sum(tf.multiply(multiplicand_1, distribution), axis=2)
        out2_vals = tf.transpose(out2_vals, [1,0])
        out_ind = p*out1_ind+(1-p)*out2_ind
        p = tf.to_float(p)
        out_vals = p*out1_vals+(1-p)*out2_vals
        return out_vals, out_ind

