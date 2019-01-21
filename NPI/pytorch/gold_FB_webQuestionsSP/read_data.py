import sys
import numpy as np
import json
import cPickle as pkl
import os
import gensim
import collections

class ReadBatchData():

    def __init__(self, param):
        np.random.seed(1)
        self.start_index = 0
        self.end_index = 1
        self.unk_index = 2
        self.pad_index = 3
        self.none_argtype_index = 0
        self.none_argtype = 'none'
        self.none_progtype_index = 0
        self.none_progtype = 'none'
        self.terminate_progtype = 'terminate'
#       self.noop_progtype = 'no-op'
#       self.noop_progtype_index = 1#bla
#       self.noop_argtype = 'none'
#       self.noop_argtype_index = 0
        self.batch_size = param['batch_size']
        self.wikidata_embed_dim = param['freebase_embed_dim']
        self.pad = '<pad>'
        self.pad_kb_symbol_index = 0
        all_questypes = {'infchain_1':'infchain_1',
                           'infchain_1_constrained':'infchain_1_constrained',
                           'infchain_2':'infchain_2',
                           'infchain_2_constrained':'infchain_2_constrained',
			   'infchain_2_constrained_date':'infchain_2_constrained_date',
			   'infchain_1_date':'infchain_1_date',
			   'infchain>2':'infchain>2',
			   'infchain>2_date':'infchain>2_date',
			   'infchain_2_date':'infchain_2_date'}
        self.all_questypes_inv = {v:k for k,v in all_questypes.items()}
        if param['question_type']=='all':	
            self.state_questype_map = all_questypes
	    param['question_type']=','.join(all_questypes.keys())	
        else:
            if not param['questype_wise_batching']:
                raise Exception(' if question_type is not "all", questype_wise_batching should be set to True')
            if ',' not in param['question_type']:
                self.state_questype_map = {self.all_questypes_inv[param['question_type']]:param['question_type']}
            else:
                self.state_questype_map = {self.all_questypes_inv[q]:q for q in param['question_type'].split(',')}
        self.wikidata_ent_embed = np.load(param['freebase_dir']+'/ent_embed_mat.npy').astype(np.float32)
        self.wikidata_ent_vocab = pkl.load(open(param['freebase_dir']+'/ent_id_map.pickle'))
        self.wikidata_ent_vocab_inv = {v:k for k,v in self.wikidata_ent_vocab.items()}
        self.wikidata_rel_embed = np.load(param['freebase_dir']+'/rel_embed_mat.npy').astype(np.float32)
        self.wikidata_rel_date_embed = np.load(param['freebase_dir']+'/rel_date_embed_mat.npy').astype(np.float32)
        self.wikidata_rel_vocab = pkl.load(open(param['freebase_dir']+'/rel_id_map.pickle'))
        self.wikidata_rel_vocab_inv = {v:k for k,v in self.wikidata_rel_vocab.items()}
        self.wikidata_rel_date_vocab = pkl.load(open(param['freebase_dir']+'/rel_date_id_map.pickle'))
        self.wikidata_rel_date_vocab_inv = {v:k for k,v in self.wikidata_rel_date_vocab.items()}
#       self.program_type_vocab = {self.none_progtype:self.none_progtype_index, self.noop_progtype:self.noop_progtype_index}
        self.program_type_vocab = {self.none_progtype:self.none_progtype_index}
        self.argument_type_vocab = {self.none_argtype:self.none_argtype_index}
        self.vocab = pkl.load(open(param['vocab_file'],'rb'))
        self.vocab_size = len(self.vocab)
        self.vocab_init_embed = np.empty([len(self.vocab.keys()), param['text_embed_dim']], dtype=np.float32)
        self.word2vec_pretrain_embed = gensim.models.KeyedVectors.load_word2vec_format(param['glove_dir']+'/GoogleNews-vectors-negative300.bin', binary=True)
        for i in xrange(self.vocab_init_embed.shape[0]):
            if self.vocab[i] in self.word2vec_pretrain_embed:
                self.vocab_init_embed[i,:] = self.word2vec_pretrain_embed[self.vocab[i]]
            elif i==4:
                self.vocab_init_embed[i,:] = np.zeros((1, self.vocab_init_embed.shape[1]), dtype=np.float32)
            else:
                self.vocab_init_embed[i,:] = np.random.rand(1, self.vocab_init_embed.shape[1]).astype(np.float32)
        count = 1
        self.rel_index = None
        self.type_index = None
        for line in open('argument_types.txt').readlines():
            line = line.strip()
            if line not in self.argument_type_vocab:
                self.argument_type_vocab[line] = count
                if line=="relation":
                    self.rel_index = count
                count+=1
        self.num_argtypes = len(self.argument_type_vocab)
        self.max_num_var = param['max_num_var']
        self.required_argtypes_for_responsetype = {}
        self.targettype_prog_map = {}
        self.prog_to_argtypes_map = {}
        program_to_argtype_temp = {}
        program_to_targettype_temp = {}
        count = 1
        for line in open('program_definition.txt').readlines():
            parts = line.strip().split('\t')
	    prog = parts[0]
	    if 'infchain_2' not in param['question_type'] and 'gen_set2' in prog:
                continue
            if 'infchain_1' not in param['question_type'] and 'gen_set1' in prog:
		continue
	    #if all(['_date' in x for x in param['question_type'].split(',')]) and 'dateconstrained' not in prog and ('gen_set2' in prog or 'gen_set1' in prog):
	    #	continue	
	    if '_date' not in param['question_type'] and 'dateconstrained' in prog:
		continue
            argtypes = parts[1].split(',')
            targettype = parts[2]
            if targettype not in self.targettype_prog_map:
                self.targettype_prog_map[targettype] = []
            self.targettype_prog_map[targettype].append(prog)
            self.prog_to_argtypes_map[prog] = argtypes
            if parts[0] not in self.program_type_vocab:
                self.program_type_vocab[parts[0]] = count
                count +=1
            program_to_argtype_temp[self.program_type_vocab[parts[0]]] = [self.argument_type_vocab[a] for a in parts[1].split(',')]
            program_to_targettype_temp[self.program_type_vocab[parts[0]]] = self.argument_type_vocab[parts[2]]
        if param['terminate_prog']:
            self.program_type_vocab[self.terminate_progtype] = len(self.program_type_vocab)
            self.terminate_progtype_index = self.program_type_vocab[self.terminate_progtype]
        self.program_type_vocab_inv = {v:k for k,v in self.program_type_vocab.items()}
        self.argument_type_vocab_inv = {v:k for k,v in self.argument_type_vocab.items()}
        self.num_progs = len(self.program_type_vocab)
        self.max_arguments = max([len(v) for v in program_to_argtype_temp.values()])
	print 'self.max_arguments ',self.max_arguments
        for k,v in program_to_argtype_temp.items():
            v = v[:min(self.max_arguments, len(v))]+[self.none_argtype_index]*max(0, self.max_arguments-len(v))
            program_to_argtype_temp[k] = v
        self.program_to_argtype = {k:[self.none_argtype_index]*self.max_arguments for k in self.program_type_vocab.values()}
        self.program_to_targettype = {k:self.none_argtype_index for k in self.program_type_vocab.values()}
        self.program_to_argtype.update(program_to_argtype_temp)
        self.program_to_targettype.update(program_to_targettype_temp)
        self.program_to_argtype = np.asarray(collections.OrderedDict(sorted(self.program_to_argtype.items())).values())
        self.program_to_targettype = np.asarray(collections.OrderedDict(sorted(self.program_to_targettype.items())).values())
        self.program_algorithm_phase = [self.program_type_vocab[x.strip()] for x in open('program_algorithm.txt').readlines()  if x.strip() in self.program_type_vocab]
        self.program_algorithm_phase.append(self.none_progtype_index)
        if param['terminate_prog']:
            self.program_algorithm_phase.append(self.terminate_progtype_index)
        self.program_variable_declaration_phase = [x for x in self.program_type_vocab.values() if x not in self.program_algorithm_phase]
        self.program_variable_declaration_phase.append(self.none_progtype_index)
	
	print 'self.argument_type_vocab ', self.argument_type_vocab
	print 'self.program_type_vocab ', self.program_type_vocab
	print 'self.program_algorithm_phase ', self.program_algorithm_phase
        print 'finished init: read data'

    def get_response_type(self, question):
        if any([question.lower().startswith(x) for x in ['when','what time','what year', 'what date']]):
            return self.argument_type_vocab['date']
        else:
            return self.argument_type_vocab['entity']

    def get_data_per_questype(self, data_dict):
        data_dict_questype = {}
        for d in data_dict:
            state = d[10]
            if state in self.state_questype_map:
                if self.state_questype_map[state] not in data_dict_questype:
                    data_dict_questype[self.state_questype_map[state]] = []
                data_dict_questype[self.state_questype_map[state]].append(d)
	for k,v in data_dict_questype.items():
		print k, ':::', len(v)
        return data_dict_questype

    def get_all_required_argtypes(self, type, reqd_argtypes):
        if type in self.targettype_prog_map:
            progs = self.targettype_prog_map[type]
            for prog in progs:
                argtypes = self.prog_to_argtypes_map[prog]
                for k,v in collections.Counter(argtypes).items():
                    if k in reqd_argtypes:
                        reqd_argtypes[k] = max(reqd_argtypes[k], v)
                    else:
                        reqd_argtypes[k] = v
                        reqd_argtypes = self.get_all_required_argtypes(k, reqd_argtypes)
        return reqd_argtypes

    def get_all_required_argtypes_matrix(self, reqd_argtypes):
        reqd_argtypes_mat = np.zeros((self.num_argtypes), dtype=np.float32)
        for k in reqd_argtypes:
            reqd_argtypes_mat[self.argument_type_vocab[k]] = float(reqd_argtypes[k])
        return reqd_argtypes_mat

    def get_batch_data(self, data_dict):
        num_data = len(data_dict)
        batch_orig_context = [data_dict[i][0] for i in range(num_data)]
        batch_context_nonkb_words = [data_dict[i][1] for i in range(num_data)]
        batch_context_kb_words = [data_dict[i][2] for i in range(num_data)]
        batch_context_kb_words = np.asarray(batch_context_kb_words)
        batch_context_kb_words[batch_context_kb_words==1]=0
        batch_context_entities = [data_dict[i][3] for i in range(num_data)]
        entity_variable_value_table = [[self.wikidata_ent_vocab_inv[e] if e!=self.pad_kb_symbol_index else \
                                        None for e in batch_context_entities[i]] for i in range(num_data)]
        batch_context_rel = [data_dict[i][4] for i in range(num_data)]
        rel_variable_value_table = [[self.wikidata_rel_vocab_inv[r] if r!=self.pad_kb_symbol_index else None \
                                     for r in batch_context_rel[i]] for i in range(num_data)]
        batch_context_rel_dates = [data_dict[i][5] for i in range(num_data)]
        rel_date_variable_value_table = [[self.wikidata_rel_date_vocab_inv[r] if r!=self.pad_kb_symbol_index else None \
                                     for r in batch_context_rel_dates[i]] for i in range(num_data)]
        batch_context_dates = [data_dict[i][6] for i in range(num_data)]
	#batch_context_dates = [[0]*5 for i in range(num_data)]
        batch_orig_response = [data_dict[i][7] for i in range(num_data)]
        batch_response_entities = [[''.join(i for i in x if ord(i)<128) for x in data_dict[i][8]] for i in range(num_data)]
        batch_response_dates = [data_dict[i][9] for i in range(num_data)]
        batch_questype = [data_dict[i][10] for i in range(num_data)]
        batch_ent_rel_kb_subgraph = [data_dict[i][11] for i in range(num_data)]
        batch_ent_rel_date_kb_subgraph = [data_dict[i][12] for i in range(num_data)]
	batch_ent_rel_rel_kb_subgraph = [data_dict[i][13] for i in range(num_data)]
	batch_ent_rel_rel_date_kb_subgraph = [data_dict[i][14] for i in range(num_data)]
	batch_ent_rel_rel_dc_kb_subgraph = [data_dict[i][15] for i in range(num_data)]
	batch_ent_rel_rel_date_dc_kb_subgraph = [data_dict[i][16] for i in range(num_data)]
        ques_id = [data_dict[i][17] for i in range(num_data)]
        batch_response_type = []
        batch_required_argtypes = []
        for i in range(num_data):
            response_type = self.get_response_type(batch_orig_context[i])
            if self.argument_type_vocab_inv[response_type] not in self.required_argtypes_for_responsetype:
                required_argtypes = self.get_all_required_argtypes(self.argument_type_vocab_inv[response_type], {})
                self.required_argtypes_for_responsetype[self.argument_type_vocab_inv[response_type]] = required_argtypes
            else:
                required_argtypes = self.required_argtypes_for_responsetype[self.argument_type_vocab_inv[response_type]]

            required_argtypes_mat = self.get_all_required_argtypes_matrix(required_argtypes)
            batch_required_argtypes.append(required_argtypes_mat)
            batch_response_type.append(response_type)

        batch_context_nonkb_words = np.asarray([[xij for xij in context_words] for context_words in batch_context_nonkb_words])
        batch_context_kb_words = np.asarray([[self.wikidata_ent_embed[xij] for xij in context_words] for context_words in batch_context_kb_words])

        batch_context_entities = np.asarray([[xij for xij in context_entities] for context_entities in batch_context_entities])
        batch_context_rel = np.asarray([[xij for xij in context_rel] for context_rel in batch_context_rel])
        batch_context_rel_dates = np.asarray([[xij for xij in context_rel_date] for context_rel_date in batch_context_rel_dates])
        if not all([len(x)==self.max_num_var for x in batch_context_entities]):
            raise Exception(str([len(x) for x in batch_context_entities]))
        if not all([len(x)==self.max_num_var for x in batch_context_rel]):
            raise Exception(str([len(x) for x in batch_context_rel]))
        if not all([len(x)==self.max_num_var for x in batch_context_rel_dates]):
            raise Exception(str([len(x) for x in batch_context_rel_dates]))
        date_variable_value_table = [[i if i!=self.pad_kb_symbol_index else None for i in dates] for dates in batch_context_dates]
        batch_response_entities = [[str(xij) for xij in response_entities] for response_entities in batch_response_entities]
        batch_context_nonkb_words = np.transpose(batch_context_nonkb_words, (1,0))
        variable_mask, variable_embed, variable_atten = self.get_variable_table_data(batch_context_entities, batch_context_rel, batch_context_rel_dates, batch_context_dates)
        #self.debug_kb_attention(batch_ent_rel_type_kb_subgraph, batch_context_entities, batch_context_rel, batch_context_types, self.wikidata_ent_vocab_inv, self.wikidata_rel_vocab_inv, self.wikidata_type_vocab_inv, '(e,r,t)')
        #self.debug_kb_attention(batch_type_rel_type_kb_subgraph, batch_context_types, batch_context_rel, batch_context_types, self.wikidata_type_vocab_inv, self.wikidata_rel_vocab_inv, self.wikidata_type_vocab_inv, '(t,r,t)')
        kb_attention_for_progs = self.get_kb_attention(batch_ent_rel_kb_subgraph, batch_ent_rel_date_kb_subgraph, batch_ent_rel_rel_kb_subgraph, batch_ent_rel_rel_date_kb_subgraph, batch_ent_rel_rel_dc_kb_subgraph, batch_ent_rel_rel_date_dc_kb_subgraph)
        variable_value_table = self.get_variable_value_table(entity_variable_value_table, rel_variable_value_table, rel_date_variable_value_table, date_variable_value_table)
        variable_value_table = np.transpose(np.asarray(variable_value_table), (1,0,2))
        #variable_value_table is of dimension batch_size x num_argtypes x max_num_var
        batch_response_type = np.asarray(batch_response_type)
        return batch_orig_context, batch_context_nonkb_words, batch_context_kb_words, \
        batch_context_entities, batch_context_rel, batch_context_rel_dates, batch_context_dates, \
        batch_orig_response, batch_response_entities, batch_response_dates,  batch_response_type, batch_required_argtypes, \
        variable_mask, variable_embed, variable_atten, kb_attention_for_progs, variable_value_table, ques_id

    def debug_kb_attention(self, kb_attention, arg1, arg2, arg3, vocab1, vocab2, vocab3, type):
        for i in range(len(kb_attention)):
            kb_attention_i = np.reshape(kb_attention[i], (self.max_num_var, self.max_num_var, self.max_num_var, self.max_num_var, self.max_num_var))
            for i1 in range(self.max_num_var):
                for i2 in range(self.max_num_var):
                    for i3 in range(self.max_num_var):
			for i4 in range(self.max_num_var):
				for i5 in range(self.max_num_var):
		                        if kb_attention_i[i1][i2][i3][i4][i5]==1.0:
                          	  		print 'batch id ', i, ':: kb attention 1.0 for ',type,' = (',vocab1[arg1[i][i1]],',',vocab2[arg2[i][i2]],',',vocab3[arg3[i][i3]],')'
        print ''

    def get_variable_value_table(self, entity_variable_value_table, rel_variable_value_table, rel_date_variable_value_table, date_variable_value_table):
#        variable_value_table = [[['']*self.max_num_var]*self.batch_size]*self.num_argtypes
        variable_value_table = [[[None]*self.max_num_var]*self.batch_size]*self.num_argtypes
        for v_type, v_type_index in self.argument_type_vocab.items():
            if v_type=="entity":
                variable_value_table[v_type_index] = entity_variable_value_table
            elif v_type=="relation":
                variable_value_table[v_type_index] = rel_variable_value_table
            elif v_type=="relation_date":
                variable_value_table[v_type_index] = rel_date_variable_value_table
            elif v_type=="date":
                variable_value_table[v_type_index] = date_variable_value_table
        return variable_value_table

    def get_kb_attention(self, batch_ent_rel_kb_subgraph, batch_ent_rel_date_kb_subgraph, batch_ent_rel_rel_kb_subgraph, batch_ent_rel_rel_date_kb_subgraph, batch_ent_rel_rel_dc_kb_subgraph, batch_ent_rel_rel_date_dc_kb_subgraph):
        kb_attention_for_progs = [None]*self.num_progs
        batch_ent_rel_kb_subgraph = np.asarray(batch_ent_rel_kb_subgraph).reshape((self.batch_size, self.max_num_var, self.max_num_var, self.max_num_var, self.max_num_var, self.max_num_var))
	batch_ent_rel_rel_kb_subgraph = np.asarray(batch_ent_rel_rel_kb_subgraph).reshape((self.batch_size, self.max_num_var, self.max_num_var, self.max_num_var, self.max_num_var, self.max_num_var))
	batch_ent_rel_rel_dc_kb_subgraph = np.asarray(batch_ent_rel_rel_dc_kb_subgraph).reshape((self.batch_size, self.max_num_var, self.max_num_var, self.max_num_var, self.max_num_var, self.max_num_var))

	if 'gen_set2_dateconstrained' not in self.program_type_vocab and 'gen_set2' not in self.program_type_vocab:
		batch_ent_rel_kb_subgraph = batch_ent_rel_kb_subgraph[:,:,:,0,0,0]
	elif 'gen_set2_dateconstrained' not in self.program_type_vocab:
                batch_ent_rel_kb_subgraph = batch_ent_rel_kb_subgraph[:,:,:,:,0,0]
		batch_ent_rel_rel_kb_subgraph = batch_ent_rel_rel_kb_subgraph[:,:,:,:,0,0]
		
        batch_ent_rel_date_kb_subgraph = np.asarray(batch_ent_rel_date_kb_subgraph).reshape((self.batch_size, self.max_num_var, self.max_num_var, self.max_num_var, self.max_num_var, self.max_num_var))
	batch_ent_rel_rel_date_kb_subgraph = np.asarray(batch_ent_rel_rel_date_kb_subgraph).reshape((self.batch_size, self.max_num_var, self.max_num_var, self.max_num_var, self.max_num_var, self.max_num_var))
	batch_ent_rel_rel_date_dc_kb_subgraph = np.asarray(batch_ent_rel_rel_date_dc_kb_subgraph).reshape((self.batch_size, self.max_num_var, self.max_num_var, self.max_num_var, self.max_num_var, self.max_num_var))
	if 'gen_set2_dateconstrained' not in self.program_type_vocab and 'gen_set2' not in self.program_type_vocab:
		batch_ent_rel_date_kb_subgraph = batch_ent_rel_date_kb_subgraph[:,:,:,0,0,0]	
	elif 'gen_set2_dateconstrained' not in self.program_type_vocab:
		batch_ent_rel_date_kb_subgraph = batch_ent_rel_date_kb_subgraph[:,:,:,:,0,0]	
		batch_ent_rel_rel_date_kb_subgraph = batch_ent_rel_rel_date_kb_subgraph[:,:,:,:,0,0]
        for prog,prog_id in self.program_type_vocab.items():
	    if 'gen_set2_dateconstrained' in self.program_type_vocab:
			kb_attention = np.ones((self.batch_size, self.max_num_var, self.max_num_var, self.max_num_var, self.max_num_var, self.max_num_var), dtype=np.float32)
	    elif 'gen_set2' in self.program_type_vocab:	
		        kb_attention = np.ones((self.batch_size, self.max_num_var, self.max_num_var, self.max_num_var), dtype=np.float32)
	    else:
		kb_attention = np.ones((self.batch_size, self.max_num_var, self.max_num_var), dtype=np.float32)		
	    #print 'Printing all shapes ', batch_ent_rel_kb_subgraph.shape, batch_ent_rel_date_kb_subgraph.shape, batch_ent_rel_rel_kb_subgraph.shape, batch_ent_rel_rel_date_kb_subgraph.shape, batch_ent_rel_rel_dc_kb_subgraph.shape, batch_ent_rel_rel_date_dc_kb_subgraph.shape
            if prog=="gen_set1":
                kb_attention = np.asarray(batch_ent_rel_kb_subgraph)
            elif prog=="gen_set1_date":
                kb_attention = np.asarray(batch_ent_rel_date_kb_subgraph)
	    elif prog=="gen_set2":
		kb_attention = np.asarray(batch_ent_rel_rel_kb_subgraph)
	    elif prog=="gen_set2_date":
		kb_attention = np.asarray(batch_ent_rel_rel_date_kb_subgraph)
	    elif prog=="gen_set2_dateconstrained":
		kb_attention = np.asarray(batch_ent_rel_rel_dc_kb_subgraph)
	    elif prog=="gen_set2_date_dateconstrained":
		kb_attention = np.asarray(batch_ent_rel_rel_date_dc_kb_subgraph)				
	    elif prog=="set_oper_ints":
		if 'gen_set2' not in self.program_type_vocab and 'gen_set2_dateconstrained' not in self.program_type_vocab:
			for i in range(self.max_num_var):
                            kb_attention[:,i,i] = 0.
		elif 'gen_set2_dateconstrained' in self.program_type_vocab:
			for i in range(self.max_num_var):
                            kb_attention[:,i,i,:,:,:] = 0.
		elif 'gen_set2' in self.program_type_vocab:
			for i in range(self.max_num_var):
                            kb_attention[:,i,i,:] = 0.
            kb_attention = np.reshape(kb_attention, (self.batch_size, -1))
            kb_attention_for_progs[prog_id]=kb_attention
        kb_attention_for_progs = np.asarray(kb_attention_for_progs)
        kb_attention_for_progs = np.transpose(kb_attention_for_progs, [1,0,2])
        return kb_attention_for_progs

    def get_variable_table_data(self, batch_context_entities, batch_context_rels, batch_context_rel_dates, batch_context_dates):
        variable_mask = np.zeros((self.num_argtypes, self.batch_size, self.max_num_var), dtype=np.float32)
        variable_embed = np.zeros((self.num_argtypes, self.batch_size, self.max_num_var, self.wikidata_embed_dim), \
                                  dtype=np.float32)
        variable_atten = np.zeros((self.num_argtypes, self.batch_size, self.max_num_var), dtype=np.float32)
        #attention will have some issue if no variables are there to begin with ??????? ####CHECK
        for v_type, v_type_index in self.argument_type_vocab.items():
            if v_type=="entity":
                mask = np.ones_like(batch_context_entities, dtype=np.float32)
                mask[batch_context_entities==self.pad_kb_symbol_index]=0.0
                #print 'for entity, mask : ', mask
                num_entities = np.sum(mask, axis=1)
                num_entities[num_entities==0] = 1e-5
                #num_entities is of dimension batch_size
                num_entities=np.tile(np.reshape(num_entities,(-1,1)), (1,self.max_num_var))
                embed = self.wikidata_ent_embed[batch_context_entities]
                #embed is of dimension batch_size x max_num_var x wikidata_embed_dim
                atten = np.copy(mask)
                atten = np.divide(atten, num_entities)
            elif v_type=="relation":
                mask = np.ones_like(batch_context_rels, dtype=np.float32)
                mask[batch_context_rels==self.pad_kb_symbol_index]=0.0
                #print 'for relation, mask : ', mask
                num_relations = np.sum(mask, axis=1)
                num_relations[num_relations==0] = 1e-5
                #num_relations is of dimension batch_size
                num_relations = np.tile(np.reshape(num_relations,(-1,1)), (1,self.max_num_var))
                embed = self.wikidata_rel_embed[batch_context_rels]
                #embed is of dimension batch_size x max_num_var x wikidata_embed_dim
                atten = np.copy(mask)
                atten = np.divide(atten, num_relations)
            elif v_type=="relation_date":
                mask = np.ones_like(batch_context_rel_dates, dtype=np.float32)
                mask[batch_context_rel_dates==self.pad_kb_symbol_index]=0.0
                #print 'for types, mask : ', mask
                num_relation_dates = np.sum(mask, axis=1)
                num_relation_dates[num_relation_dates==0]=1e-5
                #num_types is of dimension batch_size
                num_relation_dates = np.tile(np.reshape(num_relation_dates,(-1,1)), (1,self.max_num_var))
                embed = self.wikidata_rel_date_embed[batch_context_rel_dates]
                #embed is of dimension batch_size x max_num_var x wikidata_embed_dim
                atten = np.copy(mask)
                atten = np.divide(atten, num_relation_dates)
            elif v_type=="date":
                mask = np.ones_like(batch_context_dates, dtype=np.float32)
                #print 'batch_context_ints==0', batch_context_ints==self.pad_kb_symbol_index
                mask[batch_context_dates==self.pad_kb_symbol_index]=0.0
                embed = np.zeros((self.batch_size, self.max_num_var, self.wikidata_embed_dim), dtype=np.float32)
                num_dates = np.sum(mask, axis=1)
                num_dates[num_dates==0]=1e-5
                #num_ints is of dimension batch_size
                num_dates = np.tile(np.reshape(num_dates,(-1,1)), (1, self.max_num_var))
                #embed is of dimension batch_size x max_num_prepro_var x wikidata_embed_dim
                atten = np.copy(mask)
                atten = np.divide(atten, num_dates)
            else:
                mask = np.zeros_like((self.batch_size), dtype=np.float32)
                embed = np.zeros((self.batch_size, self.max_num_var, self.wikidata_embed_dim), dtype=np.float32)
                atten = np.copy(mask)
            variable_mask[v_type_index]=mask
            variable_embed[v_type_index]= embed
            variable_atten[v_type_index]=atten
        variable_mask[0,:,0] = 1.
        variable_mask = np.transpose(variable_mask, [0,2,1])
        #print 'variable mask for entities', variable_mask[self.argument_type_vocab['entity']]
        #print 'variable mask for relations', variable_mask[self.argument_type_vocab['relation']]
        #print 'variable mask for types', variable_mask[self.argument_type_vocab['type']]
        variable_embed = np.transpose(variable_embed, [0,2,1,3])
        variable_atten = np.transpose(variable_atten, [0,2,1])
        #print 'variable mask in readdata ', variable_mask
        return variable_mask, variable_embed, variable_atten


if __name__=="__main__":
    param = json.load(open(sys.argv[1]))
    if os.path.isdir(param['train_data_file']):
        training_files = [param['train_data_file']+'/'+x for x in os.listdir(param['train_data_file']) if x.endswith('.pkl')]
    elif not isinstance(param['train_data_file'], list):
        training_files = [param['train_data_file']]
    else:
        training_files = param['train_data_file']

    data = []
    for f in training_files:
        data.extend(pkl.load(open(f)))
    param['batch_size'] = len(data)
    read_data = ReadBatchData(param)
    read_data.get_batch_data(data)
