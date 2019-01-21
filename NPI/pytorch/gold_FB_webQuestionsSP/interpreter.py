import numpy as np
import msgpack
import datetime
import json

class Interpreter():
    def __init__(self, freebase_dir, num_timesteps, program_type_vocab, argument_type_vocab, printing, terminate_prog, relaxed_reward_strict, reward_function="jaccard", boolean_reward_multiplier = 0.1, relaxed_reward_till_epoch=[-1,-1], unused_var_penalize_after_epoch=[1000, 100000], length_based_penalization=False):
        np.random.seed(1)
        self.freebase_kb = json.load(open(freebase_dir+'/webQSP_freebase_subgraph.json'))
        self.argument_type_vocab = argument_type_vocab
        self.printing = printing
        self.terminate_prog = terminate_prog
        self.argument_type_vocab_inv = {v:k for k,v in self.argument_type_vocab.items()}
        self.program_type_vocab = program_type_vocab
        self.length_based_penalization = length_based_penalization
        self.relaxed_reward_strict = relaxed_reward_strict
        self.relaxed_reward_till_epoch = relaxed_reward_till_epoch
        self.unused_var_penalize_after_epoch = unused_var_penalize_after_epoch
        self.reward_function = reward_function
        if self.reward_function not in ["jaccard","recall", "f1"]:
            raise Exception('reward function must be either jaccard or recall or f1')
        self.boolean_reward_multiplier = boolean_reward_multiplier
        self.program_type_vocab_inv = {v:k for k,v in self.program_type_vocab.items()}
        self.map_program_to_func = {}
        self.map_program_to_func["gen_set1"]=self.execute_gen_set1
        self.map_program_to_func["gen_set2"]=self.execute_gen_set2
        self.map_program_to_func["gen_set1_date"]=self.execute_gen_set1
        self.map_program_to_func["gen_set2_date"]=self.execute_gen_set2_date
        self.map_program_to_func["select_oper_date_lt"]=self.execute_select_oper_date_lt
        self.map_program_to_func["select_oper_date_gt"]=self.execute_select_oper_date_gt
	self.map_program_to_func["gen_set2_dateconstrained"]=self.execute_gen_set2_dateconstrained
	self.map_program_to_func["gen_set2_date_dateconstrained"]=self.execute_gen_set2_date_dateconstrained
        self.map_program_to_func["set_oper_ints"]=self.execute_set_oper_ints
        self.map_program_to_func["none"]=self.execute_none
        self.map_program_to_func["terminate"]=self.execute_terminate
        self.HIGH_NEGATIVE_REWARD = 1
        self.HIGHEST_NEGATIVE_REWARD = 1
        self.num_timesteps = num_timesteps
        self.rewards = None
        self.parallel = 0
    def calculate_program_reward(self, args, epoch_number, overall_step_number):
        if self.parallel == 1:
            inputs, gold_entities, gold_ints, gold_bools, beam_id, batch_id = msgpack.unpackb(args, raw=False)
            inputs['variable_value_table'] = np.array(inputs['variable_value_table'], dtype = np.object)
        else:
            inputs, gold_entities, gold_dates = args
            gold_dates = [self.convert_to_date(x) for x in gold_dates]
        return self.execute_multiline_program(inputs, gold_entities, gold_dates, epoch_number, overall_step_number)

    def execute_multiline_program(self,inputs, gold_entities, gold_dates, epoch_number, overall_step_number):
        reward = 0.
        flag = 0
        self.FLAG = dict.fromkeys(['Sampling_from_empty_table','All_none_programs','Repeated_lines_of_code','First_program_none','First_program_terminate','executed_line_output_none'])
        self.FLAG['Sampling_from_empty_table'] = False
        self.FLAG['All_none_programs'] = True
        self.FLAG['First_program_none'] = False
        self.FLAG['First_program_terminate'] = False
        self.FLAG['executed_line_output_none'] = False
        variable_value_table = inputs['variable_value_table']
        if self.printing:
            print 'number of entities: ', len(variable_value_table[self.argument_type_vocab['entity']]) - (variable_value_table[self.argument_type_vocab['entity']]==None).sum()
            print 'number of relations: ', len(variable_value_table[self.argument_type_vocab['relation']]) - (variable_value_table[self.argument_type_vocab['relation']]==None).sum()
            print 'number of relation_dates: ', len(variable_value_table[self.argument_type_vocab['relation_date']]) - (variable_value_table[self.argument_type_vocab['relation_date']]==None).sum()
            print 'number of dates: ', len(variable_value_table[self.argument_type_vocab['date']]) - (variable_value_table[self.argument_type_vocab['date']]==None).sum()
        if self.parallel is not 1:
            program_probability = inputs['program_probability']
            if self.printing:
                print 'program probability', program_probability
            target_value = None
            target_type_id = None
            for k,v in inputs.items():
                if k!='variable_value_table' and k!='program_probability' and len(v)!=self.num_timesteps:
                    raise Exception('Length of '+k+' i.e. '+str(len(v))+' not!= num_timesteps i.e. '+str(self.num_timesteps))
#       variable_value_table = [] #list of len num_argtypes and each of the lists are of len max_num_var
        #variable_value_table contains the entity, relations, types, integers identifiied in the query in the preprocessing step. For the rest of the values, it is None
        lines_of_code_written = []
        num_lines_of_code = 0
        variables_unused = []
        cumulative_flag = 0.0
        max_intermediate_reward = -100000.0
        best_intermediate_timestep = -1
        #print 'variable_value_table ',variable_value_table
        for i in range(self.num_timesteps):
            program_type_id = inputs['program_type'][i]
            program_type = self.program_type_vocab_inv[program_type_id]
            argument_type_id = inputs['argument_type'][i]
            argument_type = [self.argument_type_vocab_inv[id] for id in argument_type_id]
            argument_table_index = inputs['argument_table_index'][i]
            if program_type_id !=0:
                line_of_code = str(program_type_id) + "::"
            else:
                line_of_code = ''
            argument_values = []
            argument_location = [str(arg_type_i)+'_'+str(arg_index_i) for arg_type_i, arg_index_i in zip(argument_type_id, argument_table_index)]
            for arg_type_i, arg_index_i in zip(argument_type_id, argument_table_index):
                if (arg_type_i, arg_index_i) in variables_unused:
                    variables_unused.remove((arg_type_i, arg_index_i))
                arg_variable_value_i = variable_value_table[arg_type_i][arg_index_i]
                if arg_variable_value_i is None:
                    if arg_type_i!=0:
                        if self.printing:
                            print 'FLAG: Sampling_from_empty_table'
                        self.FLAG['Sampling_from_empty_table'] = True
                        break
                argument_values.append(arg_variable_value_i)
                if program_type_id !=0:
                    line_of_code += str(arg_type_i)+"("+str(arg_index_i)+") "
            if self.FLAG['Sampling_from_empty_table']:
                break
            if 'terminate' in self.program_type_vocab and program_type_id == self.program_type_vocab['terminate']:
                if i==0:
                    if self.printing:
                        print 'FLAG: First_program_terminate'
                    self.FLAG['First_program_terminate'] = True
		else:
		    if any([x>0 for x in argument_table_index]):
			self.FLAG['FLAG: None/Terminate with bad argument'] = True			
                break
	    if 'none' in self.program_type_vocab and program_type_id == self.program_type_vocab['none']:
		self.FLAG['None/Terminate with bad argument'] = True
		
            if program_type_id != self.program_type_vocab['none']:# and ('terminate' not in self.program_type_vocab or program_type_id!=self.program_type_vocab['terminate']):
                if line_of_code in lines_of_code_written:
                    if self.printing:
                        print 'FLAG: Repeated_lines_of_code'
                    self.FLAG['Repeated_lines_of_code'] = True
                else:
                    lines_of_code_written.append(line_of_code)
                self.FLAG['All_none_programs'] = False
                num_lines_of_code += 1
                target_type_id = inputs['target_type'][i]
                target_type = self.argument_type_vocab_inv[target_type_id]
                target_table_index = inputs['target_table_index'][i]
		if self.printing:
                	print 'Invoking ', program_type, '(',
	                for arg_val in argument_values:
        	            if isinstance(arg_val, list) or isinstance(arg_val, set):
                	        if len(arg_val)>3:
                        	    print '['+','.join(list(arg_val)[:3])+' ...]',
	                        else:    
        	                    print '['+','.join(arg_val)+']',
                	    else:
                        	print arg_val,
	                    print ',',    
        	        print ')'
	
                target_value, per_step_flag = self.execute_singleline_program(program_type, argument_values, argument_location)
                variable_value_table[target_type_id][target_table_index] = target_value
                variables_unused.append((target_type_id,target_table_index))
                cumulative_flag += per_step_flag
                if target_value is None:
                    if self.printing:
                        print 'FLAG: executed_line_output_none'
                    self.FLAG['executed_line_output_none'] = True
                else:
                    if i<self.num_timesteps-1:
                        reward_till_timestep, _ = self.calculate_reward(target_value, target_type_id, cumulative_flag, num_lines_of_code, gold_entities, gold_dates, epoch_number, overall_step_number, False)
                        if reward_till_timestep>0 and reward_till_timestep>max_intermediate_reward:
                            max_intermediate_reward = reward_till_timestep
                            best_intermediate_timestep = i+1
            elif i==0:
                if self.printing:
                    print 'FLAG: First_program_none'
                self.FLAG['First_program_none'] = True

        if len(variables_unused)>1 and (epoch_number>=self.unused_var_penalize_after_epoch[0] or overall_step_number>=self.unused_var_penalize_after_epoch[1]):
            self.FLAG['Unused_variable']=1
            flag = 0.1*(len(variables_unused)-1)
        if self.FLAG['All_none_programs'] or self.FLAG['First_program_none'] or self.FLAG['First_program_terminate']:
            target_value = None
            flag = self.HIGHEST_NEGATIVE_REWARD
            num_lines_of_code = 0
            reward = -flag
            relaxed_reward = reward
        elif self.FLAG['Sampling_from_empty_table'] or self.FLAG['Repeated_lines_of_code']:
            target_value = None
            target_type_id = None
            flag = self.HIGHEST_NEGATIVE_REWARD
            reward = -flag
            relaxed_reward = reward
        elif self.FLAG['executed_line_output_none'] or target_value is None:
            flag = self.HIGHEST_NEGATIVE_REWARD
            reward = 0.0
            relaxed_reward = reward
        else:
	    if len(target_value)==0:
		target_value = None	
            flag = cumulative_flag
	    if target_value is None:
		reward = 0.0
		relaxed_reward = 1.0
	    else:
            	reward, relaxed_reward = self.calculate_reward(target_value, target_type_id, flag, num_lines_of_code, gold_entities, gold_dates, epoch_number, overall_step_number)
	#if self.FLAG['None/Terminate with bad argument']:
        #    target_value = None
        #    target_type_id = None
        #    flag = 0
        #    reward = 0.0
        if max_intermediate_reward < 0 or max_intermediate_reward<reward:
            intermediate_reward_flag = 0
            intermediate_mask = [1.]*self.num_timesteps
        else:
            intermediate_reward_flag = 1
            intermediate_mask = [1.]*best_intermediate_timestep+[0.]*(self.num_timesteps-best_intermediate_timestep)
            if len(intermediate_mask)!=self.num_timesteps:
                raise Exception('len(intermediate_mask)!=self.num_timesteps')
            if self.printing:
                print 'found intermediate reward of ', max_intermediate_reward, ' at timestep ', best_intermediate_timestep
        return target_value, reward, max_intermediate_reward, relaxed_reward, intermediate_mask, intermediate_reward_flag


    def calculate_reward(self, target_value, target_type_id, flag, num_lines_of_code, gold_entities, gold_dates, epoch_number, overall_step_number, print_flag = True):
        reward = 0
        relaxed_reward = 0
        if not self.length_based_penalization:
            num_lines_of_code = 0.0
        reward -= num_lines_of_code*0.001
        gold_type = ''
        if len(gold_dates)>0:
            gold_entities = []
            gold_type = 'set_date'
        elif len(gold_entities)>0:
            gold_dates = []
            gold_type = 'set_ent'
        if self.argument_type_vocab_inv[target_type_id]=='set_ent':
            target_value = set([str(x) for x in target_value])
            gold_entities_set = set(gold_entities)
            intersec = target_value.intersection(gold_entities_set)
            if len(gold_entities_set)!=0:
                if epoch_number <= self.relaxed_reward_till_epoch[0] and overall_step_number <= self.relaxed_reward_till_epoch[1]:
                    relaxed_reward += 1.
                if self.reward_function=="jaccard":
                    union = set([])
                    union.update(target_value)
                    union.update(gold_entities_set)
                    reward += float(len(intersec))/float(len(union))
                elif self.reward_function=="recall":
                    reward += float(len(intersec))/float(len(gold_entities_set))
                elif self.reward_function=="f1":
                    if len(target_value)==0:
                        prec = 0.0
                    else:
                        prec = float(len(intersec))/float(len(target_value))
                    rec = float(len(intersec))/float(len(gold_entities_set))
                    if prec==0 and rec==0:
                        reward += 0
                    else:
                        reward += (2.0*prec*rec)/(prec+rec)
            else:
                if epoch_number > self.relaxed_reward_till_epoch[0] or overall_step_number > self.relaxed_reward_till_epoch[1]:
                    reward += -self.HIGH_NEGATIVE_REWARD
            if print_flag:
                if self.printing:
                    print 'In calculate reward: predicted_answer=',self.argument_type_vocab_inv[target_type_id],', gold_answer=',gold_type, ' reward =', reward, ' flag = ', flag, ' relaxed_reward =', relaxed_reward
                else:
                    print 'reward =', reward, ' flag = ', flag, ' relaxed_reward =', relaxed_reward
        elif self.argument_type_vocab_inv[target_type_id]=='set_date':
            target_value = set([self.convert_to_date(x) for x in target_value])
            gold_dates = set(gold_dates)
            intersec = target_value.intersection(gold_dates)
            if len(gold_dates)!=0:
                if epoch_number <= self.relaxed_reward_till_epoch[0] and overall_step_number <= self.relaxed_reward_till_epoch[1]:
                    relaxed_reward += 1.
                if self.reward_function=="jaccard":
                    union = set([])
                    union.update(target_value)
                    union.update(gold_dates)
                    reward += float(len(intersec))/float(len(union))
                elif self.reward_function=="recall":
                    reward += float(len(intersec))/float(len(gold_dates))
                elif self.reward_function=="f1":
                    if len(target_value)==0:
                        prec = 0.0
                    else:
                        prec = float(len(intersec))/float(len(target_value))
                    rec = float(len(intersec))/float(len(gold_dates))
                    if prec==0 and rec==0:
                        reward += 0
                    else:
                        reward += (2.0*prec*rec)/(prec+rec)
            else:
                if epoch_number > self.relaxed_reward_till_epoch[0] or overall_step_number > self.relaxed_reward_till_epoch[1]:
                    reward += -self.HIGH_NEGATIVE_REWARD
            if print_flag:
                if self.printing:
                    print 'In calculate reward: predicted_answer=',self.argument_type_vocab_inv[target_type_id],', gold_answer=',gold_type, 'reward=', reward, ' flag = ', flag, ' relaxed_reward =', relaxed_reward
                else:
                    print 'reward =', reward, ' flag = ', flag, ' relaxed_reward =', relaxed_reward
        else:
            reward += -self.HIGH_NEGATIVE_REWARD
            relaxed_reward += -self.HIGH_NEGATIVE_REWARD
            if print_flag:
                if self.printing:
                    print 'In calculate reward: predicted_answer=', self.argument_type_vocab_inv[target_type_id], 'gold_answer=',gold_type,' reward =',reward, ' relaxed_reward =', -1
                else:
                    print 'reward=', reward
        reward = reward - flag
        relaxed_reward = relaxed_reward - flag
        return reward, relaxed_reward

    def is_kb_consistent(self, e, r):
        if e in self.freebase_kb and r in self.freebase_kb[e]:
            return True
        else:
            return False
        
    def execute_singleline_program(self,program_type, argument_values, argument_location):
        # debugging ########################################
        if program_type=='none':
            target_value = None
            func_name = None
            flag = None
        else:
            func = self.map_program_to_func[program_type]
            target_value, flag = func(argument_values, argument_location)
            func_name = func.__name__
        return target_value, flag

    def execute_gen_set1(self, argument_value, argument_location):
        entity = argument_value[0]
        relation = argument_value[1]
        if entity is None or relation is None:
            return set([]), 1
        tuple_set = None
        if entity in self.freebase_kb and relation in self.freebase_kb[entity]:
            tuple_set = self.freebase_kb[entity][relation]
        return tuple_set, 0

    def execute_gen_set1_date(self, argument_value, argument_location):
        entity = argument_value[0]
        relation_date = argument_value[1]
        if entity is None or relation_date is None:
            return set([]), 1
        tuple_set = None
        if entity in self.freebase_kb and relation_date in self.freebase_kb[entity]:
            tuple_set = {d:entity for d in self.freebase_kb[entity][relation_date]}
        return tuple_set, 0

    def execute_gen_set2(self, argument_value,  argument_location):
        set_ent, _ = self.execute_gen_set1(argument_value, argument_location)
        relation = argument_value[2]
        if set_ent is None or relation is None:
            return set([]), 1
        tuple_set = None
        for e in set_ent:
            if e in self.freebase_kb and relation in self.freebase_kb[e]:
                if tuple_set is None:
                    tuple_set = set(self.freebase_kb[e][relation])
                else:
                    tuple_set.update(set(self.freebase_kb[e][relation]))
        return tuple_set, 0            

    def same_year(self, tails, y):
	for t in tails:
		t = self.convert_to_date(t).year
		if t==y:
			return True
	return False		

    def execute_gen_set2_dateconstrained(self, argument_value, argument_location):
	set_ent, _ = self.execute_gen_set1(argument_value, argument_location)
	relation = argument_value[2]
	constr_rel_date = argument_value[3]
	constr_date = argument_value[4]
	if set_ent is None or relation is None or constr_rel_date is None or constr_date is None:
            return set([]), 1
	constr_year = constr_date.year
	tuple_set = None
        for e in set_ent:
		if e in self.freebase_kb and constr_rel_date in self.freebase_kb[e] and self.same_year(self.freebase_kb[e][constr_rel_date], constr_year):
			if relation not in self.freebase_kb[e]:
                                continue
			if tuple_set is None:
				tuple_set = set(self.freebase_kb[e][relation])
			else:
				tuple_set.update(set(self.freebase_kb[e][relation]))
	return tuple_set, 0
	
    def execute_gen_set2_date(self, argument_value, argument_location):
	set_ent, _ = self.execute_gen_set1(argument_value, argument_location)
        relation_date = argument_value[2]
        if set_ent is None or relation_date is None:
            return set([]), 1
        tuple_set = None
        for e in set_ent:
            if e in self.freebase_kb and relation_date in self.freebase_kb[e]:
                if tuple_set is None:
                    tuple_set = set(self.freebase_kb[e][relation_date])
                else:
                    tuple_set.update(set(self.freebase_kb[e][relation_date]))
        return tuple_set, 0

    def execute_gen_set2_date_dateconstrained(self, argument_value, argument_location):
        set_ent, _ = self.execute_gen_set1(argument_value, argument_location)
        relation_date = argument_value[2]
        constr_rel_date = argument_value[3]
        constr_date = argument_value[4]
        if set_ent is None or relation_date is None or constr_rel_date is None or constr_date is None:
            return set([]), 1
        constr_year = constr_date.year
        tuple_set = None
        for e in set_ent:
                if e in self.freebase_kb and constr_rel_date in self.freebase_kb[e] and self.same_year(self.freebase_kb[e][constr_rel_date], constr_year):
			if relation_date not in self.freebase_kb[e]:
				continue	
                        if tuple_set is None:
                                tuple_set = set(self.freebase_kb[e][relation_date])
                        else:
                                tuple_set.update(set(self.freebase_kb[e][relation_date]))
        return tuple_set, 0	
                    
    def execute_set_oper_ints(self, argument_value, argument_location):
        set_ent1 = argument_value[0]
        set_ent2 = argument_value[1]
        if set_ent1 is None or set_ent2 is None:
            return set([]), 1
        set_ent_ints = set(set_ent1).intersection(set(set_ent2))
        flag = 0
        if argument_location is not None:
            argument_location1 = argument_location[0]
            argument_location2 = argument_location[1]
            if argument_location1==argument_location2:
                flag = 0.1
        return set_ent_ints, flag
    
    def convert_to_date(self, x):
        if x.startswith('m.'):
            return None
        if 'T' in x:
                x = x.split('T')[0]
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
    
    def execute_select_oper_date_lt(self, argument_value, argument_location):
        set_date = argument_value[0]
        date = argument_value[1]
        if set_date is None or date is None:
            return set([]), 1
        set_date = set([self.convert_to_date(d) for d in set_date])
        date = self.convert_to_date(date)
        subset_date = set([])
        for d,e in set_date.items():
            if d<=date:
                subset_date.add(e)
        return subset_date, 0

    def execute_select_oper_date_gt(self, argument_value, argument_location):
        set_date = argument_value[0]
        date = argument_value[1]
        if set_date is None or date is None:
            return set([]), 1
        set_date = set([self.convert_to_date(d) for d in set_date])
        date = self.convert_to_date(date)
        subset_date = set([])
        for d,e in set_date.items():
            if d>=date:
                subset_date.add(e)
        return subset_date, 0
        
    def execute_none(self, argument_value, argument_location):
        return None, 0

    def execute_terminate(self, argument_value, argument_location):
        return None, 0
