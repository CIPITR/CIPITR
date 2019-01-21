from load_wikidata2 import load_wikidata
import numpy as np
import msgpack

class Interpreter():
    def __init__(self, wikidata_dir, num_timesteps, program_type_vocab, argument_type_vocab, printing, terminate_prog, relaxed_reward_strict, reward_function="jaccard", boolean_reward_multiplier = 0.1, relaxed_reward_till_epoch=[-1,-1], unused_var_penalize_after_epoch=[1000, 100000], length_based_penalization=False):
        np.random.seed(1)
        self.wikidata, self.reverse_wikidata, self.wikidata_type, self.reverse_wikidata_type, self.wikidata_ent_type, self.reverse_wikidata_ent_type = load_wikidata(wikidata_dir)
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
        self.map_program_to_func["gen_set"]=self.execute_gen_set
        self.map_program_to_func["gen_map1"]=self.execute_gen_map1
        self.map_program_to_func["verify"]=self.execute_verify
        self.map_program_to_func["set_oper_count"]=self.execute_set_oper_count
        self.map_program_to_func["set_oper_union"]=self.execute_set_oper_union
        self.map_program_to_func["set_oper_ints"]=self.execute_set_oper_intersec
        self.map_program_to_func["set_oper_diff"]=self.execute_set_oper_diff
        self.map_program_to_func["map_oper_count"]=self.execute_map_oper_count
        self.map_program_to_func["map_oper_union"]=self.execute_map_oper_union
        self.map_program_to_func["map_oper_ints"]=self.execute_map_oper_intersec
        self.map_program_to_func["map_oper_diff"]=self.execute_map_oper_diff
        self.map_program_to_func["select_oper_max"]=self.execute_select_oper_max
        self.map_program_to_func["select_oper_min"]=self.execute_select_oper_min
        self.map_program_to_func["select_oper_atleast"]=self.execute_select_oper_atleast
        self.map_program_to_func["select_oper_atmost"]=self.execute_select_oper_atmost
        self.map_program_to_func["select_oper_more"]=self.execute_select_oper_more
        self.map_program_to_func["select_oper_less"]=self.execute_select_oper_less
        self.map_program_to_func["select_oper_equal"]=self.execute_select_oper_equal
        self.map_program_to_func["select_oper_approx"]=self.execute_select_oper_approx
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
            inputs, gold_entities, gold_ints, gold_bools = args
        return self.execute_multiline_program(inputs, gold_entities, gold_ints, gold_bools, epoch_number, overall_step_number)

    def execute_multiline_program(self,inputs, gold_entities, gold_ints, gold_bools, epoch_number, overall_step_number):
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
            print 'number of types: ', len(variable_value_table[self.argument_type_vocab['type']]) - (variable_value_table[self.argument_type_vocab['type']]==None).sum()
            print 'number of ints: ', len(variable_value_table[self.argument_type_vocab['int']]) - (variable_value_table[self.argument_type_vocab['int']]==None).sum()
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
        for i in range(self.num_timesteps):
            program_type_id = int(inputs['program_type'][i])
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
                break
            if program_type_id != self.program_type_vocab['none']:# and ('terminate' not in self.program_type_vocab or program_type_id!=self.program_type_vocab['terminate']):
                if line_of_code in lines_of_code_written:
                    if self.printing:
                        print 'FLAG: Repeated_lines_of_code'
                    self.FLAG['Repeated_lines_of_code'] = True
                else:
                    lines_of_code_written.append(line_of_code)
                self.FLAG['All_none_programs'] = False
                num_lines_of_code += 1
                target_type_id = int(inputs['target_type'][i])
                target_type = self.argument_type_vocab_inv[target_type_id]
                target_table_index = inputs['target_table_index'][i]
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
                        reward_till_timestep, _ = self.calculate_reward(target_value, target_type_id, cumulative_flag, num_lines_of_code, gold_entities, gold_ints, gold_bools, epoch_number, overall_step_number, False)
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
        elif self.FLAG['Sampling_from_empty_table'] or self.FLAG['Repeated_lines_of_code'] or self.FLAG['executed_line_output_none']:
            target_value = None
            target_type_id = None
            flag = self.HIGHEST_NEGATIVE_REWARD
            reward = -flag
            relaxed_reward = reward
        elif target_value is None:
            flag = self.HIGHEST_NEGATIVE_REWARD
            reward = -flag
            relaxed_reward = reward
        else:
            flag = cumulative_flag
            reward, relaxed_reward = self.calculate_reward(target_value, target_type_id, flag, num_lines_of_code, gold_entities, gold_ints, gold_bools, epoch_number, overall_step_number)
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
        return reward, max_intermediate_reward, relaxed_reward, intermediate_mask, intermediate_reward_flag


    def calculate_reward(self, target_value, target_type_id, flag, num_lines_of_code, gold_entities, gold_ints, gold_bools, epoch_number, overall_step_number, print_flag = True):
        reward = 0
        relaxed_reward = 0
        if not self.length_based_penalization:
            num_lines_of_code = 0.0
        reward -= num_lines_of_code*0.001
        gold_answer = []
        gold_type = ''
        if len(gold_ints)>0:
            gold_answer = gold_ints
            gold_bools = []
            gold_entities = []
            gold_type = 'int'
        elif len(gold_bools)>0:
            gold_answer = gold_bools
            gold_ints = []
            gold_entities = []
            gold_type = 'bool'
        elif len(gold_entities)>0:
            gold_answer = gold_entities
            gold_ints = []
            gold_bools = []
            gold_type = 'set'
        if self.argument_type_vocab_inv[target_type_id]=='set' or self.argument_type_vocab_inv[target_type_id]=='entity':
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
                    print 'In calculate reward: predicted_answer=set, gold_answer=',gold_type, ' reward =', reward, ' flag = ', flag, ' relaxed_reward =', relaxed_reward
                #else:
                #    print 'reward =', reward, ' flag = ', flag, ' relaxed_reward =', relaxed_reward
        elif self.argument_type_vocab_inv[target_type_id]=='int':
            target_value = int(target_value)
            gold_ints = set(gold_ints)
            if len(gold_ints)!=0:
                if epoch_number <= self.relaxed_reward_till_epoch[0] and overall_step_number <= self.relaxed_reward_till_epoch[1]:
                    relaxed_reward += 1.
                if target_value in gold_ints:
                    if self.reward_function=="jaccard":
                        reward += 1.0/float(len(gold_ints))
                    elif self.reward_function=="recall":
                        reward += 1.0
                    elif self.reward_function=="f1":
                        prec = 1.0
                        rec = 1.0/float(len(gold_ints))
                        if prec==0 and rec==0:
                            reward += 0
                        else:
                            reward += (2.0*prec*rec)/(prec+rec)
            else:
                if epoch_number > self.relaxed_reward_till_epoch[0] or overall_step_number > self.relaxed_reward_till_epoch[1]:
                    reward += -self.HIGH_NEGATIVE_REWARD
            if print_flag:
                if self.printing:
                    print 'In calculate reward: predicted_answer=int, gold_answer=',gold_type, 'reward=', reward, ' flag = ', flag, ' relaxed_reward =', relaxed_reward
                #else:
                #    print 'reward =', reward, ' flag = ', flag, ' relaxed_reward =', relaxed_reward
        elif self.argument_type_vocab_inv[target_type_id]=='bool':
            target_value = bool(target_value)
            gold_bools = set(gold_bools)
            if len(gold_bools)!=0:
                if epoch_number <= self.relaxed_reward_till_epoch[0] and overall_step_number <= self.relaxed_reward_till_epoch[1]:
                    relaxed_reward += 1.
                if target_value in gold_bools:
                    if self.reward_function=="jaccard":
                        reward += (self.boolean_reward_multiplier*1.0)/float(len(gold_bools))
                    elif self.reward_function=="recall":
                        reward += (self.boolean_reward_multiplier*1.0)
                    elif self.reward_function=="f1":
                        prec = 1.0
                        rec = 1.0/float(len(gold_bools))
                        if prec==0 and rec==0:
                            reward += 0
                        else:
                            reward += (2.0*prec*rec)/(prec+rec)
            else:
                reward += -2.0*self.HIGH_NEGATIVE_REWARD
            if print_flag:
                if self.printing:
                    print 'In calculate reward: predicted_answer=bool, gold_answer=',gold_type, 'reward=', reward, ' flag = ', flag, ' relaxed_reward =', relaxed_reward
                #else:
                #    print 'reward=', reward, ' flag = ', flag, ' relaxed_reward =', relaxed_reward
        else:
            reward += -self.HIGH_NEGATIVE_REWARD
            relaxed_reward += -self.HIGH_NEGATIVE_REWARD
            if print_flag:
                if self.printing:
                    print 'In calculate reward: predicted_answer=', self.argument_type_vocab_inv[target_type_id], 'gold_answer=',gold_type,' reward =',reward, ' relaxed_reward =', -1
                #else:
                #    print 'reward=', reward
        reward = reward - flag
        relaxed_reward = relaxed_reward - flag
        return reward, relaxed_reward

    def execute_singleline_program(self,program_type, argument_values, argument_location):
        # debugging ########################################
        if program_type=='none':
            target_value = None
            func_name = None
        else:
            func = self.map_program_to_func[program_type]
            target_value, flag = func(argument_values, argument_location)
            func_name = func.__name__
        return target_value, flag

    def execute_gen_set(self, argument_value, argument_location):
        entity = argument_value[0]
        relation = argument_value[1]
        type = argument_value[2]
        tuple_set = None
        if isinstance(entity, list) or isinstance(entity, set):
            entities = list(entity)
            for entity in entities:
                if entity in self.wikidata_ent_type:
                    if relation in self.wikidata_ent_type[entity]:
                        if type in self.wikidata_ent_type[entity][relation]:
                            tuple_set = set(self.wikidata_ent_type[entity][relation][type])
                if entity in self.reverse_wikidata_ent_type:
                    if relation in self.reverse_wikidata_ent_type[entity]:
                        if type in self.reverse_wikidata_ent_type[entity][relation]:
                            if tuple_set is None:
                                tuple_set = set([])
                            tuple_set.update(self.reverse_wikidata_ent_type[entity][relation][type])
        else:
            if entity in self.wikidata_ent_type:
                if relation in self.wikidata_ent_type[entity]:
                    if type in self.wikidata_ent_type[entity][relation]:
                        tuple_set = set(self.wikidata_ent_type[entity][relation][type])
            if entity in self.reverse_wikidata_ent_type:
                if relation in self.reverse_wikidata_ent_type[entity]:
                    if type in self.reverse_wikidata_ent_type[entity][relation]:
                        if tuple_set is None:
                            tuple_set = set([])
                        tuple_set.update(self.reverse_wikidata_ent_type[entity][relation][type])
        return tuple_set, 0

    def execute_gen_map1(self, argument_value, argument_location):
        type1 = argument_value[0]
        relation = argument_value[1]
        type2 = argument_value[2]
        tuple_map = None
        if type1 in self.wikidata_type:
            if relation in self.wikidata_type[type1]:
                if type2 in self.wikidata_type[type1][relation]:
                    tuple_map = self.wikidata_type[type1][relation][type2]
        if type1 in self.reverse_wikidata_type:
            if relation in self.reverse_wikidata_type[type1]:
                if type2 in self.reverse_wikidata_type[type1][relation]:
                    if tuple_map is None:
                        tuple_map = {}
                    tuple_map.update(self.reverse_wikidata_type[type1][relation][type2])
        return tuple_map, 0

    def execute_set_oper_count(self, argument_value, argument_location):
        set1 = argument_value[0]
        return len(set1), 0

    def execute_map_oper_count(self, argument_value, argument_location):
        map_x_to_set = argument_value[0]
        map_x_to_int = {}
        if map_x_to_set is None:
            return None
        for x in map_x_to_set:
            map_x_to_int[x] = len(map_x_to_set[x])
        return map_x_to_int, 0

    def execute_set_oper_union(self, argument_value, argument_location):
        tuple_set1 = set(argument_value[0])
        tuple_set2 = set(argument_value[1])
        flag = 0
        if argument_location is not None:
            argument_location1 = argument_location[0]
            argument_location2 = argument_location[1]
            if argument_location1==argument_location2:
                flag = 0.1
        return tuple_set1.union(tuple_set2), flag

    def execute_set_oper_intersec(self, argument_value, argument_location):
        tuple_set1 = set(argument_value[0])
        tuple_set2 = set(argument_value[1])
        flag = 0
        if argument_location is not None:
            argument_location1 = argument_location[0]
            argument_location2 = argument_location[1]
            if argument_location1==argument_location2:
                flag = 0.1
        return tuple_set1.intersection(tuple_set2), flag

    def execute_set_oper_diff(self, argument_value, argument_location):
        tuple_set1 = set(argument_value[0])
        tuple_set2 = set(argument_value[1])
        flag = 0
        if argument_location is not None:
            argument_location1 = argument_location[0]
            argument_location2 = argument_location[1]
            if argument_location1==argument_location2:
                flag = 0.1
        return tuple_set1 - tuple_set2, flag

    def execute_map_oper_union(self, argument_value, argument_location):
        map_x_to_set1 = argument_value[0]
        map_x_to_set2 = argument_value[1]
        flag = 0
        if argument_location is not None:
            argument_location1 = argument_location[0]
            argument_location2 = argument_location[1]
            if argument_location1==argument_location2:
                flag = 0.1
        map_output = {}
        ints = set(map_x_to_set1.keys()).intersection(set(map_x_to_set2.keys()))
        for k in ints:
            map_output[k] = self.execute_set_oper_union([map_x_to_set1[k], map_x_to_set2[k]], None)[0]
        return map_output, flag

    def execute_map_oper_intersec(self, argument_value, argument_location):
        if argument_value[0] is None or argument_value[1] is None:
            self.FLAG['Sampling_from_empty_table'] = True
            return None
        flag = 0
        map_x_to_set1 = argument_value[0]
        map_x_to_set2 = argument_value[1]
        if argument_location is not None:
            argument_location1 = argument_location[0]
            argument_location2 = argument_location[1]
            if argument_location1==argument_location2:
                flag = 0.1
        map_output = {}
        ints = set(map_x_to_set1.keys()).intersection(set(map_x_to_set2.keys()))
        for k in ints:
            map_output[k] = self.execute_set_oper_intersec([map_x_to_set1[k], map_x_to_set2[k]], None)[0]
        return map_output, flag

    def execute_map_oper_diff(self, argument_value, argument_location):
        map_x_to_set1 = argument_value[0]
        map_x_to_set2 = argument_value[1]
        flag = 0
        if argument_location is not None:
            argument_location1 = argument_location[0]
            argument_location2 = argument_location[1]
            if argument_location1==argument_location2:
                flag = 0.1
        map_output = {}
        ints = set(map_x_to_set1.keys()).intersection(set(map_x_to_set2.keys()))
        for k in ints:
            map_output[k] = self.execute_set_oper_diff([map_x_to_set1[k], map_x_to_set2[k]], None)[0]
        return map_output, flag

    def execute_select_oper_min(self, argument_value, argument_location):
        tuple_map1 = argument_value[0]
        keys = tuple_map1.keys()
        if len(tuple_map1)==0:
            return None, 0
        min_value = min(tuple_map1.values())
        return [i for i in tuple_map1 if tuple_map1[i]==min_value], 0

    def execute_select_oper_max(self, argument_value, argument_location):
        tuple_map1 = argument_value[0]
        keys = tuple_map1.keys()
        if len(tuple_map1)==0:
            return None, 0
        max_value = max(tuple_map1.values())
        return [i for i in tuple_map1 if tuple_map1[i]==max_value], 0

    def execute_select_oper_atleast(self, argument_value, argument_location):
        tuple_map1 = argument_value[0]
        int1 = argument_value[1]
        if len(tuple_map1)==0:
            return None, 0
        new_tuple_map1 = {k:v for k,v in tuple_map1.items() if v >= int1}
        return new_tuple_map1, 0

    def execute_select_oper_atmost(self, argument_value, argument_location):
        tuple_map1 = argument_value[0]
        int1 = argument_value[1]
        if len(tuple_map1)==0:
            return None, 0
        new_tuple_map1 = {k:v for k,v in tuple_map1.items() if v <= int1 }
        return new_tuple_map1, 0

    def execute_select_oper_more(self, argument_value, argument_location):
        tuple_map1 = argument_value[0]
        int1 = argument_value[1]
        if len(tuple_map1)==0:
            return None, 0
        new_tuple_map1 = {k:v for k,v in tuple_map1.items() if v > int1}
        return new_tuple_map1, 0

    def execute_select_oper_less(self, argument_value, argument_location):
        tuple_map1 = argument_value[0]
        int1 = argument_value[1]
        if len(tuple_map1)==0:
            return None, 0
        new_tuple_map1 = {k:v for k,v in tuple_map1.items() if v < int1 }
        return new_tuple_map1, 0

    def execute_select_oper_equal(self, argument_value, argument_location):
        tuple_map1 = argument_value[0]
        int1 = argument_value[1]
        if len(tuple_map1)==0:
            return None, 0
        new_tuple_map1 = {k:v for k,v in tuple_map1.items() if v == int1 }
        return new_tuple_map1, 0

    def execute_select_oper_approx(self, argument_value, argument_location):
        tuple_map1 = argument_value[0]
        int1 = argument_value[1]
        if len(tuple_map1)==0:
            return None, 0
        new_tuple_map1 = {k:v for k,v in tuple_map1.items() if abs(v-int1)<2 }
        return new_tuple_map1, 0

    def execute_verify(self, argument_value, argument_location):
        entity1 = argument_value[0]
        relation = argument_value[1]
        entity2 = argument_value[2]
        if isinstance(entity1, list) or isinstance(entity1, set) or isinstance(entity2, list) or isinstance(entity2, set):
            return None, 0
        if (entity1 in self.wikidata and relation in self.wikidata[entity1] and entity2 in self.wikidata[entity1][relation]) or (entity1 in self.reverse_wikidata and relation in self.reverse_wikidata[entity1] and entity2 in self.reverse_wikidata[entity1][relation]):
            #print 'got True in execute verify ', str(argument_value)
            return True, 0
        else:
            #print 'got False in execute verify ', str(argument_value)
            return False, 0

    def execute_none(self, argument_value, argument_location):
        return None, 0

    def execute_terminate(self, argument_value, argument_location):
        return None, 0
