import torch

class consensus:
  def __init__(self, arch_flag, population, edge2index, op_dict, op_position, topk, weighted = False):
    self.arch_flag = arch_flag
    self.population = population
    self.pop_size = population.get_population_size()
    self.population.print_population_fitness()
    self.edge2index = edge2index
    self.op_dict = op_dict
    self.op_position = op_position
    self.topk = topk
    self.weighted = weighted
    self.vote_list = []
    
    self.edge2index2 = {value:key for key, value in self.edge2index.items()}
    
    
  def vote(self, individual, ind_id, verbose=False):
    arch_params = [arch_param.cpu() for arch_param in individual.get_arch_parameters()]
    arch_values, indices = [], []
    vote_index = 0
    for op_position, arch_param in zip(self.op_position, arch_params):
      if ind_id==1: vote_dict = {}
      else:         vote_dict = self.vote_list[vote_index]
      arch_value, indice = torch.topk(arch_param, self.topk)
      arch_values.append(arch_value)
      indices.append(indice)
      for idx, row in enumerate(indice):
        edge = self.edge2index2[idx]
        ops = []
        for col in row:
          ops.append(op_position[edge][str(col.item())])
        ops.sort()
        op_str = ','.join(ops)
        if verbose: print(op_str)
        if idx in vote_dict.keys():
          if op_str in vote_dict[idx].keys():
            if self.weighted: vote_dict[idx][op_str]+= -(ind_id/self.pop_size)
            else:             vote_dict[idx][op_str]+= 1
          else:
            if self.weighted: vote_dict[idx][op_str] = -(ind_id/self.pop_size)
            else:             vote_dict[idx][op_str] = 1
        else:
          if self.weighted: vote_dict[idx] = {op_str: -(ind_id/self.pop_size)}
          else:             vote_dict[idx] = {op_str: 1}
      if ind_id==1: self.vote_list.append(vote_dict)
      vote_index += 1
    if verbose: print(self.vote_list)
  
  def top_votes(self, verbose=False):
    top_votes = []
    for vote_dict in self.vote_list:
      tmp = {}
      for edge, ops_dict in vote_dict.items():
        score = float('-inf')
        for ops, value in ops_dict.items():
          if value > score: ops_str, score = ops, value
        tmp[edge] = (ops_str, score)
      top_votes.append(tmp)
    if verbose: print(f'{top_votes}')
    return top_votes
  
  def shrink(self, verbose=False):
    for pop_id, individual in enumerate(self.population.get_population()):
      self.vote(individual, pop_id+1)
    top_votes = self.top_votes(verbose=verbose)
    
    # Shrinking the search space
    new_arch_flag = [torch.zeros_like(tmp) for tmp in self.arch_flag]
    assert len(new_arch_flag) == len(top_votes), 'Mismatch lengths'
    for index, new_flag in enumerate(new_arch_flag):
      top_votes_dict = top_votes[index]
      for edge in top_votes_dict.keys():
        for op in top_votes_dict[edge][0].split(','):
          new_flag[edge][self.op_dict[op]] = True
    return new_arch_flag

  def shrink_by_one_ind(self, individual, verbose=False):
    if verbose: print(individual.arch_parameters)
    self.vote(individual, 1)
    top_votes = self.top_votes(verbose=False)
    if verbose: print(top_votes)
        
    # Shrinking the search space
    new_arch_flag = [torch.zeros_like(tmp) for tmp in self.arch_flag]
    assert len(new_arch_flag) == len(top_votes), 'Mismatch lengths'
    for index, new_flag in enumerate(new_arch_flag):
      top_votes_dict = top_votes[index]
      for edge in top_votes_dict.keys():
        for op in top_votes_dict[edge][0].split(','):
          new_flag[edge][self.op_dict[op]] = True
    return new_arch_flag

class consensus2:
  def __init__(self, arch_flag, population, edge2index, op_dict, op_position, topk, weighted = False):
    self.arch_flag = arch_flag
    self.population = population
    self.pop_size = population.get_population_size()
    self.population.print_population_fitness()
    self.edge2index = edge2index
    self.op_dict = op_dict
    self.op_position = op_position
    self.topk = topk
    self.weighted = weighted
    self.vote_list = []

    self.edge2index2 = {value:key for key, value in self.edge2index.items()}
        
        
  def vote(self, individual, ind_id, verbose=False):
    arch_params = [arch_param.cpu() for arch_param in individual.get_arch_parameters()]
    assert len(arch_params)==2, 'Wrong length of the achitecture parameter'
    arch_values, indices = [], []
    vote_index = 0
    for op_position, arch_param in zip(self.op_position, arch_params):
      if ind_id==1: vote_dict = {}
      else: vote_dict = self.vote_list[vote_index]
      arch_value, indice = torch.topk(arch_param, self.topk)
      arch_values.append(arch_value)
      indices.append(indice)
      for idx, row in enumerate(indice):
        edge = self.edge2index2[idx]
        ops = []
        for col in row:
          ops.append(op_position[edge][str(col.item())])
        if idx in vote_dict.keys():
          for op in ops:
            if op in vote_dict[idx].keys():
              if self.weighted: vote_dict[idx][op]+= -(ind_id/self.pop_size)
              else:             vote_dict[idx][op]+= 1
            else:
              if self.weighted: vote_dict[idx][op] = -(ind_id/self.pop_size)
              else:             vote_dict[idx][op] = 1
        else:
          if self.weighted: vote_dict[idx] = {op: -(ind_id/self.pop_size) for op in ops}
          else:             vote_dict[idx] = {op: 1 for op in ops}
      if ind_id==1: self.vote_list.append(vote_dict)
      vote_index += 1
    if verbose: print(self.vote_list[0], '\n', self.vote_list[1])
    
  def top_votes(self, verbose=False):
    top_votes = []
    for vote_dict in self.vote_list:
      tmp = {}            
      for edge, ops_dict in vote_dict.items():
        ops_tmp, score_tmp, k = [], [], 0
        while k < self.topk:
          max_key = max(ops_dict, key= lambda x:ops_dict[x])
          ops_tmp.append(max_key)
          score_tmp.append(ops_dict[max_key])
          ops_dict.pop(max_key)
          k += 1
        ops_str = ','.join(ops_tmp)
        score_str = ','.join(str(score_tmp))
        tmp[edge] = (ops_str, score_str)
      top_votes.append(tmp)
      if verbose: print('top_votes: ', top_votes)
    #print(f'length of top_votes: {len(top_votes)}')
    return top_votes
    
  def shrink(self, verbose=False):
    for pop_id, individual in enumerate(self.population.get_population()):
      self.vote(individual, pop_id+1, verbose)
    top_votes = self.top_votes(verbose=verbose)
        
    # Shrinking the search space
    new_arch_flag = [torch.zeros_like(tmp) for tmp in self.arch_flag]
    assert len(new_arch_flag) == len(top_votes), 'Mismatch lengths'
    for index, new_flag in enumerate(new_arch_flag):
      top_votes_dict = top_votes[index]
      for edge in top_votes_dict.keys():
        for op in top_votes_dict[edge][0].split(','):
          new_flag[edge][self.op_dict[op]] = True
    return new_arch_flag
