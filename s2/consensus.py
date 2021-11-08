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
    self.vote_dict = {}
    
    self.edge2index2 = {value:key for key, value in self.edge2index.items()}
  
  def vote(self, individual, ind_id, verbose=False):
    arch_param = individual.get_arch_parameters()[0].cpu()
    arch_values, indices = torch.topk(arch_param, self.topk)
    if verbose: print(indices)
    for idx, row in enumerate(indices):
      edge = self.edge2index2[idx]
      ops = []
      for col in row: ops.append(self.op_position[edge][str(col.item())])
      ops.sort()
      op_str = ','.join(ops)
      if verbose: print(','.join(ops))
      if idx in self.vote_dict.keys():
        if op_str in self.vote_dict[idx].keys():
          if self.weighted: self.vote_dict[idx][op_str]+= -(ind_id/self.pop_size)
          else:             self.vote_dict[idx][op_str]+= 1
        else:
          if self.weighted: self.vote_dict[idx][op_str] = -(ind_id/self.pop_size)
          else:             self.vote_dict[idx][op_str] = 1
      else:
        if self.weighted: self.vote_dict[idx] = {op_str: -(ind_id/self.pop_size)}
        else:             self.vote_dict[idx] = {op_str: 1}
    if verbose: print(self.vote_dict)
  
  def sort_vote_dict(self, verbose=False):
    sorted_vote_dict = {}
    for edge, ops_dict in self.vote_dict.items():
      sorted_vote_dict[edge] = dict(sorted(ops_dict.items(), key=lambda x: x[1], reverse=True))
    if verbose: print(sorted_vote_dict)
    return sorted_vote_dict
    
  def top_votes(self, verbose=False):
    top_votes = {}
    for edge, ops_dict in self.vote_dict.items():
      score = float('-inf')
      for ops, value in ops_dict.items():
        if value > score: ops_str, score = ops, value
      top_votes[edge] = (ops_str, score)
      if verbose: print(f'{top_votes[edge]}: {score}')
    return top_votes
        
  def shrink(self):
    for pop_id, individual in enumerate(self.population.get_population()): self.vote(individual, pop_id+1)
    #sorted_vote_dict = self.sort_vote_dict()
    top_votes_dict = self.top_votes()
    print(top_votes_dict)
    
    # Shrinking the search space
    new_arch_flag = torch.zeros_like(self.arch_flag)
    for edge in top_votes_dict.keys():
      for op in top_votes_dict[edge][0].split(','):
        new_arch_flag[edge][self.op_dict[op]] = True
    return new_arch_flag

  def shrink_by_one_ind(self, individual, verbose=False):
    if verbose: print(individual.arch_parameters)
    self.vote(individual, 1)
    top_votes_dict = self.top_votes()
    if verbose: print(top_votes_dict)
        
    # Shrinking the search space
    new_arch_flag = torch.zeros_like(self.arch_flag)
    for edge in top_votes_dict.keys():
      for op in top_votes_dict[edge][0].split(','):
        new_arch_flag[edge][self.op_dict[op]] = True
    return new_arch_flag

class consensus3:
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
    self.vote_dict = {}
    self.vote_value_dict = {}

    self.edge2index2 = {value:key for key, value in self.edge2index.items()}

  def get_vote_value(self, arch_str, ind_id):
    if arch_str in self.vote_value_dict:
      vote_value = self.vote_value_dict[arch_str]
    else:
      vote_value = -(ind_id/self.pop_size)
      self.vote_value_dict[arch_str] = vote_value
    return vote_value
    
  def vote(self, individual, ind_id, verbose=False):
    arch_param = individual.get_arch_parameters()[0].cpu()
    arch_str = individual.genotype
    arch_values, indices = torch.topk(arch_param, self.topk)
    if verbose: print(indices)
    for idx, row in enumerate(indices):
      edge = self.edge2index2[idx]
      ops = []
      for col in row:
        ops.append(self.op_position[edge][str(col.item())])
      ops.sort()
      op_str = ','.join(ops)
      if verbose: print(','.join(ops))
      if idx in self.vote_dict.keys():
        if op_str in self.vote_dict[idx].keys():
          if self.weighted: self.vote_dict[idx][op_str]+= self.get_vote_value(arch_str=arch_str, ind_id=ind_id)
          else:             self.vote_dict[idx][op_str]+= 1
        else:
          if self.weighted: self.vote_dict[idx][op_str] = self.get_vote_value(arch_str=arch_str, ind_id=ind_id)
          else:             self.vote_dict[idx][op_str] = 1
      else:
        if self.weighted: self.vote_dict[idx] = {op_str: self.get_vote_value(arch_str=arch_str, ind_id=ind_id)}
        else:             self.vote_dict[idx] = {op_str: 1}
    if verbose: print(self.vote_dict)
    
  def sort_vote_dict(self, verbose=False):
    sorted_vote_dict = {}
    for edge, ops_dict in self.vote_dict.items():
      sorted_vote_dict[edge] = dict(sorted(ops_dict.items(), key=lambda x: x[1], reverse=True))
    if verbose: print(sorted_vote_dict)
    return sorted_vote_dict
    
  def top_votes(self, verbose=False):
    top_votes = {}
    for edge, ops_dict in self.vote_dict.items():
      score = float('-inf')
      for ops, value in ops_dict.items():
        if value > score: ops_str, score = ops, value
      top_votes[edge] = (ops_str, score)
      if verbose: print(f'{top_votes[edge]}: {score}')
    return top_votes
        
  def shrink(self, verbose=False):
    for pop_id, individual in enumerate(self.population.get_population()):
      self.vote(individual, pop_id+1)
    top_votes_dict = self.top_votes(verbose=verbose)
    if verbose: print(top_votes_dict)
        
    # Shrinking the search space
    new_arch_flag = torch.zeros_like(self.arch_flag)
    for edge in top_votes_dict.keys():
      for op in top_votes_dict[edge][0].split(','):
        new_arch_flag[edge][self.op_dict[op]] = True
    return new_arch_flag
