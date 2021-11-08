import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


class MixedOp(nn.Module):

  def __init__(self, C, stride, op_dict, row):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    ## Newly Added
    self.op_position, pos = {}, 0
    ##
    for primitive in PRIMITIVES:
      #print(row[op_dict[primitive]])
      ## Newly Added
      if row[op_dict[primitive]]: 
        op = OPS[primitive](C, stride, False)
        if 'pool' in primitive:
          op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
        self._ops.append(op)
        self.op_position[str(pos)] = primitive
        pos += 1
        ##

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, op_dict, arch_flag):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier
    self.arch_flag = arch_flag

    ## Newly added
    self.op_dict = op_dict
    self.op_position = {}
    ##

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    self.idx = 0
    for i in range(self._steps):
      for j in range(2+i):
        ## Newly added
        row = self.arch_flag[self.idx]
        if j == 0: node_str = f'c_(k-2)->{i}'
        elif j == 1: node_str = f'c_(k-1)->{i}'
        else:      node_str = f'{j-2}->{i}'
        self.op_position[node_str] = {}
        ##
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride, op_dict, row)
        ## Newly added
        self.op_position[node_str] = op.op_position
        ##
        self._ops.append(op)
        ## Newly added
        self.idx += 1
        ##
    assert self.idx == len(self.op_position), "Mistake in creating the progressive cell"

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, device, steps=4, multiplier=4, stem_multiplier=3, arch_flag=None):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    #self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self._device = device

    self._k = sum(1 for i in range(self._steps) for n in range(2+i))
    
    ## Newly added for the Progressive
    if arch_flag is None:
      self._num_ops = len(PRIMITIVES)
      self.arch_flag = [torch.ones((self._k, self._num_ops), dtype=torch.bool) for _ in range(2)]
    else:
      self.arch_flag = [tmp.detach().clone() for tmp in arch_flag]
    self.op_dict = {}
    for idx, op in enumerate(PRIMITIVES): self.op_dict[op] = idx
    ##
    
    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
        arch_flag_inp = self.arch_flag[1]
      else:
        reduction = False
        arch_flag_inp = self.arch_flag[0]
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.op_dict, arch_flag_inp)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()
    self.prepare_dicts()

  #def set_criterion(self, criterion):
  #  self._criterion = criterion

  def copy_arch_parameters(self, parameters):
    for x, y in zip(self.arch_parameters(), parameters):
        #print("model arch shape: {}, pop arch shape: {}".format(x.data.shape, y.data.shape))
        x.data.copy_(y.data)

  '''def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).to(self._device)
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new'''

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.rand(k, sum(self.arch_flag[0][0].int())).to(self._device), requires_grad=False)
    self.alphas_reduce = Variable(1e-3*torch.rand(k, sum(self.arch_flag[1][0].int())).to(self._device), requires_grad=False)
    #self.alphas_normal = Variable(1e-3*torch.rand(k, num_ops).to(self._device), requires_grad=False)
    #self.alphas_reduce = Variable(1e-3*torch.rand(k, num_ops).to(self._device), requires_grad=False)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def get_alphas(self):
    '''Returns the list of architecture weights copied to cpu'''
    alphas = [a.cpu() for a in self._arch_parameters]
    return alphas
  
  def update_alphas(self, alphas):
    '''Update the architecture weights
       alphas = [normal_wts, reduction_wts]
    '''
    #self.get_alphas()[0].data.copy_(alphas)
    assert len(alphas)==2, 'Wrong dimension of the alphas'
    for alpha, value in zip(self.arch_parameters(), alphas):
      alpha.data.copy_(value)
  
  def check_alphas(self, alphas):
    '''Verify if the architecture parameter is equal to the given alphas'''
    #return np.all(self.get_alphas()[0] == alphas.get())
    for x, y in zip(self.arch_parameters(), alphas):
      #print(x, y, x.eq(y))
      if not torch.all(x.eq(y)):
        return False
    return True

  def random_alphas(self, discrete=False, k = 1.0):
    '''Generate random values for the architecture parameters and update the architecture parametr of the model
       discrete: Whether to  discretize the alphas during training
       k = value for the operations in the discrete architecture parameter
    '''
    ## Newly Added
    alphas_normal = 1e-3*torch.rand(self._k, sum(self.arch_flag[0][0].int())).to(self._device)
    alphas_reduce = 1e-3*torch.rand(self._k, sum(self.arch_flag[1][0].int())).to(self._device)
    tmp = [alphas_normal, alphas_reduce]
    self.update_alphas(tmp)
    assert self.check_alphas(tmp), "random architecture parameter failed to be copied"
    ##

    if discrete:
      discrete_tmp = self.discretize(k=k)
      self.update_alphas(discrete_tmp)
      assert self.check_alphas(discrete_tmp), "random discrete architecture parameter failed to be copied"
      return discrete_tmp
    else:
      return tmp

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    alphas_df = self.show_alphas_dataframe()[0]
    alphas_normal = torch.tensor(alphas_df[0].to_numpy())
    alphas_reduce = torch.tensor(alphas_df[1].to_numpy())
    gene_normal = _parse(F.softmax(alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(alphas_reduce, dim=-1).data.cpu().numpy())
    #gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    #gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

  def prepare_dicts(self):
    self._nodes = ['c_(k-2)', 'c_(k-1)', '0', '1', '2', '3']
    self._nodes_dict = {key:i for key, i in  enumerate(self._nodes)}
    tmp = []
    self._rows = []
    for node in range(2, len(self._nodes)):
      for prev_node in range(node):
        self._rows.append('{}->{}'.format(self._nodes_dict[prev_node], node-2))
    self._rows_dict = {key:i for key, i in  enumerate(self._rows)}
    self.edge2index = {value:key for key, value in self._rows_dict.items()}

  def show_alphas_dataframe(self, show_sum=False):
    key_list, val_list = list(self.op_dict.keys()), list(self.op_dict.values())
    df_alphas, df_softmax, df_max = [], [], []
    for alpha_num, alpha in enumerate(self.get_alphas()):
      alpha_softmax = F.softmax(alpha, dim=-1).detach().cpu().numpy()

      df   = pd.DataFrame(columns=PRIMITIVES, index=self._rows, data = np.zeros((self._k, len(PRIMITIVES)), dtype = np.float64) )
      df_s = pd.DataFrame(columns=PRIMITIVES, index=self._rows, data = np.zeros((self._k, len(PRIMITIVES)), dtype = np.float64) )
      df_m = pd.DataFrame(columns=PRIMITIVES, index=self._rows, data = np.zeros((self._k, len(PRIMITIVES)), dtype = np.float64) )
      
      for key, index in self.edge2index.items():
        row, row_softmax = alpha[index], alpha_softmax[index]
        row_flag = torch.nonzero(self.arch_flag[alpha_num][index]).squeeze(dim=1)
        op_list = [key_list[idx] for idx in row_flag]
        for idx, op in enumerate(op_list):
          df.loc[key][op] = row[idx]
          df_s.loc[key][op] = row_softmax[idx]
      
      df_m = df.astype(float).idxmax(axis = 1)
      df_alphas.append(df)
      if show_sum: df_s['sum'] = df_s.sum(axis = 1)
      df_softmax.append(df_s)
      df_max.append(df_m)
    return df_alphas, df_max, df_softmax
  
  def discretize(self, k=1.0, verbose=False):
    discrete_alphas = [torch.zeros_like(tmp) for tmp in self.arch_parameters()]
    derived_genotype = self.genotype()
    normal_cell, reduce_cell = derived_genotype.normal, derived_genotype.reduce
    
    # Normal cell
    if verbose: print('Normal Cell')
    node = 0
    for idx, (op_name, node_num) in enumerate(normal_cell):
      node_str = f'{self._nodes_dict[node_num]}->{node}'
      row_index = self.edge2index[node_str]
      for op_position, op in self.cells[0].op_position[node_str].items():
        if op == op_name:
          col_index = int(op_position)
          break
      discrete_alphas[0][row_index][col_index] = k
      if verbose: print(op_name, node_num, node_str, self.edge2index[node_str], col_index)
      if (idx%2) and not(idx==0): node += 1
    
    # Reduction cell
    if verbose: print('\nReduction Cell')
    node = 0
    for idx, (op_name, node_num) in enumerate(reduce_cell):
      node_str = f'{self._nodes_dict[node_num]}->{node}'
      row_index = self.edge2index[node_str]
      for op_position, op in self.cells[len(self.cells)//3].op_position[node_str].items():
        if op == op_name:
          col_index = int(op_position)
          break
      discrete_alphas[1][row_index][col_index] = k
      if verbose: print(op_name, node_num, node_str, self.edge2index[node_str], col_index)
      if (idx%2) and not(idx==0): node += 1

    self.update_alphas(discrete_alphas)
    
    return discrete_alphas
