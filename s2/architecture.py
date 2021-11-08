import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn

from log_utils import AverageMeter
from utils     import obtain_accuracy

class Architecture:
  def __init__(self, num_edges, search_space, value = None):
    self.num_ops = len(search_space)
    self.num_edges = num_edges
    #self.arch = np.random.rand(self.num_edges, self.num_ops)
    self.arch = torch.rand(self.num_edges, self.num_ops)#, dtype = torch.float32)
    if value is not None:
      self.update(value)
      assert self.compare(value), "Given tensor has not been successfully copied"
    #self.apply_softmax(self.arch)
    self.search_space = search_space

  def random_arch(self):
    '''
      Updating the architecture to random value
    '''
    try:
      tmp = torch.rand(self.num_edges, self.num_ops)#, dtype = torch.float32)
      self.update(tmp)
    except:
      tmp = torch.rand(self.num_edges, self.num_ops).cuda()#, dtype = torch.float32).cuda()
      self.update(tmp)

  def apply_softmax(self, arch_matrix):
    arch_matrix = torch.tensor(arch_matrix)
    m = torch.nn.Softmax(dim = 1)
    #self.arch = m(arch_matrix).cpu().numpy()
    #return m(arch_matrix).cpu().numpy()
    return m(arch_matrix)

  def get(self):
    '''Returns the architecture parameter'''
    return self.arch

  def update(self, value):
    assert isinstance(value, torch.Tensor), 'Value must be tensor'
    assert value.device == self.get().device, "Given tensor has to be in the same device"
    if value.ndim == 2:
      assert value.shape[0] == self.num_edges, 'Wrong number of edges'
      assert value.shape[1] == self.num_ops, 'Wrong number of operations'
    elif value.ndim == 1:
      assert value.shape[0] == (self.num_edges * self.num_ops), 'Input value has wrong dimension'
      value = value.reshape(self.num_edges, self.num_ops)
    
    #self.arch = value
    self.arch.data.copy_(value)
    #self.apply_softmax(value)

  def derive_architecture(self):
    self.apply_softmax()
    self.search_space(self.arch)

  def show_alphas_dataframe(self, show_sum = False, k=1):
    '''Returns the architecture weights in two pandas Dataframe.
      First dataframe is just the architecture weight and the
      second dataframe is the one hot representation of the max values in each row in first dataframe
    '''
    df = pd.DataFrame(columns = self.search_space, index = list(range(self.num_edges)))
    df_softmax = pd.DataFrame(columns = self.search_space, index = list(range(self.num_edges)))
    df_max = pd.DataFrame(columns = self.search_space, index = list(range(self.num_edges)), data = np.zeros(self.get().shape, dtype = np.int64))

    alphas = self.get().cpu().numpy()
    for index in range(self.num_edges):
      #df.loc[key] = self.arch_parameters.get()[index]
      df.loc[index] = alphas[index]
    
    alphas_softmax  = nn.functional.softmax(self.get(), dim=-1).cpu().numpy()
    #alphas_softmax = self.arch_parameters.apply_softmax(self.get_alphas()[0])
    for index in range(self.num_edges):
      df_softmax.loc[index] = alphas_softmax[index]
    
    max_idx = df.astype(float).idxmax(axis = 1)
    for column, row in enumerate(max_idx.index):
      df_max.loc[row][max_idx[column]] = k
    
    if show_sum:
      df_softmax['sum'] = df_softmax.sum(axis = 1)

    return df, df_max, df_softmax

  def multinomial_sample(self):
    for prob in self.arch:
      sample = np.random.choice(a = range(len(prob)), p = prob)

  def __str__(self):
    return '{}'.format(self.arch)

  def compare(self, value):
    '''
      Compare with the given value
    '''
    assert isinstance(value, torch.Tensor), "value has to be tensor"
    assert value.device == self.get().device, "Given tensor has to be in the same device"
    if value.size(0) == (self.num_edges * self.num_ops):
      #value = value.reshape(self.num_edges, self.num_ops)
      return torch.all(self.get() == value.view(self.num_edges, self.num_ops)).item()
    else:
      #print(f'value: {value},\t self.get(): {self.get()}, {torch.eq(self.get(), value)}')
      return torch.all(self.get() == value).item()

  def cuda(self, device):
    '''
    Send the architecture parameter to GPU
    '''
    assert isinstance(self.arch, torch.Tensor), "Architecture parameter has to be tensor"
    assert isinstance(device, torch.device), "Incorrect instance of device"
    self.arch = self.arch.to(device)
  
  def numpy(self):
    '''
    Send the architecture parameter to CPU in numpy format
    '''
    assert isinstance(self.get(), torch.Tensor), "Architecture parameter has to be tensor"
    return self.get().cpu().numpy()

  def evaluate(self, data_loader, model, criterion, device=None):
    '''
      Evaluate the architecture in the given dataset
    '''
    if device is None:
      device = self.get().device
    #alphas = self.get()
    data_time, batch_time = AverageMeter(), AverageMeter()
    arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    
    # Copying the given alphas to the model
    model.update_alphas(self.get())
    #print(f'self.get(): {self.get()}, {self.get().dtype}')
    #print(f'model.get_alphas(): {model.get_alphas()[0]}, {model.get_alphas()[0].dtype}')
    assert model.check_alphas(self.get()), "Given alphas has not been copied successfully to the model"
    
    model.eval()
    end = time.time()
    with torch.no_grad():
      for step, (inputs, targets) in enumerate(data_loader):
        #inputs = inputs.cuda(non_blocking=True)
        inputs = inputs.to(device=device, non_blocking=True)#, dtype=torch.float64)
        targets = targets.cuda(non_blocking=True)
        # measure data loading time
        data_time.update(time.time() - end)
        # prediction
        _, logits = model(inputs)
        #return logits, targets
        arch_loss = criterion(logits, targets)
        # record
        arch_prec1, arch_prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
        arch_losses.update(arch_loss.item(),  inputs.size(0))
        arch_top1.update  (arch_prec1.item(), inputs.size(0))
        arch_top5.update  (arch_prec5.item(), inputs.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        #break
    return arch_losses.avg, arch_top1.avg, arch_top5.avg
