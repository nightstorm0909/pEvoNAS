import torch
import utils

class chromosome:
  def __init__(self, num_edges, device, num_ops = None):
    self._device = device
    self._num_edges = num_edges
    
    assert num_ops is not None, 'Number of operations not given'
    self.num_ops = num_ops

    self.alphas_normal = self.generate_parameters(self._num_edges).to(self._device)
    self.arch_parameters = [self.alphas_normal]
    
    self.objs = utils.AvgrageMeter()
    self.top1 = utils.AvgrageMeter()
    self.top5 = utils.AvgrageMeter()

    self.genotype = None    

  def update(self):
    self.alphas_normal = self.arch_parameters

  def set_fitness(self, value, top1, top5):
    self.objs.avg = value
    self.top1.avg = top1
    self.top5.avg = top5

  def get_len(self):
    #return self.k * len(self.arch_parameters)
    return _num_edges 

  def get_fitness(self):
    return self.top1.avg

  def get_all_metrics(self):
    return self.objs, self.top1, self.top5

  def get_arch_parameters(self):
    return self.arch_parameters

  def generate_parameters(self, k):
    return torch.rand(k, self.num_ops)
    #return torch.nn.functional.one_hot(torch.randint(low = 0, high = self.num_ops, size = (k, )), num_classes = self.num_ops)
