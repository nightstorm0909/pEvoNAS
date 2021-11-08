import torch
from genotypes import PRIMITIVES
import ut

class chromosome:
  def __init__(self, steps, device, num_ops=None):
    self._device = device
    self._steps = steps
    self.k = sum(1 for i in range(steps) for _ in range(2 + i))
    if num_ops is None: self.num_ops = len(PRIMITIVES)
    else:               self.num_ops = num_ops

    self.alphas_normal = self.generate_parameters(self.k).to(self._device)
    self.alphas_reduce = self.generate_parameters(self.k).to(self._device)
    self.arch_parameters = [self.alphas_normal, self.alphas_reduce]
    
    self.objs = ut.AvgrageMeter()
    self.top1 = ut.AvgrageMeter()
    self.top5 = ut.AvgrageMeter()

    self.evaluated = False    
    self.encode()
    self.tmp = []

  def accumulate(self):
    self.tmp.append(self.top1.avg)

  def update(self):
    self.alphas_normal, self.alphas_reduce = self.arch_parameters
    self.encode()

  def encode(self):
    self.genes = torch.cat(self.arch_parameters).view(-1)

  def decode(self):
    tmp = self.genes.view(-1, self.num_ops)
    self.alphas_normal = tmp[: self.k]
    self.alphas_reduce = tmp[self.k :]
    self.arch_parameters = [self.alphas_normal, self.alphas_reduce]

  def set_fitness(self, value, top1, top5):
    self.objs.avg = value
    self.top1.avg = top1
    self.top5.avg = top5

  def get_len(self):
    return self.k * len(self.arch_parameters)

  def get_fitness(self):
    return self.top1.avg

  def get_all_metrics(self):
    return self.objs, self.top1, self.top5

  def get_arch_parameters(self):
    return self.arch_parameters

  def generate_parameters(self, k):
    return torch.rand(k, self.num_ops)
    #return torch.nn.functional.one_hot(torch.randint(low = 0, high = self.num_ops, size = (k, )), num_classes = self.num_ops)
