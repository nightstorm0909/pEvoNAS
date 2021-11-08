import torch

from chromosomesNAS201 import *

class Population:
  def __init__(self, pop_size, num_edges, num_ops, device = torch.device("cpu")):
    self._pop_size = pop_size
    self._device = device
    self._num_edges = num_edges
    self._num_ops = num_ops
    self.population = []
    for _ in range(pop_size):
      self.population.append(chromosome(self._num_edges, self._device, self._num_ops))

  def get_population_size(self):
    return len(self.population)

  def get_population(self):
    return self.population

  def print_population_fitness(self):
    for p in self.population:
      print(p.get_fitness(), end=', ')

  def pop_sort(self):
    self.population.sort(key = lambda x: x.get_fitness(), reverse = True)

  def random_pop(self):
    self.population = []
    for _ in range(self._pop_size):
      self.population.append(chromosome(self._num_edges, self._device, self._num_ops))
