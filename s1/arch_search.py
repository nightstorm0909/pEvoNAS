# First Train and then search architecture and shrink the search space by the elite individual

import os
import sys
sys.path.insert(0, './s1')

import copy
import logging
import random
import torch.nn as nn
import genotypes
import argparse
import numpy as np
import pandas as pd
import pickle
import torch.backends.cudnn as cudnn
import torch.utils
import torch.nn.functional as F
import torchvision
import time
import ut
import visualize

from genotypes               import PRIMITIVES
from config_utils            import load_config
from consensus               import consensus
from get_datasets            import get_dataloader
from ga                      import GeneticAlgorithm
from ut                      import AverageMeter, obtain_accuracy
from population              import *
from prog_model_search       import Network
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser("NAS201")
parser.add_argument('--data_dir', type = str, default = '../data', help = 'location of the data corpus')
parser.add_argument('--output_dir', type = str, default = None, help = 'location of trials')
parser.add_argument('--seed', type = int, default = None, help = 'random seed')
parser.add_argument('--gpu', type = int, default = 0, help = 'gpu device id')
parser.add_argument('--nas_config', type = str, default = None, help = 'location of configuration of NAS')
parser.add_argument('--learning_rate', type = float, default = 0.025, help = 'init learning rate')
parser.add_argument('--learning_rate_min', type = float, default = 0.001, help = 'min learning rate')
parser.add_argument('--momentum', type = float, default = 0.9, help = 'momentum')
parser.add_argument('--weight_decay', type = float, default = 3e-4, help = 'weight decay')
parser.add_argument('--grad_clip', type = float, default = 5, help = 'gradient clipping')
parser.add_argument('--report_freq', type = float, default = 100, help = 'report frequency')

# Added for NAS201
parser.add_argument('--dataset', type = str, default = 'cifar10', help = '["cifar10", "cifar100", "ImageNet16-120"]')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
parser.add_argument('--config_root', type=str, help='The root path of the config directory')
args = parser.parse_args()

def train(model, train_queue, criterion, optimizer, gen, args):
  losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
  model.train()
  for step, (inputs, targets) in enumerate(train_queue):
    model.random_alphas(discrete=args.train_discrete)
    
    n = inputs.size(0)
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    #inputs = inputs.cuda(non_blocking=True)
    #targets = targets.cuda(non_blocking=True)
    optimizer.zero_grad()
    logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk = (1, 5))
    losses.update(loss.item(),  inputs.size(0))
    top1.update  (prec1.item(), inputs.size(0))
    top5.update  (prec5.item(), inputs.size(0))
    
    #print(step)
    if (step) % args.report_freq == 0:
      logging.info(f"[Epoch #{gen}]: train_discrete: {args.train_discrete}")
      logging.info(f"Using Training batch #{step} with loss: {losses.avg:.5f}, prec1: {top1.avg:.5f}, prec5: {top5.avg:.5f}")
    #break
  logging.info(f"Training finished with loss: {losses.avg:.5f}, prec1: {top1.avg:.5f}, prec5: {top5.avg:.5f}")

def validation(model, valid_queue, criterion, gen, df):
  model.eval()
  for i in range(len(population.get_population())):
    valid_start = time.time()
    #Copying and checking the discretized alphas
    model.update_alphas(population.get_population()[i].arch_parameters)
    assert model.check_alphas(population.get_population()[i].arch_parameters), 'Failed to copy the architecture in the population'
    discrete_alphas = model.discretize()
    assert model.check_alphas(discrete_alphas)
    genotype_tmp = model.genotype()
    population.get_population()[i].genotype = genotype_tmp
    # Check if the architecture has already been evaluated
    series = ut.search_dataframe(df, genotype_tmp)
    if (series is not None):
      #print(series)
      generation, arch_loss, arch_top1, arch_top5 = series['generation'], series['arch_loss'], series['arch_top1'], series['arch_top5']
      logging.info(f'Already evaluated in generation #{generation}')
      population.get_population()[i].set_fitness(arch_loss, arch_top1, arch_top5)
    else:
      population.get_population()[i].objs.reset()
      population.get_population()[i].top1.reset()
      population.get_population()[i].top5.reset()
      with torch.no_grad():
        for step, (inputs, targets) in enumerate(valid_queue):
          n = inputs.size(0)
          inputs = inputs.to(device)
          targets = targets.to(device)
          logits = model(inputs)
          loss = criterion(logits, targets)
      
          prec1, prec5 = obtain_accuracy(logits, targets, topk = (1, 5))
          population.get_population()[i].objs.update(loss.data.cpu().item(), n)
          population.get_population()[i].top1.update(prec1.data.cpu().item(), n)
          population.get_population()[i].top5.update(prec5.data.cpu().item(), n)
        
          #print(step)
          #if (step + 1) % 10 == 0:
          #break
      #print("Finished in {} seconds".format((time.time() - valid_start) ))

    logging.info("[{} Generation] {}/{} finished with validation loss: {:.5f}, prec1: {:.5f}, prec5: {:.5f}".format(gen, i+1, len(population.get_population()), 
                                                      population.get_population()[i].objs.avg, 
                                                      population.get_population()[i].top1.avg, 
                                                      population.get_population()[i].top5.avg))

if args.seed is None or args.seed < 0: args.seed = random.randint(1, 100000)
nas_config = ut.NAS_config(cfg_file=args.nas_config, sectionName='S1').NAS_cfg
args.pop_size, args.progression = nas_config['populationSize'], nas_config['progression']
args.batch_size, args.valid_batch_size = nas_config['batch_size'], nas_config['valid_batch_size']
args.num_elites, args.mutate_rate = nas_config['num_elites'], nas_config['mutate_rate']
args.tsize, args.inherit_wts = nas_config['tsize'], nas_config['inherit_wts']
args.cutout, args.cutout_length = nas_config['cutout'], nas_config['cutout_length']
args.autoaug, args.split_option = nas_config['autoaug'], nas_config['split_option']
args.converged, args.epochs = nas_config['convergeBreak'], nas_config['epochs']
args.train_discrete = nas_config['train_discrete']
args.train_iterations = nas_config['train_iterations']
args.init_channels, args.layers = nas_config['init_channels'], nas_config['layers']

if args.output_dir is not None:
  if not os.path.exists(args.output_dir):
    ut.create_exp_dir(args.output_dir)
  else:
    print('The output directory already exists')
    exit(0)
  #DIR = os.path.join(args.output_dir, DIR)
  DIR = args.output_dir
else:
  DIR = "search-{}-{}".format(time.strftime("%Y%m%d-%H%M%S"), args.dataset)
  DIR = os.path.join(os.getcwd(), DIR)
  ut.create_exp_dir(DIR)
ut.create_exp_dir(os.path.join(DIR, "weights"))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(DIR, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# Initializing the summary writer
writer = SummaryWriter(os.path.join(DIR, 'runs'))

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device("cuda:{}".format(args.gpu))
cpu_device = torch.device("cpu")

torch.cuda.set_device(args.gpu)
cudnn.deterministic = True
cudnn.enabled = True
cudnn.benchmark = False

logging.info(f'python {" ".join([ar for ar in sys.argv])}')
logging.info('First Train and then search architecture and shrink the search space by the elite individual')

# Configuring dataset and dataloader
train_transform, valid_transform, train_loader, valid_loader = get_dataloader(args)
logging.info(f'train_transform: {train_transform}, \nvalid_transform: {valid_transform}')
if args.dataset == 'cifar10':    num_classes = 10
elif args.dataset == 'cifar100': num_classes = 100
logging.info("#classes: {}".format(num_classes))
train_queue, valid_queue = train_loader, valid_loader
logging.info('search_loader: {}, valid_loader: {}'.format(len(train_queue)*args.batch_size, len(valid_queue)*args.valid_batch_size))

logging.info(f'torch version: {torch.__version__}, torchvision version: {torchvision.__version__}')
logging.info("gpu device = {}".format(args.gpu))
logging.info(f'nas_config: {nas_config}')
logging.info("args =  %s", args)

# Initializing Genetic Algorithm
ga = GeneticAlgorithm(args.num_elites, args.tsize, device, args.mutate_rate)

arch_flag = None

count_steps = 0
start = time.time()
for idx, prog_number in enumerate(args.progression):
  prog_start = time.time()
  converge_flag = False
  logging.info('='*100)
  if idx==0: logging.info(f'Starting the search with {len(PRIMITIVES)} operations')
  else:      logging.info(f'Shrinking the search space to {prog_number} operations')
  if idx > 0:
    str_tmp = ''
    for p in population.get_population():
      str_tmp += f'{str(p.get_fitness())}, '
    logging.info(f'Before shrinking: {str_tmp}')

    c = consensus( arch_flag=arch_flag,
                   population=population, edge2index=edge2index, op_dict=op_dict,
                   op_position=[model.cells[0].op_position, model.cells[len(model.cells)//3].op_position],
                   topk=prog_number,
                   weighted=False)
    logging.info(f'{population.get_population()[0].arch_parameters}')
    logging.info(f'population.get_population()[0].fitness: {population.get_population()[0].get_fitness()}')
    arch_flag = c.shrink_by_one_ind(population.get_population()[0])
    ## Creating Population
    population = Population(pop_size=args.pop_size,
                            steps=model._steps,
                            device=device,
                            num_ops=prog_number)
    smaller_model = Network(C=args.init_channels, num_classes=num_classes, layers=args.layers, device=device, arch_flag=arch_flag)
    smaller_model = smaller_model.to(device)
    if args.inherit_wts:
      logging.info('[INFO] Inherting the weights from the previous one shot model')
      ut.inherit_OSM_wts(big_OSM=model, small_OSM=smaller_model, verbose=False)
    del model
    model = smaller_model
    logging.info(smaller_model.get_alphas())
    logging.info(model.get_alphas())
  else:
    # First progression
    model = Network(C=args.init_channels, num_classes=num_classes, layers=args.layers, device=device)
    model = model.to(device)
    ## Creating Population
    population = Population(pop_size=args.pop_size,
                            steps=model._steps,
                            device=device,
                            num_ops=len(PRIMITIVES))
    edge2index = model.edge2index
    op_dict = model.op_dict
  #logging.info(model)
  arch_flag = [flag_tmp.detach().cpu().clone() for flag_tmp in model.arch_flag]
  with open(os.path.join(DIR, f"arch_flag_{idx+1}.pickle"), 'wb') as f:
    pickle.dump(arch_flag, f)

  # Configuring the optimizer and the scheduler
  optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum = args.momentum, weight_decay = args.weight_decay)
  criterion = nn.CrossEntropyLoss()
  criterion.to(device)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.train_iterations), eta_min = args.learning_rate_min)
  lr = scheduler.get_lr()[0]
  logging.info(f'optimizer: {optimizer}\nCriterion: {criterion}')
  logging.info(f'Scheduler: {scheduler}')
  logging.info(f'model architecture flag:\n{arch_flag}')

  # logging the initialized architecture
  best_arch_per_epoch = []

  ## TRAINING
  logging.info('[INFO] Training the One Shot Model')
  for epoch in range(args.train_iterations):
    train_time = time.time()
    logging.info("[INFO] Epoch {} with learning rate {}".format(epoch + 1, scheduler.get_lr()[0]))
    train(model, train_queue, criterion, optimizer, epoch + 1, args)
    logging.info("[INFO] Training finished in {} minutes".format((time.time() - train_time) / 60))
    scheduler.step()
    ut.save(model, os.path.join(DIR, "weights", f"weights_{idx+1}.pt"))
 
  ## EVOLUTION
  df = pd.DataFrame(columns=['prog_id', 'generation',  'arch', 'genotype', 'arch_loss', 'arch_top1', 'arch_top5'])
  for epoch in range(args.epochs):
    start_time = time.time()
    logging.info("[INFO] Evaluating Generation {} ".format(epoch + 1))
    validation(model, valid_queue, criterion, epoch + 1, df)

    # Sorting the population according to the fitness in decreasing order
    population.pop_sort()
    
    str_tmp = ''
    for i, p in enumerate(population.get_population()):
      writer.add_scalar("{}_pop_top1_{}".format(idx+1, i + 1), p.get_fitness(), epoch + 1)
      writer.add_scalar("{}_pop_top5_{}".format(idx+1, i + 1), p.top5.avg, epoch + 1)
      writer.add_scalar("{}_pop_obj_valid_{}".format(idx+1, i + 1), p.objs.avg, epoch + 1)
      str_tmp += f'{str(p.get_fitness())}, '
      #columns=['prog_id', 'generation',  'arch', 'genotype', 'arch_loss', 'arch_top1', 'arch_top5']
      d_tmp = { 'prog_id': idx+1, 'generation': epoch+1,
                'arch': [arch_tmp.cpu().numpy() for arch_tmp in p.arch_parameters],
                'genotype': p.genotype, 'arch_loss': p.objs.avg, 'arch_top1': p.get_fitness(), 'arch_top5': p.top5.avg}
      df = df.append(d_tmp, ignore_index=True)
    logging.info(str_tmp)

    # Copying the best individual to the model
    model.update_alphas(population.get_population()[0].arch_parameters)
    assert model.check_alphas(population.get_population()[0].arch_parameters)
    genotype_tmp = model.genotype()
    logging.info(model.show_alphas_dataframe(show_sum=True))
    count_steps += 1
    tmp = genotype_tmp
    best_arch_per_epoch.append(tmp)
    logging.info(f'[INFO] Best architecture after {epoch+1} generation {genotype_tmp}')

    # Checking if the evolution has converged
    if args.converged < (len(best_arch_per_epoch)):
      tmp_list = copy.deepcopy(best_arch_per_epoch)
      tmp_list.reverse()
      if ut.converged(tmp_list[:args.converged]): converge_flag = True
   
    if converge_flag:
      logging.info(f'[INFO] Breaking the evolution since the top architecture repeated for {args.converged} generations')
      break

    # Applying Genetic Algorithm
    if epoch != (args.epochs-1):
      pop = ga.evolve(population)
      population = pop 
    
    last = time.time() - start_time
    logging.info("[INFO] Generation {} finished in {} minutes".format(epoch + 1, last / 60))

  with open(os.path.join(DIR, f"best_architectures_{idx+1}.pickle"), 'wb') as f:
    pickle.dump(best_arch_per_epoch, f)

  df.to_json(os.path.join(DIR, f'all_population_{idx+1}.json'))
  last = time.time() - prog_start
  logging.info("[INFO] Progression#{} finished in {} hours".format(idx+1, last / 3600))

writer.close()

last = time.time() - start
logging.info("[INFO] Architecture search finished in {} hours".format(last / 3600))

best_architecture = best_arch_per_epoch[-1]
with open(os.path.join(DIR, "genotype.pickle"), 'wb') as f:
  pickle.dump(best_architecture, f)
normal_cell, reduce_cell = best_architecture.normal, best_architecture.reduce
visualize.plot(genotype=normal_cell, filename=os.path.join(DIR, 'normal'))
visualize.plot(genotype=reduce_cell, filename=os.path.join(DIR, 'reduction'))
logging.info(f'[INFO] Best Architecture after the search: {best_architecture}')
logging.info(f'length best_arch_per_epoch: {len(best_arch_per_epoch)}, count_steps: {count_steps}')

print(DIR)
#model.cpu()
#del model
# Evaluating the architecture
#cmd_str = f'bash ./s1/eval_arch.sh {args.dataset} {args.gpu} {DIR} {args.batch_size}'
#logging.info(cmd_str)
#os.system(cmd_str)
