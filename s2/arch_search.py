# First Train and then search architecture and shrink the search space by the elite individual

import os
import sys
sys.path.insert(0, './s2')

import logging
import random
import torch.nn as nn
import genotypes
import argparse
import numpy as np
import pandas as pd
import pickle
import torch.utils
import torchvision
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torch.utils
import torch.nn.functional as F
import time
import utils

from cell_operations         import NAS_BENCH_201
from config_utils            import load_config
from consensus               import consensus
from collections             import Counter
from csv                     import DictWriter
from datasets                import get_datasets, get_nas_search_loaders
from gaNAS201                import GeneticAlgorithm
from log_utils               import AverageMeter
from nas_201_api             import NASBench201API as API
from populationNAS201        import *
from procedures              import get_optim_scheduler
from prog_search_model_s2    import TinyNetwork_Prog as TinyNetwork
from torch.utils.tensorboard import SummaryWriter
from torch.autograd          import Variable
from utils                   import obtain_accuracy, model_save

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
parser.add_argument('--record_filename', type = str, default = None, help = 'filename of the csv file for recording the final results')
parser.add_argument('--report_freq', type = float, default = 50, help = 'report frequency')
parser.add_argument('--init_channels', type = int, default = 16, help = 'num of init channels')

# Added for NAS201
#parser.add_argument('--channel', type = int, default = 16, help = 'initial channel for NAS201 network')
parser.add_argument('--num_cells', type = int, default = 5, help = 'number of cells for NAS201 network')
parser.add_argument('--max_nodes', type = int, default = 4, help = 'maximim nodes in the cell for NAS201 network')
parser.add_argument('--dataset', type = str, default = 'cifar10', help = '["cifar10", "cifar100", "ImageNet16-120"]')
parser.add_argument('--api_path', type = str, default = None, help = '["cifar10", "cifar10-valid","cifar100", "imagenet16-120"]')
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
parser.add_argument('--config_path', type=str, help='The config path for the one shot model')
parser.add_argument('--config_root', type=str, help='The root path of the config directory')
args = parser.parse_args()

def get_arch_score(api, arch_str, dataset, acc_type=None, use_012_epoch_training=False):
  arch_index = api.query_index_by_arch( arch_str )
  assert arch_index >= 0, 'can not find this arch : {:}'.format(arch_str)
  if use_012_epoch_training:
    info = api.get_more_info(arch_index, dataset, iepoch=None, hp='12', is_random=True)
    valid_acc, time_cost = info['valid-accuracy'], info['train-all-time'] + info['valid-per-time']
    return valid_acc, time_cost
  else:
    return api.query_by_index(arch_index=arch_index, hp = '200').get_metrics(dataset, acc_type)['accuracy']

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
    _, logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk = (1, 5))
    losses.update(loss.item(),  inputs.size(0))
    top1.update  (prec1.item(), inputs.size(0))
    top5.update  (prec5.item(), inputs.size(0))
    
    #print(step)
    if (step) % 100 == 0:
      logging.info(f"[Epoch #{gen}]: train_discrete: {args.train_discrete}")
      logging.info(f"Using Training batch #{step} with loss: {losses.avg:.5f}, prec1: {top1.avg:.5f}, prec5: {top5.avg:.5f}")
    #break
  logging.info(f"Training finished with loss: {losses.avg:.5f}, prec1: {top1.avg:.5f}, prec5: {top5.avg:.5f}")

def validation(model, valid_queue, criterion, gen, df):
  model.eval()
  for i in range(len(population.get_population())):
    valid_start = time.time()
    #Copying and checking the discretized alphas
    model.update_alphas(population.get_population()[i].arch_parameters[0])
    discrete_alphas = model.discretize()
    assert model.check_alphas(discrete_alphas)
    arch_str_tmp = model.genotype().tostr() 
    population.get_population()[i].genotype = arch_str_tmp
    if (not df.empty) and (not df[ df['genotype']==arch_str_tmp ].empty ):
      series = df[ df['genotype']==arch_str_tmp ]
      generation, arch_loss, arch_top1, arch_top5 = series['generation'].values[0], series['arch_loss'].values[0], series['arch_top1'].values[0], series['arch_top5'].values[0]
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
          _, logits = model(inputs)
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
nas_config = utils.NAS_config(cfg_file=args.nas_config, sectionName='S2').NAS_cfg
args.pop_size, args.progression = nas_config['populationSize'], nas_config['progression']
args.batch_size, args.valid_batch_size = nas_config['batch_size'], nas_config['valid_batch_size']
args.num_elites, args.mutate_rate = nas_config['num_elites'], nas_config['mutate_rate']
args.tsize, args.inherit_wts = nas_config['tsize'], nas_config['inherit_wts']
args.track_running_stats = nas_config['track_running_stats']
args.cutout, args.cutout_length = nas_config['cutout'], nas_config['cutout_length']
args.converged, args.epochs = nas_config['convergeBreak'], nas_config['epochs']
args.train_discrete, args.weighted_vote = nas_config['train_discrete'], nas_config['weighted_vote']
args.train_iterations = nas_config['train_iterations']

DIR = "search-{}-{}".format(time.strftime("%Y%m%d-%H%M%S"), args.dataset)
if args.output_dir is not None:
  if not os.path.exists(args.output_dir):
    utils.create_exp_dir(args.output_dir)
  DIR = os.path.join(args.output_dir, DIR)
else:
  DIR = os.path.join(os.getcwd(), DIR)
utils.create_exp_dir(DIR)
utils.create_exp_dir(os.path.join(DIR, "weights"))
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
assert args.api_path is not None, 'NAS201 data path has not been provided'
api = API(args.api_path, verbose = False)
logging.info(f'length of api: {len(api)}')

# Configuring dataset and dataloader
if args.dataset == 'cifar10':
  acc_type     = 'ori-test'
  val_acc_type = 'x-valid'
else:
  acc_type     = 'x-test'
  val_acc_type = 'x-valid'

datasets = ['cifar10', 'cifar100', 'ImageNet16-120']
assert args.dataset in datasets, 'Incorrect dataset'
if args.cutout:
  train_data, valid_data, xshape, num_classes = get_datasets(name = args.dataset, root = args.data_dir, cutout=args.cutout)
else:
  train_data, valid_data, xshape, num_classes = get_datasets(name = args.dataset, root = args.data_dir, cutout=-1)
logging.info("train data len: {}, valid data len: {}, xshape: {}, #classes: {}".format(len(train_data), len(valid_data), xshape, num_classes))

config = load_config(path=args.config_path, extra={'class_num': num_classes, 'xshape': xshape}, logger=None)
logging.info(f'config: {config}')
_, train_loader, valid_loader = get_nas_search_loaders(train_data=train_data, valid_data=valid_data, dataset=args.dataset,
                                                        config_root=args.config_root, batch_size=(args.batch_size, args.valid_batch_size),
                                                        workers=args.workers)
train_queue, valid_queue = train_loader, valid_loader
logging.info('search_loader: {}, valid_loader: {}'.format(len(train_queue), len(valid_queue)))

logging.info(f'torch version: {torch.__version__}, torchvision version: {torchvision.__version__}')
logging.info("gpu device = {}".format(args.gpu))
logging.info(f'nas_config: {nas_config}')
logging.info("args =  %s", args)

# Initializing Genetic Algorithm
ga = GeneticAlgorithm(args.num_elites, args.tsize, device, args.mutate_rate)

arch_flag = None
op_dict = {}
for idx, op in enumerate(NAS_BENCH_201):
  op_dict[op] = idx

count_steps = 0
start = time.time()
for idx, prog_number in enumerate(args.progression):
  prog_start = time.time()
  converge_flag = False
  logging.info('='*150)
  if idx==0: logging.info(f'Starting the search with {len(NAS_BENCH_201)} operations')
  else:      logging.info(f'Shrinking the search space to {prog_number} operations')
  # Model Initialization
  #model_config = {'C': 16, 'N': 5, 'num_classes': num_classes, 'max_nodes': 4, 'search_space': NAS_BENCH_201, 'affine': False}
  if idx > 0:
    #population.print_population_fitness()
    str_tmp = ''
    for p in population.get_population(): str_tmp += f'{str(p.get_fitness())}, '
    logging.info(f'Before shrinking: {str_tmp}')
    c = consensus( arch_flag=arch_flag,
                   population=population, edge2index=edge2index, op_dict=op_dict,
                   op_position=model.cells[0].op_position, topk=prog_number,
                   weighted=args.weighted_vote)
    logging.info(f'{population.get_population()[0].arch_parameters[0]}')
    logging.info(f'population.get_population()[0].fitness: {population.get_population()[0].get_fitness()}')
    arch_flag = c.shrink_by_one_ind(population.get_population()[0])
    ## Creating Population
    population = Population(pop_size=args.pop_size,
                            num_edges=len(edge2index),
                            num_ops=prog_number, device=device)
    smaller_model = TinyNetwork(C = args.init_channels,     N = args.num_cells,
                                max_nodes = args.max_nodes, num_classes = num_classes, 
                                search_space = NAS_BENCH_201, affine = False,
                                track_running_stats = args.track_running_stats,
                                arch_flag=arch_flag)
    smaller_model = smaller_model.to(device)
    if args.inherit_wts:
      logging.info('[INFO] Inherting the weights from the previous one shot model')
      utils.inherit_OSM_wts(big_OSM=model, small_OSM=smaller_model, verbose=False)
    del model
    model = smaller_model
    logging.info(smaller_model.arch_parameters)
    logging.info(model.arch_parameters)
  else:
    model = TinyNetwork(C = args.init_channels, N = args.num_cells, max_nodes = args.max_nodes,
                        num_classes = num_classes, search_space = NAS_BENCH_201, affine = False,
                        track_running_stats = args.track_running_stats, arch_flag=arch_flag)
    model = model.to(device)
    ## Creating Population
    population = Population(pop_size=args.pop_size,
                            num_edges=model.get_alphas()[0].shape[0],
                            num_ops=len(NAS_BENCH_201), device=device)
    edge2index = model.edge2index
  #logging.info(model)
  arch_flag = model.arch_flag.detach().clone()
  with open(os.path.join(DIR, f"arch_flag_{idx+1}.pickle"), 'wb') as f:
    pickle.dump(arch_flag.cpu(), f)

  optimizer, _, criterion = get_optim_scheduler(parameters=model.get_weights(), config=config)
  criterion = criterion.cuda()
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.train_iterations), eta_min = args.learning_rate_min)
  lr = scheduler.get_lr()[0]
  logging.info(f'optimizer: {optimizer}\nCriterion: {criterion}')
  logging.info(f'Scheduler: {scheduler}')
  logging.info(f'model architecture flag:\n{arch_flag}')

  # logging the initialized architecture
  best_arch_per_epoch = []

  arch_str = model.genotype().tostr()
  #arch_index = api.query_index_by_arch(model.genotype())
  if args.dataset == 'cifar10':
    test_acc = get_arch_score(api=api, arch_str=arch_str, dataset='cifar10', acc_type=acc_type, use_012_epoch_training=False)
    valid_acc = get_arch_score(api=api, arch_str=arch_str, dataset='cifar10-valid', acc_type=val_acc_type, use_012_epoch_training=False)
  else:
    test_acc = get_arch_score(api=api, arch_str=arch_str, dataset=args.dataset, acc_type=acc_type, use_012_epoch_training=False)
    valid_acc = get_arch_score(api=api, arch_str=arch_str, dataset=args.dataset, acc_type=val_acc_type, use_012_epoch_training=False)
  writer.add_scalar(f"{idx+1}_test_acc", test_acc, 0)
  writer.add_scalar(f"{idx+1}_valid_acc", valid_acc, 0)
  writer.add_scalar("test_acc", test_acc, count_steps)
  writer.add_scalar("valid_acc", valid_acc, count_steps)
  count_steps += 1
  tmp = (arch_str, test_acc, valid_acc)
  best_arch_per_epoch.append(tmp)

  logging.info('[INFO] Training the One Shot Model')
  for epoch in range(args.train_iterations):
    train_time = time.time()
    logging.info("[INFO] Epoch {} with learning rate {}".format(epoch + 1, scheduler.get_lr()[0]))
    train(model, train_queue, criterion, optimizer, epoch + 1, args)
    logging.info("[INFO] Training finished in {} minutes".format((time.time() - train_time) / 60))
    scheduler.step()
    model_save(model, os.path.join(DIR, "weights", f"weights_{idx+1}.pt"))
  
  df = pd.DataFrame(columns=['prog_id', 'generation',  'arch', 'genotype', 'arch_loss', 'arch_top1', 'arch_top5', 'test_acc', 'valid_acc'])
  for epoch in range(args.epochs):
    start_time = time.time()
    logging.info("[INFO] Evaluating Generation {} ".format(epoch + 1))
    validation(model, valid_queue, criterion, epoch + 1, df)

    # Sorting the population according to the fitness in decreasing order
    population.pop_sort()
    population.print_population_fitness()
    
    for i, p in enumerate(population.get_population()):
      writer.add_scalar("{}_pop_top1_{}".format(idx+1, i + 1), p.get_fitness(), epoch + 1)
      writer.add_scalar("{}_pop_top5_{}".format(idx+1, i + 1), p.top5.avg, epoch + 1)
      writer.add_scalar("()_pop_obj_valid_{}".format(idx+1, i + 1), p.objs.avg, epoch + 1)
      if args.dataset == 'cifar10':
        #columns=['prog_id', 'generation',  'arch', 'genotype', 'arch_loss', 'arch_top1', 'arch_top5', 'test_acc', 'valid_acc']
        d_tmp = { 'prog_id': idx+1, 'generation': epoch+1, 'arch': p.arch_parameters[0].cpu().numpy(), 'genotype': p.genotype,
                  'arch_loss': p.objs.avg, 'arch_top1': p.get_fitness(), 'arch_top5': p.top5.avg,
                  'test_acc': get_arch_score(api=api, arch_str=arch_str, dataset='cifar10', acc_type=acc_type, use_012_epoch_training=False),
                  'valid_acc': get_arch_score(api=api, arch_str=arch_str, dataset='cifar10-valid', acc_type=val_acc_type, use_012_epoch_training=False)}
      else:
        d_tmp = { 'prog_id': idx+1, 'generation': epoch+1, 'arch': p.arch_parameters[0].cpu().numpy(), 'genotype': p.genotype,
                  'arch_loss': p.objs.avg, 'arch_top1': p.get_fitness(), 'arch_top5': p.top5.avg,
                  'test_acc': get_arch_score(api=api, arch_str=arch_str, dataset=args.dataset, acc_type=acc_type, use_012_epoch_training=False),
                  'valid_acc': get_arch_score(api=api, arch_str=arch_str, dataset=args.dataset, acc_type=val_acc_type, use_012_epoch_training=False)}
      df = df.append(d_tmp, ignore_index=True)

    # Copying the best individual to the model
    model.update_alphas(population.get_population()[0].arch_parameters[0])
    assert model.check_alphas(population.get_population()[0].arch_parameters[0])
    arch_str = model.genotype().tostr()
    logging.info(model.show_alphas_dataframe(show_sum=True))
    #arch_index = api.query_index_by_arch(model.genotype())
    if args.dataset == 'cifar10':
      test_acc = get_arch_score(api=api, arch_str=arch_str, dataset='cifar10', acc_type=acc_type, use_012_epoch_training=False)
      valid_acc = get_arch_score(api=api, arch_str=arch_str, dataset='cifar10-valid', acc_type=val_acc_type, use_012_epoch_training=False)
    else:
      test_acc = get_arch_score(api=api, arch_str=arch_str, dataset=args.dataset, acc_type=acc_type, use_012_epoch_training=False)
      valid_acc = get_arch_score(api=api, arch_str=arch_str, dataset=args.dataset, acc_type=val_acc_type, use_012_epoch_training=False)
    writer.add_scalar(f"{idx+1}_test_acc", test_acc, epoch + 1)
    writer.add_scalar(f"{idx+1}_valid_acc", valid_acc, epoch + 1)
    writer.add_scalar("test_acc", test_acc, count_steps)
    writer.add_scalar("valid_acc", valid_acc, count_steps)
    count_steps += 1
    tmp = (arch_str, test_acc, valid_acc)
    best_arch_per_epoch.append(tmp)
    logging.info(f'[INFO] Best architecture after {epoch+1} generation{arch_str}||{test_acc}, {valid_acc}')

    # Checking if the evolution has converged
    if args.converged < (len(best_arch_per_epoch)):
      tmp_list = [arch[0] for arch in best_arch_per_epoch]
      tmp_list.reverse()
      if len(Counter(tmp_list[:args.converged])) == 1: converge_flag = True
   
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

logging.info(f'[INFO] Best Architecture after the search: {best_arch_per_epoch[-1]}')
logging.info(f'length best_arch_per_epoch: {len(best_arch_per_epoch)}, count_steps: {count_steps}')


# Recording the best architecture informations after the search finishes
tmp_a = {'arch': best_arch_per_epoch[-1][0], 'dataset': args.dataset, 'valid': best_arch_per_epoch[-1][2],
         'test': best_arch_per_epoch[-1][1], 'time': last/3600}
if os.path.exists(os.path.join(args.output_dir, args.record_filename)):
  with open(os.path.join(args.output_dir, args.record_filename), 'a', newline='') as write_obj:
    dict_writer = DictWriter(write_obj, fieldnames=list(tmp_a.keys()) )
    dict_writer.writerow(tmp_a)
else:
  with open(os.path.join(args.output_dir, args.record_filename), 'w', newline='') as write_obj:
    dict_writer = DictWriter(write_obj, fieldnames=list(tmp_a.keys()) )
    dict_writer.writeheader()
    dict_writer.writerow(tmp_a)
  #df = pd.DataFrame()
  #df = df.append(tmp_a, ignore_index=True)
  #df = df.set_index('arch')
  #df.to_csv(os.path.join(args.output_dir, args.record_filename))
