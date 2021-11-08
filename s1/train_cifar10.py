import os
import sys
sys.path.insert(0,'./')

import time
import glob
import numpy as np
import torch
import ut
import logging
import argparse
import torch.nn as nn
import genotypes
import visualize
import random
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision

from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from model import NetworkCIFAR as Network

import pickle

parser = argparse.ArgumentParser("cifar10")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dir', type=str, default = None, help='location of population')
parser.add_argument('--pop_num', type=int, default=50, help='Generation number')
parser.add_argument('--ga', action='store_true', default=False, help='use ga generated population for evaluation')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=None, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

if args.seed is None or args.seed < 0: args.seed = random.randint(1, 100000)
args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
if args.dir is not None:
  args.save = os.path.join(args.dir, args.save)
ut.create_exp_dir(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'eval_log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info('[INFO] torch version: {}, torchvision version: {}'.format(torch.__version__, torchvision.__version__))

CIFAR_CLASSES = 10
writer = SummaryWriter(os.path.join(args.save, 'runs'))

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  device = torch.device("cuda:{}".format(args.gpu))
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  cudnn.deterministic = True
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  #genotype = eval("genotypes.%s" % args.arch)
  
  if os.path.exists(os.path.join(args.dir, "genotype.pickle")):
    with open(os.path.join(args.dir, "genotype.pickle"), 'rb') as f:
      genotype = pickle.load(f)
    print("Unpickling genotype.pickle")
    visualize.plot(genotype.normal, os.path.join(args.save, "normal"), False)
    visualize.plot(genotype.reduce, os.path.join(args.save, "reduction"), False)
  
  elif args.dir is not None:
    print("No genotype in the given directory")
    return

  logging.info(genotype)

  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  model = model.cuda()

  logging.info("param size = %fMB", ut.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
      )

  train_transform, valid_transform = ut._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
  logging.info("[INFO] len(train_data): {}, len(valid_data): {}".format(len(train_data), len(valid_data)))

  train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
    valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

  test_error = []
  best_acc_top1 = 0.0

  for epoch in range(args.epochs):
    logging.info('[INFO] epoch (%d/%d) lr %e', epoch + 1, args.epochs,scheduler.get_last_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    epoch_start = time.time()
    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    logging.info('[INFO] train_acc %f finished in %f minutes', train_acc, (time.time()-epoch_start) / 60)
    writer.add_scalar("train_acc", train_acc, epoch + 1)
    writer.add_scalar("train_obj", train_obj, epoch + 1)
    scheduler.step()

    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('[INFO] valid_acc %f', valid_acc)
    writer.add_scalar("valid_acc", valid_acc, epoch + 1)
    writer.add_scalar("valid_obj", valid_obj, epoch + 1)
    writer.add_scalar("test_error", 100 - valid_acc, epoch + 1)

    ut.save(model, os.path.join(args.save, 'weights.pt'))
    test_error.append(100 - valid_acc)

    if valid_acc > best_acc_top1:
      ut.save(model, os.path.join(args.save, 'best_weights.pt'))
      best_acc_top1 = valid_acc
    writer.add_scalar("best_acc", best_acc_top1, epoch + 1)
    writer.add_scalar("best_test_error", 100 - best_acc_top1, epoch + 1)
    logging.info(f'[INFO] Epoch finished in {(time.time() - epoch_start) / 60}')
    logging.info('='*100)

  logging.info(f'best_acc: {best_acc_top1.item()}, valid_acc: {valid_acc.item()}')

  with open("{}/test_error.pickle".format(args.save), 'wb') as f:
    pickle.dump(test_error, f)

def train(train_queue, model, criterion, optimizer):
  objs = ut.AvgrageMeter()
  top1 = ut.AvgrageMeter()
  top5 = ut.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda()

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = ut.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data, n)
    top1.update(prec1.data, n)
    top5.update(prec5.data, n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
      #break

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = ut.AvgrageMeter()
  top1 = ut.AvgrageMeter()
  top5 = ut.AvgrageMeter()
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = Variable(input, volatile=True).cuda()
      target = Variable(target, volatile=True).cuda()

      logits, _ = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = ut.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.data, n)
      top1.update(prec1.data, n)
      top5.update(prec5.data, n)

      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
        #break

  return top1.avg, objs.avg


if __name__ == '__main__':
  start_time = time.time()
  main()
  logging.info("[INFO] Training finished in {} hours".format((time.time() - start_time) / 3600))
  writer.close()
