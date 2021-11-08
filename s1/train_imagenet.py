import sys
sys.path.insert(0,'./')

import argparse
import genotypes
import glob
import logging
import numpy as np
import os
import pickle
import random
import time
import torch
import utils
import ut
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from model import NetworkImageNet as Network
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser("training imagenet")
parser.add_argument('--dir', type=str, default = None, help='location of population')
parser.add_argument('--gpu', type=int, default=None, help='gpu device id')
parser.add_argument('--gpu_list', type=str, default=None, help='gpu device id list')
parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')
parser.add_argument('--workers', type=int, default=16, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.5, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default=None, help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default=None, help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--lr_scheduler', type=str, default='linear', help='lr scheduler, linear or cosine')
parser.add_argument('--tmp_data_dir', type=str, default='/tmp/cache/', help='temp data dir')
#parser.add_argument('--note', type=str, default='try', help='note for this run')

args, unparsed = parser.parse_known_args()

if args.gpu_list is not None:
  gpu_list = [int(item) for item in args.gpu_list.split(',')]

if args.save is None:
  args.save = 'IMAGENET-eval-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
  #ut.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
  args.save = os.path.join(args.dir, args.save)
  utils.create_exp_dir(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
  format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

writer = SummaryWriter(os.path.join(args.save, 'runs'))

CLASSES = 1000

class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss

def main():
  if not torch.cuda.is_available():
    logging.info('No GPU device available')
    sys.exit(1)
  np.random.seed(args.seed)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info("args = %s", args)
  logging.info("unparsed_args = %s", unparsed)
  #num_gpus = torch.cuda.device_count()   
  print('---------Genotype---------')
  if args.arch is not None:
    genotype = eval("genotypes.%s" % args.arch)
  
  if args.dir is not None:
    with open(os.path.join(args.dir, "genotype.pickle"), 'rb') as f:
      genotype = pickle.load(f)
    logging.info("Unpickling genotype.pickle")
  logging.info(genotype)
  print('--------------------------') 
  model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
  if args.parallel:
    if args.gpu_list is None:
      model = nn.DataParallel(model)
      model = model.cuda()
    else:
      model = nn.DataParallel(model, device_ids=gpu_list)
      #model = model.cuda()
      model.to(f'cuda:{model.device_ids[0]}')
  else:
    model = model.cuda()
    torch.cuda.set_device(args.gpu)
  logging.info("param size = %fMB", ut.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
  criterion_smooth = criterion_smooth.cuda()

  optimizer = torch.optim.SGD(
    model.parameters(),
    args.learning_rate,
    momentum=args.momentum,
    weight_decay=args.weight_decay
    )
  data_dir = os.path.join(args.tmp_data_dir, 'imagenet')
  traindir = os.path.join(data_dir, 'train')
  validdir = os.path.join(data_dir, 'val')
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  train_data = dset.ImageFolder(
    traindir,
    transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.2),
      transforms.ToTensor(),
      normalize,
    ]))
  valid_data = dset.ImageFolder(
    validdir,
    transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
    ]))

  train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)

  valid_queue = torch.utils.data.DataLoader(
    valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
  logging.info(f'[INFO] train_queue: {len(train_queue)}, valid_queue: {len(valid_queue)}')

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
  best_acc_top1 = 0
  best_acc_top5 = 0
  start_epoch = 0

  # Checking if checkpoint already exists in args.save
  if os.path.exists(os.path.join(args.save, 'checkpoint.pth.tar')):
    checkpoint = torch.load(os.path.join(args.save, 'checkpoint.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    best_acc_top1 = checkpoint['best_acc_top1']
    best_acc_top5 =checkpoint['best_acc_top5']
    start_epoch = checkpoint['epoch']

  for epoch in range(start_epoch, args.epochs):
    if args.lr_scheduler == 'cosine':
      #scheduler.step()
      current_lr = scheduler.get_last_lr()[0]
    elif args.lr_scheduler == 'linear':
      current_lr = adjust_lr(optimizer, epoch)
    else:
      print('Wrong lr type, exit')
      sys.exit(1)
    logging.info('Epoch: (%d/%d) lr %e', epoch+1, args.epochs, current_lr)
    if epoch < 5 and args.batch_size > 256:
      for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr * (epoch + 1) / 5.0
      logging.info('Warming-up Epoch: %d, LR: %e', epoch+1, current_lr * (epoch + 1) / 5.0)
    #if num_gpus > 1:
    if args.parallel:
      model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
    else:
      model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
    epoch_start = time.time()
    train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer)
    logging.info('Train_acc: %f finished in %.5f minutes', train_acc, (time.time() - epoch_start) / 60)
    writer.add_scalar("train_acc", train_acc, epoch + 1)
    writer.add_scalar("train_obj", train_obj, epoch + 1)
    if args.lr_scheduler == 'cosine':
      scheduler.step()

    valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
    logging.info('Valid_acc_top1: %f', valid_acc_top1)
    logging.info('Valid_acc_top5: %f', valid_acc_top5)
    epoch_duration = time.time() - epoch_start
    logging.info('Epoch time: %dmins.', epoch_duration / 60)
    writer.add_scalar("valid_acc_top1", valid_acc_top1, epoch + 1)
    writer.add_scalar("valid_acc_top5", valid_acc_top5, epoch + 1)
    writer.add_scalar("valid_obj", valid_obj, epoch + 1)
    writer.add_scalar("test_error_top1", 100 - valid_acc_top1, epoch + 1)
    writer.add_scalar("test_error_top5", 100 - valid_acc_top5, epoch + 1)
    is_best = False
    if valid_acc_top5 > best_acc_top5:
      best_acc_top5 = valid_acc_top5
    if valid_acc_top1 > best_acc_top1:
      best_acc_top1 = valid_acc_top1
      is_best = True
    ut.save_checkpoint({
      'epoch': epoch + 1,
      'state_dict': model.state_dict(),
      'best_acc_top1': best_acc_top1,
      'optimizer' : optimizer.state_dict(),
      'scheduler' : scheduler.state_dict(),
      'best_acc_top1' : best_acc_top1,
      'best_acc_top5' : best_acc_top5
      }, is_best, args.save)
    logging.info('='*100)
  logging.info(f'[INFO] best_acc_top1: {best_acc_top1}, best_acc_top5: {best_acc_top5}')
    
def adjust_lr(optimizer, epoch):
  # Smaller slope for the last 5 epochs because lr * 1/250 is relatively large
  if args.epochs -  epoch > 5:
    lr = args.learning_rate * (args.epochs - 5 - epoch) / (args.epochs - 5)
  else:
    lr = args.learning_rate * (args.epochs - epoch) / ((args.epochs - 5) * 5)
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  return lr    

def train(train_queue, model, criterion, optimizer):
  objs = ut.AvgrageMeter()
  top1 = ut.AvgrageMeter()
  top5 = ut.AvgrageMeter()
  batch_time = ut.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    #target = target.cuda(non_blocking=True)
    target = target.to(f'cuda:{model.device_ids[0]}')
    #input = input.cuda(non_blocking=True)
    input = input.to(f'cuda:{model.device_ids[0]}')
    b_start = time.time()
    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    batch_time.update(time.time() - b_start)
    prec1, prec5 = ut.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      end_time = time.time()
      if step == 0:
        duration = 0
        start_time = time.time()
      else:
        duration = end_time - start_time
        start_time = time.time()
      logging.info('TRAIN Step: %04d Objs: %e R1: %f R5: %f lr: %e Duration: %02ds BTime: %.3fs', 
                  step, objs.avg, top1.avg, top5.avg, optimizer.param_groups[0]['lr'], duration, batch_time.avg)
      #break

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = ut.AvgrageMeter()
  top1 = ut.AvgrageMeter()
  top5 = ut.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    #input = input.cuda()
    input = input.to(f'cuda:{model.device_ids[0]}')
    #target = target.cuda(non_blocking=True)
    target = target.to(f'cuda:{model.device_ids[0]}')
    with torch.no_grad():
      logits, _ = model(input)
      loss = criterion(logits, target)

    prec1, prec5 = ut.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      end_time = time.time()
      if step == 0:
        duration = 0
        start_time = time.time()
      else:
        duration = end_time - start_time
        start_time = time.time()
      logging.info('VALID Step: %03d Objs: %e R1: %f R5: %f Duration: %ds', step, objs.avg, top1.avg, top5.avg, duration)
      #break

  return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
  main_start = time.time()
  main()
  logging.info(f'[INFO] Training finished in {(time.time() - main_start)/3600} hours')
  writer.close()
