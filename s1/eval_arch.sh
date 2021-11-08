#!/bin/bash
# bash ./s1/eval_arch.sh cifar10 gpu arch_dir batch_size

echo script name: $0
echo $# arguments

dataset=$1
gpu=$2
arch_dir=$3
if [ -z "$4" ]
  then
    batch_size=96
  else
    batch_size=$4
fi
epochs=600

data_path="../../../data"

if [ "$dataset" == "cifar10" ] ; then
  python ./s1/train_cifar10.py --cutout --auxiliary --epochs ${epochs} --dir ${arch_dir} --data ${data_path} --gpu ${gpu} --batch_size ${batch_size}
else
  python ./s1/train_cifar100.py --cutout --auxiliary --epochs ${epochs} --dir ${arch_dir} --data ${data_path} --gpu ${gpu} --batch_size ${batch_size}
fi
