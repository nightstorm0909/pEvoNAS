#!/bin/bash
# bash ./s2/arch_search_s2.sh cifar10 gpu outputs nas_config_path

echo script name: $0
echo $# arguments

dataset=$1
gpu=$2
channel=16
num_cells=5
max_nodes=4
output_dir=$3

if [ "$dataset" == "cifar10" ] || [ "$dataset" == "cifar100" ]; then
  data_path="../../../data"
else
  data_path="../../../data/ImageNet16"
fi
api_path="../../../NAS_Bench_201/NAS-Bench-201-v1_1-096897.pth"
config_path="./s2/configs/pEvoNAS.config"
config_root="./s2/configs"
#nas_config="./configs/s2_configs.cfg"
nas_config=$4
record_filename=info.csv

for index in {1..3..1}
do
  python ./s2/arch_search.py --gpu ${gpu} --max_nodes ${max_nodes} --init_channel ${channel} --num_cells ${num_cells} \
                                 --dataset ${dataset} --data_dir ${data_path} --output_dir ${output_dir} --record_filename ${record_filename} \
                                 --api_path ${api_path} --config_path ${config_path} --config_root ${config_root}  --nas_config ${nas_config}
done
