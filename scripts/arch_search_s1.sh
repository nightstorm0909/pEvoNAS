#!/bin/bash
# bash ./scripts/arch_search_s1_prog1_test.sh cifar10 gpu outputs nas_config_path

echo script name: $0
echo $# arguments

dataset=$1
gpu=$2
dir=$3
data_path="../../../data"
config_root="./s2/configs"
nas_config=$4

today=`date +%Y.%m.%d`
rand=$(date +%s)
output_dir="${dir}/search-${today}-${rand}-${dataset}"
while [[ -d "$output_dir" ]]
do
  output_dir="${dir}/search-${today}-${rand}-${dataset}"
done
echo $output_dir

for index in {1..1..1}
do
  python ./s1/arch_search.py --gpu ${gpu} --dataset ${dataset} --data_dir ${data_path} --output_dir ${output_dir} --config_root ${config_root} --nas_config ${nas_config}
done

echo $output_dir
bash ./s1/eval_arch.sh ${dataset} ${gpu} ${output_dir} #${batch_size}
