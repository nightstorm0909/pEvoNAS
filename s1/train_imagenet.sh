# bash train_imagenet.sh batch_size genotype_dir dataset_dir workers

python ./s1/train_imagenet.py --batch_size $1 --tmp_data_dir $3 --dir $2 --auxiliary --parallel --workers $4 --report_freq 100
