#!/bin/bash
GPU=$2
data_name="ITS" # 'NH-HAZE', 'DENSE-HAZE', 'ITS'
model_mode="SG"  # "base" "SG"
pre_data='nyud'  # 'nyud' 'voc'

seg_class_num=40 # 21 for voc, 40 for nyud
model_suffix=${pre_data}_8000_1v10
config=${data_name}_${model_mode}_${model_suffix}
echo ${config}

seg_ckpt_path=../../semantic_segmentation/light-weight-refinenet/ckpt/50_${pre_data}.ckpt

if [ $1 = "train" ]; then
    if [ $data_name = "NH-HAZE" ]; then
      CUDA_VISIBLE_DEVICES=$GPU python -u train.py --data_name ${data_name} --crop --num_epochs 8000 \
          --neg_num 16 --train_batch_size 16 --val_batch_size 1 --display_iter 5 --snapshot_iter 20 --model_mode ${model_mode} \
          --seg_dataset ${pre_data} --seg_class_num ${seg_class_num} --seg_dim 16 --seg_ckpt_path ${seg_ckpt_path} \
          --snapshots_folder ./train_logs/${config} |tee train_logs/${config}_train.log
    elif [ $data_name = "DENSE-HAZE" ]; then
      CUDA_VISIBLE_DEVICES=$GPU python -u train.py --data_name ${data_name} --crop --num_epochs 8000 \
          --neg_num 16 --train_batch_size 16 --val_batch_size 1 --display_iter 5 --snapshot_iter 20 --model_mode ${model_mode} \
          --seg_dataset ${pre_data} --seg_class_num ${seg_class_num} --seg_dim 16 --seg_ckpt_path ${seg_ckpt_path} \
          --snapshots_folder ./train_logs/${config} |tee train_logs/${config}_train.log
    elif [ $data_name = "ITS" ]; then
      CUDA_VISIBLE_DEVICES=$GPU python -u train.py --data_name ${data_name} --crop --num_epochs 80000 \
          --neg_num 10 --train_batch_size 16 --val_batch_size 10 --display_iter 20 --snapshot_iter 500 --model_mode ${model_mode} \
          --seg_dataset ${pre_data} --seg_class_num ${seg_class_num} --seg_dim 16 --seg_ckpt_path ${seg_ckpt_path} \
          --snapshots_folder ./train_logs/${config} |tee train_logs/${config}_train.log
    fi
elif [ $1 = "test" ]; then
    CUDA_VISIBLE_DEVICES=$GPU python -u test.py --data_name ${data_name} --model_mode ${model_mode} \
        --seg_dataset ${pre_data} --seg_class_num ${seg_class_num} --seg_dim 16 --seg_ckpt_path ${seg_ckpt_path} \
        --model_path ./pretrained_models/DH_train.pk  --test_save --test_output_folder ./test_results/${config}
fi
