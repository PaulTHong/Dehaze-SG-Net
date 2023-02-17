#!/bin/bash
GPU=$2 # 0
data="OTS"  # "ITS" "OTS"
layer=50  # 50, 101, 152
pre_data='nyud'  # 'nyud' 'voc'

prefix=${data}_${pre_data}${layer}

echo $prefix

if [ $1 = "train" ]; then
    CUDA_VISIBLE_DEVICES=$GPU python -u sl_train.py --num_epochs 10 --train_batch_size 8 --init_lr 0.001 \
        --multiseg --display_iter 100 --dataset ${data} --seg_model_res ${layer} --seg_class_num 40 --seg_dataset ${pre_data} \
        --milestones 2 5 8 --loss_func l1 --snapshots_folder ./train_logs/${prefix}/snapshots/ \
        --seg_dim 16 --valid_not_save --sample_output_folder ./train_logs/${prefix}/samples/ \
        --model_path ./train_logs/${prefix}/dehazer.pth \
        2>&1 |tee train_logs/${prefix}_train.log

elif [ $1 = "valid" ]; then
    CUDA_VISIBLE_DEVICES=$GPU python -u validate.py --model_path ./train_logs/${prefix}/dehazer.pth \
        --sample_output_folder ./train_logs/${prefix}/samples/
elif [ $1 = "test" ]; then
    CUDA_VISIBLE_DEVICES=$GPU python -u test.py --resize \
        --multiseg --seg_model_res ${layer} --seg_class_num 40 --seg_dataset ${pre_data} \
        --seg_dim 16 --model_path ./train_logs/${prefix}/snapshots/dehazer.pth \
        --test_output_folder ./test_results/${prefix}
elif [ $1 = "dehaze" ]; then
    CUDA_VISIBLE_DEVICES=$GPU python -u dehaze.py --test_dir ../../../Dataset/demos/indoor \
        --model_path train_logs/${prefix}/snapshots/dehazer.pth
    #CUDA_VISIBLE_DEVICES=$GPU python -u dehaze.py --test_dir ../../../Dataset/demos/outdoor \
        #--model_path train_logs/${prefix}/snapshots/dehazer.pth
fi


