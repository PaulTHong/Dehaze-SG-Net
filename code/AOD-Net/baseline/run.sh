#!/bin/bash
data="ITS"  # "OTS"

GPU=$2 # 0 or 1 or 0,1 
if [ $1 = "train" ]; then
    CUDA_VISIBLE_DEVICES=$GPU python -u train.py --num_epochs 10 --train_batch_size 8 --init_lr 0.001 --milestones 2 5 8 \
        --loss_func l1 --valid_not_save --snapshots_folder ./train_logs/${data}_base/snapshots/ \
        --dataset ${data} --sample_output_folder ./train_logs/${data}_base/samples/
elif [ $1 = "test" ]; then
    CUDA_VISIBLE_DEVICES=$GPU python -u test.py --resize --model_path ./train_logs/${data}_base/snapshots/dehazer.pth
elif [ $1 = "dehaze" ]; then
    CUDA_VISIBLE_DEVICES=$GPU python -u dehaze.py --test_dir ../test_images
fi


