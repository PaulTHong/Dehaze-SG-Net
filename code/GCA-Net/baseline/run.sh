#!/bin/bash
GPU=$2 # 0 or 1 or 0,1
data='ITS'  # 'OTS'
if [ $1 = "train" ]; then
    # === ITS:
    CUDA_VISIBLE_DEVICES=$GPU python -u train.py --add_edge --only_residual \
        --num_epochs 100 --train_batch_size 8 --milestones 40 80 --loss_func l1 --display_iter 100 \
        --dataset ${data} --valid_not_save --model_path ./train_logs/${data}/dehaze.ckpt \
        --sample_output_folder ./train_logs/${data}/samples/
#    === OTS:
#    CUDA_VISIBLE_DEVICES=$GPU python -u train.py --add_edge --only_residual \
#        --num_epochs 30 --train_batch_size 8 --milestones 15 25 --loss_func l1 --display_iter 100 \
#        --dataset ${data} --valid_not_save --model_path ./train_logs/${data}/dehaze.ckpt \
#        --sample_output_folder ./train_logs/${data}/samples/
elif [ $1 = "test" ]; then
    CUDA_VISIBLE_DEVICES=$GPU python -u test.py --add_edge --only_residual --resize \
        --model_path ./train_logs/${data}/dehaze.ckpt \
        --test_output_folder ./train_logs/${data}/test_results/
elif [ $1 = "dehaze" ]; then
    CUDA_VISIBLE_DEVICES=$GPU python -u 255_dehaze.py --add_edge --only_residual \
        --test_dir ../demo_images --output_dir ./demo_results
fi


