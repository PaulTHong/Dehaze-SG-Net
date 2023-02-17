#!/bin/bash
GPU=$2 # 0 or 1 or 0,1 
data="ITS"  # "ITS"  "OTS"
layer=50  # 50, 101, 152
pre_data='nyud'  # 'nyud' 'voc'
prefix=${data}_${pre_data}${layer}_seg_attention
echo ${prefix}

if [ $1 = "train" ]; then
    # === ITS:
    CUDA_VISIBLE_DEVICES=$GPU python -u train.py --add_edge --only_residual \
        --valid_interval 5 --num_epochs 100 --train_batch_size 8 --milestones 40 80 --loss_func l1 --display_iter 100 \
        --dataset ${data} --seg_model_res ${layer} --seg_class_num 40 --seg_dataset ${pre_data} \
        --valid_not_save --model_path ./train_logs/${prefix}/dehaze.ckpt \
        --sample_output_folder ./train_logs/${prefix}/samples/
    # === OTS:
    #CUDA_VISIBLE_DEVICES=$GPU python -u train.py --add_edge --only_residual \
        #--valid_interval 2 --num_epochs 30 --train_batch_size 8 --milestones 15 25 --loss_func l1 --display_iter 100 \
        #--dataset ${data} --seg_model_res ${layer} --seg_class_num 40 --seg_dataset ${pre_data} \
        #--valid_not_save --model_path ./train_logs/${prefix}/dehaze.ckpt \
        #--sample_output_folder ./train_logs/${prefix}/samples/
elif [ $1 = "test" ]; then
    CUDA_VISIBLE_DEVICES=$GPU python -u test.py --add_edge --only_residual --resize \
        --seg_model_res ${layer} --seg_class_num 40 --seg_dataset ${pre_data} \
        --model_path ./train_logs/${prefix}/best_dehaze.ckpt \
        --test_output_folder ./test_results/${prefix}/
elif [ $1 = "dehaze" ]; then
    CUDA_VISIBLE_DEVICES=$GPU python -u 255_dehaze.py --add_edge --only_residual \
        --test_dir ../../../Dataset/demos/indoor --model_path train_logs/${prefix}/best_dehaze.ckpt
    #CUDA_VISIBLE_DEVICES=$GPU python -u 255_dehaze.py --add_edge --only_residual \
        #--test_dir ../../../Dataset/demos/outdoor --model_path train_logs/${prefix}/best_dehaze.ckpt
fi


