#!/bin/bash
GPU=$2 
data='its'  # 'its' 'ots'
pre_data='nyud'  # 'nyud' 'voc'
model_suffix=${pre_data}_SG
echo ${data}_${model_suffix}

if [ $1 = "train" ]; then
    # === ITS:
    CUDA_VISIBLE_DEVICES=$GPU python -u SG_main.py --net='ffa' --crop --crop_size=240 --blocks=19 --gps=3 --bs=5 \
        --lr=0.0001 --trainset=${data}'_train' --testset=${data}'_test' --steps=500000 --eval_step=5000 \
        --insert_seg --model_suffix ${model_suffix} --seg_dataset ${pre_data} --seg_class_num 40 \
        |tee logs/${data}_${model_suffix}_train.log
    # === OTS:
    #CUDA_VISIBLE_DEVICES=$GPU python -u SG_main.py --net='ffa' --crop --crop_size=240 --blocks=19 --gps=3 --bs=5 \
        #--lr=0.0001 --trainset=${data}'_train' --testset=${data}'_test' --steps=1000000 --eval_step=5000 \
        #--insert_seg --model_suffix ${model_suffix} --seg_dataset ${pre_data} --seg_class_num 40 \
        #|tee logs/${data}_${model_suffix}_train.log
elif [ $1 = "test" ]; then
    CUDA_VISIBLE_DEVICES=$GPU python SG_test.py --model_path ./trained_models/${data}_${model_suffix}.pk
elif [ $1 = "dehaze" ]; then
    CUDA_VISIBLE_DEVICES=$GPU python SG_dehaze.py --task ${data} --test_imgs ../../../Dataset/demos/indoor \
        --model_path ./trained_models/${data}_${model_suffix}.pk --output_dir ./demo_results/${data}_${model_suffix}
    #CUDA_VISIBLE_DEVICES=$GPU python SG_dehaze.py --task ${data} --test_imgs ../../../Dataset/demos/outdoor \
        #--model_path ./trained_models/${data}_${model_suffix}.pk --output_dir ./demo_results/${data}_${model_suffix}
fi
