#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

BASE_EXPNAME="output/output/emb/config_0"
BASE_CONFIG="arguments_MoDE/emb/config_0"

SCENES=("coffee_martini" "cook_spinach" "cut_roasted_beef" "flame_salmon_1" "flame_steak" "sear_steak")

ITERATIONS=(14000 20000 30000)
BASE_DATASET="/home/mhj/database2/ETRI/dataset/n3v"


for SCENE in "${SCENES[@]}"; do
    python train_emb.py \
        -s "${BASE_DATASET}/${SCENE}" \
        --expname "${BASE_EXPNAME}/${SCENE}" \
        --configs "${BASE_CONFIG}/${SCENE}.py"

    for ITER in "${ITERATIONS[@]}"; do
        python render_emb.py \
            --model_path "${BASE_EXPNAME}/${SCENE}" \
            --skip_train \
            --configs "${BASE_CONFIG}/${SCENE}.py" \
            --iteration $ITER
    done
done

CUDA_VISIBLE_DEVICES=2 python train_emb.py \
        -s "/home/mhj/database2/ETRI/dataset/n3v/coffee_martini" \
        --expname "/home/mhj/database2/ETRI/MoDE_rot/config_0/coffee_martini" \
        --configs "arguments_MoDE/dynerf/config_rot_0/coffee_martini.py"

CUDA_VISIBLE_DEVICES=1 python render_emb.py --skip_test \
    --model_path /home/mhj/test/MoDE_samples/emb/coffee_martini \
    --skip_train --skip_test \
    --configs "arguments_MoDE/emb/config_0/coffee_martini.py" \
    --iteration 14000

