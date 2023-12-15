#!/bin/bash

# Define an array of batch sizes
batch_sizes=(64 128 256 512 1024 2048)

# Loop over each batch size
for batch_size in "${batch_sizes[@]}"
do
    CUDA_VISIBLE_DEVICES=1 python -m training.main \
        --imagenet-val /dfs/data/data/ILSVRC2012/val/ \
        --model RN50 \
        --pretrained openai \
        --batch-size=$batch_size \
        --max-batch-size=128
done
