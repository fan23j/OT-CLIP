#!/bin/bash

# Define an array of batch sizes
batch_sizes=(8 16 32)

# Loop over each batch size
for batch_size in "${batch_sizes[@]}"
do
    python -m training.main \
        --imagenet-val /dfs/data/data/ILSVRC2012/val/ \
        --model RN50 \
        --pretrained openai \
        --batch-size=$batch_size \
        --max-batch-size=128 \
        --eval-ot
done