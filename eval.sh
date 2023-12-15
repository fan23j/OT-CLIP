python -m training.main \
    --imagenet-val /dfs/data/data/ILSVRC2012/val/ \
    --model RN50 \
    --pretrained openai \
    --batch-size=32 \
    --eval-ot