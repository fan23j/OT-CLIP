python -m training.main \
    --imagenet-val /dfs/data/data/ILSVRC2012/val/ \
    --model RN50 \
    --batch-size=1000 \
    --pretrained openai \
    --eval-ot
