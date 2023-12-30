python -m training.main \
    --imagenet-val /dfs/data/data/ILSVRC2012/val/ \
    --model RN50 \
    --resume /dfs/data/main/OT-CLIP/logs/sinkhorn_ce/checkpoints/epoch_30.pt \
    --batch-size=2500 \
#     --eval-ot
