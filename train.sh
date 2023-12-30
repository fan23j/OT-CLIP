torchrun --nproc_per_node 4 -m training.main \
    --train-data '/dfs/data/main/data/cc3m/cc3m_train/{00000..00331}.tar' \
    --train-num-samples 2669756 \
    --dataset-type webdataset \
    --batch-size=256 \
    --lr=5e-4 \
    --wd=0.1 \
    --epochs=30 \
    --model RN50 \
    --precision amp \
    --workers 8 \
    --imagenet-val /dfs/data/data/ILSVRC2012/val \
#     --resume 'logs/orig/checkpoints/epoch_25.pt'