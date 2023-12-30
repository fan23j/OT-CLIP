import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import ot

from open_clip import (
    get_input_dtype,
    get_tokenizer,
    build_zero_shot_classifier,
    IMAGENET_CLASSNAMES,
    OPENAI_IMAGENET_TEMPLATES,
    SIMPLE_IMAGENET_TEMPLATES,
    IMAGENET_A_CLASSNAMES,
)
from .precision import get_autocast
from .zero_shot_utils import sinkhorn, sinkhorn_knopp

def plot_heatmap(tensor, name="test.png"):
    """
    Plots a heatmap of the given tensor.
    
    Parameters:
    tensor (torch.Tensor): A 2D tensor to be visualized as a heatmap.
    """
    import matplotlib.pyplot as plt

    if tensor.ndim != 2:
        raise ValueError("Input tensor should be 2D")
    
    # Convert the tensor to a numpy array for plotting
    array = tensor.numpy()
    
    # Create the heatmap
    plt.imshow(array, cmap='viridis', interpolation='nearest')
    
    # Add a color bar for reference
    plt.colorbar()

    # Show the plot
    plt.savefig(name)

def accuracy(output, target, topk=(1,)):
    
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]

def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    
    with torch.no_grad():
        top1, top5, n = 0.0, 0.0, 0.0
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=args.device, dtype=input_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = (
                    output["image_features"] if isinstance(output, dict) else output[0]
                )
                logits = F.softmax(image_features @ classifier, dim=1)
                
            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            
            top1 += acc1
            top5 += acc5
            n += images.size(0)
    
    top1 = top1 / n
    top5 = top5 / n
    return top1, top5

# def run_ot(model, classifier, dataloader, args, num_iters=7):
#     autocast = get_autocast(args.precision)
#     input_dtype = get_input_dtype(args.precision)

#     # Load clusters
#     clusters = torch.load('clusters_2000.pt')

#     with torch.no_grad():
#         top1, top5, n = 0., 0., 0.
#         for cluster_id, cluster_data in tqdm(clusters.items()):


#             with autocast():
#                 # Convert the cluster data to PyTorch tensors and send to the correct device
#                 image_features = torch.tensor(cluster_data["features"], dtype=input_dtype).to(args.device)
#                 targets = torch.tensor(cluster_data["targets"]).to(args.device)

#                 sigma_1 = torch.matmul(image_features, image_features.T)  # [bs, bs]
#                 sigma_2 = torch.matmul(classifier.T, classifier)  # [num_class, num_class]

#                 C = 1.0 - torch.matmul(image_features, classifier).to(args.device)  # [bs, num_class]
#                 P = F.softmax(-C, dim=1)
               
#                 for iteration in range(num_iters):
#                     C = C - sigma_1 @ P * 0.01
#                     P = F.softmax(-C, dim=1)

#                 logits = P

#             # Measure accuracy for the cluster
#             acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
#             top1 += acc1
#             top5 += acc5
#             n += image_features.size(0)

#     top1 = (top1 / n)
#     top5 = (top5 / n)
#     return top1, top5

def run_sinkhorn(model, classifier, dataloader, args, num_iters=3):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    features = []
    targets = []
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        index = 0
        
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            
            images = images.to(device=args.device, dtype=input_dtype)
            target = target.to(args.device)

            with autocast():
                # Predict
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0] # [bs, 1024]
                features.append(image_features)
                targets.append(target)
            n += images.size(0)
    #n = 50000
    #image_features = torch.load("image_features.pt").to(args.device)
    #targets = torch.load("targets.pt").to(args.device)
    image_features = torch.cat(features, dim=0)
    targets = torch.cat(targets, dim=0)
    a = torch.ones((image_features.shape[0],)).to(args.device)
    b_d = torch.full((classifier.shape[1],), 1.0 * image_features.shape[0]).to(args.device)
    b_u = torch.full((classifier.shape[1],), 1.0 * image_features.shape[0]).to(args.device)

    P = sinkhorn(1.0 - image_features @ classifier, a, b_u, b_d)
    acc1, acc5 = accuracy(P, targets, topk=(1, 5))
    top1 = (acc1 / n)
    top5 = (acc5 / n)
    return top1, top5

def run_fused_gw(model, classifier, dataloader, args, num_iters=3):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
#     features = []
#     targets = []
#     with torch.no_grad():
#         top1, top5, n = 0., 0., 0.
#         index = 0
        
#         for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            
#             images = images.to(device=args.device, dtype=input_dtype)
#             target = target.to(args.device)

#             with autocast():
#                 # Predict
#                 output = model(image=images)
#                 image_features = output['image_features'] if isinstance(output, dict) else output[0] # [bs, 1024]
#                 features.append(image_features)
#                 targets.append(target)
#             n += images.size(0)

    image_features = torch.load("image_features.pt").to(args.device)
    targets = torch.load("targets.pt").to(args.device)
    C1 = torch.matmul(image_features, image_features.T)
    C2 = 1.0 - torch.matmul(classifier.T, classifier)
    M = 1.0 - torch.matmul(image_features, classifier)
    

    n = 50000
    P = ot.fused_gromov_wasserstein(M, C1, C2, alpha=0.01)
                                
    acc1, acc5 = accuracy(P, targets, topk=(1, 5))
    top1 = (acc1 / n)
    top5 = (acc5 / n)
    return top1, top5


def get_eval_fn(args):
    if args.eval_ot:
        return run_sinkhorn
    else:
        return run


def zero_shot_eval(model, data, epoch, args, tokenizer=None):
    if "imagenet-val" not in data and "imagenet-v2" not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    logging.info("Starting zero-shot imagenet.")
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    logging.info("Building zero-shot classifier")
    autocast = get_autocast(args.precision)
    with autocast():
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=IMAGENET_CLASSNAMES,
            templates=OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )
        #classifier = torch.load("classifier.pt")

    logging.info("Using classifier")
    results = {}
    
    eval_fn = get_eval_fn(args)
    if "imagenet-val" in data:
        top1, top5 = eval_fn(model, classifier, data["imagenet-val"].dataloader, args)
        results["imagenet-zeroshot-val-top1"] = top1
        results["imagenet-zeroshot-val-top5"] = top5
    if "imagenet-v2" in data:
        top1, top5 = eval_fn(model, classifier, data["imagenet-v2"].dataloader, args)
        results["imagenetv2-zeroshot-val-top1"] = top1
        results["imagenetv2-zeroshot-val-top5"] = top5

    logging.info("Finished zero-shot imagenet.")

    return results
