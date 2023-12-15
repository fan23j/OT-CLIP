import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm

from open_clip import (
    get_input_dtype,
    get_tokenizer,
    build_zero_shot_classifier,
    IMAGENET_CLASSNAMES,
    OPENAI_IMAGENET_TEMPLATES,
)
from .precision import get_autocast


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]

def sinkhorn(C, ratio, reg=0.1, num_iters=7, tol=1e-3):
    N = C.shape[0]

    K = torch.exp(-C / reg).to(C.device)

    u = torch.ones(N, dtype=C.dtype, device=C.device) / N
    v = ratio.to(C.device)

    for _ in range(num_iters):
        u = u / (K @ v)
        v = ratio / (K.T @ u)

    # Compute the transport matrix
    transport_matrix = torch.diag(u) @ K @ torch.diag(v)

    return transport_matrix

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
                logits = 100.0 * image_features @ classifier.T

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = top1 / n
    top5 = top5 / n
    return top1, top5

def run_ot(model, classifier, dataloader, args, num_iters=10):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=args.device, dtype=input_dtype)
            target = target.to(args.device)
        
            with autocast():
                # predict
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0] # [bs, 1024]

                sigma_1 = torch.matmul(image_features, image_features.T) # [bs, bs]
                sigma_2 = torch.matmul(classifier, classifier.T) # [num_class, num_class]
                C = 1.0 - torch.matmul(image_features, classifier.T).to(args.device) #[bs, num_class]
                P = F.softmax(-C, dim=1)

                tolerance = 1e-2  # Define a small tolerance
                prev_C = C.clone()  # Initialize with the current value of C
                import pudb; pudb.set_trace()
                for iteration in range(num_iters):
                    C = C - sigma_1 @ P
                    P = F.softmax(-C, dim=1)

                    # Check for convergence
                    delta = torch.norm(C - prev_C)  # Calculate the change in C
                    if delta < tolerance:
                        break  # Stop if change is smaller than tolerance

                    prev_C = C.clone()  # Update previous value for next iteration

                logits = P

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5

def get_eval_fn(args):
    if args.eval_ot:
        return run_ot
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
