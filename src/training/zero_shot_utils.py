import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import cuml
from tqdm import tqdm
import numpy as np
import cupy as cp

def concatenate_features(highest_confidence_features_per_class):
    # Convert the dictionary values to a list of feature vectors
    feature_list = list(highest_confidence_features_per_class.values())

    # Convert the list of numpy arrays to a list of PyTorch tensors
    feature_tensors = [torch.tensor(feature, dtype=torch.float32) for feature in feature_list]

    # Concatenate the feature tensors into a single tensor
    concatenated_features = torch.stack(feature_tensors)

    return concatenated_features


def get_high_conf(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    # Dictionary to store highest confidence features per class
    highest_confidence_features_per_class = {}
    highest_confidence_scores_per_class = {}

    with torch.no_grad():
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

                # Process each prediction in the batch
                for i in range(logits.size(0)):
                    class_index = logits[i].argmax().item()
                    confidence = logits[i].max().item()

                    # Update if this is the highest confidence for the class
                    if class_index not in highest_confidence_scores_per_class or \
                       confidence > highest_confidence_scores_per_class[class_index]:
                        highest_confidence_scores_per_class[class_index] = confidence
                        highest_confidence_features_per_class[class_index] = image_features[i].cpu().numpy()

    return concatenate_features(highest_confidence_features_per_class)


def plot_class_acc(preds, args):
    # Ensure preds is a list of booleans or 1s/0s
    chunk_size = 50
    num_classes = len(preds) // chunk_size

    # Initialize a list to store accuracy for each class
    class_acc = []

    for class_index in range(num_classes):
        # Calculate start and end indices for each chunk
        start_idx = class_index * chunk_size
        end_idx = start_idx + chunk_size
        # Calculate accuracy for each class
        acc = sum(preds[start_idx:end_idx]) / chunk_size
        class_acc.append(acc.cpu())

    # Plotting
    plt.bar(range(num_classes), class_acc)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title(f'Class-wise Accuracy - Batch Size: {args.batch_size}')
    plt.savefig(f'{args.batch_size}_class_accuracy.png')
    plt.close()
    
def get_centroids(model, dataloader, args):
    model.to(args.device)
    features = []

    with torch.no_grad():
        for images, _ in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(args.device)
            feature_vectors = model(images)
            # Convert PyTorch tensors to cuPy arrays and accumulate
            features.append(cp.asarray(feature_vectors.cpu().numpy()))

    # Concatenate features in cuPy
    features = cp.concatenate(features, axis=0)

    # Perform KMeans clustering with cuML
    kmeans = cuml.KMeans(n_clusters=1000)
    kmeans.fit(features)
    centroids = kmeans.cluster_centers_
    import pudb; pudb.set_trace()
    return centroids
    
    
def sinkhorn(P, a, b_d, b_u, max_iters=1000, epsilon=1e-8):
    
    P = torch.exp(-P / 0.01)
    
    for _ in range(max_iters):
        P_prev = P
        # proj c1
        P = torch.diag(a / P.sum(dim=1)) @ P
        # proj c2
        P = P @ torch.diag(torch.max(b_d / P.t().sum(dim=1), torch.ones(P.shape[1]).to(P.device)))
        # proj c3
        P = P @ torch.diag(torch.min(b_u / P.t().sum(dim=1), torch.ones(P.shape[1]).to(P.device)))
        # Check for convergence
        if torch.norm(P - P_prev) < epsilon:
            break
    return P

def sinkhorn_knopp(a, b, M, reg=0.01, numItermax=1000, stopThr=1e-9, **kwargs):
    """
    a: 1_bz
    b: ratios [r_1,r_2,...]
    M: Sim/Cost
    """

    # init data
    dim_a = a.shape[0]
    dim_b = b.shape[0]

    u = torch.ones(dim_a, dtype=M.dtype, device=M.device) / dim_a
    v = torch.ones(dim_b, dtype=M.dtype, device=M.device) / dim_b

    K = torch.exp(M / (-reg))

    Kp = (1 / a).reshape(-1, 1) * K

    err = 1
    for ii in range(numItermax):
        uprev = u
        vprev = v
        KtransposeU = K.t() @ u
        v = b / KtransposeU
        u = 1. / (Kp @ v)

        if (torch.any(KtransposeU == 0)
                or torch.any(torch.isnan(u)) or torch.any(torch.isnan(v))
                or torch.any(torch.isinf(u)) or torch.any(torch.isinf(v))):
            u = uprev
            v = vprev
            break
        if ii % 10 == 0:
            tmp2 = torch.einsum('i,ij,j->j', u, K, v)
            err = torch.norm(tmp2 - b)  # violation of marginal

            if err < stopThr:
                break

    return u.reshape((-1, 1)) * K * v.reshape((1, -1))

def uot_badmm(x: torch.Tensor, p0: torch.Tensor, q0: torch.Tensor,
              num: int = 4, eps: float = 1e-8) -> torch.Tensor:
    """
    Solving regularized optimal transport via Bregman ADMM algorithm (entropic regularizer)
    :param x: (B, N, D), a matrix with N samples and each sample is D-dimensional
    :param p0: (B, 1, D), the marginal prior of dimensions
    :param q0: (B, N, 1), the marginal prior of samples
    :param num: the number of Bregman ADMM iterations
    :param eps: the epsilon to avoid numerical instability
    :return:
        t: (N, D), the optimal transport matrix
    """
    log_p0 = torch.log(p0)  # (B, 1, D)
    log_q0 = torch.log(q0 + eps) # (B, N, 1)
    log_t = (log_q0 + log_p0)  # (B, N, D)
    log_s = (log_q0 + log_p0)  # (B, N, D)
    log_mu = torch.log(p0)  # (B, 1, D)
    log_eta = torch.log(q0 + eps)  # (B, N, 1)
    z = torch.zeros_like(log_t)  # (B, N, D)
    z1 = torch.zeros_like(p0)  # (B, 1, D)
    z2 = torch.zeros_like(q0)  # (B, N, 1)
    for k in range(num):
        n = min([k, a1.shape[0] - 1])

        # update logP
        y = ((x - z) + log_s)  # (B, N, D)
        log_t = (log_eta - torch.logsumexp(y, dim=1, keepdim=True)) + y  # (B, N, D)
        # update logS
        y = (z * log_t) # (B, N, D)
        ymin, _ = torch.min(y, dim=1, keepdim=True)
        ymax, _ = torch.max(ymin - ymin + y, dim=0, keepdim=True)  # (B, 1, D)
        # (B, N, D)
        log_s = (log_mu - torch.log(torch.sum(torch.exp((y - ymax)), dim=0, keepdim=True)) - ymax) + y
        # update dual variables
        t = torch.exp(log_t)
        s = torch.exp(log_s)
        z = z * (t - s)
        y = ( log_mu + log_p0 - z1)  # (B, 1, D)
        log_mu = y - torch.logsumexp(y, dim=1, keepdim=True)  # (B, 1, D)
        y = (log_eta + log_q0 - z2)  # (B, N, 1)
        ymin, _ = torch.min(y, dim=1, keepdim=True)
        ymax, _ = torch.max(ymin - ymin + y, dim=0, keepdim=True)  # (B, 1, D)
        log_eta = (y - torch.log(
            torch.sum(torch.exp((y - ymax)), dim=0, keepdim=True)) - ymax)  # (B, N, 1)
        # update dual variables
        z1 = z1 + (torch.exp(log_mu) - torch.sum(s, dim=0, keepdim=True))  # (B, 1, D)
        z2 = z2 + (torch.exp(log_eta) - torch.sum(t, dim=1, keepdim=True))  # (B, N, 1)
    return torch.exp(log_t)