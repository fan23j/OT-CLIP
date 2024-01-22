import torch.nn as nn
from torch.nn import functional as F
import torch
import ot

class Cross_Entropy_Loss(nn.Module):
    def __init__(self):
        super(Cross_Entropy_Loss, self).__init__()
        
    def get_logits(self, image_features, text_features, logit_scale):
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, all_image_features, all_text_features, logit_scale, labels):
        logits_per_image, logits_per_text = self.get_logits(all_image_features, all_text_features, logit_scale)
        loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2
        
        return {"cross_entropy_loss": loss}

class DBOT_Sinkhorn_Loss(nn.Module):
    def __init__(self):
        super(DBOT_Sinkhorn_Loss, self).__init__()
        
    def get_logits(self, image_features, text_features):
        logits_per_image = image_features @ text_features.T
        logits_per_text = text_features @ image_features.T
        
        return logits_per_image, logits_per_text
    
    def dbot_sinkhorn(self, P, max_iters=5):
        n = P.shape[0]
        device = P.device

        a = torch.ones((n,)).to(device)
        b_d = torch.full((n,), 0.1 * n).to(device)
        b_u = torch.full((n,), 0.9 * n).to(device)
        P = torch.exp(-P)

        for _ in range(max_iters):
            P_prev = P

            sum_P = P.sum(dim=1)
            P = torch.diag(a / sum_P) @ P

            sum_P_t = P.t().sum(dim=1)
            P = P @ torch.diag(torch.max(b_d / sum_P_t, torch.ones(P.shape[1]).to(P.device)))

            sum_P_t = P.t().sum(dim=1)
            P = P @ torch.diag(torch.min(b_u / sum_P_t, torch.ones(P.shape[1]).to(P.device)))
        return P
    
    def forward(self, all_image_features, all_text_features, logit_scale, labels):
        logits_per_image, logits_per_text = self.get_logits(all_image_features, all_text_features)
        loss = (
            F.cross_entropy(self.dbot_sinkhorn(1.0 - logits_per_image), labels) +
            F.cross_entropy(self.dbot_sinkhorn(1.0 - logits_per_text), labels)
        ) / 2
        
        return {"dbot_sinkhorn_loss": loss}
    
class Entropic_Sinkhorn_Loss(nn.Module):
    def __init__(self):
        super(Entropic_Sinkhorn_Loss, self).__init__()
        
    def get_logits(self, image_features, text_features):
        logits_per_image = image_features @ text_features.T
        logits_per_text = text_features @ image_features.T
        
        return logits_per_image, logits_per_text
    
    def sinkhorn(self, M, labels, reg=0.01, numItermax=5):
        """
        a: 1_bz
        b: ratios [r_1,r_2,...]
        M: Sim/Cost
        """
        device = M.device
        a = torch.full((M.shape[0],),1).to(device)
        b = torch.full((M.shape[0],),1/ M.shape[0]).to(device)

        # init data
        dim_a = a.shape[0]
        dim_b = b.shape[0]

        u = torch.ones(dim_a, dtype=M.dtype, device=M.device) / dim_a
        v = torch.ones(dim_b, dtype=M.dtype, device=M.device) / dim_b

        K = torch.exp(M / (-reg))

        Kp = (1 / a).reshape(-1, 1) * K

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

        P = u.reshape((-1, 1)) * K * v.reshape((1, -1))
        
        return F.cross_entropy(P, labels)

    def forward(self, all_image_features, all_text_features, logit_scale, labels):
        logits_per_image, logits_per_text = self.get_logits(all_image_features, all_text_features)
        
        loss = (
            self.sinkhorn(1.0 - logits_per_image, labels) +
            self.sinkhorn(1.0 - logits_per_text, labels)
        ) / 2
        
        return {"entropic_sinkhorn_loss": loss}
    
class InfoNCE_Loss(nn.Module):
    def __init__(self):
        super(InfoNCE_Loss, self).__init__()
        
    def get_logits(self, image_features, text_features, logit_scale):
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text
    
    def infonce(self, scaled_sim):
        positive_samples = torch.diag(scaled_sim)
        
        # log sum exp
        denominator = torch.log(torch.sum(torch.exp(scaled_sim), dim=1) + 1e-8)

        per_sample_loss = -positive_samples + denominator

        loss = torch.mean(per_sample_loss)

        return loss
    
    def forward(self, all_image_features, all_text_features, logit_scale, labels):
        logits_per_image, logits_per_text = self.get_logits(all_image_features, all_text_features, logit_scale)
        loss = (
            self.infonce(logits_per_image) +
            self.infonce(logits_per_text)
        ) / 2
        
        return {"infonce_loss": loss}
    
    
class Gromov_OT_Loss(nn.Module):
    def __init__(self):
        super(Gromov_OT_Loss, self).__init__()
        
    def get_logits(self, image_features, text_features):
        logits_per_image = image_features @ text_features.T
        logits_per_text = text_features @ image_features.T
        
        return logits_per_image, logits_per_text
    
    def sinkhorn(self, P, max_iters=100, epsilon=1e-8):
        n = P.shape[0]
        device = P.device

        a = torch.ones((n,)).to(device)
        b = torch.full((n,), 1.0 / n).to(device)
        P = torch.exp(-P / 0.01)

        for _ in range(max_iters):
            P_prev = P.clone()

            # Scale rows to meet the constraints defined by vector a
            sum_P = P.sum(dim=1, keepdim=True)
            P = P * (a / sum_P)

            # Scale columns to meet the constraints defined by vector b
            sum_P_t = P.sum(dim=0, keepdim=True)
            P = P * (b / sum_P_t)

            # Check for convergence
            if torch.norm(P - P_prev, p='fro') < epsilon:
                break

        return P
    
    def iterate_P(self, sim_matrix, m, num_iterations=5):
        P = torch.exp(sim_matrix)

        for _ in range(num_iterations):
            # C1
            row_sums_P = P.sum(dim=1)
            scaling_factors = torch.max(
                row_sums_P,
                torch.tensor(1.0).to(P.device),
            )
            P = torch.div(P, scaling_factors.unsqueeze(1))

            P = P * m / P.sum()
        return P
    
    def dbot_sinkhorn(self, P, max_iters=5):
        n = P.shape[0]
        device = P.device

        a = torch.ones((n,)).to(device)
        b_d = torch.full((n,), 0.1 * n).to(device)
        b_u = torch.full((n,), 0.9 * n).to(device)
        P = torch.exp(-P)

        for _ in range(max_iters):
            P_prev = P

            sum_P = P.sum(dim=1)
            P = torch.diag(a / sum_P) @ P

            sum_P_t = P.t().sum(dim=1)
            P = P @ torch.diag(torch.max(b_d / sum_P_t, torch.ones(P.shape[1]).to(P.device)))

            sum_P_t = P.t().sum(dim=1)
            P = P @ torch.diag(torch.min(b_u / sum_P_t, torch.ones(P.shape[1]).to(P.device)))
        return P

    
    def gromov(self, image_features, text_features, num_iters=7):
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features)
        sigma_1 = torch.matmul(image_features, image_features.T)
        sigma_2 = torch.matmul(text_features, text_features.T)

        C = (1.0 - logits_per_image).to(image_features.device)  # [bs, num_class]
        P = F.softmax(-C, dim=1)
        a = torch.ones(P.shape[0]).to(image_features.device)
        b = torch.full((P.shape[0],), 1/P.shape[0]).to(image_features.device)
        
        for iteration in range(num_iters):
            C = C + sigma_1 @ P * 0.01
            C = C - P @ sigma_2 * 0.01
            #P = ot.bregman.sinkhorn(a, b, C, 0.01)
            P = self.iterate_P(C, P.shape[0])
        return P
    
    def forward(self, all_image_features, all_text_features, logit_scale, labels):

        loss = F.cross_entropy(self.gromov(all_image_features, all_text_features) * logit_scale, labels)
        
        return {"gromov_ot_loss": loss}

class Triplet_Loss(nn.Module):
    def __init__(self, margin=1.0):
        super(Triplet_Loss, self).__init__()
        self.margin = margin

    def forward(self, all_image_features, all_text_features, logit_scale, labels):
        """
        Calculates the Triplet Loss. Assumes that for each index i in the batch, 
        all_image_features[i] and all_text_features[i] form a positive pair, and 
        all other combinations are negative pairs.

        Parameters:
        all_image_features (Tensor): Batch of image feature representations.
        all_text_features (Tensor): Batch of text feature representations.
        logit_scale (float): Scaling factor for logits (not used in loss calculation here).
        labels (Tensor): Labels indicating the correct pairs (not used in loss calculation here).

        Returns:
        Tensor: The computed Triplet Loss.
        """

        batch_size = all_image_features.size(0)
        losses = []

        for i in range(batch_size):
            anchor = all_image_features[i].unsqueeze(0)
            positive = all_text_features[i].unsqueeze(0)
            # All other features in the batch are considered as negatives.
            negatives = torch.cat([all_text_features[:i], all_text_features[i+1:]])

            # Calculate distances
            distance_positive = F.pairwise_distance(anchor, positive, p=2)
            distance_negative = F.pairwise_distance(anchor, negatives, p=2).min()

            # Calculate triplet loss for this particular triplet
            loss = F.relu(distance_positive - distance_negative + self.margin)
            losses.append(loss)

        # Calculate mean of all losses
        loss = torch.mean(torch.stack(losses))

        return {"triplet_loss": loss}
    
class UOTBadmmLoss(nn.Module):
    def __init__(self):
        super(UOTBadmmLoss, self).__init__()
        
    def get_logits(self, image_features, text_features):
        logits_per_image = image_features @ text_features.T
        logits_per_text = text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def uot_badmm(self, x: torch.Tensor, p0: torch.Tensor, q0: torch.Tensor,
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
            n = k

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
            log_mu = y - torch.logsumexp(y, dim=0, keepdim=True)  # (B, 1, D)
            y = (log_eta + log_q0 - z2)  # (B, N, 1)
            ymin, _ = torch.min(y, dim=0, keepdim=True)
            ymax, _ = torch.max(ymin - ymin + y, dim=0, keepdim=True)  # (B, 1, D)
            log_eta = (y - torch.log(
                torch.sum(torch.exp((y - ymax)), dim=0, keepdim=True)) - ymax)  # (B, N, 1)
            # update dual variables
            z1 = z1 + (torch.exp(log_mu) - torch.sum(s, dim=0, keepdim=True))  # (B, 1, D)
            z2 = z2 + (torch.exp(log_eta) - torch.sum(t, dim=0, keepdim=True))  # (B, N, 1)
        return torch.exp(log_t)

    def forward(self, all_image_features, all_text_features, logit_scale, labels):
        logits_per_image, logits_per_text = self.get_logits(all_image_features, all_text_features)
        device = logits_per_image.device
        n = logits_per_image.shape[0]
        a = torch.ones(n).to(device)
        b = torch.full((n,), 1/n).to(device)
        
        loss = F.cross_entropy(self.uot_badmm(logits_per_image, a, b), labels)
        
        return {"uotbadmm": loss}