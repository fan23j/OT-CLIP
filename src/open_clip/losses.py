import torch.nn as nn
from torch.nn import functional as F
import torch

class Cross_Entropy_Loss(nn.Module):
    def __init__(self):
        super(Cross_Entropy_Loss, self).__init__()
        
    def get_logits(self, image_features, text_features, logit_scale):
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, all_image_features, all_text_features, logit_scale, labels, **kwargs):
        logits_per_image, logits_per_text = self.get_logits(all_image_features, all_text_features, logit_scale)
        loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2
        
        return {"cross_entropy_loss": loss}

class DBOT_Sinkhorn_Loss(nn.Module):
    def __init__(self):
        super(DBOT_Sinkhorn_Loss, self).__init__()
        
    def get_logits(self, image_features, text_features, logit_scale):
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        
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
    
    def forward(self, all_image_features, all_text_features, logit_scale, labels, **kwargs):
        logits_per_image, logits_per_text = self.get_logits(all_image_features, all_text_features, logit_scale)
        loss = (
            F.cross_entropy(self.dbot_sinkhorn(1.0 - logits_per_image), labels) +
            F.cross_entropy(self.dbot_sinkhorn(1.0 - logits_per_text), labels)
        ) / 2
        
        return {"dbot_sinkhorn_loss": loss}
