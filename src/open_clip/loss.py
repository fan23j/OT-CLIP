import torch
import torch.nn as nn
from torch.nn import functional as F
from .losses import DBOT_Sinkhorn_Loss, Cross_Entropy_Loss

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None
    
def gather_tensor(tensor, with_grad=False, world_size=1, rank=0, local_loss=False):
    if with_grad:
        return torch.cat(torch.distributed.nn.all_gather(tensor), dim=0)

    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)

    if not local_loss:
        gathered_tensors[rank] = tensor

    return torch.cat(gathered_tensors, dim=0)


def gather_features(
    image_features,
    image_aug_features,
    text_features,
    text_aug_features,
    unpaired_image_features=None,
    unpaired_image_aug_features=None,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
):
    assert (
        has_distributed
    ), "torch.distributed did not import correctly, please use a PyTorch version with support."

    all_image_features = gather_tensor(
        image_features, gather_with_grad, world_size, rank, local_loss
    )
    all_text_features = gather_tensor(
        text_features, gather_with_grad, world_size, rank, local_loss
    )

    return (
        all_image_features,
        all_text_features,
    )

_loss_factory = {
    'dbot_sinkhorn': DBOT_Sinkhorn_Loss,
    'cross_entropy': Cross_Entropy_Loss,
}


class ClipLoss(nn.Module):

    def __init__(
            self,
            losses,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.losses = [_loss_factory[name] for name in losses]
        
        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        image_features, text_features = gather_features(
            image_features, text_features,
            self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        loss_states = {}
        for loss in self.losses:
            loss_output = loss(image_features, text_features, logit_scale, labels)
            loss_dict = loss_output
            loss_states.update(loss_dict)

        return loss_states