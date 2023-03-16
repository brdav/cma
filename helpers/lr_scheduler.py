import torch
from torch.optim import Optimizer


class LinearWarmupLinearLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 max_steps: int = None,
                 warmup_iters: int = 1500,
                 warmup_ratio: float = 1e-6,
                 min_lr=0.,
                 last_epoch=-1):
        self.max_updates = max_steps
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):

        # warmup phase
        if self.last_epoch < self.warmup_iters:
            k = (1 - self.last_epoch / self.warmup_iters) * \
                (1 - self.warmup_ratio)
            return [_lr * (1 - k) for _lr in self.base_lrs]

        # poly phase
        else:
            coeff = 1 - (self.last_epoch - self.warmup_iters) / \
                     float(self.max_updates - self.warmup_iters)
            return [(base_lr - self.min_lr) * coeff + self.min_lr for base_lr in self.base_lrs]
