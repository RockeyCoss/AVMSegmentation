import torch
from torch.optim.lr_scheduler import _LRScheduler

from builder import SCHEDULER


@SCHEDULER.registry_module()
class PolyScheduler(_LRScheduler)
def poly_scheduler(optimizer: torch.optim.Optimizer,
                   max_iters: int,
                   init_lr=1e-3,
                   min_lr=1e-5,
                   power=0.9999,
                   ):
    def schdule()
    assert max_iters >= 1
    self.
