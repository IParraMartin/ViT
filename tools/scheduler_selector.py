import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR, SequentialLR, LinearLR
from transformers import get_scheduler

# MAKE LINEAR WITH WARMUP (use kwargs for multiple scheduler options?)

def set_scheduler(optimizer: optim.Optimizer, scheduler: str = 'plateau') -> optim.lr_scheduler:

    assert scheduler in ['linear', 'step', 'plateau', 'cosine', 'sequential', 'none'], 'Invalid scheduler. Choose between linear, step, plateau, cosine, sequential, or none'

    if scheduler == 'linear':
        return LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=0.00001,
            total_iters=62500
        )

    if scheduler == 'step': # every x epochs (step_size) reduce lr multiplying it by y (gamma)
        return StepLR(
            optimizer,
            step_size=10,
            gamma=0.1,
            last_epoch=90
        )
    
    elif scheduler == 'plateau': # factor = Factor by which the learning rate will be reduced
        return ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.1,
            patience=3,
            min_lr=1e-6
        )
    
    elif scheduler == 'cosine': # T_max = number of epochs/steps after which the scheduler will reset, eta_min = minimum learning rate
        return CosineAnnealingLR(
            optimizer, 
            T_max=100,
            eta_min=1e-6
        )
    
    elif scheduler == 'sequential': # milestones = List of epoch indices. Must be increasing.
        return SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(
                    optimizer,
                    start_factor=0.001,
                    end_factor=1.0,
                    total_iters=2000
                ),
                LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=0.001,
                    total_iters=62500
                )
            ],
            milestones=[2000]
        )
    
    elif scheduler == 'none':
        return None
    
    else:
        raise ValueError('Invalid scheduler. Choose between linear, step, plateau, cosine, sequential, or none')