import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR, SequentialLR, ConstantLR, LinearLR
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup 


def set_scheduler(optimizer: optim.Optimizer, scheduler: str = 'plateau') -> optim.lr_scheduler:

    assert scheduler in ['step', 'plateau', 'cosine', 'sequential', 'linear_warmup', 'none'], 'Invalid scheduler. Choose between step, plateau, cosine, or sequential'

    if scheduler == 'step': # every x epochs (step_size) reduce lr multiplying it by y (gamma)
        return StepLR(
            optimizer,
            step_size=25,
            gamma=0.1
        )
    
    elif scheduler == 'plateau': # factor = Factor by which the learning rate will be reduced
        return ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1,
            patience=5
        )
    
    elif scheduler == 'cosine': # T_max = number of epochs after which the scheduler will reset, eta_min = minimum learning rate
        return CosineAnnealingLR(
            optimizer, 
            T_max=10,
            eta_min=1e-6
        )
    
    elif scheduler == 'sequential': # milestones = List of epoch indices. Must be increasing.
        return SequentialLR(
            optimizer,
            schedulers=[
                ConstantLR(
                    optimizer, 
                    factor=1.0, 
                    total_iters=5
                ),
                CosineAnnealingLR(
                    optimizer,
                    T_max=10,
                    eta_min=1e-5
                )
            ],
            milestones=[10]
        )
    
    elif scheduler == 'linear_warmup':
        return create_lr_scheduler_with_warmup(
            lr_scheduler=LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=0.0001,
            ),
            warmup_start_value=0.0,
            warmup_duration=5000,
            warmup_end_value=0.1
        )
    
    elif scheduler == 'none':
        return None
    
    else:
        raise ValueError('Invalid scheduler. Choose between step, plateau, or cosine')