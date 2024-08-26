from torch import optim
import torch

def set_optimizer(optimizer: str, model: torch.nn.Module, learning_rate: float) -> optim.Optimizer:
    assert optimizer in ['Adam', 'SGD', 'AdamW'], 'Invalid optimizer. Choose between Adam, SGD, or AdamW.'
    if optimizer == 'SGD': # Worked good. Same optim and params as ResNet paper (He et al., 2015)
        return optim.SGD(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=0.01, 
            momentum=0.9
        )
    elif optimizer == 'AdamW': # Not tested yet
        return optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-5
        )
    else: # Params for Adam from Vaswani et al., (2017)
        return optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=0.01,
            betas=(0.9, 0.99),
            eps=1e-9
        )