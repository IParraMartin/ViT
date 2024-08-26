import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import numpy as np

from train import train, evaluate
from vit import VisionTransformer

import certifi
import os
import yaml
import random

from tools.optim_selector import set_optimizer
from tools.scheduler_selector import set_scheduler

os.environ['SSL_CERT_FILE'] = certifi.where()



if __name__ == '__main__':

    with open('configs/cifar10.yaml', 'r') as f:
        config = yaml.safe_load(f)

    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    set_seed(config['seed'])

    transformation = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    cifar_10_train = torchvision.datasets.CIFAR10(
        root='./cifar',
        train=True,
        download=True,
        transform=transformation
    )

    cifar_10_test = torchvision.datasets.CIFAR10(
        root='./cifar',
        train=False,
        download=True,
        transform=transformation
    )

    train_size = int(0.8 * len(cifar_10_train))
    test_size = int(0.1 * len(cifar_10_train))
    val_size = len(cifar_10_train) - train_size - test_size

    seed_generator = torch.Generator().manual_seed(config['seed'])
    train_split, val_split, test_split = random_split(cifar_10_train, [train_size, val_size, test_size], generator=seed_generator)
    train_dataloader = DataLoader(train_split, config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_split, config['batch_size'], shuffle=False)
    test_dataloader = DataLoader(test_split, config['batch_size'], shuffle=False)

    print(f'{'-'*50}\nTrain size: {len(train_split)} - Val size: {len(val_split)} - Test size: {len(test_split)}\n{'-'*50}')

    model = VisionTransformer(**config['model_config'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()
    optimizer = set_optimizer(config['optimizer'], model, config['learning_rate'])
    scheduler = set_scheduler(optimizer, config['scheduler'])

    train(
        model=model,
        n_epochs=config['n_epochs'],
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_classes=config['num_classes'],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        wandb_log=False,
        checkpoint_interval=config['checkpoint_interval'],
        ACCUMULATION_STEPS=config['accumulation_steps'],
        grad_clip=config['grad_clip']
    )

    evaluate(
        test_dataloader=test_dataloader,
        model=model,
        num_classes=config['num_classes'],
        criterion=criterion,
        device=device
    )