import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import numpy as np
import random
import yaml
import wandb
import argparse

from vit import VisionTransformer
from data import AudioData

from train import train, evaluate
from tools.optim_selector import set_optimizer
from tools.scheduler_selector import set_scheduler


if __name__ == '__main__':

    # For reproducibility, set the seed for all random number generators
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    parser = argparse.ArgumentParser(description='Train a Vision Transformer model on audio data')
    parser.add_argument('--config', type=str, default='configs/vit_base.yaml', help='Path to the configuration file')
    parser.add_argument('--device', type=str, default='mps', help='Device on which the model will be trained.')
    parser.add_argument('--log_wandb', dest='log_wandb', action='store_true', help='Log metrics to wandb')
    parser.add_argument('--no_log_wandb', dest='log_wandb', action='store_false', help='Do not log metrics to wandb')
    parser.set_defaults(log_wandb=False)
    args = parser.parse_args()

    # Load configuration file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Set seed for reproducibility
    set_seed(config['seed'])

    # Data transformation
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=config['sample_rate'],
        n_fft=config['n_fft'],
        hop_length=config['hop_length'],
        n_mels=config['n_mels']
    )

    # Dataset
    dataset = AudioData(
        annotations_dir=config['annotations_path'], 
        audio_dir=config['audio_path'],
        transformation=mel_spectrogram,
        target_sample_rate=config['sample_rate'],
        n_samples=config['n_samples'],
        augment=config['augment'],
        n_augment=config['n_augment']
    )

    # Split dataset into training and validation sets and set dataloaders
    generator = torch.Generator().manual_seed(config['seed'])
    train_size = int((1 - config['validation_split']) * len(dataset))
    test_size = int(config['test_split'] * len(dataset))
    val_size = len(dataset) - train_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Model, device, and loss
    def set_device(use_device: str = 'cuda'):
        assert use_device in ['cuda', 'mps', 'cpu'], "Device Error. Torch devices available: 'cuda', 'mps', or 'cpu'"
        if use_device == 'cuda':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                raise RuntimeError('CUDA is not available. Choose another device.')
        elif use_device == 'mps':
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                raise RuntimeError('MPS is not available. Choose another device.')
        elif use_device == 'cpu':
            device = torch.device('cpu')
        else:
            raise RuntimeError(f"Unknown device: {use_device}. Please use 'cuda', 'mps', or 'cpu'")
        return device
    
    device = set_device(args.device)
    model = VisionTransformer(**config['model_config'])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {n_params / 1e6:.2f}M')
    print(f'Number of classes: {config['num_classes']}')

    criterion = nn.CrossEntropyLoss()

    if args.log_wandb:
        wandb.login()
        wandb.init(**config['wandb_config'])

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
        wandb_log=args.log_wandb,
        checkpoint_interval=config['checkpoint_interval'],
        ACCUMULATION_STEPS=config['accumulation_steps'],
        grad_clip=config['grad_clip'],
        save_path=config['chekpoints_path']
    )

    evaluate(
        test_dataloader=test_dataloader,
        model=model,
        num_classes=config['num_classes'],
        criterion=criterion,
        device=device
    )

    # Finish wandb run
    if args.log_wandb:
        wandb.finish() 
