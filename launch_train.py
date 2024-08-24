import numpy as np
import random
import yaml
import wandb

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, random_split

from train import train, evaluate
from data import AudioData
from vit import VisionTransformer
from tools.optim_selector import set_optimizer
from tools.scheduler_selector import set_scheduler


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

# Set wandb to False if you don't want to log the training process
log_wandb = False

# Load config file
with open('vit_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Paths (from config.py)
annotations = config['annotations_path']
audio_dir = config['audio_dir']
sample_rate = config['sample_rate']
n_fft = config['n_fft']
hop_length=config['hop_length']
n_mels=config['n_mels']

# Train parameters (from config.py)
batch_size = config['batch_size']
validation_split = config['validation_split']
test_split = config['test_split']
learning_rate = config['learning_rate']
n_epochs = config['n_epochs']
optimizer = config['optimizer']
scheduler = config['scheduler']

# Data transformation
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels
)

# Dataset
dataset = AudioData(
    annotations_dir=annotations, 
    audio_dir=audio_dir,
    transformation=mel_spectrogram,
    target_sample_rate=sample_rate,
    n_samples=22050,
    augment=True,
    n_augment=5
)

# Split dataset into training and validation sets and set dataloaders
generator = torch.Generator().manual_seed(42)
train_size = int((1 - validation_split) * len(dataset))
test_size = int(test_split * len(dataset))
val_size = len(dataset) - train_size - test_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Model, device, and loss
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
model = VisionTransformer(**config['model_config'])
# Try weighting classes to see if it improves the model
criterion = nn.CrossEntropyLoss()

if log_wandb:
    wandb.login()
    wandb.init(**config['wandb_config'])

optimizer = set_optimizer(optimizer, model, learning_rate)
scheduler = set_scheduler(optimizer, scheduler)

train(
    model=model,
    n_epochs=n_epochs,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    wandb=False,
    checkpoint_dir='checkpoints',
    checkpoint_interval=10,
    accumulation_steps=4
)

evaluate(
    test_dataloader=test_dataloader,
    model=model,
    criterion=criterion,
    device=device
)

# Finish wandb run
if log_wandb:
    wandb.finish() 