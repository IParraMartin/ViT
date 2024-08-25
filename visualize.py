import torch
import torch.nn as nn
import yaml
import torchaudio
from data import AudioData
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
import random
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)


# Load configuration file
with open('configs/vit_base.yaml', 'r') as file:
    config = yaml.safe_load(file)


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
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)


class PatchEmbeddingsConv(nn.Module):

    def __init__(self, img_size: int, patch_size: int = 7, in_channels: int = 1, d_model: int = 512):
        super().__init__()
        assert img_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.projection = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=d_model,
                kernel_size=patch_size, 
                stride=patch_size,
                bias=False 
            ),
            nn.Flatten(start_dim=2)
        )

    def forward(self, x) -> torch.Tensor:
        x = self.projection[0](x)
        return x

def view_examples(dataloader):
    data_iter = iter(dataloader)
    images, _ = next(data_iter)
    image = images[1]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image.squeeze().numpy(), cmap='viridis')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

view_examples(dataloader=train_dataloader)

def visualize_patches(model, dataloader):
    data_iter = iter(dataloader)
    images, _ = next(data_iter)
    image = images[1]
    patches = model(image.unsqueeze(0))
    patches = patches.squeeze(0)
    n_patches = patches.shape[0]
    grid_size = int(n_patches ** 0.5)
    
    patch_size = model.patch_size
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < n_patches:
            patch = image[:, i // grid_size * patch_size:(i // grid_size + 1) * patch_size, i % grid_size * patch_size:(i % grid_size + 1) * patch_size]
            ax.imshow(patch.squeeze().cpu().numpy(), cmap='viridis')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

img_size    = 256
patch_size  = 16
in_channels = 1
d_model     = 512

model = PatchEmbeddingsConv(img_size, patch_size, in_channels, d_model)
visualize_patches(model, train_dataloader)

