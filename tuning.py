import torch
import torchvision
from torchvision import transforms
import torch.utils
import torch.utils.data

import yaml

from vit import VisionTransformer
from train import train
from tools.scheduler_selector import set_scheduler

CHECKPOINT_PATH = '/Users/inigoparra/Desktop/ViT/checkpoints/checkpoint_61.pt'

with open('configs/cifar10.yaml', 'r') as file:
    config = yaml.safe_load(file)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

checkpoint = torch.load(CHECKPOINT_PATH)
model = VisionTransformer(**config['model_config'])
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
# Change to SGD to improve generalization
optimizer = torch.optim.SGD(model.parameters(), config['learning_rate'], momentum=0.9, weight_decay=0.01)
scheduler = set_scheduler(optimizer, config['scheduler'])
criterion = torch.nn.CrossEntropyLoss()

transformation = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.4813,), (0.2386,))
])

cifar_10_train = torchvision.datasets.CIFAR10(
    root='./cifar',
    train=True,
    download=True,
    transform=transformation
)

train_size = int(0.8 * len(cifar_10_train))
val_size = len(cifar_10_train) - train_size

seed_generator = torch.Generator().manual_seed(config['seed'])
train_split, val_split = torch.utils.data.random_split(cifar_10_train, [train_size, val_size], generator=seed_generator)
train_dataloader = torch.utils.data.DataLoader(train_split, config['batch_size'], shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_split, config['batch_size'], shuffle=False)

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
