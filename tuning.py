import torch
from vit import VisionTransformer
import yaml


with open('configs/cifar10.yaml', 'r') as file:
    config = yaml.safe_load(file)

CHECKPOINT_PATH = '/Users/inigoparra/Desktop/ViT/checkpoints/best_model.pt'

model = VisionTransformer(**config['model_config'])

optimizer = torch.optim.SGD(
    model.parameters(), 
    lr=0.001, 
    momentum=0.9, 
    weight_decay=0.01
)

checkpoint = torch.load(CHECKPOINT_PATH)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epochs = checkpoint['epoch']

