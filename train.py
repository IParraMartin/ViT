import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.nn.utils import clip_grad_norm_

import numpy as np
import random
import yaml
import os

from vit import VisionTransformer
from tools.optim_selector import set_optimizer
from tools.scheduler_selector import set_scheduler
from data import AudioData

import wandb


"""
TO DO:
    - More work on setting optimal parameters for the MelSpectrogram
    - Explore other transformations that capture the target features better
    - Try different number of samples
    - Analyze the performance with different h_dim values
    - Analyze the performance with different non-linearities
"""


# For reproducibility, set the seed for all random number generators
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)


def save_checkpoint(model, optimizer, scheduler, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, path)


# TRAINING
def train(
        n_epochs: int, 
        model: nn.Module, 
        train_dataloader: DataLoader, 
        val_dataloader: DataLoader, 
        criterion: nn.Module, 
        optimizer: optim.Optimizer, 
        scheduler: optim.lr_scheduler, 
        device: torch.device, 
        wandb: bool = False,
        checkpoint_dir: str = 'checkpoints',
        checkpoint_interval: int = 20
    ):
    
    print(f"{'-'*50}\nDevice: {device}")
    print(f"Scheduler: {type(scheduler).__name__}\n{'-'*50}")
    print(f"Training...")
    
    model.to(device)
    if wandb:
        global_step = 0
        log_interval = 10

    # Make a checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(n_epochs):
        # TRAIN
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        for batch_idx, (signals, labels) in enumerate(train_dataloader):
            signals, labels = signals.to(device), labels.to(device)
            
            # expected signals shape should be [batch_size, channels, height, width]
            if len(signals.shape) != 4:
                signals = signals.unsqueeze(1)
            
            outputs = model(signals)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if wandb:
                global_step += 1

            # Print step metrics in the local console
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{n_epochs}] - Step [{batch_idx+1}/{len(train_dataloader)}] - Loss: {loss.item():.3f}')

            # Log metrics to wandb
            if wandb and global_step % log_interval == 0:
                wandb.log({
                    'step': global_step,
                    'train_loss': loss.item(),
                    'train_accuracy': train_accuracy,
                    'learning_rate': scheduler.get_last_lr()
                })
        
        train_accuracy = (correct_train / total_train) * 100
        epoch_train_loss = running_train_loss / len(train_dataloader)
        # Print epoch metrics in the local console
        print(f'Epoch [{epoch+1}/{n_epochs}] - Train Loss: {epoch_train_loss:.3f} || Acc: {train_accuracy:.3f}')


        # VALIDATION
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for signals, labels in val_dataloader:
                signals, labels = signals.to(device), labels.to(device)
                
                if len(signals.shape) == 4:
                    signals = signals.squeeze(1)
                
                signals = signals.unsqueeze(1)
                
                outputs = model(signals)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                # Count predictions per class to see if there's an imbalance
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_dataloader)
        val_accuracy = (correct / total) * 100

        # Pass loss to scheduler and update learning rate (if needed)
        if scheduler is not None:
            scheduler.step()

        #Log validation metrics to wandb
        if wandb:
            wandb.log({
                'step': global_step,
                'val_loss': epoch_val_loss,
                'val_accuracy': val_accuracy
            })

        # Print LR and summary
        print(f'Learning rate: {scheduler.get_last_lr()}')
        print(f'Epoch [{epoch+1}/{n_epochs}] - Train Loss: {epoch_train_loss:.3f} - Val Loss: {epoch_val_loss:.3f} || Val Accuracy: {val_accuracy:.3f}')

        # Save checkpoint every x epochs
        if epoch % checkpoint_interval == 0 and epoch != 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{epoch+1}.pt')
            save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path)

    print("Training complete.")


# EVALUATION IN TEST SET
def evaluate(model: nn.Module, test_dataloader: DataLoader, criterion: nn.Module, device: torch.device):
    print("Evaluating...")
    model.to(device)
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for signals, labels in test_dataloader:
            signals, labels = signals.to(device), labels.to(device)

            if len(signals.shape) == 4:
                signals = signals.squeeze(1)

            signals = signals.unsqueeze(1)

            outputs = model(signals)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_dataloader)
    test_accuracy = (correct / total) * 100
    
    # Evaluation results
    print(f'Test Loss: {test_loss:.3f} || Test Accuracy: {test_accuracy:.3f}')
    print("Evaluation complete.")



if __name__ == '__main__':

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
        wandb=False
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
