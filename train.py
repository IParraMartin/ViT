import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.nn.utils import clip_grad_norm_
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassF1Score, MulticlassRecall

import numpy as np
import random
import yaml
import os
import wandb
import argparse

from vit import VisionTransformer
from tools.optim_selector import set_optimizer
from tools.scheduler_selector import set_scheduler
from data import AudioData


"""
TO DO:
    - More work on setting optimal parameters for the MelSpectrogram
    - Explore other transformations that capture the target features better
    - Try different number of samples
    - Analyze the performance with different non-linearities
    - Try weighting classes (CELoss) to see if it improves the model
"""

def save_checkpoint(model, optimizer, scheduler, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, path)


def train(
        n_epochs: int, 
        model: nn.Module, 
        train_dataloader: DataLoader, 
        val_dataloader: DataLoader,
        num_classes: int,
        criterion: nn.Module, 
        optimizer: optim.Optimizer, 
        scheduler: optim.lr_scheduler, 
        device: torch.device, 
        wandb_log: bool = False,
        checkpoint_interval: int = 20,
        ACCUMULATION_STEPS: int = 4,
        grad_clip: bool = False
    ):
    
    print(f"{'-'*50}\nDevice: {device}")
    print(f"Scheduler: {type(scheduler).__name__}\n{'-'*50}")
    print(f"Training...")
    
    model.to(device)

    if wandb_log:
        global_step = 0
        log_interval = 10

    os.makedirs('checkpoints', exist_ok=True)
    best_val_loss = float('inf')

    # Training
    for epoch in range(n_epochs):

        model.train()
        running_train_loss = 0.0
        train_accuracy = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
        f1_macro = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
        f1_micro = MulticlassF1Score(num_classes=num_classes, average='micro').to(device)
        precision = MulticlassPrecision(num_classes=num_classes, average='micro').to(device)
        recall = MulticlassRecall(num_classes=num_classes, average='micro').to(device)

        for batch_idx, (signals, labels) in enumerate(train_dataloader):
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            loss = criterion(outputs, labels)
            running_train_loss += loss.item()       # Get the actual loss to print

            # Normalize the gradients for the accumulation steps (gives training stability)
            loss = loss / ACCUMULATION_STEPS
            loss.backward()
            
            if grad_clip:
                clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights
            if ((batch_idx + 1) % ACCUMULATION_STEPS == 0) or (batch_idx + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()

                if wandb_log:
                    global_step += 1
                    if global_step % log_interval == 0:
                        wandb.log({
                            'step': global_step,
                            'train_loss': running_train_loss / (batch_idx + 1),
                            'learning_rate': scheduler.get_last_lr()[0]
                        }) 

                train_accuracy.update(outputs, labels)
                f1_macro.update(outputs, labels)
                f1_micro.update(outputs, labels)
                precision.update(outputs, labels)
                recall.update(outputs, labels)

            if (batch_idx + 1) % 10 == 0: # Maybe the same as ACCUMULATION_STEPS to match the gradient accumulation?
                avg_loss = running_train_loss / (batch_idx + 1)
                print(f'Epoch [{epoch+1}/{n_epochs}] - Step [{batch_idx+1}/{len(train_dataloader)}] - Loss: {avg_loss:.3f} - Lr: {optimizer.param_groups[0]["lr"]:.6f}')
        
        epoch_train_loss = running_train_loss / len(train_dataloader)
        train_accuracy, train_f1_macro, train_f1_micro, train_precision, train_recall = train_accuracy.compute(), f1_macro.compute(), f1_micro.compute(), precision.compute(), recall.compute()

        if wandb_log:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': epoch_train_loss,
                'train_accuracy': train_accuracy.item(),
                'train_f1_macro': train_f1_macro.item(),
                'train_f1_micro': train_f1_micro.item(),
                'train_precision': train_precision.item(),
                'train_recall': train_recall.item()
            })
        
        print(f'Epoch [{epoch+1}/{n_epochs}] - Train Loss: {epoch_train_loss:.3f} || Train Accuracy: {train_accuracy.item():.3f}')
        print(f'Train F1-Macro: {train_f1_macro.item():.3f} || Train F1-Micro: {train_f1_micro.item():.3f} || Precision: {train_precision.item():.3f} || Recall: {train_recall.item():.3f}')


        # Validation
        model.eval()
        running_val_loss = 0.0
        val_accuracy = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
        val_f1_macro = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
        val_f1_micro = MulticlassF1Score(num_classes=num_classes, average='micro').to(device)
        val_precision = MulticlassPrecision(num_classes=num_classes, average='micro').to(device)
        val_recall = MulticlassRecall(num_classes=num_classes, average='micro').to(device)

        with torch.no_grad():
            for signals, labels in val_dataloader:
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                val_accuracy.update(outputs, labels)
                val_f1_macro.update(outputs, labels)
                val_f1_micro.update(outputs, labels)
                val_precision.update(outputs, labels)
                val_recall.update(outputs, labels)

        epoch_val_loss = running_val_loss / len(val_dataloader)
        val_accuracy, val_f1_macro, val_f1_micro, val_precision, val_recall = val_accuracy.compute(), val_f1_macro.compute(), val_f1_micro.compute(), val_precision.compute(), val_recall.compute()

        if scheduler is not None:
            scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, 'checkpoints/best_model.pt')

        if wandb_log:
            wandb.log({
                'epoch': epoch + 1,
                'val_loss': epoch_val_loss,
                'val_accuracy': val_accuracy.item(),
                'val_f1_macro': val_f1_macro.item(),
                'val_f1_micro': val_f1_micro.item(),
                'val_precision': val_precision.item(),
                'val_recall': val_recall.item()
            })

        print(f'Learning rate: {scheduler.get_last_lr()[0]}')
        print(f'Epoch [{epoch+1}/{n_epochs}] - Train Loss: {epoch_train_loss:.3f} - Val Loss: {epoch_val_loss:.3f} || Val Accuracy: {val_accuracy.item():.3f}')
        print(f'Val F1-Macro: {val_f1_macro.item():.3f} || Val F1-Micro: {val_f1_micro.item():.3f} || Val Precision: {val_precision.item():.3f} || Val Recall: {val_recall.item():.3f}')


        if epoch % checkpoint_interval == 0 and epoch != 0:
            checkpoint_path = os.path.join('checkpoints', f'checkpoint_{epoch+1}.pt')
            save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path)

    print("Training complete.")


# EVALUATION IN TEST SET
def evaluate(model: nn.Module, test_dataloader: DataLoader, num_classes: int, criterion: nn.Module, device: torch.device):

    print("Evaluating...")

    model.to(device)
    model.eval()
    test_loss = 0.0
    test_accuracy = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
    test_f1_macro = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    test_f1_micro = MulticlassF1Score(num_classes=num_classes, average='micro').to(device)
    test_precision = MulticlassPrecision(num_classes=num_classes, average='micro').to(device)
    test_recall = MulticlassRecall(num_classes=num_classes, average='micro').to(device)

    with torch.no_grad():
        for signals, labels in test_dataloader:
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            test_accuracy.update(outputs, labels)
            test_f1_macro.update(outputs, labels)
            test_f1_micro.update(outputs, labels)
            test_precision.update(outputs, labels)
            test_recall.update(outputs, labels)

    test_loss = test_loss / len(test_dataloader)
    test_accuracy, test_f1_macro, test_f1_micro, test_precision, test_recall = test_accuracy.compute(), test_f1_macro.compute(), test_f1_micro.compute(), test_precision.compute(), test_recall.compute()
    
    # Evaluation results
    print(f'Test Loss: {test_loss:.3f} || Test Accuracy: {test_accuracy:.3f}')
    print(f'Test F1-Macro: {test_f1_macro.item():.3f} || Test F1-Micro: {test_f1_micro.item():.3f} || Test Precision: {test_precision.item():.3f} || Test Recall: {test_recall.item():.3f}')

    print("Evaluation complete.")


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
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
        grad_clip=config['grad_clip']
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
