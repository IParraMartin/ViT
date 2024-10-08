import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassF1Score, MulticlassRecall

import os
import wandb

"""
TO DO:
    - More work on setting optimal parameters for the MelSpectrogram
    - Explore other transformations that capture the target features better
    - Try different number of samples
    - Analyze the performance with different non-linearities
    - Try weighting classes (CELoss) to see if it improves the model
"""

def save_checkpoint(model, optimizer, scheduler, epoch, path):
    if scheduler is not None:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, path)
    else:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
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
        grad_clip: bool = False,
        save_path: str = 'checkpoints'
    ):
    
    print(f"{'-'*50}\nDevice: {device}")
    print(f"Scheduler: {type(scheduler).__name__}\n{'-'*50}")
    print(f"Training...")
    
    model.to(device)

    if wandb_log:
        global_step = 0
        log_interval = 10

    os.makedirs(save_path, exist_ok=True)
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

                if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step()

                if wandb_log:
                    global_step += 1
                    if global_step % log_interval == 0:
                        wandb.log({
                            'step': global_step,
                            'train_loss': running_train_loss / (batch_idx + 1),
                            'learning_rate': optimizer.param_groups[0]['lr']
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

        if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_val_loss)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, f'{save_path}/best_model.pt')

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

        print(f' LR: {optimizer.param_groups[0]['lr']}')
        print(f'Epoch [{epoch+1}/{n_epochs}] - Train Loss: {epoch_train_loss:.3f} - Val Loss: {epoch_val_loss:.3f} || Val Accuracy: {val_accuracy.item():.3f}')
        print(f'Val F1-Macro: {val_f1_macro.item():.3f} || Val F1-Micro: {val_f1_micro.item():.3f} || Val Precision: {val_precision.item():.3f} || Val Recall: {val_recall.item():.3f}')

        if epoch % checkpoint_interval == 0 and epoch != 0:
            if scheduler is not None:
                checkpoint_path = os.path.join(save_path, f'checkpoint_{epoch+1}.pt')
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
