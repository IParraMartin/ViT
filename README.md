## Some Guidelines

Use ```pip install -r requirements.txt``` to install all necessary libraries.

## Note on device (SOLVED)
If training is run on a Windows system with non-compatible GPU it may raise errors. Make sure that the device is selected properly in the torch.device() assignment.

## Dataset
You can find the original ESC-50 dataset here: https://github.com/karolpiczak/ESC-50

## How to train
1 - Clone the repo in the local machine:
```
git clone https://github.com/IParraMartin/ViT.git
```
2 - Navigate to the directory using the terminal, for example:
```
cd Desktop/ViT
```
3 - To train the base model, run the following command:
```
python3 train.py --config configs/vit_base.yaml --no_log_wandb
```

To train the model on the CIFAR10 dataset, use the following command:
```
python3 cifar10_train.py --config configs/cifar10.yaml --no_log_wandb
```
