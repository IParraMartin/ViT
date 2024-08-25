## Some Guidelines

Use ```pip install -r requirements.txt``` to install all necessary libraries.

## Note on device
If training is run on a Windows system with non-compatible GPU it may raise errors. Make sure that the device is selected properly in the torch.device() assignment.

## How to train
1 - Clone the repo in the local machine:
```
git clone https://github.com/IParraMartin/ViT.git
```
2 - Navigate to the directory using the terminal, for example:
```
cd Desktop/ViT
```
3 - Run the following command:
```
python3 train.py --config configs/vit_config_mini.yaml --no_log_wandb
```
