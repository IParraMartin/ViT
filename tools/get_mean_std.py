import torch

def get_mean_and_std(dataset):
    mean = torch.zeros(1)                           # Make a 1-dim tensor to fill with mean values
    std = torch.zeros(1)                            # Make a 1-dim tensor to fill with mean values
    for images, _ in dataset:                       # Iterate over the images in the train set
        mean += torch.mean(images)                  # get mean for each channel
        std += torch.std(images)                    # get std for each channel
    mean /= len(dataset)                            # Average
    std /= len(dataset)
    return mean, std