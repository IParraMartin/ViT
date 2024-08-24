import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import random

from torch.utils.data import Dataset
import torchaudio
import torch

from tools.augmentate import Augmentator



# For reproducibility, set the seed for all random number generators
# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     np.random.seed(seed)
#     random.seed(seed)

# set_seed(42)


class AudioData(Dataset):
    
    def __init__(self, annotations_dir: str, audio_dir: str, target_sample_rate: int, n_samples: int, transformation = None, n_augment: int = 5, augment: bool = False) -> None:
        super().__init__()
        self.annotations = pd.read_csv(annotations_dir)
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.n_samples = n_samples
        self.transformation = transformation
        self.to_db = torchaudio.transforms.AmplitudeToDB()

        """
        Data augmentation helped to improve the model's performance.
        """
        self.n_augment = n_augment
        self.augment = augment
        if self.augment:
            self.augmentator = Augmentator()
    

    def __len__(self):
        # Multiply by the number of augmentations (may be limited by the types of augmentations)
        return len(self.annotations) * (self.n_augment if self.augment else 1)


    def __getitem__(self, index):
        # Original index is calculated by performing floor division of the given index by the number of augmentations.
        # This maps the larger dataset index back to the index of the original sample in the annotations.
        original_index = index // self.n_augment
        
        # Augment index is calculated by taking the modulus of the given index with the number of augmentations.
        # This determines which version (original or augmented) of the sample is being accessed.
        augment_index = index % self.n_augment

        audio_sample_path = self.get_audio_sample_path(original_index)
        label = self.get_audio_sample_label(original_index)

        if not os.path.exists(audio_sample_path):
            raise FileNotFoundError(f'Audio file {audio_sample_path} not found.')
        
        try:
            signal, sr = torchaudio.load(audio_sample_path)
        except Exception as e:
            raise RuntimeError(f'Error loading {audio_sample_path}. {e}')

        signal = self.resample_if_necessary(signal, sr)
        signal = self.mixdown_if_necessary(signal)
        signal = self.pad_if_necessary(signal)
        signal = self.truncate_if_necessary(signal)

        """
        Augmentation section:
            1 - Check boolean flag: If True, we augment the signal
            2 - Transform to numpy: Some transformations require the signal to be in numpy format
            3 - Augment with the augmentator
            4 - Transform back to torch.Tensor: we need to convert back to torch.Tensor
        """

        if self.augment:
            # transform to numpy float32
            signal = signal.numpy()
            signal = signal.astype(np.float32)
            # If the augment_index is not 0 (because of the division), we augment the signal

            if augment_index != 0:
                signal = self.augmentator.augmentate(signal, sample_rate=self.target_sample_rate)

        if self.transformation is not None:
            # Convert back to torch.Tensor if necessary after augmentation
            if self.augment:
                signal = torch.from_numpy(signal)
            
            mel_spec = self.transformation(signal)
            
            # Convert to decibels
            mel_spec_db = self.to_db(mel_spec)
            
            # Square the spectrogram
            squared_spec = torch.nn.functional.interpolate(
                mel_spec_db.unsqueeze(0),
                size=(mel_spec_db.shape[1], mel_spec_db.shape[1]),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            signal = squared_spec

        return signal, label
        

    def get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 2]
    

    def resample_if_necessary(self, signal, sr):
        """
        We are resampling the sound to the sample rate we need
        """
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal


    def mixdown_if_necessary(self, signal: torch.Tensor):
        """
        We are mixing stereo to mono: (n_channels, samples) => (2, 16000) -> (1, 16000)
        """
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    

    def pad_if_necessary(self, signal: torch.Tensor):
        """
        Right pad the signal if the samples are not met.
        """
        len_signal = signal.shape[1]
        if len_signal < self.n_samples:
            n_missing = self.n_samples - len_signal
            r_padding = (0, n_missing)
            signal = torch.nn.functional.pad(signal, r_padding)
        return signal
 

    def truncate_if_necessary(self, signal: torch.Tensor):
        """
        Truncate the signal if it is too long.
        """
        if signal.shape[1] > self.n_samples:
            signal = signal[:, :self.n_samples]
        return signal
    

    def get_audio_sample_path(self, index):
        # folder = f'fold{self.annotations.iloc[index, 2]}'
        # path = os.path.join(self.audio_dir, folder, self.annotations.iloc[index, 0])
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 0])
        return path
    

    @staticmethod
    def plot_example(signal, title: str = None, save: bool = False):
        plt.figure()
        plt.imshow(signal.squeeze().numpy(), aspect='equal', cmap='inferno', origin='lower')
        plt.tight_layout()
        plt.title(title)
        plt.show()
    

if __name__ == "__main__": 
    annotations = '/Users/inigoparra/Desktop/ESC-50-master-2/meta/esc50.csv'
    audio_dir = '/Users/inigoparra/Desktop/ESC-50-master-2/audio'
    sample_rate = 22050

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=2048,
        hop_length=256,
        n_mels=256 # Can be 128 for more resolution
    )

    usd = AudioData(
        annotations_dir=annotations, 
        audio_dir=audio_dir,
        transformation=mel_spectrogram,
        target_sample_rate=sample_rate,
        n_samples=44100,
        n_augment=5, # Increase the number x5 (5 different augmentations max)
        augment=True
    )

    assert str(torchaudio.list_audio_backends()) is not None, 'Try <pip install soundfile> or <pip3 install soundfile>'

    print(f'Total length of the dataset: {len(usd)}')
    spectrogram, label = usd[15]
    print(f'Shape of squared spectrogram: {spectrogram.shape}')
    usd.plot_example(spectrogram, title=f"Label: {label}")
