from audiomentations import Compose, AddGaussianNoise, PitchShift, HighPassFilter, LowPassFilter, Gain
import librosa
import soundfile as sf


"""
TO DO:
    - Try with background noise augmentation (added the sound sample in the folder).
    - Make sure the augmented signals are paired with labels correctly.
"""


class Augmentator():

    """
    Class to augmentate the audio signals.
    Uses the audiomentations library.
    """

    def __init__(self) -> None:
        self.augmentate = Compose([
            AddGaussianNoise(
                min_amplitude=0.001, 
                max_amplitude=0.015, 
                p=1
            ),
            PitchShift(
                min_semitones=-2, 
                max_semitones=2, 
                p=1
            ),
            HighPassFilter(
                max_cutoff_freq=2500, 
                min_cutoff_freq=400, 
                p=1
            ),
            LowPassFilter(
                max_cutoff_freq=7000, 
                min_cutoff_freq=500, 
                p=1
            ),
            Gain(
                min_gain_in_db=-10, 
                max_gain_in_db=5, 
                p=1
            )
        ])


if __name__ == '__main__':
    
    signal, sr = librosa.load('/Users/inigoparra/Desktop/ESC-50-master/audio/1-5996-A-6.wav', sr=22050)
    augmentator = Augmentator()
    augmented_signal = augmentator.augmentate(signal, sample_rate=sr)
    sf.write('augmented.wav', augmented_signal, sr)
