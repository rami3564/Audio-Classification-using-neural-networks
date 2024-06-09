import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from audio_processing import mel_filter_banks, time_shift

def load_audio_files(file_paths, augment=False):
    """
    Load audio files and extract Mel spectrograms, with optional augmentation.

    Args:
        file_paths (list of str): List of paths to audio files.
        augment (bool): Whether to apply time shift augmentation.

    Returns:
        list of np.ndarray: List of Mel spectrograms.
    """
    melspectrograms = []
    for file_path in file_paths:
        y, sr = librosa.load(file_path)
        if augment:
            y = time_shift(y, 0.05)
        melspectrogram = mel_filter_banks(file_path)
        melspectrograms.append(melspectrogram)
    return melspectrograms

class AudioDataset(Dataset):
    """
    Custom Dataset for loading audio data and corresponding labels.

    Args:
        audios (list of np.ndarray): List of audio data.
        labels (list of int): List of labels corresponding to the audio data.
    """
    def __init__(self, audios, labels):
        self.audios = audios
        self.labels = labels

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):
        audio = self.audios[idx]
        audio = np.expand_dims(audio, axis=0)
        audio = torch.tensor(audio, dtype=torch.float32)
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)
        return audio, label
