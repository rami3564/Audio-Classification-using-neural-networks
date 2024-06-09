import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

from data_loading import load_audio_files, AudioDataset
from models import get_resnet
from training import train_model

def main():
    # File paths
    input_path = 'C:/Users/ramis/OneDrive - nyu.edu/Desktop/New ML Project/train_output'
    num_files = 11886
    
    # Load and preprocess audio data
    train_file_paths = [f'{input_path}/{i}/vocals.wav' for i in range(num_files)]
    train_mel_spectrograms = load_audio_files(train_file_paths)
    train_labels = np.load('path/to/your/labels.npy')  # Load your labels from a file

    # Split data into training and validation sets
    train_mel_spectrograms, val_mel_spectrograms, train_labels, val_labels = train_test_split(
        train_mel_spectrograms, train_labels, test_size=0.2, random_state=42)
    
    # Create dataset instances for training and validation sets
    train_dataset = AudioDataset(train_mel_spectrograms, train_labels)
    val_dataset = AudioDataset(val_mel_spectrograms, val_labels)
    
    # Create data loaders for batching
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    print("Data loaders are set up and ready.")
    
    # Instantiate the model
    model = get_resnet(num_classes=4)
    
    # Train the model
    train_model(model, train_loader, val_loader, epochs=40, lr=0.0001)

if __name__ == "__main__":
    main()

