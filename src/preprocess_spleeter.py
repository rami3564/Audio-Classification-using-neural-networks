import os
from spleeter.separator import Separator

def separate_audio(input_path, output_path, num_files):
    """
    Separates audio files into stems using Spleeter.

    Args:
        input_path (str): Path to the directory containing input audio files.
        output_path (str): Path to the directory where output files will be saved.
        num_files (int): Number of audio files to process.
    """
    separator = Separator('spleeter:2stems')
    train_paths = [os.path.join(input_path, f"{i}.mp3") for i in range(num_files)]

    for train_path in train_paths:
        separator.separate_to_file(train_path, output_path)
        print(f'Separated {train_path}')

if __name__ == "__main__":
    # File paths
    input_path = 'C:/Users/ramis/OneDrive - nyu.edu/Desktop/New ML Project/train_mp3s'
    output_path = 'C:/Users/ramis/OneDrive - nyu.edu/Desktop/New ML Project/train_output'
    
    # Number of files to process
    num_files = 11886
    
    # Separate audio files
    separate_audio(input_path, output_path, num_files)
