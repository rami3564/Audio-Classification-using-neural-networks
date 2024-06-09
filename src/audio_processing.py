import numpy as np
import librosa

def mel_filter_banks(path):
    """
    Compute Mel filter banks for an audio file.

    Args:
        path (str): Path to the audio file.

    Returns:
        np.ndarray: Mel filter banks of the audio file.
    """
    # Load audio file
    y, sr = librosa.load(path, sr=16000)

    # Define frame parameters
    frame_size, frame_stride = 0.025, 0.01
    frame_length, frame_step = int(round(sr * frame_size)), int(round(sr * frame_stride))
    signal_length = 3 * sr  # Extract 0-3 seconds part
    frame_num = int(np.ceil((signal_length - frame_length) / frame_step)) + 1
    pad_frame = (frame_num - 1) * frame_step + frame_length - signal_length
    pad_y = np.append(y, np.zeros(pad_frame))
    signal_len = signal_length + pad_frame

    # Frame splitting
    indices = np.tile(np.arange(0, frame_length), (frame_num, 1)) + np.tile(np.arange(0, frame_num * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_y[indices]
    frames *= np.hamming(frame_length)

    # FFT and power spectra
    NFFT = 1024
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = mag_frames ** 2 / NFFT

    # Define Mel filter bank parameters
    mel_N = 128
    mel_low, mel_high = 0, (2595 * np.log10(1 + (sr / 2) / 700))
    mel_freq = np.linspace(mel_low, mel_high, mel_N + 2)
    hz_freq = (700 * (10 ** (mel_freq / 2595) - 1))
    bins = np.floor((NFFT + 1) * hz_freq / sr)

    # Construct Mel filter bank
    fbank = np.zeros((mel_N, int(NFFT / 2 + 1)))
    for m in range(1, mel_N + 1):
        f_m_minus = int(bins[m - 1])  # left
        f_m = int(bins[m])  # center
        f_m_plus = int(bins[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])

    # Apply Mel filter bank
    filter_banks = np.matmul(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Avoid log of zero
    filter_banks = 20 * np.log10(filter_banks)
    filter_banks = (filter_banks - np.mean(filter_banks)) / np.std(filter_banks)

    return filter_banks

def time_shift(audio, shift_limit):
    """
    Apply time shift to an audio signal.

    Args:
        audio (np.ndarray): The audio signal to shift.
        shift_limit (float): The maximum proportion of the audio length to shift.

    Returns:
        np.ndarray: The time-shifted audio signal.
    """
    shift_amt = int(np.random.uniform(-shift_limit, shift_limit) * len(audio))
    return np.roll(audio, shift_amt)

if __name__ == "__main__":
    # Example usage
    example_path = 'path/to/your/audio/file.mp3'
    filter_banks = mel_filter_banks(example_path)
    print("Mel Filter Banks:", filter_banks)

    # Example time shift
    y, sr = librosa.load(example_path, sr=16000)
    shifted_audio = time_shift(y, 0.1)
    print("Time-shifted Audio:", shifted_audio)
