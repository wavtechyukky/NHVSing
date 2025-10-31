import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np

# --- データセットクラス ---

class VocoderDataset(Dataset):
    """Dataset class that recursively searches for npz files in a given directory and returns the data.

    Args:
        dataset_dir (str or Path): Root directory containing the .npz files.
    """
    def __init__(self, dataset_dir: str):
        self.dataset_path = Path(dataset_dir)
        # .rglob("*.npz") を使ってサブディレクトリ内も再帰的に検索
        self.file_paths = sorted(list(self.dataset_path.rglob("*.npz")))
        
        if not self.file_paths:
            print(f"警告: '{dataset_dir}' 内に .npz ファイルが見つかりませんでした。")
        else:
            print(f"Found {len(self.file_paths)} files in {dataset_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        npz_path = self.file_paths[idx]
        npz = np.load(npz_path)

        f0 = npz['f0']
        log_melspc = npz['log_melspc']
        wav = npz['wav']

        # フレーム数 × hop_size に長さを厳密に合わせる
        hop_size = 256
        n_frames = log_melspc.shape[0]
        expected_wav_len = n_frames * hop_size
        
        # wavが長い場合は切り捨て、短い場合はゼロパディング
        if len(wav) > expected_wav_len:
            wav = wav[:expected_wav_len]
        elif len(wav) < expected_wav_len:
            wav = np.pad(wav, (0, expected_wav_len - len(wav)))

        f0, uv = norm_interp_f0(f0)
        
        return f0[np.newaxis].astype('float32'), log_melspc, wav.astype('float32'), uv[np.newaxis]
    

# https://github.com/MoonInTheRiver/DiffSinger/blob/master/utils/pitch_utils.py
def norm_interp_f0(f0):
    """Interpolate F0.

    This function interpolates the fundamental frequency (F0) contour.
    Unvoiced frames (where F0 is 0) are linearly interpolated from voiced frames.

    Args:
        f0 (np.ndarray or torch.Tensor): Input F0 contour.

    Returns:
        tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
            - The interpolated F0 contour.
            - The unvoiced/voiced (UV) flag (1 for unvoiced, 0 for voiced).
    """
    is_torch = isinstance(f0, torch.Tensor)
    if is_torch:
        device = f0.device
        f0 = f0.data.cpu().numpy()
    uv = f0 == 0

    if sum(uv) == len(f0):
        f0[uv] = 0
    elif sum(uv) > 0:
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])

    uv = 1 * uv
    uv = np.array(uv)

    if is_torch:
        uv = torch.FloatTensor(uv)
        f0 = torch.FloatTensor(f0)
        f0 = f0.to(device)
    return f0, uv


# https://github.com/xushengyuan/FastSing2
def collate_fn_padd(batch):
    """Pads data in a batch to the maximum length.

    This function is used as a collate_fn for a DataLoader. It takes a list of
    tuples (representing the dataset items) and pads each element to the maximum
    length in the batch.

    Args:
        batch (list[tuple]): A list of data samples, where each sample is a tuple
            containing f0, log_melspc, wav, and uv.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
            - Padded F0 contours.
            - Padded log-mel spectrograms.
            - Padded waveforms.
            - Padded UV flags.
            - A mask tensor indicating the padded regions.
    """
    fine_f0s = []
    log_melspcs = []
    wavs = []
    uvs = []

    wav_lengths = []

    for t in batch:
        fine_f0s.append(t[0][0])
        log_melspcs.append(t[1])
        wavs.append(t[2])
        uvs.append(t[3][0])

        wav_lengths.append(len(t[2]))

    fine_f0s = pad_1D(fine_f0s)
    log_melspcs = pad_2D(log_melspcs)
    wavs = pad_1D(wavs)
    uvs = pad_1D(uvs)

    fine_f0s = np.array(fine_f0s)[:, np.newaxis, :] # (batch,1,frame)
    log_melspcs = np.array(log_melspcs) # (batch,frame,80)
    wavs = np.array(wavs) # (batch,n_sample)
    uvs = np.array(uvs)[:, np.newaxis, :]

    mask = get_mask_from_lengths(wav_lengths)

    return fine_f0s, log_melspcs, wavs, uvs, mask


def pad_1D(inputs):
    """Pads a list of 1D arrays to the same length.

    Args:
        inputs (list[np.ndarray]): List of 1D numpy arrays.

    Returns:
        list[np.ndarray]: List of padded 1D numpy arrays.
    """
    max_len = 0
    for input in inputs:
        if max_len < len(input):
            max_len = len(input)
    
    for i in range(len(inputs)):
        inputs[i] = np.pad(inputs[i], (0, max_len-inputs[i].shape[0]))

    return inputs


def pad_2D(inputs):
    """Pads a list of 2D arrays to the same length in the first dimension.

    Args:
        inputs (list[np.ndarray]): List of 2D numpy arrays (frame, dim).

    Returns:
        list[np.ndarray]: List of padded 2D numpy arrays.
    """
    max_len = 0
    for input in inputs:
        if max_len < len(input):
            max_len = len(input)
    
    for i in range(len(inputs)):
        inputs[i] = np.pad(inputs[i], ((0,max_len-inputs[i].shape[0]),(0,0)))

    return inputs


def get_mask_from_lengths(lengths):
    """Create a boolean mask from a tensor of sequence lengths.

    Args:
        lengths (torch.Tensor): A 1D tensor containing the length of each sequence
            in a batch.

    Returns:
        torch.Tensor: A boolean mask of shape (batch_size, max_len)
            where `True` indicates a padded position.
    """
    lengths = torch.Tensor(lengths)
    batch_size = lengths.shape[0]
    max_len = int(torch.max(lengths).item())
    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1)
    mask = (ids >= lengths.unsqueeze(1).expand(-1, max_len))
    return mask
