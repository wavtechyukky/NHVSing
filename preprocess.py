import os
import sys
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import shutil

import matplotlib
# Important: Specify the backend before importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import numpy as np
import soundfile as sf
import pyworld as pw
from pydub import AudioSegment
from pydub.silence import split_on_silence
from scipy.signal import resample_poly
import librosa

from dsp import frame_center_log_mel_spectrogram

# --- Utility Functions ---

@contextmanager
def suppress_stdout_stderr():
    """Temporarily suppress stdout and stderr, e.g., from C/C++ libraries."""
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            yield

def load_config(path: str) -> dict:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --- Dataset Creation Steps ---

def step_resample_wavs(input_dir: Path, output_dir: Path, sample_rate: int, prefix: str):
    """Resample WAV files, add a prefix, and save to a flat directory."""
    print(f"--- Step 1: Resampling and Adding Prefix ---")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    wav_paths = sorted(list(input_dir.rglob("*.wav")))
    
    if not wav_paths:
        print("Warning: No WAV files found in the input directory.")
        return

    for in_path in tqdm(wav_paths, desc="Resampling"):
        # Save with prefix in a flat structure
        out_path = output_dir / f"{prefix}_{in_path.stem}.wav"
        if out_path.exists():
            continue
        
        wav, sr = sf.read(in_path, always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)

        if sr != sample_rate:
            wav = resample_poly(wav, sample_rate, sr)

        sf.write(out_path, wav, sample_rate)

def step_cut_wavs(input_dir: Path, output_dir: Path, cfg: dict):
    """Split WAV files by silent sections."""
    print(f"\n--- Step 2: Cutting WAVs at silent sections ---")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    wav_paths = sorted(list(input_dir.rglob("*.wav")))

    if not wav_paths:
        print("Warning: No WAV files found in the input directory.")
        return

    max_chunk_len_ms = 0 # Variable to record the maximum length (in milliseconds)

    for wav_path in tqdm(wav_paths, desc="Cutting"):
        # Load with pydub
        sound = AudioSegment.from_file(wav_path)
        
        # Split on silence
        chunks = split_on_silence(
            sound,
            min_silence_len=cfg['min_silence_len'],
            silence_thresh=cfg['silence_thresh'],
            keep_silence=cfg['keep_silence']
        )
        
        # Save the split files
        for i, chunk in enumerate(chunks):
            # Update the max chunk length
            if len(chunk) > max_chunk_len_ms:
                max_chunk_len_ms = len(chunk)

            save_path = output_dir / f"{wav_path.stem}_{i:03d}.wav"
            chunk.export(save_path, format='wav')

    # Convert max length to seconds and print
    if max_chunk_len_ms > 0:
        max_chunk_len_s = max_chunk_len_ms / 1000
        print(f"Max length of cut audio: {max_chunk_len_s:.2f} seconds")

def step_create_npz(input_dir: Path, output_dir: Path, cfg: dict, spec_type='nhv'):
    """Extract features from WAV files and save them in NPZ format."""
    print(f"\n--- Step 3: Creating NPZ files ---")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    wav_paths = sorted(list(input_dir.glob("*.wav")))
    
    if not wav_paths:
        print("Warning: No WAV files found in the input directory.")
        return

    for wav_path in tqdm(wav_paths, desc="Creating NPZ"):
        # Determine output path and skip if it exists
        save_path = output_dir / f"{wav_path.stem}.npz"
        if save_path.exists():
            continue
        
        y, sr = sf.read(wav_path)
        y = y * cfg['scale']
        assert sr == cfg['sample_rate']
        
        frame_size = cfg['frame_size']
        y = y[:frame_size * (len(y) // frame_size)]

        # Flag to determine whether to align F0 frames with mel-spectrogram center frames
        should_align_f0 = cfg.get('align_f0_to_spec', True)
        y_for_f0 = y

        if should_align_f0:
            # To align F0 with the center of the mel-spectrogram frames,
            # shift the audio to the left by frame_size / 2 (i.e., cut the beginning)
            # and pad the end to maintain length.
            # This makes the 0th frame of harvest analyze the center of the original audio's frame_size position.
            shift_amount = frame_size // 2
            y_for_f0 = np.pad(y[shift_amount:], (0, shift_amount), 'constant')

        with suppress_stdout_stderr():
            # F0 estimation (Harvest)
            # To match the hop_size of F0 and mel-spectrogram, calculate frame_period based on hop_size
            f0_harvest, _ = pw.harvest(
                y_for_f0.astype(np.float64), sr, # Use the padded waveform
                f0_floor=cfg['f0_min'], f0_ceil=cfg['f0_max'], 
                frame_period=cfg['hop_size'] / sr * 1000
            )

        # Branch for mel-spectrogram calculation based on flag
        if spec_type == 'nhv':
            # log mel-spectrogram calculation
            wav_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
            log_melspc = frame_center_log_mel_spectrogram(
                wav_tensor, frame_size, frame_size * 4, 'hann',
                cfg['fft_size'], cfg['sample_rate'], cfg['mel_dim'], 
                cfg['mel_min'], cfg['mel_max'], cfg['min_level_db']
            ).squeeze(0)
            # Normalization process
            # Commented out because training somehow didn't go well
            #log_melspc = torch.clip((log_melspc - cfg['min_level_db']) / -cfg['min_level_db'], 0, 1)
        else:
            raise ValueError(f"Unknown spec_type: {spec_type}")


        '''
        elif spec_type == 'power':
            # Log power mel-spectrogram using librosa
            # Default minimum value is -80 (dB)
            # Training is possible with this in train.py, but other scripts
            # do not support this mel-spectrogram, so it's commented out.
            S = librosa.feature.melspectrogram(
                y=y,
                sr=cfg['sample_rate'],
                n_fft=cfg['fft_size'],
                hop_length=cfg['hop_size'],
                win_length=cfg['hop_size'] * 4, # Equivalent to NHV's window_size
                n_mels=cfg['mel_dim'],
                fmin=cfg['mel_min'],
                fmax=cfg['mel_max']
            )
            # Convert to dB scale with power_to_db
            log_melspc_np = librosa.power_to_db(S, ref=np.max)
            # Transpose to (Frames, Mels) shape and convert to Tensor
            log_melspc = torch.from_numpy(log_melspc_np.T).float()
        '''

        # Align the lengths of F0, mel-spectrogram, and waveform
        hop_size = cfg['hop_size']

        # 1. Get the minimum number of frames
        n_frames = min(len(f0_harvest), len(log_melspc))

        # 2. Calculate the corresponding number of audio samples for this frame count and compare with actual audio length
        expected_wav_len = n_frames * hop_size
        final_wav_len = min(expected_wav_len, len(y))

        # 3. Recalculate the number of frames from the final audio length to determine the final frame count
        final_n_frames = final_wav_len // hop_size

        # 4. Trim everything to the final length
        f0_trimmed = f0_harvest[:final_n_frames]
        log_melspc_trimmed = log_melspc[:final_n_frames]
        wav_trimmed = y[:final_n_frames * hop_size]

        # Convert Tensor to Numpy array
        if isinstance(log_melspc_trimmed, torch.Tensor):
            log_melspc_np = log_melspc_trimmed.numpy()
        else:
            log_melspc_np = log_melspc_trimmed

        # Save to NPZ file
        np.savez(
            save_path,
            f0=f0_trimmed,
            log_melspc=log_melspc_np,
            wav=wav_trimmed
        )

def step_plot_f0_validation(npz_dir: Path, img_dir: Path, cfg: dict):
    """Plots F0 and mel-spectrogram from NPZ files and saves them as images."""
    print(f"\n--- Step 4: Starting F0 Validation Plot Creation ---")
    print(f"Input source: {npz_dir}")
    print(f"Output destination: {img_dir}")

    img_dir.mkdir(parents=True, exist_ok=True)
    npz_paths = sorted(list(npz_dir.glob("*.npz")))

    if not npz_paths:
        print("Warning: No NPZ files found in the input directory.")
        return

    # Parameters for mel scale conversion
    mel_dim = cfg['mel_dim']
    mel_min_hz = cfg['mel_min']
    mel_max_hz = cfg['mel_max']
    mel_min = librosa.hz_to_mel(mel_min_hz)
    mel_max = librosa.hz_to_mel(mel_max_hz)

    for npz_path in tqdm(npz_paths, desc="Plotting"):
        # Determine output path and skip if it exists
        img_path = img_dir / f"{npz_path.stem}.png"
        if img_path.exists():
            continue

        data = np.load(npz_path)
        f0_hz = data['f0']
        log_melspc = data['log_melspc']

        # Convert F0 to mel bin index
        f0_hz_with_nan = np.copy(f0_hz).astype(float)
        f0_hz_with_nan[f0_hz_with_nan == 0] = np.nan
        f0_mel = librosa.hz_to_mel(f0_hz_with_nan)
        f0_mel_bins = (f0_mel - mel_min) * (mel_dim - 1) / (mel_max - mel_min)

        # Create plot
        fig, ax = plt.subplots(figsize=(15, 6))
        
        img = ax.imshow(log_melspc.T, origin='lower', aspect='auto', cmap='magma')
        fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Magnitude (dB)')

        ax.plot(
            np.arange(len(f0_mel_bins)), f0_mel_bins,
            color='cyan', linestyle='-', marker='.', markersize=2, linewidth=1,
            label='F0 (on Mel-bin scale)'
        )

        # Limit Y-axis to 1000Hz equivalent
        limit_hz = 1000.0
        limit_mel = librosa.hz_to_mel(limit_hz)
        limit_mel_bin = (limit_mel - mel_min) * (mel_dim - 1) / (mel_max - mel_min)
        ax.set_ylim(0, limit_mel_bin)

        ax.set_title(f'Log-Mel Spectrogram and F0: {npz_path.stem}')
        ax.set_xlabel('Frame Index')
        ax.set_ylabel(f'Mel Bin Index (0-{mel_dim-1})')
        ax.legend()
        
        # Save to file and release memory
        fig.savefig(img_path, bbox_inches='tight')
        plt.close(fig)

def step_filter_npz(npz_dir: Path, f0_img_dir: Path, cfg: dict):
    """Filters NPZ files based on frame length and moves corresponding F0 plots.
    If the extracted audio is too short, F0 is often not estimated correctly.
    Also, very long audio can cause out-of-memory issues during training.
    If memory allows, longer audio can be tolerated.
    """
    print(f"\n--- Step 5: Starting NPZ File Filtering ---")
    
    filter_cfg = cfg.get('data_filtering')
    if not filter_cfg:
        print("Warning: 'data_filtering' not found in config file, skipping.")
        return

    min_frames = filter_cfg.get('min_frames', 0)
    max_frames = filter_cfg.get('max_frames', float('inf'))
    backup_dir = Path(filter_cfg.get('backup_dir', 'dataset/npz_backup'))
    
    # Also set the backup destination for F0 plot images
    f0_img_backup_dir = backup_dir.with_name(backup_dir.name + '_f0_imgs')

    print(f"Input source (NPZ): {npz_dir}")
    print(f"Backup destination (NPZ): {backup_dir}")
    print(f"Backup destination (F0 Imgs): {f0_img_backup_dir}")
    print(f"Allowed frame length: {min_frames} - {max_frames}")

    backup_dir.mkdir(parents=True, exist_ok=True)
    f0_img_backup_dir.mkdir(parents=True, exist_ok=True)
    
    npz_paths = sorted(list(npz_dir.glob("*.npz")))

    if not npz_paths:
        print("Warning: No NPZ files found in the input directory.")
        return

    moved_count = 0
    for npz_path in tqdm(npz_paths, desc="Filtering NPZ"):
        try:
            with np.load(npz_path) as data:
                if 'log_melspc' in data:
                    num_frames = data['log_melspc'].shape[0]
                else:
                    print(f"Warning: 'log_melspc' not found in {npz_path.name}. Skipping.")
                    continue
            
            if not (min_frames <= num_frames <= max_frames):
                # Move NPZ file
                shutil.move(str(npz_path), str(backup_dir / npz_path.name))
                
                # Move corresponding F0 plot image
                img_path = f0_img_dir / f"{npz_path.stem}.png"
                if img_path.exists():
                    shutil.move(str(img_path), str(f0_img_backup_dir / img_path.name))
                
                moved_count += 1

        except Exception as e:
            print(f"Error processing {npz_path.name}: {e}")

    print(f"Filtering complete. {moved_count} files moved to backup destination.")

'''
def plot_compare_mel_spectrogram(wav_path: Path, cfg: dict, save_path: Path = None):
    """
    Extracts two types of mel-spectrograms (NHV and power) from a single WAV file and creates a comparison plot.
    """
    y, sr = sf.read(wav_path)
    y = y * cfg['scale']
    assert sr == cfg['sample_rate']
    frame_size = cfg['frame_size']
    y = y[:frame_size * (len(y) // frame_size)]

    # NHV method
    wav_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    log_melspc_nhv = frame_center_log_mel_spectrogram(
        wav_tensor, frame_size, frame_size * 4, 'hann',
        cfg['fft_size'], cfg['sample_rate'], cfg['mel_dim'], 
        cfg['mel_min'], cfg['mel_max'], -10.0
    ).squeeze(0).numpy()

    # Log power mel-spectrogram
    S = librosa.feature.melspectrogram(
        y=y,
        sr=cfg['sample_rate'],
        n_fft=cfg['fft_size'],
        hop_length=cfg['hop_size'],
        win_length=cfg['hop_size'] * 4,
        n_mels=cfg['mel_dim'],
        fmin=cfg['mel_min'],
        fmax=cfg['mel_max']
    )
    log_melspc_power = librosa.power_to_db(S, ref=np.max).T

    # Align lengths
    min_len = min(log_melspc_nhv.shape[0], log_melspc_power.shape[0])
    log_melspc_nhv = log_melspc_nhv[:min_len]
    log_melspc_power = log_melspc_power[:min_len]

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    axes[0].imshow(log_melspc_nhv.T, origin='lower', aspect='auto', cmap='magma')
    axes[0].set_title('NHV log-mel spectrogram')
    axes[0].set_ylabel('Mel Bin')

    axes[1].imshow(log_melspc_power.T, origin='lower', aspect='auto', cmap='magma')
    axes[1].set_title('FS2 log-mel spectrogram')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Mel Bin')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
'''

# --- Main Processing ---
def main():
    parser = argparse.ArgumentParser(description="Audio dataset preprocessing script", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file")
    parser.add_argument("--step", type=str, default="all", choices=["all", "resample", "cut", "npz", "plot", "filter"], help="Select the processing step to execute")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        p_cfg = cfg['preprocess']
    except (FileNotFoundError, KeyError) as e:
        print(f"Error: Failed to load configuration file '{args.config}'. Details: {e}")
        return

    raw_wav_dir = Path(p_cfg['raw_wav_dir'])
    resample_wav_dir = Path(p_cfg['resample_wav_dir'])
    cut_wav_dir = Path(p_cfg['cut_wav_dir'])
    npz_dir = Path(p_cfg['npz_dir'])
    f0_img_dir = Path(p_cfg['f0_img_dir'])
    
    if args.step in ["all", "resample"]:
        step_resample_wavs(raw_wav_dir, resample_wav_dir, p_cfg['sample_rate'], p_cfg['prefix'])
    
    if args.step in ["all", "cut"]:
        step_cut_wavs(resample_wav_dir, cut_wav_dir, p_cfg)

    if args.step in ["all", "npz"]:
        step_create_npz(cut_wav_dir, npz_dir, p_cfg)
    
    if args.step in ["all", "plot"]:
        step_plot_f0_validation(npz_dir, f0_img_dir, p_cfg)

    if args.step in ["all", "filter"]:
        step_filter_npz(npz_dir, f0_img_dir, p_cfg)
    
    print("All processing steps completed.")

if __name__ == "__main__":
    main()
