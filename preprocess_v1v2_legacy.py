import sys
import os
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil

import matplotlib
# Important: Specify the backend before importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import librosa

try:
    import pyworld as pw
    _PYWORLD_AVAILABLE = True
except ImportError:
    _PYWORLD_AVAILABLE = False

from tools.cut_by_phrases import detect_regions, build_segments


def _interpolate_f0(f0: np.ndarray) -> np.ndarray:
    """Linearly interpolate F0 over unvoiced (zero) regions."""
    f0 = f0.copy()
    voiced = f0 > 0
    if not voiced.any():
        return f0
    indices = np.arange(len(f0))
    f0 = np.interp(indices, indices[voiced], f0[voiced])
    return f0

# --- Utility Functions ---

def load_config(path: str) -> dict:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --- Worker function for parallel NPZ creation ---

def _process_one_wav(wav_path_str: str, output_dir_str: str, cfg: dict) -> str:
    """1 ファイルを処理するワーカー関数（ProcessPoolExecutor から呼ばれる）。

    トップレベルに定義しないと pickle できないため、ここに置く。
    戻り値は "ok" / "skip" / "error:<メッセージ>" の文字列。
    """
    wav_path   = Path(wav_path_str)
    output_dir = Path(output_dir_str)
    save_path  = output_dir / f"{wav_path.stem}.npz"

    if save_path.exists():
        return "skip"

    y, sr = sf.read(wav_path)
    y = y * cfg['scale']
    if sr != cfg['sample_rate']:
        return f"error:sample_rate mismatch {sr} != {cfg['sample_rate']} ({wav_path.name})"

    frame_size = cfg['frame_size']
    y = y[:frame_size * (len(y) // frame_size)]

    f0_extractor = cfg.get('f0_extractor', 'harvest')

    if f0_extractor == 'rmvpe':
        import sys as _sys
        _project_root = cfg.get('_project_root', '.')
        if _project_root not in _sys.path:
            _sys.path.insert(0, _project_root)
        from tools.f0.algorithms.rmvpe import RMVPEPitchAlgorithm
        _abs_max = np.abs(y).max()
        y_norm = (y / _abs_max if _abs_max > 1e-6 else y).astype(np.float32)
        y_norm = np.clip(y_norm, -1.0, 1.0)
        algo = RMVPEPitchAlgorithm(
            sample_rate=sr,
            hop_size=cfg['hop_size'],
            fmin=cfg['f0_min'],
            fmax=cfg['f0_max'],
            device=cfg.get('rmvpe_device', 'cpu'),
        )
        f0_raw, voiced_flag, _ = algo.extract_pitch(y_norm)
        f0_raw[~voiced_flag] = 0.0  # unvoiced frames → 0 before interpolation
        f0_raw = _interpolate_f0(f0_raw)
    else:
        if not _PYWORLD_AVAILABLE:
            return "error:pyworld not installed. Use f0_extractor: rmvpe or install pyworld."
        frame_period = cfg['hop_size'] / cfg['sample_rate'] * 1000
        f0_raw, _ = pw.harvest(
            y.astype(np.float64), sr,
            f0_floor=cfg['f0_min'],
            f0_ceil=cfg['f0_max'],
            frame_period=frame_period,
        )

    S = librosa.feature.melspectrogram(
        y=y,
        sr=cfg['sample_rate'],
        n_fft=cfg['fft_size'],
        hop_length=cfg['hop_size'],
        win_length=cfg['hop_size'] * 4,
        n_mels=cfg['mel_dim'],
        fmin=cfg['mel_min'],
        fmax=cfg['mel_max'],
        center=True,
    )
    log_melspc = librosa.power_to_db(S, ref=1.0).T  # (T, D)

    hop_size = cfg['hop_size']
    n_frames = min(len(f0_raw), len(log_melspc))
    final_n_frames = min(n_frames * hop_size, len(y)) // hop_size

    np.savez(
        save_path,
        f0=f0_raw[:final_n_frames],
        log_melspc=log_melspc[:final_n_frames],
        wav=y[:final_n_frames * hop_size],
    )
    return "ok"


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
    """Split WAV files into phrase-based segments using the cut_by_phrases algorithm."""
    print(f"\n--- Step 2: Cutting WAVs by phrases ---")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    wav_paths = sorted(list(input_dir.rglob("*.wav")))

    if not wav_paths:
        print("Warning: No WAV files found in the input directory.")
        return

    cut_cfg = cfg.get('cut_wavs', {})
    silence_thresh_db = cut_cfg.get('silence_thresh', -50.0)
    min_silence_dur   = cut_cfg.get('min_silence_dur', 0.10)
    max_dur           = cut_cfg.get('max_dur', 9.0)
    long_silence      = cut_cfg.get('long_silence', 1.0)
    pad               = cut_cfg.get('pad', 0.10)

    print(f"  silence_thresh={silence_thresh_db} dBFS, min_silence_dur={min_silence_dur}s, "
          f"max_dur={max_dur}s, long_silence={long_silence}s, pad={pad}s")

    max_seg_len_s = 0.0

    for wav_path in tqdm(wav_paths, desc="Cutting"):
        audio, sr = sf.read(str(wav_path), always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        regions = detect_regions(audio, sr,
                                 silence_thresh_db=silence_thresh_db,
                                 min_silence_dur=min_silence_dur)

        segments = build_segments(regions, sr,
                                  max_dur=max_dur,
                                  long_silence=long_silence,
                                  pad=pad,
                                  total_samples=len(audio))

        for idx, (s, e) in enumerate(segments):
            chunk = audio[s:e]
            dur = (e - s) / sr
            if dur > max_seg_len_s:
                max_seg_len_s = dur
            save_path = output_dir / f"{wav_path.stem}_{idx:04d}.wav"
            sf.write(str(save_path), chunk, sr)

    if max_seg_len_s > 0:
        print(f"Max length of cut audio: {max_seg_len_s:.2f} seconds")

def step_create_npz(input_dir: Path, output_dir: Path, cfg: dict, num_workers: int):
    """Extract features from WAV files and save them in NPZ format."""
    print(f"\n--- Step 3: Creating NPZ files ---")
    print(f"Input dir  : {input_dir}")
    print(f"Output dir : {output_dir}")
    print(f"num_workers: {num_workers}")
    print(f"f0_min={cfg['f0_min']}, f0_max={cfg['f0_max']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    wav_paths = sorted(list(input_dir.glob("*.wav")))

    if not wav_paths:
        print("Warning: No WAV files found in the input directory.")
        return

    cfg = dict(cfg)
    cfg['_project_root'] = str(Path(__file__).resolve().parent)

    if cfg.get('f0_extractor') == 'rmvpe' and num_workers > 1:
        print(f"  Note: RMVPE loads a neural model per worker. Forcing num_workers=1.")
        num_workers = 1

    n_ok = n_skip = n_err = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_process_one_wav, str(p), str(output_dir), cfg): p
            for p in wav_paths
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Creating NPZ"):
            result = future.result()
            if result == "ok":
                n_ok += 1
            elif result == "skip":
                n_skip += 1
            else:
                n_err += 1
                print(f"\n  ERROR: {result}")

    print(f"完了: {n_ok} 処理, {n_skip} スキップ, {n_err} エラー")

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


def step_split_train_test(npz_dir: Path, train_dir: Path, test_dir: Path, cfg: dict):
    """Copy NPZ files from npz_dir into train/test directories based on the configured ratio.

    Files are sorted for reproducibility, then distributed so that every
    (train + test)-th file starting at index `train` goes to test.
    Example with train=100, test=1: indices 100, 201, 302, ... → test.
    """
    print(f"\n--- Step 6: Splitting NPZ into train/test ---")

    split_cfg = cfg.get('train_test_split', {})
    train_ratio = split_cfg.get('train', 100)
    test_ratio  = split_cfg.get('test', 1)
    total_ratio = train_ratio + test_ratio

    print(f"Input source: {npz_dir}")
    print(f"Train dir:    {train_dir}  (ratio: {train_ratio})")
    print(f"Test dir:     {test_dir}  (ratio: {test_ratio})")

    npz_paths = sorted(list(npz_dir.glob("*.npz")))
    if not npz_paths:
        print("Warning: No NPZ files found in the input directory.")
        return

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    n_train = n_test = 0
    for i, npz_path in enumerate(npz_paths):
        if i % total_ratio >= train_ratio:
            dst = test_dir / npz_path.name
            n_test += 1
        else:
            dst = train_dir / npz_path.name
            n_train += 1
        shutil.copy2(str(npz_path), str(dst))

    print(f"Split complete: train={n_train}, test={n_test}  (total={len(npz_paths)})")


# --- Main Processing ---
def main():
    parser = argparse.ArgumentParser(description="Audio dataset preprocessing script", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file")
    parser.add_argument("--step", type=str, default="all", choices=["all", "resample", "cut", "npz", "plot", "filter", "split"], help="Select the processing step to execute")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of parallel workers for the npz step")
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
    train_dir = Path(cfg['training']['train_dir'])
    test_dir  = Path(cfg['training']['test_dir'])

    # RMVPE availability check — fail fast before any processing starts
    if p_cfg.get('f0_extractor') == 'rmvpe' and args.step in ["all", "npz"]:
        project_root = str(Path(__file__).resolve().parent)
        import sys as _sys
        if project_root not in _sys.path:
            _sys.path.insert(0, project_root)
        try:
            from tools.f0.algorithms.rmvpe import RMVPEPitchAlgorithm  # noqa: F401
            print("✅ RMVPE: import OK")
        except Exception as e:
            print(f"❌ f0_extractor='rmvpe' が指定されていますが、RMVPEをインポートできません: {e}")
            print("   tools/f0/algorithms/rmvpe.py が存在するか確認してください。")
            return

    if args.step in ["all", "resample"]:
        step_resample_wavs(raw_wav_dir, resample_wav_dir, p_cfg['sample_rate'], p_cfg['prefix'])

    if args.step in ["all", "cut"]:
        step_cut_wavs(resample_wav_dir, cut_wav_dir, p_cfg)

    if args.step in ["all", "npz"]:
        step_create_npz(cut_wav_dir, npz_dir, p_cfg, args.num_workers)

    if args.step in ["all", "plot"]:
        step_plot_f0_validation(npz_dir, f0_img_dir, p_cfg)

    if args.step in ["all", "filter"]:
        step_filter_npz(npz_dir, f0_img_dir, p_cfg)

    if args.step in ["all", "split"]:
        step_split_train_test(npz_dir, train_dir, test_dir, p_cfg)

    print("All processing steps completed.")

if __name__ == "__main__":
    main()
