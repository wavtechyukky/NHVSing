import argparse
import time
import yaml
from pathlib import Path
import os

import torch
import numpy as np
import soundfile as sf
import librosa
import onnxruntime as ort

from dataset import norm_interp_f0
from model import NHVSing, NHVSingV2, repeat_interpolate


def _active_rms(y: np.ndarray, silence_thresh_db: float = -40.0) -> float:
    """RMS computed from active (non-silent) samples only.

    Samples whose amplitude is below silence_thresh_db are excluded so that
    long silent passages do not drag down the RMS and cause over-amplification.
    Falls back to full-signal RMS when fewer than 1 % of samples are active.
    """
    thresh = 10.0 ** (silence_thresh_db / 20.0)
    active = y[np.abs(y) > thresh]
    if len(active) < max(1, int(len(y) * 0.01)):
        return float(np.sqrt(np.mean(y ** 2)))
    return float(np.sqrt(np.mean(active ** 2)))


# --- Utility Functions ---

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(snapshot_path: str, config_path: str, device):
    """Loads the model from a snapshot and attempts JIT compilation."""
    print(f"--- Loading and preparing the model ---")

    cfg = load_config(config_path)
    print(f"✅ Configuration file loaded: '{config_path}'")

    ltv_filter_cfg = cfg['model']['ltv_filter']
    ModelClass = NHVSingV2 if ltv_filter_cfg.get('use_shared_trunk', False) else NHVSing
    model = ModelClass(
        vocoder_cfg=cfg['model']['vocoder'],
        ltv_filter_cfg=ltv_filter_cfg,
    )

    snapshot = torch.load(snapshot_path, map_location=device)
    if isinstance(snapshot, dict) and 'model' in snapshot:
        model.load_state_dict(snapshot['model'])
    else:
        model.load_state_dict(snapshot)
    model.remove_weight_norm()
    model.eval()
    model.to(device)
    print(f"✅ Snapshot loaded: '{snapshot_path}'")

    try:
        scripted_model = torch.jit.script(model)
        print("✅ JIT compilation successful.")
    except Exception as e:
        print(f"❌ JIT compilation failed: {e}")
        scripted_model = None

    return model, scripted_model, cfg


def extract_features_from_wav(wav_path: Path, cfg: dict):
    """Extracts F0 and log mel-spectrogram from a WAV file.

    F0 extractor is selected from config:
      harvest — pyworld harvest (V1 default)
      rmvpe   — neural F0 estimator (V2, auto-downloaded on first run)
    """
    print(f"--- Extracting features from WAV ---")
    p_cfg = cfg['preprocess']
    sr = p_cfg['sample_rate']
    f0_extractor = p_cfg.get('f0_extractor', 'harvest')

    y, read_sr = sf.read(wav_path)
    if y.ndim == 2:
        y = y.mean(axis=1)
    assert read_sr == sr, f"Sample rate mismatch: {read_sr} vs {sr}"

    # RMS normalization (required for models trained on M4Singer)
    target_rms = p_cfg.get('target_rms', None)
    if target_rms:
        rms = _active_rms(y.astype(np.float32))
        if rms > 1e-6:
            y = y * (target_rms / rms)
        y = np.clip(y, -1.0, 1.0)
        print(f"  RMS normalized to {target_rms} (active RMS was {rms:.4f})")

    # F0 extraction
    if f0_extractor == 'rmvpe':
        from tools.f0.algorithms.rmvpe import RMVPEPitchAlgorithm
        y_norm = y.astype(np.float32)
        abs_max = np.abs(y_norm).max()
        if abs_max > 1e-6:
            y_norm = y_norm / abs_max
        y_norm = np.clip(y_norm, -1.0, 1.0)
        algo = RMVPEPitchAlgorithm(
            sample_rate=sr,
            hop_size=p_cfg['hop_size'],
            fmin=p_cfg['f0_min'],
            fmax=p_cfg['f0_max'],
            device='cpu',
        )
        f0_raw, voiced_flag, *_ = algo.extract_pitch(y_norm)
        f0_raw[~voiced_flag] = 0.0  # unvoiced frames → 0 before interpolation
    else:
        import pyworld as pw
        frame_period = p_cfg['hop_size'] / sr * 1000
        f0_raw, _ = pw.harvest(
            y.astype(np.float64), sr,
            f0_floor=p_cfg['f0_min'],
            f0_ceil=p_cfg['f0_max'],
            frame_period=frame_period,
        )

    # Log mel-spectrogram
    S = librosa.feature.melspectrogram(
        y=y.astype(np.float32), sr=sr,
        n_fft=p_cfg['fft_size'], hop_length=p_cfg['hop_size'],
        win_length=p_cfg['hop_size'] * 4,
        n_mels=p_cfg['mel_dim'], fmin=p_cfg['mel_min'], fmax=p_cfg['mel_max'],
        center=True,
    )
    log_melspc = librosa.power_to_db(S, ref=1.0).T  # (T, D)

    # Align lengths
    min_len = min(len(f0_raw), len(log_melspc))
    f0_raw = f0_raw[:min_len]
    log_melspc = log_melspc[:min_len]

    print(f"✅ Feature extraction complete ({f0_extractor}). Frames: {min_len}")
    return torch.from_numpy(f0_raw).float(), torch.from_numpy(log_melspc).float()


# --- Unified ONNX Model ---

class UnifiedONNXModel:
    """Single unified ONNX model for vocoder inference.
    Load the .onnx file created by: python export.py --full_onnx
    """
    def __init__(self, model_path: str, config: dict):
        print(f"--- Initializing Unified ONNX Model ---")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path, sess_options, providers=['CPUExecutionProvider']
        )
        print(f"✅ Unified ONNX model loaded: {model_path}")

        self.hop_size = config['model']['vocoder']['hop_size']
        self.noise_std = config['model']['vocoder']['noise_std']

    def run(self, log_melspc: np.ndarray, f0: np.ndarray) -> np.ndarray:
        """
        log_melspc: (T, D) float32
        f0: (T,) float32 — should be interpolated (no zeros)
        Returns: (T * hop_size,) float32 waveform
        """
        x   = log_melspc[np.newaxis].astype(np.float32)          # (1, T, D)
        cf0 = f0[np.newaxis, np.newaxis].astype(np.float32)       # (1, 1, T)
        z   = np.random.normal(
            0.0, self.noise_std, (1, 1, x.shape[1] * self.hop_size)
        ).astype(np.float32)

        result = self.session.run(None, {'log_melspc': x, 'f0': cf0, 'z': z})
        return result[0].flatten()


# --- Performance Measurement ---

def measure_performance(model_to_test, model_name: str, device, x, cf0, num_trials: int):
    """Measures average inference time for a PyTorch model."""
    print(f"\n--- Measurement started: [{model_name}] on [{device.type.upper()}] ---")

    model_to_test.to(device)
    x_d, cf0_d = x.to(device), cf0.to(device)

    # Warm-up
    with torch.no_grad():
        _ = model_to_test(x_d, cf0_d)

    start_time = time.perf_counter()
    for _ in range(num_trials):
        with torch.no_grad():
            output_waveform = model_to_test(x_d, cf0_d)
    avg_time = (time.perf_counter() - start_time) / num_trials

    return avg_time, output_waveform


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Neural Vocoder Inference Script")
    parser.add_argument("input_path", type=str, help="Path to input file (.npz or .wav)")
    parser.add_argument("--snapshot", type=str, required=True, help="Path to model snapshot (.pth)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save generated audio")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials for RTF measurement")
    parser.add_argument("--onnx", type=str, default=None, help="Path to unified ONNX model (.onnx)")
    args = parser.parse_args()

    devices_to_test = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices_to_test.append(torch.device("cuda"))
        print(f"Available devices: CPU, GPU")
    else:
        print(f"Available devices: CPU")

    # 1. Load PyTorch model
    native_model, scripted_model, cfg = load_model(
        args.snapshot, args.config, torch.device("cpu")
    )

    # 2. Prepare input data
    input_path = Path(args.input_path)
    if input_path.suffix == '.npz':
        print(f"\n--- Loading features from NPZ file ---")
        data = np.load(input_path)
        f0 = torch.from_numpy(data['f0']).float()
        log_melspc = torch.from_numpy(data['log_melspc']).float()
        print(f"✅ Loading complete. Number of frames: {len(f0)}")
    elif input_path.suffix == '.wav':
        f0, log_melspc = extract_features_from_wav(input_path, cfg)
    else:
        raise ValueError("Unsupported input file format. Please specify .npz or .wav.")

    # Apply F0 interpolation (same as training dataset)
    f0, _ = norm_interp_f0(f0)

    x   = log_melspc.unsqueeze(0)       # (1, T, D)
    cf0 = f0.unsqueeze(0).unsqueeze(0)  # (1, 1, T)

    # 3. RTF measurement
    all_results = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = cfg['preprocess']['sample_rate']

    for device in devices_to_test:
        # Native model
        avg_time, waveform = measure_performance(
            native_model, "Native Python", device, x, cf0, args.trials
        )
        all_results.append({'model': 'Native Python', 'device': device.type.upper(), 'time': avg_time})
        sf.write(output_dir / f"output_native_{device.type}.wav",
                 waveform.squeeze().cpu().numpy(), sample_rate)

        # JIT model
        if scripted_model:
            avg_time, waveform = measure_performance(
                scripted_model, "JIT Script", device, x, cf0, args.trials
            )
            all_results.append({'model': 'JIT Script', 'device': device.type.upper(), 'time': avg_time})
            sf.write(output_dir / f"output_jit_{device.type}.wav",
                     waveform.squeeze().cpu().numpy(), sample_rate)

    # Unified ONNX (CPU only)
    if args.onnx:
        onnx_model = UnifiedONNXModel(args.onnx, cfg)
        f0_np = f0.numpy().astype(np.float32)
        mel_np = log_melspc.numpy().astype(np.float32)

        print(f"\n--- Measurement started: [Unified ONNX] on [CPU] ---")
        _ = onnx_model.run(mel_np, f0_np)  # warm-up

        start_time = time.perf_counter()
        for _ in range(args.trials):
            waveform_np = onnx_model.run(mel_np, f0_np)
        avg_time = (time.perf_counter() - start_time) / args.trials

        all_results.append({'model': 'Unified ONNX', 'device': 'CPU', 'time': avg_time})
        sf.write(output_dir / "output_onnx.wav", waveform_np, sample_rate)

    # 4. Display results
    ref_path = output_dir / "output_native_cpu.wav"
    if ref_path.exists():
        ref_wav, _ = sf.read(ref_path)
        audio_duration = len(ref_wav) / sample_rate
    else:
        audio_duration = x.shape[1] * cfg['preprocess']['hop_size'] / sample_rate

    print("\n" + "=" * 65)
    print("  Performance Comparison Results")
    print("=" * 65)
    print(f"  Audio duration : {audio_duration:.4f} s")
    print(f"  Trials / model : {args.trials}\n")
    print(f"  {'Model':<22} | {'Device':<6} | {'Avg Time':<14} | RTF")
    print(f"  {'-'*22} | {'-'*6} | {'-'*14} | {'-'*10}")
    for res in all_results:
        rtf = res['time'] / audio_duration
        print(f"  {res['model']:<22} | {res['device']:<6} | {res['time']:.6f} sec   | {rtf:.6f}")
    print("=" * 65)
    print(f"\n✅ Generated audio files saved to '{output_dir}'.")


if __name__ == "__main__":
    main()
