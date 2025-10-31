
import os
import argparse
import yaml
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
import pyworld as pw
import onnxruntime as ort
from contextlib import contextmanager, redirect_stdout, redirect_stderr

# --- Project Imports ---
# Add project root to sys.path to allow direct imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsp import frame_center_log_mel_spectrogram
from model import NHVSing
from inference_onnx import NHVSing_with_ONNX
from model import repeat_interpolate # For hybrid model
from dsp import generate_impulse_train, complex_cepstrum_to_imp, ltv_fir # For hybrid model

# --- Utilities ---

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            yield

def load_config(path: str) -> dict:
    """Loads a YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# --- F0 Processing ---

def norm_interp_f0(f0):
    """
    Linearly interpolates F0 and returns a uv flag.
    (Copied from dataset.py)
    """
    is_torch = isinstance(f0, torch.Tensor)
    if is_torch:
        device = f0.device
        f0 = f0.data.cpu().numpy()
    
    uv = f0 == 0
    if sum(uv) == len(f0):
        # All unvoiced frames, do nothing.
        pass
    elif sum(uv) > 0:
        # Interpolate unvoiced frames.
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])

    # The model itself doesn't use the uv flag, but we return f0.
    # The original function returns uv as well.
    uv = 1 * uv
    uv = np.array(uv)

    if is_torch:
        uv = torch.FloatTensor(uv)
        f0 = torch.FloatTensor(f0)
        f0 = f0.to(device)
    return f0, uv


# --- WAV Preprocessing ---

class WavPreprocessor:
    """
    Extracts features (F0, Mel Spectrogram) from a WAV file based on preprocess.py logic.
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg['preprocess']
        self.p_cfg = cfg # full config for model params

    def process(self, wav_path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Processes a single WAV file.
        Returns:
            f0 (np.ndarray): F0 contour.
            log_melspc (np.ndarray): Log-Mel spectrogram.
        """
        y, sr = sf.read(wav_path)
        if y.ndim > 1:
            y = y.mean(axis=1) # convert to mono
        y = y.astype(np.float32)
        
        if 'scale' in self.cfg:
            y = y * self.cfg['scale']
        
        if sr != self.cfg['sample_rate']:
            raise ValueError(f"Sample rate mismatch: expected {self.cfg['sample_rate']}, got {sr}")

        frame_size = self.cfg['frame_size']
        y = y[:frame_size * (len(y) // frame_size)]

        # Align F0 to mel-spectrogram frames
        shift_amount = frame_size // 2
        y_for_f0 = np.pad(y[shift_amount:], (0, shift_amount), 'constant')

        with suppress_stdout_stderr():
            f0_harvest, _ = pw.harvest(
                y_for_f0.astype(np.float64), sr,
                f0_floor=self.cfg['f0_min'], f0_ceil=self.cfg['f0_max'],
                frame_period=self.cfg['hop_size'] / sr * 1000
            )

        # Calculate log-mel spectrogram
        wav_tensor = torch.from_numpy(y).float().unsqueeze(0)
        log_melspc = frame_center_log_mel_spectrogram(
            wav_tensor, frame_size, frame_size * 4, 'hann',
            self.cfg['fft_size'], self.cfg['sample_rate'], self.cfg['mel_dim'],
            self.cfg['mel_min'], self.cfg['mel_max'], self.cfg['min_level_db']
        ).squeeze(0)
        
        # Normalize
        #log_melspc = torch.clip((log_melspc - self.cfg['min_level_db']) / -self.cfg['min_level_db'], 0, 1)

        # Align lengths
        n_frames = min(len(f0_harvest), len(log_melspc))
        f0_trimmed = f0_harvest[:n_frames]
        
        # Interpolate F0
        f0_interpolated, _ = norm_interp_f0(f0_trimmed)

        # Cast f0 to float32 to match the model's expected dtype (torch.float)
        f0_final = f0_interpolated.astype(np.float32)
        log_melspc_trimmed = log_melspc[:n_frames].numpy()

        return f0_final, log_melspc_trimmed

# --- Hybrid ONNX + PyTorch Model ---

class HybridONNXPyTorchModel(torch.nn.Module):
    """
    Uses ONNX for convolutions and PyTorch for DSP post-processing.
    """
    def __init__(self, onnx_path: str, vocoder_cfg: dict, ltv_filter_cfg: dict):
        super().__init__()
        # Store DSP parameters
        self.fs = vocoder_cfg['sample_rate']
        self.hop_size = vocoder_cfg['hop_size']
        self.fft_size = ltv_filter_cfg['fft_size']
        self.noise_std = vocoder_cfg['noise_std']
        
        # Load ONNX session
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def forward(self, x, cf0):
        """
        x: (B, T, D) - Mel-cepstrum tensor
        cf0: (B, 1, T) - Continuous F0 tensor
        """
        # Generate noise
        z_shape = (x.size(0), 1, x.size(1) * self.hop_size)
        z = torch.normal(0.0, self.noise_std, z_shape).to(x.device)

        # Run ONNX convolutions
        # Transpose input from (B, T, D) to (B, D, T) for ONNX Conv1d
        #x_onnx = x.transpose(1, 2).cpu().numpy()
        x_onnx = x.cpu().numpy()
        ccep_harm_np, ccep_noise_np = self.session.run(
            self.output_names, {self.input_name: x_onnx}
        )
        ccep_harm = torch.from_numpy(ccep_harm_np).to(x.device)
        ccep_noise = torch.from_numpy(ccep_noise_np).to(x.device)

        # --- PyTorch DSP Processing (copied from NHVSing) ---
        cf0_resampled = repeat_interpolate(cf0, self.hop_size)
        harmonic_source = generate_impulse_train(cf0_resampled, 200, float(self.fs))

        imp_harm = complex_cepstrum_to_imp(ccep_harm, self.fft_size)
        sig_harm = ltv_fir(harmonic_source, imp_harm, self.hop_size)

        imp_noise = complex_cepstrum_to_imp(ccep_noise, self.fft_size)
        sig_noise = ltv_fir(z, imp_noise, self.hop_size)
        
        y = sig_harm + sig_noise
        y = torch.clamp(y, -1, 1)
        
        return y.reshape(x.size(0), -1)

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Inference checker for exported models.")
    parser.add_argument("input_path", type=str, help="Path to the input WAV file.")
    parser.add_argument("save_folder_path", type=str, help="Path to the folder to save output audio.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file.")
    parser.add_argument("--pth_path", type=str, help="Path to the .pth model file.")
    parser.add_argument("--pt_path", type=str, help="Path to the .pt JIT model file.")
    parser.add_argument("--onnx_path", type=str, help="Path to the .onnx model file.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (e.g., 'cpu', 'cuda').")
    args = parser.parse_args()

    # --- Setup ---
    cfg = load_config(args.config)
    save_dir = Path(args.save_folder_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # --- Preprocessing ---
    print(f"Processing input file: {args.input_path}")
    preprocessor = WavPreprocessor(cfg)
    f0, mel = preprocessor.process(args.input_path)
    
    # Prepare inputs for models, explicitly setting dtype to float32
    mel_tensor = torch.from_numpy(mel).to(torch.float32).unsqueeze(0).to(device)
    f0_tensor = torch.from_numpy(f0).to(torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    mel_np = np.expand_dims(mel, 0)
    f0_np = np.expand_dims(np.expand_dims(f0, 0), 0)

    # --- Inference ---

    # 1. PTH model
    if args.pth_path:
        print("\n--- Running PTH Inference ---")
        model = NHVSing(cfg['model']['vocoder'], cfg['model']['ltv_filter'])
        model.load_state_dict(torch.load(args.pth_path, map_location='cpu'))
        model.remove_weight_norm()
        model.to(device)
        model.eval()
        with torch.no_grad():
            output_wav = model(mel_tensor, f0_tensor).squeeze().cpu().numpy()
        save_path = save_dir / "output_pth.wav"
        sf.write(save_path, output_wav, cfg['preprocess']['sample_rate'])
        print(f"Saved PTH output to {save_path}")

    # 2. PT (JIT) model
    if args.pt_path:
        print("\n--- Running JIT (.pt) Inference ---")
        model = torch.jit.load(args.pt_path, map_location=device)
        model.eval()
        with torch.no_grad():
            output_wav = model(mel_tensor, f0_tensor).squeeze().cpu().numpy()
        save_path = save_dir / "output_pt.wav"
        sf.write(save_path, output_wav, cfg['preprocess']['sample_rate'])
        print(f"Saved JIT output to {save_path}")

    # 3. ONNX models
    if args.onnx_path:
        # 3a. ONNX + PyTorch (Hybrid) - RUNNING THIS FIRST AS REQUESTED
        print("\n--- Running ONNX + PyTorch (Hybrid) Inference ---")
        hybrid_model = HybridONNXPyTorchModel(args.onnx_path, cfg['model']['vocoder'], cfg['model']['ltv_filter'])
        hybrid_model.to(device)
        hybrid_model.eval()
        with torch.no_grad():
            output_wav = hybrid_model(mel_tensor, f0_tensor).squeeze().cpu().numpy()
        save_path = save_dir / "output_onnx_pytorch.wav"
        sf.write(save_path, output_wav, cfg['preprocess']['sample_rate'])
        print(f"Saved ONNX+PyTorch output to {save_path}")

        # 3b. ONNX + NumPy
        print("\n--- Running ONNX + NumPy Inference ---")
        onnx_numpy_model = NHVSing_with_ONNX(args.onnx_path, cfg['model']['vocoder'], cfg['model']['ltv_filter'])
        noise_std = cfg['model']['vocoder']['noise_std']
        z_shape = (1, 1, mel_np.shape[1] * cfg['preprocess']['hop_size'])
        z = np.random.normal(0.0, noise_std, z_shape).astype(np.float32)
        # The ONNX model expects (B, T, D) input, so no transpose is needed.
        # mel_np is already in the correct shape (1, 1111, 128)
        output_wav = onnx_numpy_model.inference(z, mel_np, f0_np).flatten()
        save_path = save_dir / "output_onnx_numpy.wav"
        sf.write(save_path, output_wav, cfg['preprocess']['sample_rate'])
        print(f"Saved ONNX+NumPy output to {save_path}")

    print("\nAll inferences complete.")

if __name__ == "__main__":
    main()
