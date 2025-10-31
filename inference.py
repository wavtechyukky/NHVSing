import argparse
import time
import yaml
from pathlib import Path
import os

import torch
import numpy as np
import soundfile as sf
import onnxruntime as ort
import pyworld as pw

from dsp import frame_center_log_mel_spectrogram, generate_impulse_train, complex_cepstrum_to_imp, ltv_fir
from model import NHVSing, repeat_interpolate

# --- Utility Functions ---

def load_config(path: str) -> dict:
    """Loads a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_model(snapshot_path: str, config_path: str, device):
    """Loads the model from a snapshot and configuration, then JIT compiles it."""
    print(f"--- Loading and preparing the model ---")
    
    # 1. Load configuration file
    cfg = load_config(config_path)
    print(f"✅ Configuration file loaded: '{config_path}'")

    # 2. Instantiate model
    model = NHVSing(
        vocoder_cfg=cfg['model']['vocoder'],
        ltv_filter_cfg=cfg['model']['ltv_filter'],
    )
    
    # 3. Load snapshot
    snapshot = torch.load(snapshot_path, map_location=device)
    model.load_state_dict(snapshot['model'])
    model.remove_weight_norm()
    model.eval()
    model.to(device)
    print(f"✅ Snapshot loaded: '{snapshot_path}'")
    
    # 4. JIT compilation
    try:
        scripted_model = torch.jit.script(model)
        print("✅ JIT compilation successful.")
    except Exception as e:
        print(f"❌ JIT compilation failed: {e}")
        scripted_model = None
        
    return model, scripted_model, cfg

def export_onnx_core(model, save_path, dummy_x):
    """Exports the core part of the model (NHVConvsONNX) to ONNX format."""
    print(f"Exporting ONNX core model to {save_path}...")
    core_model = model.convs_onnx

    # Specify variable dimensions with `dynamic_axes`.
    # The input 'log_melspc' is (B, T, D), and the first dimension (time/frame) is made variable with the name 'time'.
    # The two outputs 'ccep_harm' and 'ccep_noise' are (B, T, ccep_size),
    # and the first dimension (time/frame) is made variable with the name 'time'.
    dynamic_axes = {
        'log_melspc': {0: 'batch_size', 1: 'time'},
        'ccep_harm': {0: 'batch_size', 1: 'time'},
        'ccep_noise': {0: 'batch_size', 1: 'time'}
    }

    torch.onnx.export(
        core_model,
        dummy_x,
        save_path,
        opset_version=11,
        input_names=['log_melspc'],
        # The model returns two tensors, so two output names are specified.
        output_names=['ccep_harm', 'ccep_noise'],
        dynamic_axes=dynamic_axes
    )
    print("Done.")

def extract_features_from_wav(wav_path: Path, cfg: dict):
    """Extracts F0 and log mel-spectrogram from a WAV file."""
    print(f"--- Extracting features from WAV ---")
    p_cfg = cfg['preprocess']
    
    y, sr = sf.read(wav_path)
    if y.ndim == 2:
        y = y.mean(axis=1) # Convert to mono
    assert sr == p_cfg['sample_rate'], f"Sample rate mismatch: {sr} vs {p_cfg['sample_rate']}"
    
    y = y.astype(np.float64)
    frame_size = p_cfg['frame_size']
    
    # F0 estimation (Harvest)
    f0, _ = pw.harvest(
        y, sr, 
        f0_floor=p_cfg['f0_min'], f0_ceil=p_cfg['f0_max'],
        frame_period=frame_size / sr * 1000
    )
    
    # Log mel-spectrogram calculation
    y_tensor = torch.from_numpy(y).float().unsqueeze(0)
    log_melspc = frame_center_log_mel_spectrogram(
        y_tensor, frame_size, frame_size * 4, 'hann',
        p_cfg['fft_size'], p_cfg['sample_rate'], p_cfg['mel_dim'],
        p_cfg['mel_min'], p_cfg['mel_max'], -10.0
    ).squeeze(0)
    
    # Align lengths
    min_len = min(len(f0), len(log_melspc))
    f0 = f0[:min_len]
    log_melspc = log_melspc[:min_len]
    
    print(f"✅ Feature extraction complete. Number of frames: {min_len}")
    return torch.from_numpy(f0).float(), log_melspc

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

def measure_performance(model_to_test, model_name: str, device, x, cf0, num_trials):
    """Measures inference speed and RTF for the specified model and device."""
    print(f"\n--- Measurement started: [{model_name}] on [{device.type.upper()}] ---")
    
    model_to_test.to(device)
    x, cf0 = x.to(device), cf0.to(device)
    
    # Warm-up
    with torch.no_grad():
        _ = model_to_test(x, cf0)
    
    # Time measurement
    start_time = time.perf_counter()
    for _ in range(num_trials):
        with torch.no_grad():
            output_waveform = model_to_test(x, cf0)
    end_time = time.perf_counter()
    
    avg_inference_time = (end_time - start_time) / num_trials
    
    return avg_inference_time, output_waveform

# --- Main Processing ---

def main():
    parser = argparse.ArgumentParser(description="Neural Vocoder Inference Script")
    parser.add_argument("input_path", type=str, help="Path to input file (.npz or .wav)")
    parser.add_argument("--snapshot", type=str, required=True, help="Path to model snapshot (.pth)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save generated audio files")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials for RTF measurement")
    args = parser.parse_args()

    devices_to_test = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices_to_test.append(torch.device("cuda"))
        print(f"Available devices: CPU, GPU")
    else:
        print(f"Available devices: CPU")

    # 1. Load model (initially to CPU)
    native_model, scripted_model, cfg = load_model(args.snapshot, args.config, torch.device("cpu"))
    
    # Temporary ONNX model export
    onnx_temp_path = Path("/tmp/core_model_temp.onnx")
    # Dummy input has shape (B, T, D), where D is mel_dim
    dummy_x_onnx = torch.randn(1, 100, cfg['preprocess']['mel_dim'], dtype=torch.float32)
    export_onnx_core(native_model, str(onnx_temp_path), dummy_x_onnx)

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

    # Convert to model input format
    x = log_melspc.unsqueeze(0)
    cf0 = f0.unsqueeze(0).unsqueeze(0)
    
    # 3. RTF comparison measurement
    all_results = []

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = cfg['preprocess']['sample_rate']

    for device in devices_to_test:
        # Native model
        avg_time, waveform = measure_performance(native_model, "Native Python", device, x, cf0, args.trials)
        all_results.append({'model': 'Native Python', 'device': device.type.upper(), 'time': avg_time})
        sf.write(output_dir / f"output_native_{device.type}.wav", waveform.squeeze().cpu().numpy(), sample_rate)

        # JIT model
        if scripted_model:
            avg_time, waveform = measure_performance(scripted_model, "JIT Script", device, x, cf0, args.trials)
            all_results.append({'model': 'JIT Script', 'device': device.type.upper(), 'time': avg_time})
            sf.write(output_dir / f"output_jit_{device.type}.wav", waveform.squeeze().cpu().numpy(), sample_rate)

        # ONNX + PyTorch (Hybrid) model
        print(f"\n--- Measurement started: [ONNX + PyTorch Hybrid] on [{device.type.upper()}] ---")
        hybrid_model = HybridONNXPyTorchModel(str(onnx_temp_path), cfg['model']['vocoder'], cfg['model']['ltv_filter'])
        hybrid_model.to(device)
        hybrid_model.eval()
        
        # Warm-up
        with torch.no_grad():
            _ = hybrid_model(x, cf0)
        
        # Time measurement
        start_time = time.perf_counter()
        for _ in range(args.trials):
            with torch.no_grad():
                output_waveform = hybrid_model(x, cf0)
        end_time = time.perf_counter()
        
        avg_inference_time = (end_time - start_time) / args.trials
        all_results.append({'model': 'ONNX + PyTorch Hybrid', 'device': device.type.upper(), 'time': avg_inference_time})
        sf.write(output_dir / f"output_onnx_pytorch_{device.type}.wav", output_waveform.squeeze().cpu().numpy(), sample_rate)
    
    # 4. Display results and save audio
    sample_rate = cfg['preprocess']['sample_rate']
    # Use output from any model to get waveform length
    # Assuming the length of Native Python model's CPU output here
    # In reality, the output length of each model should be the same
    dummy_waveform_path = output_dir / f"output_native_cpu.wav"
    if dummy_waveform_path.exists():
        dummy_waveform, _ = sf.read(dummy_waveform_path)
        num_samples = len(dummy_waveform)
    else:
        # If Native Python model was not executed on CPU (e.g., only GPU specified)
        # or if the file has not been written yet, infer appropriate length
        # Calculated here from x's number of frames and hop_size
        num_samples = x.shape[1] * cfg['preprocess']['frame_size']

    audio_duration = num_samples / sample_rate

    print("\n" + "="*65)
    print("  🚀 Overall Performance Comparison Results 🚀")
    print("="*65)
    print(f"  - Generated audio length: {audio_duration:.4f} seconds")
    print(f"  - Number of measurements (each model): {args.trials} times\n")
    print(f"  {'Model Type':<15} | {'Device':<8} | {'Avg. Inference Time':<22} | {'RTF':<15}")
    print(f"  {'-'*15} | {'-'*8} | {'-'*22} | {'-'*15}")

    for res in all_results:
        rtf = res['time'] / audio_duration
        print(f"  {res['model']:<15} | {res['device']:<8} | {res['time']:.6f} sec               | {rtf:.6f}")
    
    print("="*65)
    
    # Save as audio file
    print(f"\n✅ Generated audio files saved to '{output_dir}'.")

    # Delete temporary ONNX file
    if onnx_temp_path.exists():
        os.remove(onnx_temp_path)
        print(f"✅ Temporary ONNX file '{onnx_temp_path}' deleted.")

if __name__ == "__main__":
    main()