import time
import yaml
from pathlib import Path
import numpy as np
import torch
import argparse
import onnxruntime as ort
import sys

# --- モデルのインポート ---
# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import model
from dsp import generate_impulse_train, complex_cepstrum_to_imp, ltv_fir
from model import repeat_interpolate
from inference_onnx import NHVSing_with_ONNX as NHVSing_ONNX_NumPy_Backend

# --- 設定とパス ---
CONFIG_PATH = "config.yaml"
ONNX_MODEL_PATH = "exported_models/core_model.onnx"
JIT_MODEL_PATH = "exported_models/model_jit.pt"
PYTORCH_MODEL_PATH = "exported_models/model.pth"


# --- モデルラッパークラスの定義 ---

class NHVSingPyTorch:
    """標準的なPyTorchモデルのラッパー"""
    def __init__(self, vocoder_cfg, ltv_filter_cfg, model_path):
        self.model = model.NHVSing(vocoder_cfg, ltv_filter_cfg)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        self.noise_std = vocoder_cfg['noise_std']
        self.hop_size = vocoder_cfg['hop_size']

    def to(self, device):
        self.model.to(device)
        return self

    def __call__(self, x, cf0):
        return self.model(x, cf0)

class NHVSingJIT:
    """JITコンパイル済みモデルのラッパー"""
    def __init__(self, vocoder_cfg, ltv_filter_cfg, model_path, device='cpu'):
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        self.device = device
        # JITモデルはノイズ生成を内包していると仮定
    
    def to(self, device):
        # JITモデルはロード時にデバイスが決定される
        if self.device != device:
            print(f"Warning: JIT model was loaded on {self.device}, requested {device}.")
        return self

    def __call__(self, x, cf0):
        return self.model(x, cf0)

class NHVSing_ONNX_PyTorchDSP:
    """ONNX + PyTorch DSPモデルのラッパー"""
    def __init__(self, vocoder_cfg, ltv_filter_cfg, onnx_path, device='cpu'):
        self.device = device
        self.fs = vocoder_cfg['sample_rate']
        self.hop_size = vocoder_cfg['hop_size']
        self.fft_size = ltv_filter_cfg['fft_size']
        self.noise_std = vocoder_cfg['noise_std']
        
        providers = ['CPUExecutionProvider']
        if device == 'cuda':
            providers.insert(0, 'CUDAExecutionProvider')
        self.ort_session = ort.InferenceSession(onnx_path, providers=providers)

    def to(self, device):
        # ONNX Runtimeセッションのデバイスは初期化時に決定
        if self.device != device:
             print(f"Warning: ONNX (PyTorch DSP) model was loaded on {self.device}, requested {device}.")
        return self

    def __call__(self, x, cf0):
        z_shape = (x.size(0), 1, x.size(1) * self.hop_size)
        z = torch.normal(0, self.noise_std, z_shape).to(self.device)

        ort_inputs = {self.ort_session.get_inputs()[0].name: x.cpu().numpy()}
        ccep_harm_np, ccep_noise_np = self.ort_session.run(None, ort_inputs)
        
        ccep_harm = torch.from_numpy(ccep_harm_np).to(self.device)
        ccep_noise = torch.from_numpy(ccep_noise_np).to(self.device)

        cf0_resampled = repeat_interpolate(cf0, self.hop_size)
        harmonic_source = generate_impulse_train(cf0_resampled, 200, float(self.fs))

        imp_harm = complex_cepstrum_to_imp(ccep_harm, self.fft_size)
        sig_harm = ltv_fir(harmonic_source, imp_harm, self.hop_size)

        imp_noise = complex_cepstrum_to_imp(ccep_noise, self.fft_size)
        sig_noise = ltv_fir(z, imp_noise, self.hop_size)
        
        y = sig_harm + sig_noise
        return torch.clamp(y, -1, 1).reshape(x.size(0), -1)

class NHVSing_ONNX_NumPyDSP:
    """ONNX + NumPy DSPモデルのラッパー"""
    def __init__(self, vocoder_cfg, ltv_filter_cfg, onnx_path, device='cpu'):
        self.model = NHVSing_ONNX_NumPy_Backend(onnx_path, vocoder_cfg, ltv_filter_cfg)
        self.noise_std = vocoder_cfg['noise_std']
        self.hop_size = vocoder_cfg['hop_size']
        # NumPyバックエンドはCPUでのみ動作
        self.device = 'cpu'

    def to(self, device):
        if device != 'cpu':
            print("Warning: ONNX+NumPy backend only runs on CPU.")
        return self

    def __call__(self, x, cf0):
        z_shape = (x.size(0), 1, x.size(1) * self.hop_size)
        z = torch.normal(0, self.noise_std, z_shape)

        x_np = x.cpu().numpy()
        cf0_np = cf0.cpu().numpy()
        z_np = z.cpu().numpy()

        y_np = self.model.inference(z_np, x_np, cf0_np)
        
        return torch.from_numpy(y_np).to(x.device)

# --- ベンチマーク実行関数 ---

def benchmark(model, model_name, npz_data, sample_rate, device):
    print(f"--- Benchmarking: {model_name} on {device.upper()} ---")
    model.to(device)
    
    log_melspc = torch.from_numpy(npz_data['log_melspc']).unsqueeze(0).float().to(device)
    f0 = torch.from_numpy(npz_data['f0']).unsqueeze(0).unsqueeze(0).float().to(device)

    # ウォームアップ
    print("Running warmup inference...")
    with torch.no_grad():
        _ = model(log_melspc, f0)

    # ベンチマーク
    iterations = 50
    timings = []
    rtfs = []
    print(f"Running benchmark ({iterations} iterations)...")
    for i in range(iterations):
        start_time = time.time()
        with torch.no_grad():
            y = model(log_melspc, f0)
        end_time = time.time()
        
        inference_time = end_time - start_time
        timings.append(inference_time)
        
        audio_duration = y.shape[-1] / sample_rate
        rtf = inference_time / audio_duration
        rtfs.append(rtf)

    avg_time = np.mean(timings)
    std_time = np.std(timings)
    avg_rtf = np.mean(rtfs)
    std_rtf = np.std(rtfs)
    
    print(f"Result: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms per inference")
    print(f"        RTF: {avg_rtf:.4f} ± {std_rtf:.4f}")
    print("-" * (30 + len(model_name) + len(device)))
    return avg_time, avg_rtf

# --- メイン処理 ---

def main():
    parser = argparse.ArgumentParser(description="Benchmark inference speed of NHVSing models.")
    parser.add_argument("npz_path", type=str, help="Path to the input .npz file.")
    parser.add_argument("--device", type=str, default=None, help="Device: 'cpu' or 'cuda'. Auto-detects if not set.")
    args = parser.parse_args()

    if not Path(args.npz_path).exists():
        print(f"Error: NPZ file not found: {args.npz_path}")
        return

    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    
    npz_data = np.load(args.npz_path)
    vocoder_cfg = cfg['model']['vocoder']
    ltv_filter_cfg = cfg['model']['ltv_filter']

    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Benchmarking on device: {device.upper()}")
    print("-" * 60)

    models_to_benchmark = {
        "PyTorch": NHVSingPyTorch(vocoder_cfg, ltv_filter_cfg, PYTORCH_MODEL_PATH),
        "JIT": NHVSingJIT(vocoder_cfg, ltv_filter_cfg, JIT_MODEL_PATH, device),
        "ONNX+PyTorchDSP": NHVSing_ONNX_PyTorchDSP(vocoder_cfg, ltv_filter_cfg, ONNX_MODEL_PATH, device),
        "ONNX+NumPyDSP": NHVSing_ONNX_NumPyDSP(vocoder_cfg, ltv_filter_cfg, ONNX_MODEL_PATH, device),
    }

    results = {}
    for name, model_instance in models_to_benchmark.items():
        # NumPyバックエンドはCPUでのみ実行
        bench_device = 'cpu' if name == "ONNX+NumPyDSP" else device
        try:
            avg_time, avg_rtf = benchmark(model_instance, name, npz_data, vocoder_cfg['sample_rate'], bench_device)
            results[name] = (avg_time, avg_rtf)
        except Exception as e:
            print(f"Error during {name} benchmark: {e}")
            import traceback
            traceback.print_exc()
            results[name] = (-1, -1)

    print("\n" + "="*22 + " Benchmark Summary " + "="*21)
    print(f"Device: {device.upper()}")
    print(f"{'Model':<20} | {'Avg Time (ms)':<15} | {'Avg RTF':<15}")
    print("-" * 60)
    for name, (avg_time, avg_rtf) in results.items():
        if avg_time != -1:
            print(f"{name:<20} | {avg_time*1000:<15.2f} | {avg_rtf:<15.4f}")
        else:
            print(f"{name:<20} | {'FAILED':<15} | {'FAILED':<15}")
    print("=" * 60)

if __name__ == "__main__":
    main()
