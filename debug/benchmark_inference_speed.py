import time
import yaml
from pathlib import Path
import numpy as np
import torch
import argparse
import onnxruntime as ort

# --- モデルのインポート ---
# プロジェクトのルートから実行することを想定
import model
import refactored_model
from dsp import generate_impulse_train, complex_cepstrum_to_imp, ltv_fir
from model import repeat_interpolate

# --- 設定とパス ---
CONFIG_PATH = "config.yaml"
# ONNXモデルのパス
ONNX_MODEL_PATH = "debug/nhvsing_convs.onnx"

# --- ONNXランタイムを使った推論モデルの定義 ---

class NHVSingWithONNXRuntime:
    """
    畳み込み部分をONNXランタイムで実行するNHVSingのラッパークラス。
    refactored_model.NHVSingRefactoredのDSP部分を再利用します。
    """
    def __init__(self, vocoder_cfg, ltv_filter_cfg, onnx_path, device='cpu'):
        print("Initializing NHVSingWithONNXRuntime")
        self.device = device
        self.fs = vocoder_cfg['sample_rate']
        self.hop_size = vocoder_cfg['hop_size']
        self.fft_size = ltv_filter_cfg['fft_size']
        self.noise_std = vocoder_cfg['noise_std']

        # ONNXランタイムセッションを初期化
        print(f"Loading ONNX model from: {onnx_path}")
        providers = ['CPUExecutionProvider']
        if device == 'cuda':
            providers.insert(0, 'CUDAExecutionProvider')
        self.ort_session = ort.InferenceSession(onnx_path, providers=providers)
        print(f"ONNX session created for device: {self.ort_session.get_providers()}")

        # DSP関数
        self.impulse_generator = generate_impulse_train

    def to(self, device):
        """デバイス転送用のダミーメソッド"""
        self.device = device
        # Note: ONNX Runtimeセッションのデバイスは初期化時に決定される
        return self

    def __call__(self, x, cf0):
        """推論実行"""
        # ノイズを生成
        z_shape = (x.size(0), 1, x.size(1) * self.hop_size)
        z = torch.normal(0, self.noise_std, z_shape).to(self.device)

        # --- Part 1: ONNXランタイムでの畳み込み実行 ---
        ort_inputs = {self.ort_session.get_inputs()[0].name: x.cpu().numpy()}
        ccep_harm_np, ccep_noise_np = self.ort_session.run(None, ort_inputs)
        
        ccep_harm = torch.from_numpy(ccep_harm_np).to(self.device)
        ccep_noise = torch.from_numpy(ccep_noise_np).to(self.device)

        # --- Part 2: PyTorchでのDSP処理 ---
        cf0_resampled = repeat_interpolate(cf0, self.hop_size)
        harmonic_source = self.impulse_generator(cf0_resampled, 200, float(self.fs))

        imp_harm = complex_cepstrum_to_imp(ccep_harm, self.fft_size)
        sig_harm = ltv_fir(harmonic_source, imp_harm, self.hop_size)

        imp_noise = complex_cepstrum_to_imp(ccep_noise, self.fft_size)
        sig_noise = ltv_fir(z, imp_noise, self.hop_size)
        
        y = sig_harm + sig_noise
        y = torch.clamp(y, -1, 1)
        
        return y.reshape(x.size(0), -1)


def benchmark(model_class, model_name, vocoder_cfg, ltv_filter_cfg, npz_data, device):
    """指定されたモデルの推論速度を計測する"""
    print(f"--- Benchmarking: {model_name} ---")
    
    # モデルの初期化
    if model_name == "NHVSing_with_ONNX":
        model = model_class(vocoder_cfg, ltv_filter_cfg, ONNX_MODEL_PATH, device)
    else:
        model = model_class(vocoder_cfg, ltv_filter_cfg)
    
    model.to(device)
    
    # 入力データを準備
    log_melspc = torch.from_numpy(npz_data['log_melspc']).unsqueeze(0).float().to(device)
    f0 = torch.from_numpy(npz_data['f0']).unsqueeze(0).unsqueeze(0).float().to(device)
    
    # ノイズはNHVSingのオリジナル版のみ必要
    noise = None
    if model_name == "NHVSing (Original)":
        noise_shape = (1, 1, log_melspc.size(1) * vocoder_cfg['hop_size'])
        noise = torch.normal(0, vocoder_cfg['noise_std'], noise_shape).to(device)

    # ウォームアップ実行
    print("Running warmup inference...")
    with torch.no_grad():
        if model_name == "NHVSing (Original)":
            _ = model(noise, log_melspc, f0)
        else:
            _ = model(log_melspc, f0)

    # ベンチマーク実行
    iterations = 20
    timings = []
    rtfs = []
    print(f"Running benchmark ({iterations} iterations)...")
    for i in range(iterations):
        start_time = time.time()
        with torch.no_grad():
            if model_name == "NHVSing (Original)":
                y = model(noise, log_melspc, f0)
            else:
                y = model(log_melspc, f0)
        end_time = time.time()
        
        # 計測結果を保存
        inference_time = end_time - start_time
        timings.append(inference_time)
        
        # RTFを計算
        audio_duration = y.shape[-1] / vocoder_cfg['sample_rate']
        rtf = inference_time / audio_duration
        rtfs.append(rtf)

    avg_time = np.mean(timings)
    std_time = np.std(timings)
    avg_rtf = np.mean(rtfs)
    std_rtf = np.std(rtfs)
    
    print(f"Result: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms per inference")
    print(f"        RTF: {avg_rtf:.4f} ± {std_rtf:.4f}")
    print("-" * (20 + len(model_name)))
    return avg_time, avg_rtf

def main():
    parser = argparse.ArgumentParser(description="Benchmark inference speed of different NHVSing models.")
    parser.add_argument("npz_path", type=str, help="Path to the input .npz file for benchmarking.")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g., 'cpu', 'cuda'). Defaults to cuda if available.")
    args = parser.parse_args()

    npz_path = Path(args.npz_path)
    if not npz_path.exists():
        print(f"エラー: 指定されたNPZファイルが見つかりません: {npz_path}")
        return

    # 設定ファイルを読み込み
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    
    print(f"Using NPZ file: {npz_path}")
    npz_data = np.load(npz_path)

    # デバイスを設定
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ベンチマーク対象のモデル
    models_to_benchmark = [
        (model.NHVSing, "NHVSing (Original)"),
        (refactored_model.NHVSingRefactored, "NHVSingRefactored"),
        (NHVSingWithONNXRuntime, "NHVSing_with_ONNX"),
    ]

    results = {}
    for model_cls, name in models_to_benchmark:
        try:
            avg_time, avg_rtf = benchmark(model_cls, name, cfg['model']['vocoder'], cfg['model']['ltv_filter'], npz_data, device)
            results[name] = (avg_time, avg_rtf)
        except Exception as e:
            print(f"エラー: {name} のベンチマーク中にエラーが発生しました: {e}")
            results[name] = (-1, -1)

    print("\n--- Benchmark Summary ---")
    print(f"Device: {device.upper()}")
    print(f"{ 'Model':<25} | {'Avg Time (ms)':<15} | {'Avg RTF':<15}")
    print("-" * 60)
    for name, (avg_time, avg_rtf) in results.items():
        if avg_time != -1:
            print(f"{name:<25} | {avg_time*1000:<15.2f} | {avg_rtf:<15.4f}")
        else:
            print(f"{name:<25} | {'FAILED':<15} | {'FAILED':<15}")
    print("-" * 60)


if __name__ == "__main__":
    main()