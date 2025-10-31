import argparse
import os
import yaml
import torch
import numpy as np
import onnxruntime

from model import NHVSing
# dsp.pyの代わりにinference_onnx.pyからNumPy関数をインポート
from inference_onnx import np_ltv_fir, np_complex_cepstrum_to_imp

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_model(checkpoint_path, config):
    """チェックポイントからモデルを読み込む"""
    model = NHVSing(
        vocoder_cfg=config['model']['vocoder'],
        ltv_filter_cfg=config['model']['ltv_filter'],
    )
    snapshot = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(snapshot['model'])
    model.eval()
    return model

def export_pytorch_model(model, save_path):
    """PyTorchモデルのstate_dictを保存する"""
    print(f"Exporting PyTorch model state_dict to {save_path}...")
    torch.save(model.state_dict(), save_path)
    print("Done.")

def export_jit_model(model, save_path):
    """JITコンパイル済みモデルを保存する"""
    print(f"Exporting JIT model to {save_path}...")
    scripted_model = torch.jit.script(model)
    scripted_model.save(save_path)
    print("Done.")

def export_onnx_core(model, save_path, dummy_x):
    """モデルの中核部分(NHVConvsONNX)をONNX形式でエクスポートする"""
    print(f"Exporting ONNX core model to {save_path}...")
    core_model = model.convs_onnx

    # 可変にしたい次元を `dynamic_axes` で指定します。
    # 入力'log_melspc'は (B, T, D) で、1番目の次元(時間/フレーム)を'time'という名前で可変にします。
    # 2つの出力'ccep_harm', 'ccep_noise'は (B, T, ccep_size) で、
    # 1番目の次元(時間/フレーム)を'time'という名前で可変にします。
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
        # モデルは2つのテンソルを返すため、出力名も2つ指定します
        output_names=['ccep_harm', 'ccep_noise'],
        dynamic_axes=dynamic_axes
    )
    print("Done.")

class OnnxInferenceWrapper:
    """ONNXモデルとNumPy DSP関数を組み合わせて音声生成を行うクラス"""
    def __init__(self, onnx_path, config):
        print("Initializing ONNX inference wrapper...")
        self.session = onnxruntime.InferenceSession(onnx_path)
        
        # DSP処理に必要なパラメータを設定
        self.frame_size = config['training']['frame_size']
        self.fft_size = config['model']['ltv_filter']['fft_size']
        self.noise_std = config['model']['vocoder']['noise_std']
        print("ONNX wrapper initialized.")

    def __call__(self, log_melspc, cf0):
        """
        Args:
            log_melspc (np.ndarray): log mel-spectrogram (B, C, T_mel)
            cf0 (np.ndarray): continuous F0 (B, 1, T_f0). Used for determining output length.
        Returns:
            np.ndarray: generated waveform
        """
        # 1. ONNXモデルで複素ケプストラムを生成
        inputs = {'log_melspc': log_melspc.astype(np.float32)}
        ccep = self.session.run(None, inputs)[0] # Shape: [B, n_cep, n_frames] 
        
        # 2. 複素ケプストラムからフィルタのインパルス応答を計算 (NumPy)
        #    np_complex_cepstrum_to_impは最後の次元に沿って処理するため、入力形式を [B, n_frames, n_cep] に変更
        ccep_transposed = ccep.transpose(0, 2, 1)
        filters = np_complex_cepstrum_to_imp(ccep_transposed, self.fft_size) # Shape: [B, n_frames, fft_size]

        # 3. ノイズのソース信号を生成
        n_samples = cf0.shape[-1] * self.frame_size
        noise_source = np.random.normal(0, self.noise_std, (log_melspc.shape[0], 1, n_samples)).astype(np.float32)

        # 4. LTV-FIRフィルタを適用 (NumPy)
        wav = np_ltv_fir(
            noise_source,
            filters,
            frame_size=self.frame_size,
        )
        return wav

def main():
    parser = argparse.ArgumentParser(description="Export trained models for inference.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint (.pth).")
    parser.add_argument("--config", type=str, required=True, help="Path to the training config file (config.yaml).")
    parser.add_argument("--output_dir", type=str, default="exported_models", help="Directory to save exported models.")
    parser.add_argument("--all", action="store_true", help="Export all model types.")
    parser.add_argument("--pytorch", action="store_true", help="Export full PyTorch model.")
    parser.add_argument("--jit", action="store_true", help="Export JIT-scripted model.")
    parser.add_argument("--onnx", action="store_true", help="Export ONNX core model and wrapper.")

    args = parser.parse_args()

    # 出力ディレクトリを作成
    os.makedirs(args.output_dir, exist_ok=True)

    # 設定ファイルを読み込み
    config = load_config(args.config)

    # モデルを読み込み
    print("Loading model from checkpoint...")
    model = load_model(args.checkpoint, config)
    print("Model loaded.")

    # エクスポート用のダミー入力を作成
    # ONNXエクスポートには (B, T, D) の形状のダミー入力が必要です
    dummy_x = torch.randn(1, 100, config['preprocess']['mel_dim'], dtype=torch.float32)
    dummy_cf0 = torch.randn(1, 1, 100, dtype=torch.float32)
    
    # モデルをエクスポート
    if args.all or args.pytorch:
        export_pytorch_model(model, os.path.join(args.output_dir, "model.pth"))

    if args.all or args.jit:
        export_jit_model(model, os.path.join(args.output_dir, "model_jit.pt"))

    if args.all or args.onnx:
        export_onnx_core(model, os.path.join(args.output_dir, "core_model.onnx"), dummy_x)
        
        print("\n--- ONNX Wrapper Usage Example ---")
        try:
            onnx_wrapper = OnnxInferenceWrapper(os.path.join(args.output_dir, "core_model.onnx"), config)
            # ONNXラッパーのテストには (B, T, D) の形状が必要です
            log_melspc_np = dummy_x.numpy()
            cf0_np = dummy_cf0.numpy()
            
            print("Running inference with ONNX wrapper...")
            wav = onnx_wrapper(log_melspc_np, cf0_np)
            print(f"Inference successful. Output waveform shape: {wav.shape}")
        except Exception as e:
            print(f"An error occurred during ONNX wrapper test: {e}")

if __name__ == "__main__":
    main()