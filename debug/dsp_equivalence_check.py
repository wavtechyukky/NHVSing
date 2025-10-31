
import torch
import numpy as np
import sys
import os

# プロジェクトルートをPythonパスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# プロジェクトのモジュールをインポート
from dsp import generate_impulse_train
from inference_onnx import np_generate_impulse_train, np_repeat_interpolate
from dataset import norm_interp_f0

# --- 定数 ---
SAMPLING_RATE = 44100
N_HARMONIC = 200
HOP_SIZE = 256  # F0フレームのアップサンプリングに使用
N_FRAMES = 100  # テスト用のF0フレーム数

def compare_repeat_interpolate():
    """
    PyTorchのrepeat_interleaveとNumPyのrepeatの動作を比較検証します。
    これは dsp.py の repeat_interpolate() と inference_onnx.py の np_repeat_interpolate() に相当します。
    """
    print("--- Verifying repeat_interpolate vs np_repeat_interpolate ---")
    
    # 1. ランダムなF0データを生成 (NumPy)
    # (Batch, 1, Frames)
    f0_np = np.random.rand(1, 1, N_FRAMES).astype(np.float32) * 500
    f0_np[0, 0, 20:40] = 0 # 無音区間を作成
    
    # 2. 線形補完
    f0_np_interp, _ = norm_interp_f0(f0_np.squeeze())
    f0_np_interp = f0_np_interp[np.newaxis, np.newaxis, :]

    # 3. PyTorch版のF0を作成
    f0_torch = torch.from_numpy(f0_np_interp)

    # 4. NumPy版のアップサンプリング
    np_resampled = np_repeat_interpolate(f0_np_interp, HOP_SIZE)

    # 5. PyTorch版のアップサンプリング
    torch_resampled = torch.repeat_interleave(f0_torch, HOP_SIZE, dim=-1)

    # 6. 結果を比較
    are_close = np.allclose(np_resampled, torch_resampled.numpy(), atol=1e-6)
    
    print(f"Outputs are equivalent: {are_close}")
    if not are_close:
        diff = np.abs(np_resampled - torch_resampled.numpy()).max()
        print(f"Max difference: {diff}")
    print("\n")

def compare_generate_impulse_train():
    """
    generate_impulse_train (PyTorch) と np_generate_impulse_train (NumPy) の
    動作を比較検証します。
    """
    print("--- Verifying generate_impulse_train vs np_generate_impulse_train ---")
    
    # 1. ランダムなF0データを生成 (NumPy)
    f0_np = np.random.rand(1, 1, N_FRAMES).astype(np.float32) * 500
    f0_np[0, 0, 40:60] = 0 # 無音区間

    # 2. 線形補完とアップサンプリング
    f0_np_interp, _ = norm_interp_f0(f0_np.squeeze())
    f0_np_interp = f0_np_interp[np.newaxis, np.newaxis, :]
    f0_resampled_np = np.repeat(f0_np_interp, HOP_SIZE, axis=-1)

    # 3. PyTorch版のF0を作成
    f0_resampled_torch = torch.from_numpy(f0_resampled_np)

    # 4. NumPy版の関数を実行
    # 注意: np_generate_impulse_trainは警告を出す不完全な実装です
    print("Running NumPy version...")
    impulse_np = np_generate_impulse_train(f0_resampled_np, N_HARMONIC, SAMPLING_RATE)

    # 5. PyTorch版の関数を実行
    print("Running PyTorch version...")
    impulse_torch = generate_impulse_train(f0_resampled_torch, N_HARMONIC, float(SAMPLING_RATE))
    
    # 6. 結果を比較
    are_close = np.allclose(impulse_np, impulse_torch.numpy(), atol=1e-5)
    
    print(f"Outputs are equivalent: {are_close}")
    if not are_close:
        diff = np.abs(impulse_np - impulse_torch.numpy()).max()
        print(f"Max difference: {diff}")
        
        # 最初の10サンプルを表示して違いを確認
        print("First 10 samples (NumPy):")
        print(impulse_np[0, 0, :10])
        print("First 10 samples (PyTorch):")
        print(impulse_torch.numpy()[0, 0, :10])
    print("\n")


if __name__ == "__main__":
    print("Starting DSP function equivalence check...")
    compare_repeat_interpolate()
    compare_generate_impulse_train()
    print("Check finished.")
