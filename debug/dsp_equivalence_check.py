
import torch
import numpy as np
import sys
import os

# プロジェクトルートをPythonパスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# プロジェクトのモジュールをインポート
from dsp import (
    generate_impulse_train, freq_multiplier, fftpad, complex_cepstrum_to_fft, 
    complex_cepstrum_to_imp, frame_signal, unframe_signal, time_corr, ltv_fir
)
from inference_onnx import (
    np_generate_impulse_train, np_repeat_interpolate, np_freq_multiplier,
    np_fftpad, np_complex_cepstrum_to_fft, np_complex_cepstrum_to_imp,
    np_frame_signal, np_unframe_signal, np_time_corr_framewise, np_ltv_fir
)
from dataset import norm_interp_f0

# --- 定数 ---
SAMPLING_RATE = 44100
N_HARMONIC = 200
HOP_SIZE = 256  # F0フレームのアップサンプリングに使用
N_FRAMES = 100  # テスト用のF0フレーム数
CCEP_SIZE = 512 # ケプストラムのサイズ
FFT_SIZE = 2048 # FFTサイズ
FRAME_SIZE = 1024 # フレームサイズ
FILTER_SIZE = 512 # フィルタサイズ (for time_corr)

def compare_time_corr():
    """
    dsp.time_corr と inference_onnx.np_time_corr_framewise の動作を比較検証します。
    ltv_firで使われるケースを想定しています。
    """
    print("--- Verifying time_corr vs np_time_corr_framewise ---")

    # 1. Create random framed signals
    framed_x_np = np.random.randn(1, N_FRAMES, FRAME_SIZE).astype(np.float32)
    framed_y_np = np.random.randn(1, N_FRAMES, FILTER_SIZE).astype(np.float32)
    framed_x_torch = torch.from_numpy(framed_x_np)
    framed_y_torch = torch.from_numpy(framed_y_np)

    # 2. Run PyTorch version
    torch_corr = time_corr(framed_x_torch, framed_y_torch)

    # 3. Run NumPy version
    np_corr = np_time_corr_framewise(framed_x_np, framed_y_np)

    # 4. Compare results
    are_close = np.allclose(np_corr, torch_corr.numpy(), atol=1e-5)

    print(f"Outputs are equivalent: {are_close}")
    if not are_close:
        print(f"Max difference: {np.abs(np_corr - torch_corr.numpy()).max()}")

    print("\n")

def compare_ltv_fir():
    """
    dsp.ltv_fir と inference_onnx.np_ltv_fir の動作を比較検証します。
    """
    print("--- Verifying ltv_fir vs np_ltv_fir ---")

    # 1. Create random signals
    n_sample = N_FRAMES * FRAME_SIZE
    x_np = np.random.randn(1, 1, n_sample).astype(np.float32)
    filters_np = np.random.randn(1, N_FRAMES, FILTER_SIZE).astype(np.float32)
    
    x_torch = torch.from_numpy(x_np)
    filters_torch = torch.from_numpy(filters_np)

    # 2. Run PyTorch version
    torch_out = ltv_fir(x_torch, filters_torch, FRAME_SIZE)

    # 3. Run NumPy version
    np_out = np_ltv_fir(x_np, filters_np, FRAME_SIZE)

    # 4. Compare results
    min_len = min(torch_out.shape[-1], np_out.shape[-1])
    are_close = np.allclose(
        np_out[..., :min_len],
        torch_out.numpy()[..., :min_len],
        atol=1e-4 # 浮動小数点演算の差を考慮し、許容誤差を少し大きく設定
    )

    print(f"Outputs are equivalent: {are_close}")
    if not are_close:
        diff = np.abs(np_out[..., :min_len] - torch_out.numpy()[..., :min_len])
        print(f"Max difference: {diff.max()}")
        print(f"Shape (NumPy): {np_out.shape}")
        print(f"Shape (PyTorch): {torch_out.shape}")

    print("\n")

def compare_unframe_signal():
    """
    dsp.unframe_signal と inference_onnx.np_unframe_signal の動作を比較検証します。
    """
    print("--- Verifying unframe_signal vs np_unframe_signal ---")

    # 1. Create random framed signal
    framed_np = np.random.randn(1, FRAME_SIZE, N_FRAMES).astype(np.float32)
    framed_torch = torch.from_numpy(framed_np)

    # 2. Run PyTorch version
    torch_unframed = unframe_signal(framed_torch, HOP_SIZE)

    # 3. Run NumPy version
    np_unframed = np_unframe_signal(framed_np, HOP_SIZE)

    # 4. Compare results
    # Note: The PyTorch `fold` operation might have a slightly different shape handling
    # for the output compared to a manual implementation. We trim to the shortest length.
    min_len = min(torch_unframed.shape[-1], np_unframed.shape[-1])
    are_close = np.allclose(
        np_unframed[..., :min_len],
        torch_unframed.numpy()[..., :min_len],
        atol=1e-5
    )

    print(f"Outputs are equivalent: {are_close}")
    if not are_close:
        diff = np.abs(np_unframed[..., :min_len] - torch_unframed.numpy()[..., :min_len])
        print(f"Max difference: {diff.max()}")

    print("\n")

def compare_frame_signal():
    """
    dsp.frame_signal と inference_onnx.np_frame_signal の動作を比較検証します。
    """
    print("--- Verifying frame_signal vs np_frame_signal ---")

    # 1. Create random signal
    signal_len = FRAME_SIZE + (N_FRAMES - 1) * HOP_SIZE
    signal_np = np.random.randn(1, 1, signal_len).astype(np.float32)
    signal_torch = torch.from_numpy(signal_np)

    # 2. Run PyTorch version
    torch_framed = frame_signal(signal_torch, FRAME_SIZE, HOP_SIZE)

    # 3. Run NumPy version
    np_framed = np_frame_signal(signal_np, FRAME_SIZE, HOP_SIZE)

    # 4. Compare results
    are_close = np.allclose(np_framed, torch_framed.numpy(), atol=1e-5)

    print(f"Outputs are equivalent: {are_close}")
    if not are_close:
        print(f"Max difference: {np.abs(np_framed - torch_framed.numpy()).max()}")

    print("\n")

def compare_fftpad():
    """
    dsp.fftpad と inference_onnx.np_fftpad の動作を比較検証します。
    """
    print("--- Verifying fftpad vs np_fftpad ---")

    # --- Test Case 1: Even size tensor ---
    test_tensor_np = np.random.randn(1, 1, 10).astype(np.float32)
    test_tensor_torch = torch.from_numpy(test_tensor_np)
    padding = 5

    # Run functions
    np_out = np_fftpad(test_tensor_np, padding)
    torch_out = fftpad(test_tensor_torch, padding).numpy()

    # Compare
    are_close = np.allclose(np_out, torch_out)
    print(f"Outputs are equivalent: {are_close}")
    if not are_close:
        print(f"Max difference: {np.abs(np_out - torch_out).max()}")

    # --- Test Case 2: Odd size tensor ---
    test_tensor_np_odd = np.random.randn(1, 1, 11).astype(np.float32)
    test_tensor_torch_odd = torch.from_numpy(test_tensor_np_odd)

    np_out_odd = np_fftpad(test_tensor_np_odd, padding)
    torch_out_odd = fftpad(test_tensor_torch_odd, padding).numpy()
    are_close_odd = np.allclose(np_out_odd, torch_out_odd)
    print(f"Outputs (odd size) are equivalent: {are_close_odd}")
    if not are_close_odd:
        print(f"Max difference (odd size): {np.abs(np_out_odd - torch_out_odd).max()}")

    # --- Test Case 3: No padding ---
    np_out_no_pad = np_fftpad(test_tensor_np, 0)
    torch_out_no_pad = fftpad(test_tensor_torch, 0).numpy()
    are_close_no_pad = np.allclose(np_out_no_pad, torch_out_no_pad)
    print(f"Outputs (no padding) are equivalent: {are_close_no_pad}")
    if not are_close_no_pad:
        print(f"Max difference (no padding): {np.abs(np_out_no_pad - torch_out_no_pad).max()}")

    print("\n")


def compare_complex_cepstrum_to_fft():
    """
    dsp.complex_cepstrum_to_fft と inference_onnx.np_complex_cepstrum_to_fft の動作を比較検証します。
    """
    print("--- Verifying complex_cepstrum_to_fft vs np_complex_cepstrum_to_fft ---")

    # 1. Create random cepstrum
    ccep_np = np.random.randn(1, CCEP_SIZE, N_FRAMES).astype(np.float32)
    ccep_torch = torch.from_numpy(ccep_np)

    # 2. Run PyTorch version
    torch_X, torch_log_mag, torch_phase = complex_cepstrum_to_fft(ccep_torch, FFT_SIZE)

    # 3. Run NumPy version
    np_X, np_log_mag, np_phase = np_complex_cepstrum_to_fft(ccep_np, FFT_SIZE)

    # 4. Compare results
    are_close_X = np.allclose(np_X, torch_X.numpy(), atol=1e-5)
    are_close_log_mag = np.allclose(np_log_mag, torch_log_mag.numpy(), atol=1e-5)
    are_close_phase = np.allclose(np_phase, torch_phase.numpy(), atol=1e-5)

    print(f"FFT outputs are equivalent: {are_close_X}")
    if not are_close_X:
        print(f"Max difference (FFT): {np.abs(np_X - torch_X.numpy()).max()}")

    print(f"Log-magnitude outputs are equivalent: {are_close_log_mag}")
    if not are_close_log_mag:
        print(f"Max difference (Log-Mag): {np.abs(np_log_mag - torch_log_mag.numpy()).max()}")

    print(f"Phase outputs are equivalent: {are_close_phase}")
    if not are_close_phase:
        print(f"Max difference (Phase): {np.abs(np_phase - torch_phase.numpy()).max()}")

    print("\n")


def compare_complex_cepstrum_to_imp():
    """
    dsp.complex_cepstrum_to_imp と inference_onnx.np_complex_cepstrum_to_imp の動作を比較検証します。
    """
    print("--- Verifying complex_cepstrum_to_imp vs np_complex_cepstrum_to_imp ---")

    # 1. Create random cepstrum
    ccep_np = np.random.randn(1, CCEP_SIZE, N_FRAMES).astype(np.float32)
    ccep_torch = torch.from_numpy(ccep_np)

    # 2. Run PyTorch version
    torch_imp = complex_cepstrum_to_imp(ccep_torch, FFT_SIZE)

    # 3. Run NumPy version
    np_imp = np_complex_cepstrum_to_imp(ccep_np, FFT_SIZE)

    # 4. Compare results
    are_close = np.allclose(np_imp, torch_imp.numpy(), atol=1e-5)

    print(f"Impulse responses are equivalent: {are_close}")
    if not are_close:
        print(f"Max difference: {np.abs(np_imp - torch_imp.numpy()).max()}")

    print("\n")


def compare_freq_multiplier():
    """
    PyTorchのfreq_multiplierとNumPyのnp_freq_multiplierの動作を比較検証します。
    """
    print("--- Verifying freq_multiplier vs np_freq_multiplier ---")

    # PyTorch版
    torch_multiplier = freq_multiplier(N_HARMONIC, torch.device('cpu'))

    # NumPy版
    np_multiplier = np_freq_multiplier(N_HARMONIC)

    # 結果を比較
    are_close = np.allclose(np_multiplier, torch_multiplier.numpy(), atol=1e-6)

    print(f"Outputs are equivalent: {are_close}")
    if not are_close:
        diff = np.abs(np_multiplier - torch_multiplier.numpy()).max()
        print(f"Max difference: {diff}")
    print("\n")

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
    compare_freq_multiplier()
    compare_repeat_interpolate()
    compare_generate_impulse_train()
    compare_fftpad()
    compare_complex_cepstrum_to_fft()
    compare_complex_cepstrum_to_imp()
    compare_frame_signal()
    compare_unframe_signal()
    compare_time_corr()
    compare_ltv_fir()
    print("Check finished.")

