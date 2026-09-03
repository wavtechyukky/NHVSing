"""
ONNX-compatible implementation of generate_impulse_train.

Phase 2: Generate impulse train matching PyTorch reference.
Reference: dsp.py::generate_impulse_train

Key differences from original:
1. No @torch.jit.script decorator (JIT removed for ONNX compatibility)
2. No freq_multiplier() call inside function (pre-computed as buffer)
3. cumsum を float64 で実行 — PyTorch の cumsum は内部で float64 アキュムレータを
   使用するため、ONNX 側でも明示的に float64 へ昇格して一致させる
"""

import torch
import torch.nn as nn
import math
from torch import Tensor
from typing import Optional


def freq_multiplier_onnx(n_harmonic: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Generate harmonic multiplier [1, 2, ..., n_harmonic] as [1, n_harmonic, 1]"""
    return torch.arange(1, n_harmonic + 1, dtype=dtype, device=device).view(1, n_harmonic, 1)


class GenerateImpulseTrainONNX(nn.Module):
    """
    ONNX-compatible impulse train generator.
    Exact match to dsp.py::generate_impulse_train (without @torch.jit.script).
    
    Input: f0_t [batch, 1, time]  (fundamental frequency, float32)
    Output: source [batch, 1, time]  (impulse train, float32)
    
    Parameters:
        n_harmonic: Number of harmonics (int, inherited from config)
        sampling_rate: Sampling rate in Hz (float, inherited from config)
    """
    
    def __init__(self, n_harmonic: int, sampling_rate: float):
        super().__init__()
        self.n_harmonic = n_harmonic
        self.sampling_rate = float(sampling_rate)
        
        # Harmonic multiplier [1, n_harmonic, 1]
        # Only pre-compute this (non-varying parameter)
        multiplier = freq_multiplier_onnx(n_harmonic, torch.float32, torch.device('cpu'))
        self.register_buffer("multiplier", multiplier)
        # fm/sr を float64 で前計算しておく。位相計算で cumsum(f0) の「後」にこのテンソルを掛けることで、
        # スカラ (1/sr) を cumsum に掛けた時に ONNX オプティマイザが起こす cumsum(f0/sr) 融合(=小値和で
        # runtime 間非決定 ~0.047)を防ぐ。cumsum(f0)(大値)は runtime 間でビット一致する。
        self.register_buffer("mult_over_sr", multiplier.double() / self.sampling_rate)
        
    def forward(self, f0_t: Tensor) -> Tensor:
        """
        Generate impulse train from fundamental frequency.
        Exact implementation matching dsp.py::generate_impulse_train
        
        Args:
            f0_t: [batch, 1, time] fundamental frequency in Hz
            
        Returns:
            source: [batch, 1, time] impulse train signal
        """
        # Ensure input is float32
        f0_t = f0_t.float()
        
        # Move multiplier to same device as input
        multiplier = self.multiplier.to(f0_t.device).float()
        
        # Reference implementation line-by-line replication:
        # f0_map = freq_multiplier(n_harmonic, f0_t.device) * f0_t
        f0_map = multiplier * f0_t
        
        # weight_map = torch.sigmoid(-(f0_map - sampling_rate / 2.0))
        weight_map = torch.sigmoid(-(f0_map - self.sampling_rate / 2.0))
        
        # w0_map_cum = (
        #     f0_t.cumsum(dim=-1) * 2.0 * math.pi / sampling_rate *
        #     freq_multiplier(n_harmonic, f0_t.device)
        # )
        # 位相は float64 で cycles を累積 → mod 1.0 で [0,1) へ折り返し → float32 で ×2π → cos。
        # 旧実装は float64 cumsum の直後に float32 へ戻す no-op(pure float32 と 1bit 一致)で、累積値
        # (フル長尺で ~1e8, ULP≈10)の丸めにより位相が劣化し倍音間に滲みが出ていた。ONNX Runtime は
        # Cos(double) 非実装のため、折り返して [0,2π) の小さい値にしてから float32 cos に渡す
        # (dsp.py::generate_impulse_train と同一計算。位相精度は float64 品質を保つ)。
        # ★fm/sr は前計算テンソル mult_over_sr を cumsum(f0) の後に掛ける(スカラ 1/sr を cumsum に
        #   掛けると ONNX が cumsum(f0/sr) に融合し runtime 間で ~0.047 ズレる。cumsum(f0) はビット一致)。
        cycles = f0_t.to(torch.float64).cumsum(dim=-1) * self.mult_over_sr.to(f0_t.device)
        w0_map_cum = ((cycles - torch.floor(cycles)) * (2.0 * math.pi)).to(torch.float32)
        source = torch.sum(torch.cos(w0_map_cum) * weight_map, dim=1, keepdim=True)
        return source * 0.01


def export_impulse_train_onnx(
    output_path: str,
    n_harmonic: int = 200,
    sampling_rate: float = 44100.0,
    opset_version: int = 16
) -> None:
    """
    Export GenerateImpulseTrainONNX to ONNX format.
    
    Args:
        output_path: Output file path for .onnx model
        n_harmonic: Number of harmonics (default 40)
        sampling_rate: Sampling rate in Hz (default 44100)
        opset_version: ONNX opset version (default 16)
    """
    print(f"Exporting GenerateImpulseTrainONNX to {output_path}")
    print(f"  - n_harmonic: {n_harmonic}")
    print(f"  - sampling_rate: {sampling_rate}")
    print(f"  - opset_version: {opset_version}")
    
    model = GenerateImpulseTrainONNX(n_harmonic, sampling_rate)
    model.eval()
    
    # Create dummy input: [batch=1, 1, time=1024]
    # Use float32 (matches our implementation)
    dummy_input = torch.randn(1, 1, 1024, dtype=torch.float32)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['f0_t'],
        output_names=['source'],
        dynamic_axes={
            'f0_t': {0: 'batch_size', 2: 'time'},
            'source': {0: 'batch_size', 2: 'time'}
        },
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False
    )
    
    print(f"✅ Successfully exported to {output_path}")


if __name__ == "__main__":
    import os
    
    # Ensure output directory exists
    os.makedirs("dsp_rebuild", exist_ok=True)
    
    # Export to ONNX
    export_impulse_train_onnx("dsp_rebuild/impulse_train.onnx")
