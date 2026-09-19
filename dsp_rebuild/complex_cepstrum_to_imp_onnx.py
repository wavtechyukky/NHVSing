"""
ONNX-compatible implementation of complex_cepstrum_to_imp.
Production version: outputs only the impulse response (no debug intermediates).
"""

import os
import torch
import torch.nn.functional as _F
import torch.nn as nn
import math
import yaml
from torch import Tensor
from typing import List

try:                                    # torch>=2.6 相当。無ければ非分割へフォールバック
    from torch._higher_order_ops.scan import scan as _scan
except Exception:                       # noqa: BLE001
    _scan = None



# --- Helper functions (bug-fixed version matching dsp.py) ---

def reshape_zeros_like(x: Tensor, dim: int, length: int) -> Tensor:
    shape = list(x.shape)
    shape[dim] = length
    return torch.zeros(shape, dtype=x.dtype, device=x.device)

def fftpad(x: Tensor, padding: int) -> Tensor:
    size = x.size(-1)
    half = size // 2
    first_half = torch.narrow(x, -1, 0, size - half)
    second_half = torch.narrow(x, -1, size - half, half)
    zeros = reshape_zeros_like(x, -1, padding)
    return torch.cat([first_half, zeros, second_half], dim=-1)


class ComplexCepstrumToImpONNX(nn.Module):
    """
    Converts complex cepstrum to impulse response.
    Production version: returns only the impulse response tensor.
    """

    def __init__(self, fft_size: int, use_float64: bool = True, frame_block: int = 0):
        super().__init__()
        self.fft_size = fft_size
        self.use_float64 = use_float64
        # フレーム方向のブロック長。0 以下で非分割（従来どおり）。
        # ここの FFT も [B, n_frame, fft_size, 2] を全長で作るため、長さに比例して増える
        # （fft_size=1024・hop256 で 8KB/フレーム × 6 本前後）。FFT は dim=-1（フレーム内）
        # のみでフレーム間の総和が無いので、**分割はビット一致**する。
        self.frame_block = int(frame_block) if frame_block else int(os.environ.get('NHV_CCEP_BLOCK', 128))

    def forward(self, ccep: Tensor) -> Tensor:
        if self.frame_block > 0 and _scan is not None and ccep.dim() == 3:
            return self._forward_blocked(ccep)
        return self._forward_whole(ccep)

    def _forward_blocked(self, ccep: Tensor) -> Tensor:
        """フレームを frame_block 本ずつ scan で処理する（ビット一致）。

        ONNX では Python の for がトレース時に展開されフレーム数が固定されるため、
        `torch._higher_order_ops.scan`（ONNX の Scan 演算子）で動的回数のループにする。
        バッチは 1 を前提（この経路の呼び出し元は常に B=1）。
        """
        n_frame = ccep.size(1); csz = ccep.size(-1); blk = self.frame_block
        npad = (-n_frame) % blk
        xs = _F.pad(ccep, (0, 0, 0, npad)).reshape(-1, blk, csz).detach()   # [nb, blk, ccep_size]
        # detach: view_as_complex は autograd 経路があると scan 内で落ちる（推論専用）

        def body(carry, x):
            return carry.clone(), (self._forward_whole(x[0].unsqueeze(0))[0],)

        c0 = torch.zeros(1, dtype=torch.float32, device=ccep.device)
        _, (ys,) = _scan(body, c0, (xs,))                              # [nb, blk, fft_size]
        return ys.reshape(ccep.size(0), -1, self.fft_size).narrow(1, 0, n_frame)

    def _forward_whole(self, ccep: Tensor) -> Tensor:
        ccep_size = ccep.size(-1)

        if self.use_float64:
            ccep_work = ccep.to(torch.float64)
        else:
            ccep_work = ccep.float()

        ccep_padded = fftpad(ccep_work, self.fft_size - ccep_size)

        X_hat_c = torch.fft.fft(ccep_padded, dim=-1)
        X_hat_ri = torch.view_as_real(X_hat_c)

        log_magnitude = X_hat_ri[..., 0]
        phase = X_hat_ri[..., 1]
        magnitude = torch.exp(log_magnitude.clamp(max=10.0))

        X_real = magnitude * torch.sin(math.pi / 2.0 - phase)
        X_imag = magnitude * torch.sin(phase)

        X_ri = torch.stack([X_real, X_imag], dim=-1)
        X_c = torch.view_as_complex(X_ri)

        impulse_response = torch.fft.ifft(X_c, dim=-1).real.to(torch.float32)

        return impulse_response


def export_complex_cepstrum_to_imp_onnx(
    output_path: str,
    ccep_size: int,
    fft_size: int,
    use_float64: bool = True,
    opset_version: int = 18
) -> List[str]:
    """
    Exports ComplexCepstrumToImpONNX to ONNX format.
    """
    precision = "float64" if use_float64 else "float32"
    print(f"Exporting ComplexCepstrumToImpONNX ({precision}) to {output_path}")

    model = ComplexCepstrumToImpONNX(fft_size, use_float64=use_float64)
    model.eval()

    dummy_input = torch.randn(1, 1, ccep_size, dtype=torch.float32)

    output_names = ["impulse_response"]

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['ccep'],
        output_names=output_names,
        dynamic_axes={
            'ccep': {0: 'batch_size', 1: 'channels'},
            "impulse_response": {0: 'batch_size', 1: 'channels'},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False
    )

    print(f"✅ Successfully exported to {output_path}")
    return output_names


if __name__ == "__main__":
    import os
    
    os.makedirs("dsp_rebuild", exist_ok=True)
    
    print("Loading config.yaml to get model parameters...")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    ccep_size = config["model"]["ltv_filter"]["ccep_size"]
    fft_size = config["model"]["ltv_filter"]["fft_size"]
    
    # Export with float64 precision (float32 gives SNR ~56dB < 80dB target)
    export_complex_cepstrum_to_imp_onnx(
        "dsp_rebuild/complex_cepstrum_to_imp.onnx",
        ccep_size=ccep_size,
        fft_size=fft_size
    )
