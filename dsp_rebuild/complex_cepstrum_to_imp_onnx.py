"""
ONNX-compatible implementation of complex_cepstrum_to_imp.
Production version: outputs only the impulse response (no debug intermediates).
"""

import torch
import torch.nn as nn
import math
import yaml
from torch import Tensor
from typing import List


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

    def __init__(self, fft_size: int, use_float64: bool = True):
        super().__init__()
        self.fft_size = fft_size
        self.use_float64 = use_float64

    def forward(self, ccep: Tensor) -> Tensor:
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
