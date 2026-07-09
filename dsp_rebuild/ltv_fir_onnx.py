"""
ONNX-exportable implementation of the ltv_fir function from dsp.py.

The core idea is to replace problematic PyTorch operators with ONNX-friendly equivalents:
- `torch.nn.functional.conv1d` with dynamic groups is replaced by FFT-based convolution.
  (conv1d with dynamic groups fails in the dynamo ONNX exporter because the number of
   groups must be a compile-time constant.)
- `torch.nn.functional.fold` (col2im) is replaced by a manual overlap-add (OLA) using `scatter_add`.
"""
import torch
import torch.nn as nn
import os
import yaml

class LTVFirONNX(nn.Module):
    """
    Flattened ONNX-exportable implementation of the ltv_fir function.
    Combines frame_signal, fftshift, FFT-based convolution, and OLA (scatter_add)
    into a single module to avoid nested module tracing issues.
    """
    def __init__(self, frame_size: int, filter_size: int = 0):
        super().__init__()
        self.frame_size = frame_size
        # Pre-compute the FFT length for frame_size + filter_size.
        # ONNX Runtime's DFT op is fast ONLY for powers of two (a non-pow2 length such as
        # 1280 is ~4-8x slower in ORT than 2048). We therefore pad to the next power of two:
        # slightly more samples, but ~2x faster end-to-end in ORT, and bit-exact (FFT is FFT).
        if filter_size > 0:
            L_min = frame_size + filter_size - 1
            self._fast_fft_len = self._next_pow2(L_min)
        else:
            self._fast_fft_len = 0

    @staticmethod
    def _next_pow2(n: int) -> int:
        """Next power of two >= n (ORT DFT is fast only for powers of two)."""
        p = 1
        while p < n:
            p *= 2
        return p

    @staticmethod
    def _next_fast_len(n: int) -> int:
        """Next 2/3/5-smooth size >= n (kept for reference; ORT prefers _next_pow2)."""
        while True:
            m = n
            while m % 2 == 0:
                m //= 2
            while m % 3 == 0:
                m //= 3
            while m % 5 == 0:
                m //= 5
            if m == 1:
                return n
            n += 1

    def forward(self, x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
        """
        Linear time-varying FIR filter with a square OLA window.

        Args:
            x: [n_batch, 1, n_sample]
            filters: [n_batch, n_frame, filter_size]
                     Filter FIRs stored as time-wrapped signals.

        Returns:
            striped_y: [n_batch, 1, n_sample]
        """
        n_sample = x.size(-1)
        filter_size = filters.size(-1)

        # === Step 1: frame_signal (inline) ===
        # x: [n_batch, 1, n_sample] -> [n_batch, 1, n_sample, 1]
        x_2d = x.unsqueeze(-1)
        # unfold: [n_batch, frame_size, n_frame]
        framed_x = torch.nn.functional.unfold(
            x_2d,
            kernel_size=(self.frame_size, 1),
            stride=(self.frame_size, 1)  # Use frame_size as stride (no overlap)
        )
        # transpose: [n_batch, n_frame, frame_size]
        framed_x = framed_x.transpose(1, 2)

        # === Step 2: fftshift (inline) ===
        split_point = (filter_size + 1) // 2
        filters = torch.cat((filters[..., split_point:], filters[..., :split_point]), dim=-1)

        # === Step 3: Linear convolution via FFT ===
        # Replaces grouped conv1d which fails with dynamic shapes in the dynamo
        # ONNX exporter (groups must be a compile-time constant).
        #
        # Equivalence: conv1d(x, flip(h), full_padding) == IFFT(FFT(x) * FFT(h))
        # This means we don't need to flip the filters at all.
        Nx = framed_x.size(-1)
        Ny = filters.size(-1)
        L_min = Nx + Ny - 1
        # Pad FFT size to a "fast" length (small prime factors only).
        # e.g. 1143 = 3^2 * 127 (slow) -> 1152 = 2^7 * 3^2 (fast)
        L = self._fast_fft_len if self._fast_fft_len >= L_min else L_min
        # Full complex fft/ifft: rfft/irfft is blocked by ONNX exporter limitations
        # (constant_pad_nd on complex-valued tensors not supported, and ONNX Runtime
        # doesn't support DFT with is_onesided=True and inverse=True simultaneously).
        X_f = torch.fft.fft(framed_x, n=L, dim=-1)
        F_f = torch.fft.fft(filters, n=L, dim=-1)
        framed_z = torch.fft.ifft(X_f * F_f, n=L, dim=-1).real
        # framed_z: [n_batch, n_frame, Nx + Ny - 1]

        # === Step 4: Overlap-Add (OLA) with scatter_add (inline) ===
        # framed_z: [B, n_frame, N_out]
        # Permute to [B, N_out, n_frame] to match scatter logic
        framed_z_t = framed_z.permute(0, 2, 1)
        n_batch, frame_sz, n_frame = framed_z_t.shape

        # Output length for OLA
        ola_n_sample = frame_sz + (n_frame - 1) * self.frame_size

        # Initialize output buffer
        output = torch.zeros(n_batch, 1, ola_n_sample, dtype=framed_z_t.dtype, device=framed_z_t.device)

        # Create indices for scatter_add
        frame_indices = torch.arange(n_frame, device=framed_z_t.device, dtype=torch.long)
        position_indices = torch.arange(frame_sz, device=framed_z_t.device, dtype=torch.long)

        # indices[i, j] = i + j * frame_size
        indices = position_indices.unsqueeze(1) + frame_indices.unsqueeze(0) * self.frame_size
        indices = indices.unsqueeze(0).expand(n_batch, -1, -1)

        # Flatten for scatter_add. Use contiguous() before view/reshape for safety.
        framed_z_flat = framed_z_t.contiguous().view(n_batch, 1, frame_sz * n_frame)
        indices_flat = indices.contiguous().view(n_batch, 1, frame_sz * n_frame)

        # Use scatter_add_ to perform OLA
        y = output.scatter_add(2, indices_flat, framed_z_flat)

        # === Step 5: Slice to match original n_sample ===
        start_slice = filter_size // 2
        # The output length of striped_y should match n_sample, so we slice exactly that many samples
        striped_y = y.narrow(2, start_slice, n_sample)
        
        return striped_y


def export_ltv_fir_onnx(output_path: str, config_path: str):
    """
    Exports the LTVFirONNX module to an ONNX file.
    """
    print(f"Exporting LTVFirONNX to ONNX at {output_path}...")
    
    # Load parameters from config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        frame_size = config['model']['vocoder']['hop_size']
        fft_size = config['model']['ltv_filter']['fft_size']
        filter_size = fft_size
    except (FileNotFoundError, KeyError) as e:
        print(f"Warning: Could not load parameters from config.yaml ({e}). Using default values.")
        frame_size = 256
        filter_size = 2048

    model = LTVFirONNX(frame_size, filter_size=filter_size)
    fast_len = model._fast_fft_len
    L_min = frame_size + filter_size - 1
    print(f"  frame_size={frame_size}, filter_size={filter_size}")
    print(f"  FFT size: {L_min} -> {fast_len} (optimized)")
    model.eval()

    # Dummy inputs simulating a few frames of audio
    # Note: n_sample must be a multiple of frame_size for this logic
    n_frame = 5
    n_sample = frame_size * n_frame
    dummy_x = torch.randn(1, 1, n_sample, dtype=torch.float32)
    dummy_filters = torch.randn(1, n_frame, filter_size, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy_x, dummy_filters),
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['x', 'filters'],
        output_names=['output'],
        dynamic_axes={
            'x': {0: 'batch', 2: 'n_sample'},
            'filters': {0: 'batch', 1: 'n_frame'},
            'output': {0: 'batch', 2: 'n_sample_out'}
        }
    )
    print(f"✅ Successfully exported LTVFirONNX to {output_path}")

if __name__ == '__main__':
    # Create directory if it doesn't exist
    output_dir = "dsp_rebuild"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output path and config path
    onnx_model_path = os.path.join(output_dir, "ltv_fir.onnx")
    config_file_path = "config.yaml"
    
    # Run the export
    export_ltv_fir_onnx(onnx_model_path, config_file_path)
