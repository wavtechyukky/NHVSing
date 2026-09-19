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
import torch.nn.functional as F
import os
import yaml

try:                                    # torch>=2.6 相当。無ければ非分割へフォールバック
    from torch._higher_order_ops.scan import scan as _scan
except Exception:                       # noqa: BLE001
    _scan = None

class LTVFirONNX(nn.Module):
    """
    Flattened ONNX-exportable implementation of the ltv_fir function.
    Combines frame_signal, fftshift, FFT-based convolution, and OLA (scatter_add)
    into a single module to avoid nested module tracing issues.
    """
    def __init__(self, frame_size: int, filter_size: int = 0, use_hann: bool = False,
                 frame_block: int = 0):
        super().__init__()
        self.frame_size = frame_size
        # フレーム方向のブロック長。0 以下で非分割（従来どおり）。
        # FFT 段の中間テンソルは [B, n_frame, L, 2] で、分割しないと長さに比例して巨大化する
        # （L=2048・hop256 で 1 テンソルあたり 16KB/フレーム、これが 10 本前後同時に生きる）。
        # FFT は dim=-1（フレーム内）のみなので、フレーム方向の分割は演算として厳密等価。
        # OLA の加算順序だけが変わるため、出力は float32 の丸め分（実測 -142dB）だけずれる。
        self.frame_block = int(frame_block) if frame_block else int(os.environ.get('NHV_LTV_BLOCK', 128))
        # use_hann=True: same Hann WOLA (window length 2*frame_size, 50% overlap) as the eager
        # reference dsp.hann_ltv_fir (plain-PyTorch implementation used as the ground truth).
        # use_hann=False: legacy square OLA (bit-exact with the old graph). Set this to match
        # whatever ola_mode the model was trained with (ola_mode: hann -> use_hann=True).
        self.use_hann = bool(use_hann)
        # Analysis frame length: square = frame_size (non-overlapping), hann = 2*frame_size (50% overlap).
        analysis_len = frame_size * 2 if self.use_hann else frame_size
        # Pre-compute the FFT length for analysis_len + filter_size.
        # ONNX Runtime's DFT op is fast ONLY for powers of two (a non-pow2 length such as
        # 1280 is ~4-8x slower in ORT than 2048). We therefore pad to the next power of two:
        # slightly more samples, but ~2x faster end-to-end in ORT, and bit-exact (FFT is FFT).
        if filter_size > 0:
            L_min = analysis_len + filter_size - 1
            self._fast_fft_len = self._next_pow2(L_min)
        else:
            self._fast_fft_len = 0
        if self.use_hann:
            # Same as the eager hann_ltv_fir: periodic Hann of length 2*frame_size.
            # Constant buffer (persistent=False -> not saved into the state_dict).
            self.register_buffer(
                'hann_win', torch.hann_window(frame_size * 2, periodic=True), persistent=False)

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
        if self.frame_block > 0 and _scan is not None:
            return self._forward_blocked(x, filters)
        return self._forward_whole(x, filters)

    def _forward_blocked(self, x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
        """フレームを frame_block 本ずつ scan で処理する。ピークがブロック長で決まる定数になる。

        重いのは FFT 段の [B, n_frame, L, 2]（L=2048 で 16KB/フレーム × 10 本前後）であって、
        フレーム化した信号 [B, n_frame, ws]（ws=512 で 2KB/フレーム）は軽い。そこで
        **フレーム化までは全長で行い、FFT 段だけをブロック化**する。こうすると
        ブロック長を入力長から計算する必要が無くなり（reshape だけで切れる）、
        ONNX で長さが定数に焼き付く問題も起きない。

        ONNX では Python の for がトレース時に展開されフレーム数が固定されるため、
        `torch._higher_order_ops.scan`（ONNX の Scan 演算子）で動的回数のループにする。
        """
        n_sample = x.size(-1)
        filter_size = filters.size(-1)
        fs = self.frame_size
        ws = fs * 2 if self.use_hann else fs
        blk = self.frame_block
        L_min = ws + filter_size - 1
        L = self._fast_fft_len if self._fast_fft_len >= L_min else L_min

        # === Step 1-2: フレーム化 + fftshift（非分割版と同一・全長のまま）===
        if self.use_hann:
            xp = F.pad(x, [fs, fs - 1])
        else:
            xp = x
        framed_x = F.unfold(xp.unsqueeze(-1), kernel_size=(ws, 1),
                            stride=(fs, 1)).transpose(1, 2)          # [B, n_frame_x, ws]
        if self.use_hann:
            framed_x = framed_x * self.hann_win.to(framed_x.dtype)
        sp = (filter_size + 1) // 2
        filters = torch.cat((filters[..., sp:], filters[..., :sp]), dim=-1)

        # フレーム数を filters 側に合わせる（非分割版は filters の n_frame で OLA する）
        n_frame = filters.size(1)
        framed_x = framed_x.narrow(1, 0, n_frame)

        # === ブロックへ reshape（長さ依存の算術は不要）===
        # **バッチ次元は scan の中へ持ち込む**（[nb, B, blk, *]）。[B*nb, blk, *] に畳むと
        # 別バッチのブロックが 1 本のストリームへ OLA され、B>1 で壊れる。
        B = x.size(0)
        npad = (-n_frame) % blk
        fx = (F.pad(framed_x, (0, 0, 0, npad)).reshape(B, -1, blk, ws)
              .transpose(0, 1).contiguous())                                # [nb, B, blk, ws]
        fl = (F.pad(filters, (0, 0, 0, npad)).reshape(B, -1, blk, filter_size)
              .transpose(0, 1).contiguous())                                # [nb, B, blk, fsz]
        nb = fl.size(0)

        blk_n = L + (blk - 1) * fs
        idx = (torch.arange(blk, device=x.device).unsqueeze(1) * fs
               + torch.arange(L, device=x.device).unsqueeze(0)).reshape(1, 1, -1)

        def body(carry, xs):
            fr, fi = xs[0], xs[1]                                    # 各 [B, blk, *]
            # FFT は dim=-1（フレーム内）だけなので、フレーム方向に切っても**各フレームの
            # FFT は 1 ビットも変わらない**。非分割版との差は OLA の加算順序だけから出る。
            # 実測（use_hann 2 通り × B∈{1,2,3} × フレーム数 6 通り）:
            #   frame_block=128（既定） … **ビット一致**（36/36）
            #   frame_block=32 / 64 / 256 … -153 / -158 / -161dB
            # なお fold(Col2Im) でも書ける。ONNX は 7.5MB→3.3MB と小さくなるが、
            # blk=128 以外で加算順序が変わる（-157dB）ので、既定のビット一致を優先して採らない。
            fz = torch.fft.ifft(torch.fft.fft(fr, n=L, dim=-1)
                                * torch.fft.fft(fi, n=L, dim=-1),
                                n=L, dim=-1).real                    # [B, blk, L]
            nB = fz.size(0)
            buf = torch.zeros(nB, 1, blk_n, dtype=fz.dtype, device=fz.device)
            buf = buf.scatter_add(2, idx.expand(nB, 1, -1), fz.reshape(nB, 1, -1))
            return carry.clone(), (buf.reshape(nB, blk_n),)          # [B, blk_n]

        c0 = torch.zeros(1, dtype=x.dtype, device=x.device)
        _, (ys,) = _scan(body, c0, (fx, fl))                         # [nb, B, blk_n]

        # === ブロック間の OLA（ブロック i は i*blk*fs から blk_n サンプル）===
        total = (nb - 1) * (blk * fs) + blk_n
        idx2 = (torch.arange(nb, device=x.device).unsqueeze(1) * (blk * fs)
                + torch.arange(blk_n, device=x.device).unsqueeze(0)).reshape(1, 1, -1)
        y = torch.zeros(B, 1, total, dtype=x.dtype, device=x.device)
        y = y.scatter_add(2, idx2.expand(B, 1, -1),
                          ys.transpose(0, 1).reshape(B, 1, -1))      # [B, nb*blk_n]
        start = filter_size // 2 + (fs if self.use_hann else 0)
        return y.narrow(2, start, n_sample)

    def _forward_whole(self, x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
        n_sample = x.size(-1)
        filter_size = filters.size(-1)

        # === Step 1: frame_signal (inline) ===
        if self.use_hann:
            # Same as eager hann_ltv_fir: pad x by [frame_size, frame_size-1], slice
            # length-2*frame_size windows at hop=frame_size (50% overlap), then multiply the
            # periodic Hann window. The OLA hop stays frame_size (same as square), so a
            # constant/identity filter is transparent in the middle (COLA=1); frame-boundary
            # steps are cross-faded and the filtered spillover is tapered at the edges
            # (this is what removes the high-pitch 1-period dropout).
            wl = self.frame_size
            wr = self.frame_size - 1
            x = torch.nn.functional.pad(x, [wl, wr])
            ws = self.frame_size * 2
            x_2d = x.unsqueeze(-1)
            framed_x = torch.nn.functional.unfold(
                x_2d, kernel_size=(ws, 1), stride=(self.frame_size, 1))   # [B, ws, n_frame]
            framed_x = framed_x.transpose(1, 2)                            # [B, n_frame, ws]
            framed_x = framed_x * self.hann_win.to(framed_x.dtype)
        else:
            # Legacy square OLA: non-overlapping frame_size frames.
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
        # square: filter_size//2 (same as eager ltv_fir).
        # hann: same as eager hann_ltv_fir's [filter_size//2 + wl : ...] (wl = frame_size left pad).
        start_slice = filter_size // 2 + (self.frame_size if self.use_hann else 0)
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
