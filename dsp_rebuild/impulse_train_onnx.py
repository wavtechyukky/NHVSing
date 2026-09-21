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

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from typing import Optional

try:                                    # torch>=2.6 相当。無ければ非分割へフォールバック
    from torch._higher_order_ops.scan import scan as _scan
except Exception:                       # noqa: BLE001
    _scan = None


def _tracing() -> bool:
    """ONNX エクスポート等のトレース中か（dynamo / torch.export / TorchScript trace）。

    トレース中は Python の for が展開されて長さが固定されるので scan 版へ、eager では Python ループへ分岐する。
    """
    comp = getattr(torch, 'compiler', None)
    for name in ('is_compiling', 'is_exporting'):
        fn = getattr(comp, name, None)
        if fn is not None and fn():
            return True
    return bool(torch.jit.is_tracing())


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
    
    def __init__(self, n_harmonic: int, sampling_rate: float, block: int = 0):
        super().__init__()
        self.n_harmonic = n_harmonic
        self.sampling_rate = float(sampling_rate)
        # 素の実装は cos(w0) を [B, n_harmonic, n_sample] で一度に作るため、長さに比例して
        # 巨大化する（n_harmonic=200・44.1kHz で 35MB/秒）。
        #
        # eager 実行（推論・bench）… 時間方向のブロック分割を **Python の for** で回す。
        #                     ピークは n_harmonic×block の定数で、一括版と**ビット一致**
        #                     （要素ごとの式も dim=1 の総和順序も同じ）。
        # ONNX エクスポート（トレース中）… Python の for は展開されて長さが固定されるので、
        #                     `torch._higher_order_ops.scan`（ONNX の Scan）で動的回数にする。
        #                     どちらの向きに回すかを mode で選ぶ。
        #   'harmonic'（既定）… **倍音方向のループ**。[B,1,n] の累算器に n_harmonic 回足す。
        #                     中間が [B,1,n] だけになるのでピークは O(n)＝入力長にしか依らない。
        #                     V3X のエクスポートもこれでないと通らない（'time' は ConversionError）。
        #                     総和が逐次加算になるので一括版とは float32 の丸めぶん違う。
        #   'time'        … **時間方向のブロック分割**を scan で。ピークが n_harmonic×block。
        # 'off'         … 対策なし（従来の一括実装）。eager でもエクスポートでも一括。対照実験用。
        self.mode = os.environ.get('NHV_IMP_MODE', 'harmonic')
        # 時間ブロック長 [samples]。eager と 'time' が使う。'harmonic' / 'off' では参照しない。
        self.block = int(block) if block else int(os.environ.get('NHV_IMP_BLOCK', 16384))
        
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

        # Reference implementation (dsp.py::generate_impulse_train):
        #   f0_map = freq_multiplier(n_harmonic, f0_t.device) * f0_t
        #   weight_map = torch.sigmoid(-(f0_map - sampling_rate / 2.0))
        #   w0_map_cum = f0_t.cumsum(dim=-1) * 2π / sampling_rate * freq_multiplier(...)
        #   source = sum_h cos(w0_map_cum) * weight_map
        # 位相は float64 で cycles を累積 → mod 1.0 で [0,1) へ折り返し → float32 で ×2π → cos。
        # 旧実装は float64 cumsum の直後に float32 へ戻す no-op(pure float32 と 1bit 一致)で、累積値
        # (フル長尺で ~1e8, ULP≈10)の丸めにより位相が劣化し倍音間に滲みが出ていた。ONNX Runtime は
        # Cos(double) 非実装のため、折り返して [0,2π) の小さい値にしてから float32 cos に渡す
        # (dsp.py::generate_impulse_train と同一計算。位相精度は float64 品質を保つ)。
        # ★fm/sr は前計算テンソル mult_over_sr を cumsum(f0) の後に掛ける(スカラ 1/sr を cumsum に
        #   掛けると ONNX が cumsum(f0/sr) に融合し runtime 間で ~0.047 ズレる。cumsum(f0) はビット一致)。
        # 位相 cumsum は [B, 1, n_sample] のまま全長で持つ（1 秒あたり 0.2MB と安い）。
        # 重いのは「×n_harmonic 本」の展開（weight_map / cos）なので、そこだけ分割する。
        phase = f0_t.to(torch.float64).cumsum(dim=-1)                     # [B, 1, n_sample]
        tracing = _tracing()
        if self.mode == 'off' or (tracing and _scan is None):
            return self._one_shot(f0_t, phase, multiplier) * 0.01
        if tracing and self.mode == 'harmonic':
            return self._loop_harmonics(f0_t, phase) * 0.01
        if self.block <= 0:                       # ブロック長が無い＝対策なし
            return self._one_shot(f0_t, phase, multiplier) * 0.01
        if tracing:                               # 'time'
            return self._blocked(f0_t, phase, multiplier) * 0.01
        return self._blocked_eager(f0_t, phase, multiplier) * 0.01

    def _one_shot(self, f0_t: Tensor, phase: Tensor, multiplier: Tensor) -> Tensor:
        """従来の一括実装。[B, n_harmonic, n_sample] を展開する（長さに比例してメモリを食う）。"""
        weight_map = torch.sigmoid(-(multiplier * f0_t - self.sampling_rate / 2.0))
        cycles = phase * self.mult_over_sr.to(f0_t.device)
        w0_map_cum = ((cycles - torch.floor(cycles)) * (2.0 * math.pi)).to(torch.float32)
        return torch.sum(torch.cos(w0_map_cum) * weight_map, dim=1, keepdim=True)

    def _blocked_eager(self, f0_t: Tensor, phase: Tensor, multiplier: Tensor) -> Tensor:
        """時間ブロック × Python の for（eager 専用）。一括版と**ビット一致**。

        各ブロックで `_one_shot` と同じ式を同じ形 [B, n_harmonic, blk] で計算し dim=1 で総和するので、
        要素ごとの演算も総和の順序も一括版と同じ。ピークは n_harmonic×block（既定 16384 で ~40MB）。
        scan 版と違い展開の中間を一切持ち越さないので、eager では最速かつ最小メモリ。
        """
        n = f0_t.size(-1)
        mos = self.mult_over_sr.to(f0_t.device)
        sr_half = self.sampling_rate / 2.0
        out = torch.empty_like(f0_t)
        for s in range(0, n, self.block):
            e = min(n, s + self.block)
            weight_map = torch.sigmoid(-(multiplier * f0_t[..., s:e] - sr_half))
            cycles = phase[..., s:e] * mos
            w0 = ((cycles - torch.floor(cycles)) * (2.0 * math.pi)).to(torch.float32)
            out[..., s:e] = torch.sum(torch.cos(w0) * weight_map, dim=1, keepdim=True)
        return out

    def _loop_harmonics(self, f0_t: Tensor, phase: Tensor) -> Tensor:
        """倍音方向のループ。**累算器 [B,1,n] だけを持ち回り、スキャン出力は捨て値**。

        ONNX の Loop/Scan は loop-carried dependency のメモリを累算器サイズに固定するので、
        ピークが O(n) になる（= dsp/harmonic_excitation の `_harm_sum_loop` と同じ特性）。
        `[B, n_harmonic, n]` を一度も作らない代わりに n_harmonic 回走査するので遅い。

        総和が逐次加算になるため、非分割版（`torch.sum` の pairwise 加算）とはビット一致しない。
        """
        H = int(self.n_harmonic)
        sr = float(self.sampling_rate)
        sr_half = sr / 2.0
        dev = f0_t.device
        ks = torch.arange(1, H + 1, dtype=torch.float64, device=dev).reshape(H, 1)   # [H, 1]

        # 累算器は **1 次元**にする。[B,1,n] のように size 1 の次元があると stride が一意に
        # 決まらず（(0,0,1) にも (n,n,1) にもなる）、scan が「init と carry の metadata 不一致」
        # で落ちる。1 次元なら stride は (1,) しかない。
        ph1 = phase.reshape(-1)                            # [n] float64
        f01 = f0_t.reshape(-1)                             # [n] float32

        def body(carry, xs):
            k = xs[0]                                      # [1] float64
            cyc = ph1 * (k / sr)                           # [n]
            w0 = ((cyc - torch.floor(cyc)) * (2.0 * math.pi)).to(torch.float32)
            wm = torch.sigmoid(-(k.to(torch.float32) * f01 - sr_half))
            # スキャン出力はスカラ 1 個（H 個ぶんしか溜まらない＝実質ゼロ）
            return carry + torch.cos(w0) * wm, (torch.zeros(1, dtype=torch.float32, device=dev),)

        acc0 = torch.zeros_like(f01)
        acc, _ = _scan(body, acc0, (ks,))
        return acc.reshape(f0_t.shape)

    def _blocked(self, f0_t: Tensor, phase: Tensor, multiplier: Tensor) -> Tensor:
        """時間ブロック × scan。非分割版と **ビット一致**（dim=1 の総和順序が同じ）。

        ONNX では Python の for がトレース時に展開され長さが固定されてしまうため、
        `torch._higher_order_ops.scan`（ONNX の Scan 演算子）で動的回数のループにする。
        """
        # ★係数は **body の中で定数から作り直す**。register_buffer をクロージャで捕らえると
        #   scan の追加入力になり、torch.export が形まで記号化して [1,1,blk] との broadcast に
        #   失敗する（scan が 2 つ以上あるグラフで顕在化）。arange から作れば形はリテラル。
        H = int(self.n_harmonic)
        sr = float(self.sampling_rate)
        dev = f0_t.device
        blk = self.block
        n = f0_t.size(-1)
        pad = (-n) % blk                                # ブロック長の倍数へ 0 詰め（末尾は捨てる）
        ph = F.pad(phase, (0, pad)).reshape(-1, 1, blk)                 # [nb, 1, blk]
        f0p = F.pad(f0_t, (0, pad)).reshape(-1, 1, blk)                 # [nb, 1, blk]
        sr_half = sr / 2.0

        def body(carry, xs):
            p, f = xs[0], xs[1]                          # 各 [1, blk]
            k64 = torch.arange(1, H + 1, dtype=torch.float64, device=dev).reshape(1, H, 1)
            k32 = torch.arange(1, H + 1, dtype=torch.float32, device=dev).reshape(1, H, 1)
            cyc = p.unsqueeze(0) * (k64 / sr)            # [1, n_harmonic, blk]
            w0 = ((cyc - torch.floor(cyc)) * (2.0 * math.pi)).to(torch.float32)
            wm = torch.sigmoid(-(k32 * f.unsqueeze(0) - sr_half))
            y = torch.sum(torch.cos(w0) * wm, dim=1, keepdim=True)      # [1, 1, blk]
            return carry.clone(), (y.squeeze(0),)        # carry は使わないが scan が要求する

        c0 = torch.zeros(1, dtype=torch.float32, device=f0_t.device)
        _, (ys,) = _scan(body, c0, (ph, f0p))            # ys: [nb, 1, blk]
        return ys.reshape(f0_t.size(0), 1, -1).narrow(2, 0, n)


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
