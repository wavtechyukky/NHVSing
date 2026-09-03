'''
Copyright 2021 Zhijun Liu

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import torch
from torch import Tensor
from torch.fft import fft, ifft, fftshift
from torch.nn.functional import pad
from functools import lru_cache
from math import pi
import math
from typing import Union, Optional, Iterable, Tuple
from librosa.filters import mel as mel_fn


def freq_multiplier(n_harmonic: int, device: torch.device) -> Tensor:
    """Generate the frequency multiplier [[[1], [2], ..., [n_harmonic]]]
    This function is LRU cached.
    Returns:
        multiplier: [1, n_harmonic, 1]
    """
    return torch.as_tensor(
        [[1.0 * k for k in range(1, n_harmonic + 1)]]
    ).reshape(1, n_harmonic, 1).to(device)





@torch.jit.script
def generate_impulse_train(
        f0_t: Tensor, n_harmonic: int,
        sampling_rate: float) -> Tensor:

    fm = freq_multiplier(n_harmonic, f0_t.device)
    weight_map = torch.sigmoid(-(fm * f0_t - sampling_rate / 2.0))
    # 位相は float64 で cycles 単位に累積し、mod 1.0 で [0,1) へ折り返してから float32 の cos に渡す。
    # float32 の cumsum はフル長尺(15s+)で累積値が ~1e8 に達し ULP≈10 まで粗くなって位相が劣化 →
    # 倍音間に滲み(サイドバンド)が出る。学習は短 crop(cumsum 小)で鋭いのに eval/配備はフル長尺で劣化する
    # train/deploy 不一致の原因だった。float64 で累積し折り返して精度を確保、cos は [0,2π) の小さい値
    # なので float32 で十分(ONNX Runtime は Cos(double) 非対応 → ONNX 側と同一計算に統一)。
    # ★スカラ (1/sr) を cumsum に直接掛けてはいけない: ONNX オプティマイザが cumsum の中へ融合し
    #   cumsum(f0/sr)(=小値和)になって runtime 間で加算順序依存の非決定性が出る。cumsum(f0)(大値)は
    #   runtime 間でビット一致するので、fm/sr を float64 テンソルで先に作り cumsum の「後」にテンソル乗算する。
    # v3.1 の ckpt はこの励起で学習(旧 v3 以前の ckpt もそのまま動く: 変わるのは励起の位相精度のみ)。
    cycles = f0_t.double().cumsum(dim=-1) * (fm.double() / sampling_rate)
    w0_map_cum = ((cycles - torch.floor(cycles)) * (2.0 * math.pi)).to(f0_t.dtype)
    source = torch.sum(torch.cos(w0_map_cum) * weight_map, dim=1, keepdim=True)
    return source * 0.01




@torch.jit.script
def generate_impulse_train_v3beta(
        f0_t: Tensor, n_harmonic: int,
        sampling_rate: float) -> Tensor:
    """v3beta 励起(v3 以前の実験・未採用): Nyquist soft mask + 1/k spectral tilt(-6dB/oct)。

    既存 generate_impulse_train は全高調波を均等振幅(white)で足し Nyquist soft mask のみ
    → 高周波 harmonic 過剰(ビリビリの正体)+ Nyquist 直下 alias。v3beta は:
      ① Nyquist soft mask(sigmoid)で alias 抑制。hard cut は L2 正規化が cut で失った
         高周波エネルギーを低周波に押し付け低周波過剰になるため不採用(ユーザー指摘)。
         Nyquist 超えは tilt(1/k)で <1% なので soft mask + tilt で十分抑制。
      ② harmonic 番号 k で 1/k 減衰(-6dB/oct)。声門波 source(-12dB/oct)+唇放射(+6dB/oct)
         = 声の最終 spectral tilt -6dB/oct を励起に物理的に付与(impulse が white なのを是正)。
      ③ L2 正規化で tilt で下がったエネルギーを white(soft mask)相当に補う(ccep が無理に増幅しない)。
    ccep は残りの声道フォルマント整形を担う。後方互換のため既存関数は不変(これは新関数)。
    """
    k = freq_multiplier(n_harmonic, f0_t.device)             # [1, n_harmonic, 1] = [1,2,...,n]
    f0_map = k * f0_t
    mask = torch.sigmoid(-(f0_map - sampling_rate / 2.0))    # Nyquist soft mask(hard cut でない)
    tilt_w = mask / k                                         # 1/k tilt(-6dB/oct)
    # エネルギー保存(ユーザー指摘): tilt+Nyquist cut でエネルギーが激減(peak 1/22)し ccep が
    # 無理に増幅する羽目になる。tilt は「相対バランス(高周波減衰)」だけにし、各時刻の総エネルギー
    # (L2 norm = RMS)は white(mask)相当に正規化する。peak は tilt で鈍る(=まろやか)がエネルギー保存。
    norm_ref = torch.sqrt((mask ** 2).sum(dim=1, keepdim=True) + 1e-8)    # white の L2
    norm_tilt = torch.sqrt((tilt_w ** 2).sum(dim=1, keepdim=True) + 1e-8)  # tilt の L2
    weight_map = tilt_w * (norm_ref / norm_tilt)             # L2 を white に合わせる
    w0_map_cum = (
        f0_t.cumsum(dim=-1) * 2.0 * math.pi / sampling_rate * k
    )
    source = torch.sum(torch.cos(w0_map_cum) * weight_map, dim=1, keepdim=True)
    return source * 0.01


def envelope_loss(y: Tensor, y_hat: Tensor, kernel_size: int, stride: int,
                  directional: bool = False) -> Tensor:
    """Amplitude envelope MAE loss (RefineGAN §2.5.1, eq.2).
    Extracts upper and lower envelopes via 1D max-pooling and computes MAE.
    Penalises jagged/rough amplitude envelopes in the generated waveform.

    directional=False(既定): 中央パディング max-pool(従来挙動, ビット不変)。
    directional=True: 上下包絡を min(後ろ向き max, 前向き max) で取る。中央 max は tall peak を
        ±kernel/2 に膨張(dilation)させ、高→低の急降下後 ~kernel/2 サンプルは包絡が高いまま
        張り付く盲点を作る。min 方式はこの膨張をゼロにして急降下を正確に追従(crest 感度は
        維持)。返り値は同じ upper_MAE + lower_MAE の2項和なので loss のオーダーは不変
        (min で1本に合成してから MAE を取る=1/2 補正は不要)。

    Args:
        y, y_hat: [n_batch, 1, n_sample]
        kernel_size: max-pool window (= 2 * hop_size recommended)
        stride: max-pool stride (= hop_size recommended)
        directional: True で min(causal, anticausal) 方式(膨張/盲点を除去)
    Returns:
        loss: scalar
    """
    import torch.nn.functional as F
    K = kernel_size

    def _env(sig):
        if not directional:
            return F.max_pool1d(sig, K, stride=stride, padding=K // 2)
        # min(過去K の max, 未来K の max): 膨張ゼロで急降下を追従。左右で K-1 パディング
        causal = F.max_pool1d(F.pad(sig, (K - 1, 0), value=-1e9), K, stride=stride)
        anti   = F.max_pool1d(F.pad(sig, (0, K - 1), value=-1e9), K, stride=stride)
        t = min(causal.size(-1), anti.size(-1))
        return torch.minimum(causal[..., :t], anti[..., :t])

    upper_y    = _env(y)
    upper_yhat = _env(y_hat)
    lower_y    = _env(-y)
    lower_yhat = _env(-y_hat)
    # trim to same length to guard against rounding differences
    t = min(upper_y.size(-1), upper_yhat.size(-1))
    return ((upper_y[..., :t] - upper_yhat[..., :t]).abs().mean() +
            (lower_y[..., :t] - lower_yhat[..., :t]).abs().mean())


def stft_loss(
    x: Tensor, y: Tensor, fft_lengths: Iterable[int], window_lengths: Iterable[int], hop_lengths: Iterable[int], 
    loss_scale_type: str) -> Tensor:
    """Compute STFTLoss. The length of provided configuration lists should be
    the same.
    Args:
        x, y: [n_batch, 1, n_sample]
    Returns:
        loss: []
    """
    x, y = x.squeeze(1), y.squeeze(1)
    loss = 0.0
    n_res = 0
    batch_size = x.size(0)
    z = torch.cat([x, y], dim=0)  # shape: [2 x Batch, T]
    for fft_length, window_length, hop_length in zip(fft_lengths, window_lengths, hop_lengths):
        n_res += 1
        window = torch.hann_window(window_length, device=x.device)
        Z = torch.stft(z, fft_length, hop_length, window_length, window, return_complex=False)
        # shape: [2 x Batch, Frame, 2]
        SquareZ = Z.pow(2).sum(dim=-1) + 1e-10  # shape: [2 x Batch, Frame]
        SquareX, SquareY = SquareZ.split(batch_size, dim=0)
        MagZ = SquareZ.sqrt()
        MagX, MagY = MagZ.split(batch_size, dim=0)
        if loss_scale_type == "log_linear":
            loss += (MagX - MagY).abs().mean() + \
                    0.5 * (SquareX.log() - SquareY.log()).abs().mean()
        elif loss_scale_type == "linear":
            loss += (MagX - MagY).abs().mean()
        elif loss_scale_type == "nsf":
            # NSF-HiFiGAN(SingingVocoders)流: spectral convergence + log magnitude。
            # SC = ||MagY-MagX||_F / ||MagY||_F(相対フロベニウス, 大バンドに依存しない),
            # logmag = |log MagX - log MagY|.mean()。最後に解像度で平均(下の n_res 除算)。
            sc = torch.norm(MagY - MagX, p='fro') / (torch.norm(MagY, p='fro') + 1e-10)
            logmag = (MagX.log() - MagY.log()).abs().mean()
            loss = loss + sc + logmag
        elif isinstance(loss_scale_type, float):
            loss += (MagX - MagY).abs().mean() + \
                    0.5 * loss_scale_type * \
                          (SquareX.log() - SquareY.log()).abs().mean()
        else:
            raise RuntimeError(f"Unrecognized STFT loss scale type.")
    if loss_scale_type == "nsf":
        loss = loss / max(1, n_res)   # NSF 流は解像度で平均(log_linear は従来通り合算)
    return loss


@lru_cache(maxsize=4)
def _mel_basis_cached(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float):
    mb = mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    return torch.from_numpy(mb).float()


def wav_to_mel_torch(wav: Tensor, sr: int, n_fft: int, hop: int, win: int,
                     n_mels: int, fmin: float, fmax: float) -> Tensor:
    """波形 → diffsinger ln-mel。preprocess_singing_db.wav_to_mel と一致(center=False +
    reflect pad (n_fft-hop)//2, hann, ln clamp 1e-5)。GPU/微分可能。mel_loss と
    pitch_augment(リサンプル波形の mel 再計算)の単一情報源。
    wav: [B, T] or [B, 1, T] → 返り [B, n_mels, frame]。"""
    if wav.dim() == 3:
        wav = wav.squeeze(1)
    window = torch.hann_window(win, device=wav.device)
    pad = (n_fft - hop) // 2
    w = torch.nn.functional.pad(wav.unsqueeze(1), (pad, pad), mode='reflect').squeeze(1)
    S = torch.stft(w, n_fft, hop, win, window, center=False, return_complex=True)
    # eps を入れて padding 部(0→stft=0)の 0/0 微分(勾配 NaN)を回避(mel_loss 由来)。
    mag = torch.sqrt(S.real ** 2 + S.imag ** 2 + 1e-7)   # [B, freq, frame]
    mb = _mel_basis_cached(sr, n_fft, n_mels, fmin, fmax).to(wav.device)
    return torch.log(torch.clamp(torch.matmul(mb, mag), min=1e-5))  # [B, n_mels, frame] (ln)


def mel_loss(wav_gen: Tensor, target_mel: Tensor, sr: int, n_fft: int, hop: int,
             win: int, n_mels: int, fmin: float, fmax: float,
             mask: Tensor = None) -> Tensor:
    """生成波形の diffsinger mel と target_mel(入力 log_melspc)の L1 loss。
    mel 計算は wav_to_mel_torch(preprocess と一致)に委譲。
    NSF-HiFiGAN/SingingVocoders 流の主再構成 loss。

    Args:
        wav_gen: [B, 1, T] 生成波形(target wav と同スケール)
        target_mel: [B, T_frames, n_mels] or [B, n_mels, T_frames] 入力 log_melspc
    Returns:
        loss: []  (L1, mel 値域 -6〜1.5 スケール)
    """
    mel = wav_to_mel_torch(wav_gen, sr, n_fft, hop, win, n_mels, fmin, fmax)  # [B, n_mels, frame]
    if target_mel.size(1) != n_mels:          # [B, T, n_mels] -> [B, n_mels, T]
        target_mel = target_mel.transpose(1, 2)
    T = min(mel.size(-1), target_mel.size(-1))
    diff = (mel[..., :T] - target_mel[..., :T]).abs()  # [B, n_mels, T]
    if mask is not None:
        # mask: [B, T_frames] (True=padding)。padding 部(est=0→mel -6 vs target 0)を
        # 含めると固定 loss=6 が乗り勾配を汚すので、valid フレームのみで平均する。
        valid = (~mask[..., :T]).to(diff.dtype).unsqueeze(1)   # [B, 1, T]
        return (diff * valid).sum() / valid.sum().clamp(min=1.0) / diff.size(1)
    return diff.mean()


@torch.jit.script
def reshape_zeros_like(x: Tensor, dim: int, length: int) -> Tensor:
    """Return torch.zeros_like(x), while change shape of the `dim` dimension."""
    shape = list(x.shape)
    shape[dim] = length
    return torch.zeros(shape, dtype=x.dtype, device=x.device)

@torch.jit.script
def fftpad(x: Tensor, padding: int) -> Tensor:
    """Insert zeros in x with the following pattern:
        [x, y] => [x, 0, y];
        [x, y, z] => [x, y, 0, z];
    """
    size = x.size(-1)
    half = size // 2
    first_half = torch.narrow(x, -1, 0, size - half)
    second_half = torch.narrow(x, -1, size - half, half)
    zeros = reshape_zeros_like(x, -1, padding)
    return torch.cat([first_half, zeros, second_half], dim=-1)

@torch.jit.script
def complex_cepstrum_to_fft(
    ccep: Tensor, fft_size: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert complex cepstrums to its corresponding fft.
    Args:
        ccep: ccep.size(dim) is ccep_size.
        fft_size: Target fft_size, should be substantially larger than
            ccep_size to avoid time wraping error.
        dim: Dimension storing the complex cepstrum. Defaults to -1.
    Returns:
        X: FFT of the impulse response x
        log(|X|): Log scale magnitude
            responses at FFT sampling points.
        arg(X): Phase responses at FFT samples.
    """
    ccep_size = ccep.size(-1)
    assert fft_size >= ccep_size, "FFT size should be greater than CCep size."
    ccep = fftpad(ccep, fft_size - ccep_size)
    X_hat = torch.fft.fft(ccep, dim=-1)  # [fft_size@dim]
    log_magnitude_responses = X_hat.real
    phase_responses = X_hat.imag
    magnitude_responses = log_magnitude_responses.clamp(max=10.0).exp()
    X_real = magnitude_responses * torch.cos(phase_responses)
    X_imag = magnitude_responses * torch.sin(phase_responses)
    X = torch.complex(X_real, X_imag)
    return X, log_magnitude_responses, phase_responses

@torch.jit.script
def complex_cepstrum_to_imp(
    ccep: Tensor, fft_size: int) -> Tensor:
    """Convert complex cepstrums to corresponding magnitude responses,
    phase responses, and impulse responses.
    Args:
        ccep: [ccep_size@dim]
        fft_size: Target fft_size, should be greater than ccep_size.
    Returns:
        impulse_responses: [fft_size@dim]. Approximated time wrapped
            impulse responses.
    """
    X, _, _ = complex_cepstrum_to_fft(ccep, fft_size)
    x = torch.fft.ifft(X, dim=-1).real  # [fft_size@dim]
    return x


# ── 最小位相ヘルパ(著者 dsp/cepstrum.py より移植。dim=-1 固定、eager)─────────










@torch.jit.script
def frame_signal(x: Tensor, frame_size: int, frame_shift: int) -> Tensor:
    """Frame a signal with given frame size and frame shift.
    The first frame starts from 0. When the signal x is not long enough
    to fill a frame, the sampling points are dropped.
    You should pad appropriately to preserve these sampling points.
    NOTE: n_sample >= frame_size must be true

    Args:
        x: [n_batch, 1, n_sample]

    Returns:
        framed_x (Tensor):
        [n_batch, frame_size, (n_sample - frame_size + frame_shift) // frame_shift]
    """
    return torch.nn.functional.unfold(
        x.unsqueeze(-1), kernel_size=(frame_size, 1), stride=(frame_shift, 1)
    )

@torch.jit.script
def time_corr(x: Tensor, y: Tensor) -> Tensor:
    """Compute batched 1D correlation of signal x and signal y in the
    time domain.

    Shapes:
        shape of x and y must be broadcastable except for the last dimension.
        x: [..., nx]
        y: [..., ny]
        returns: [..., nx + ny - 1]

    Args:
        flip (bool, optional): Defaults to False. When set to True,
        framed_y is flipped in time.
    """

    # Implemented with dark magic. Must broadcast in the first step.
    #x = x.view(*x.shape, 1)
    x = x.unsqueeze(-1)
    #y = y.view(*y.shape[:-1], 1, y.size(-1))
    y = y.unsqueeze(-2)
    x, y = torch.broadcast_tensors(x, y)
    x = x[..., 0]
    y = y[..., 0, :]
    # End reshaping.

    nx = x.size(-1)
    batch_size = x.shape[:-1]
    multiplied_batch_size = 1
    for sz in batch_size:
        multiplied_batch_size *= sz
    ny = y.size(-1)
    y = y.flip(-1)
    x = x.reshape(1, multiplied_batch_size, nx)  # (1, *batch_size, nx)
    x = pad(x, [ny - 1, ny - 1], mode="constant")
    y = y.reshape(multiplied_batch_size, 1, ny)  # (*batch_size, 1, ny)
    framed_z = torch.nn.functional.conv1d(x, y, groups=multiplied_batch_size)
    framed_z = framed_z.reshape(batch_size + (framed_z.size(-1),))

    return framed_z

@torch.jit.script
def unframe_signal(x: Tensor, frame_shift: int) -> Tensor:
    """This function uses overlap and add to unframe a framed signal.
    Shapes:
        x: [n_batch, frame_size, n_frame]
        returns: [n_batch, 1, frame_size + (n_frame - 1) * frame_shift]
    """
    _, frame_size, n_frame = x.shape
    n_sample = frame_size + (n_frame - 1) * frame_shift
    return torch.nn.functional.fold(
        x,
        output_size=(n_sample, 1),
        kernel_size=(frame_size, 1),
        stride=(frame_shift, 1),
    ).squeeze(-1)

@torch.jit.script
def fft_multiply(x: Tensor, y: Tensor, flip: bool) -> Tensor:
    """周波数領域の積。flip=False で畳み込み、True で相関(著者 dsp/conv.py 準拠)。"""
    fft_x = fft(x, dim=-1)
    fft_y = fft(y, dim=-1)
    flip_sign = -1.0 if flip else 1.0
    fft_z_real = fft_x.real * fft_y.real - flip_sign * fft_x.imag * fft_y.imag
    fft_z_imag = flip_sign * fft_x.real * fft_y.imag + fft_x.imag * fft_y.real
    fft_z = torch.complex(fft_z_real, fft_z_imag)
    return ifft(fft_z, dim=-1).real

@torch.jit.script
def fft_corr(x: Tensor, y: Tensor, flip: bool) -> Tensor:
    """time_corr の FFT 等価版(エイリアス回避 pad 込み)。flip=True で畳み込み。
    fft_size は次の 2 の冪(著者 default の 2**ceil(log2(nx+ny-1)) と一致)。"""
    nx = x.size(-1)
    ny = y.size(-1)
    fft_size = 1
    while fft_size < nx + ny - 1:
        fft_size = fft_size * 2
    padded_y = pad(y, [0, fft_size - ny])
    if flip:
        padded_x = pad(x, [0, fft_size - nx])
    else:
        padded_x = pad(x, [ny - 1, fft_size - nx - ny + 1])
    z = fft_multiply(padded_x, padded_y, not flip)
    return z[..., :nx + ny - 1]

@torch.jit.script
def _framewise_corr_ola(
    framed_x: Tensor, framed_y: Tensor, frame_shift: int,
    method: str = "fft") -> Tensor:
    """Computes time-varying 1D correlation in time or frequency domain.
    This function receives framed signal x. It filters framed signal x,
    and unframe the signal with OLA. This function preserves all
    non-zero entries, which is commonly referred to as the 'valid'
    padding.

    Args:
        framed_x: [n_batch, n_frame, nx]
        framed_y: [n_batch, n_frame, ny]
        method (str, optional): Either 'time' or 'fft'.
        flip (bool, optional): Defaults to False. When set to True, the
        signal y is flipped in time in each frame.

    Returns:
        z: [n_batch, 1, (nx + ny - 1) + (n_frame - 1) * frame_shift]
    """
    if method == "fft":
        # FFT 等価実装(著者 dsp/conv.py の fft_corr, flip=True=畳み込み)。
        # time_corr(grouped conv1d)とビット等価で CPU 約8倍速。
        framed_z = fft_corr(framed_x, framed_y, True)
    else:
        framed_z = time_corr(framed_x, framed_y)
    output = unframe_signal(framed_z.transpose(-1, -2), frame_shift)

    return output

@torch.jit.script
def ltv_fir(
    x: Tensor, filters: Tensor, frame_size: int, method: str = "fft"
) -> Tensor:
    """Linear time-varying FIR filter with a square OLA window.
    Notice that this implements a convolution rather than a correlation.

    Args:
        x: [n_batch, 1, n_sample]
        filters: [n_batch, n_frame, filter_size].
                 Filter FIRs. FIRs in each frame are assumed to be stored
                 as time wrapped signals. Function fftshift convert it
                 back to continuous time order.
        frame_size: The frame size in sampling points.
        method: Either 'time' or 'fft'. Defaults to 'fft'.

    Returns: [n_batch, 1, n_sample]
             n_sample: n_frame * frame_size
    """
    filter_size = filters.size(-1)
    n_sample = x.size(-1)
    framed_x = frame_signal(x, frame_size, frame_size).transpose(-1, -2)
    # [n_batch, n_frame, frame_size]
    filters = fftshift(filters, dim=-1)
    y = _framewise_corr_ola(framed_x, filters, frame_size, method)
    striped_y = y[..., filter_size // 2: n_sample + filter_size // 2]
    return striped_y





