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
from torch.fft import fft, fftshift
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



def freq_antialias_mask(sampling_rate: Union[int, float], freq_tensor: Tensor,
                        hard_boundary: Optional[bool] = True) -> Tensor:
    """Return a harmonic amplitude mask that silence any harmonics above
    sampling_rate / 2.
    Args:
        sampling_rate (Number): The sampling rate in Hertz.
        freq_tensor (Tensor): Tensor of any shape (...), values in Hertz.
    Returns:
        mask (Tensor): Mask tensor of the same shape as freq_tensor.
        mask[freq_tensor > fs / 2] are zeroed.
    """
    if hard_boundary:
        return (freq_tensor < sampling_rate / 2.0).float()
    else:
        return torch.sigmoid(-(freq_tensor - sampling_rate / 2.0))

def harmonic_amplitudes_to_signal(f0_t: Tensor, harmonic_amplitudes_t: Tensor,
                                  sampling_rate: int, min_f0: float) -> Tensor:
    """Generate harmonic signal from given frequency and harmonic amplitudes.
    The phase of sinusoids are assumed to be all zero. The periodic function
    used is SINE.

    Args:
        f0_t: [n_batch, 1, n_sample]. Fundamental frequency per
            sampling point in Hertz.
        harmonic_amplitudes_t: [n_batch, n_harmonic, n_sample].
            Harmonic amplitudes per sampling point.
        sampling_rate: Sampling rate in Hertz.
        min_f0: Minimum f0 to accept. All f0_t below min_f0 are ignored.

    Returns:
        signal: [n_batch, 1, n_sample]. Sum of sinusoids with given harmonic
            amplitudes and fundamental frequencies.
    """
    _, n_harmonic, _ = harmonic_amplitudes_t.shape
    f0_map = freq_multiplier(n_harmonic, f0_t.device) * f0_t
    # [n_batch, n_harmonic, n_sample]
    weight_map = (
        freq_antialias_mask(sampling_rate, f0_map) * harmonic_amplitudes_t
    )  # [n_batch, n_harmonic, n_sample]
    f0_map_cum = f0_t.cumsum(dim=-1) * freq_multiplier(
        n_harmonic, f0_t.device
    )  # [n_batch, n_harmonic, n_sample]
    w0_map_cum = f0_map_cum * 2.0 * pi / sampling_rate
    source = torch.sum(
        torch.sin(w0_map_cum) * weight_map, dim=-2, keepdim=True
    )  # [n_batch, 1, n_sample]
    source = (~(f0_t < min_f0)).float() * source
    return source * 0.01

@torch.jit.script
def generate_impulse_train(
        f0_t: Tensor, n_harmonic: int,
        sampling_rate: float) -> Tensor:

    f0_map = freq_multiplier(n_harmonic, f0_t.device) * f0_t
    weight_map = torch.sigmoid(-(f0_map - sampling_rate / 2.0))
    w0_map_cum = (
        f0_t.cumsum(dim=-1) * 2.0 * math.pi / sampling_rate *
        freq_multiplier(n_harmonic, f0_t.device)
    )
    source = torch.sum(torch.cos(w0_map_cum) * weight_map, dim=1, keepdim=True)
    return source * 0.01

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
    batch_size = x.size(0)
    z = torch.cat([x, y], dim=0)  # shape: [2 x Batch, T]
    for fft_length, window_length, hop_length in zip(fft_lengths, window_lengths, hop_lengths):
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
        elif isinstance(loss_scale_type, float):
            loss += (MagX - MagY).abs().mean() + \
                    0.5 * loss_scale_type * \
                          (SquareX.log() - SquareY.log()).abs().mean()
        else:
            raise RuntimeError(f"Unrecognized STFT loss scale type.")
    return loss

@torch.jit.script
def reshape_zeros_like(x: Tensor, dim: int, length: int) -> Tensor:
    """Return torch.zeros_like(x), while change shape of the `dim` dimension."""
    shape = list(x.shape)
    shape[dim] = length
    return torch.zeros(x.shape, dtype=x.dtype, device=x.device)

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
    magnitude_responses = log_magnitude_responses.exp()
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
def _framewise_corr_ola(
    framed_x: Tensor, framed_y: Tensor, frame_shift: int) -> Tensor:
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
    framed_z = time_corr(framed_x, framed_y)
    output = unframe_signal(framed_z.transpose(-1, -2), frame_shift)

    return output

@torch.jit.script
def ltv_fir(
    x: Tensor, filters: Tensor, frame_size: int
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
    y = _framewise_corr_ola(framed_x, filters, frame_size)
    striped_y = y[..., filter_size // 2: n_sample + filter_size // 2]
    return striped_y

def bare_stft(x: Tensor, padded_window: Tensor, hop_size: int) -> Tensor:
    """Compute STFT of real 1D signal.
    This function does not handle padding of x, and the window tensor.
    This function assumes fft_size = window_size.
    Args:
        x: [..., n_sample]
        padded_window: [fft_size], a window padded to fft_size.
        hop_size: Also referred to as the frame shift.
    Returns:
        n_frame: see frame_signal definition.
        X: [..., n_frame, fft_size],
            where n_frame = n_sample // hop_size
    """
    fft_size = len(padded_window)
    # Squash x's batch_sizes
    batch_size = x.shape[:-1]
    n_sample = x.size(-1)
    squashed_x = x.reshape(-1, 1, n_sample)
    # shape: [prod(batch_size), 1, n_sample]
    framed_squashed_x = frame_signal(squashed_x, fft_size, hop_size)
    # shape: [prod(batch_size), fft_size, n_frame]
    windowed_framed_squashed_x = \
        framed_squashed_x * padded_window.unsqueeze(-1)
    squashed_X = fft(
        windowed_framed_squashed_x.transpose(-1, -2), dim=-1
    )  # shape: [prod(batch_size), n_frame, fft_size]
    X = squashed_X.reshape(*batch_size, *(squashed_X.shape[1:]))
    # shape: [*batch_size, n_frame, fft_size]
    return X

def get_window(window_type: str,
               window_size: int,
               device: Union[str, torch.device],
               periodic: Optional[bool] = True,
               padding: Optional[Tuple[int, int]] = (0, 0)
               ) -> Tensor:
    """LRU cached window functions. Wrapper around PyTorch functions.
    Examples:
        >>> get_window("hann", 10, "cpu", periodic=True)
        tensor([0.0000, 0.0955, 0.3455, 0.6545, 0.9045, 1.0000, 0.9045, \
            0.6545, 0.3455, 0.0955])
    """
    window = getattr(torch, f"{window_type}_window")(
        window_size, device=device, periodic=periodic
    )
    window = pad(window, padding, mode="constant", value=0.0)
    return window

def frame_center_stft(
    x: Tensor,
    hop_size: int,
    window_size: int,
    window_type: str,
    fft_size: int,
) -> Tensor:
    """STFT analysis where the window is located at around the center
    of each frame.
    Args:
        x: [n_batch, 1, n_sample].
        window_type: STFT window type. See `get_window` for details.
    Returns:
        X: [n_batch, n_frame, fft_size], 
            where n_frame = n_sample // hop_size
    """
    assert fft_size >= window_size >= hop_size

    padding_left = window_size // 2 - hop_size // 2
    padding_right = fft_size - padding_left - hop_size
    window = get_window(
        window_type, window_size, x.device, periodic=False,
        padding=(0, fft_size - window_size),
    )
    x = pad(x, [padding_left, padding_right])
    X = bare_stft(x, window, hop_size)
    return X.squeeze(1)

def linear_mel_matrix(
    sampling_rate: int, fft_size: int, mel_size: int,
    mel_min_f0: Union[int, float],
    mel_max_f0: Union[int, float],
    device: torch.device
) -> Tensor:
    """
    Args:
        sampling_rate: Sampling rate in Hertz.
        fft_size: FFT size, must be an even number.
        mel_size: Number of mel-filter banks.
        mel_min_f0: Lowest frequency in the mel spectrogram.
        mel_max_f0: Highest frequency in the mel spectrogram.
        device: Target device of the transformation matrix.

    Returns:
        basis: [mel_size, fft_size // 2 + 1].
    """

    #import inspect
    #print("mel_fn ->", mel_fn)
    #print("module  ->", getattr(mel_fn, "__module__", None))
    #print("name    ->", getattr(mel_fn, "__name__", None))
    #print("sig     ->", inspect.signature(mel_fn))

    basis = torch.FloatTensor(
        mel_fn(sr=sampling_rate, n_fft=fft_size, n_mels=mel_size, fmin=mel_min_f0, fmax=mel_max_f0)
    ).transpose(-1, -2)
    return basis.to(device)

def stft_magnitude_to_mel_scale_log_magnitude(
    sampling_rate: int,
    mag_X: Tensor,
    fft_size: int,
    mel_size: int,
    mel_min_f0: float,
    mel_max_f0: float,
    mel_log_min_clip: float
) -> Tensor:
    """Project onesided STFT magnitude to mel scale log magnitude.
    Args:
        sampling_rate: Sampling rate in Hertz.
        mag_X: [..., fft_size].
        fft_size: The fft_size of mag_X.
        log_mel_min_clip: A (mostly negative) float number to clip
                          the log mel scaled spectrogram.
    Returns:
        mel_magnitude_x: [..., mel_size].
    """
    assert fft_size % 2 == 0
    projection_matrix = linear_mel_matrix(
        sampling_rate, fft_size, mel_size, mel_min_f0,
        mel_max_f0, mag_X.device)
    return mag_X[..., :(fft_size // 2 + 1)].matmul(projection_matrix) \
        .log().clamp(min=mel_log_min_clip)

def frame_center_log_mel_spectrogram(
    x: Tensor, hop_size: int, window_size: int,
    window_type: str, fft_size: int, sampling_rate: int,
    mel_size: int, mel_min_f0: float, mel_max_f0: float,
    mel_log_min_clip: float
) -> Tensor:
    """Frame center log mel spectrogram.
    Args:
        x: [n_batch, 1, n_sample].
    Returns:
        mag_X: [n_batch, n_frame, mel_size]
    """
    X = frame_center_stft(
        x, hop_size, window_size, window_type, fft_size
    )
    mag_X = X.abs()
    return stft_magnitude_to_mel_scale_log_magnitude(
        sampling_rate, mag_X, fft_size, mel_size, mel_min_f0, mel_max_f0,
        mel_log_min_clip)