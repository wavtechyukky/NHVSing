import numpy as np
import onnxruntime as ort
from numpy.fft import fft, ifft
from typing import Union, Optional
from scipy.special import expit

# --- NumPy ports of DSP functions ---

def np_freq_multiplier(n_harmonic: int) -> np.ndarray:
    """Generate the frequency multiplier [[[1], [2], ..., [n_harmonic]]]
    Returns:
        multiplier: [1, n_harmonic, 1]
    """
    return np.arange(1, n_harmonic + 1, dtype=np.float32).reshape(1, n_harmonic, 1)


def np_repeat_interpolate(x, frame_size):
    return np.repeat(x, frame_size, axis=-1)

def np_freq_antialias_mask(sampling_rate: Union[int, float], freq_tensor: np.ndarray,
                        hard_boundary: Optional[bool] = True) -> np.ndarray:
    """NumPy port of freq_antialias_mask from dsp.py with improved numerical stability using expit."""
    if hard_boundary:
        return (freq_tensor < sampling_rate / 2.0).astype(np.float32)
    else:
        x = -(freq_tensor - sampling_rate / 2.0)
        return expit(x)

def np_generate_impulse_train(f0_t, n_harmonic, sampling_rate):
    f0_map = np_freq_multiplier(n_harmonic) * f0_t
    weight_map = np_freq_antialias_mask(sampling_rate, f0_map, hard_boundary=False)
    w0_map_cum = (
        f0_t.cumsum(axis=-1) * 2.0 * np.pi / sampling_rate *
        np_freq_multiplier(n_harmonic)
    )
    source = np.sum(np.cos(w0_map_cum % (2 * np.pi)) * weight_map, axis=1, keepdims=True)
    return source * 0.01

def np_fftpad(x, padding):
    size = x.shape[-1]
    half = size // 2
    first_half = x[..., 0:size - half]
    second_half = x[..., size - half:]
    # The original torch implementation has a bug where it uses the input size instead of padding size.
    # We replicate this bug here for equivalence.
    zeros = np.zeros_like(x)
    return np.concatenate([first_half, zeros, second_half], axis=-1)

def np_complex_cepstrum_to_fft(ccep, fft_size):
    """NumPy port of complex_cepstrum_to_fft from dsp.py"""
    ccep_size = ccep.shape[-1]
    if fft_size < ccep_size:
        raise ValueError("FFT size should be greater than CCep size.")

    ccep_padded = np_fftpad(ccep, fft_size - ccep_size)
    X_hat = fft(ccep_padded, axis=-1)

    log_magnitude_responses = X_hat.real
    phase_responses = X_hat.imag

    magnitude_responses = np.exp(log_magnitude_responses)
    X = magnitude_responses * (np.cos(phase_responses) + 1j * np.sin(phase_responses))

    return X, log_magnitude_responses, phase_responses

def np_complex_cepstrum_to_imp(ccep, fft_size):
    X, _, _ = np_complex_cepstrum_to_fft(ccep, fft_size)
    x = ifft(X, axis=-1).real
    return x

def np_frame_signal(x, frame_size, frame_shift):
    """
    NumPy implementation of torch.nn.functional.unfold.
    x: (B, 1, n_sample)
    Returns: (B, frame_size, n_frame)
    """
    n_sample = x.shape[-1]
    # Note: Ensure n_sample >= frame_size, or this will be negative.
    if n_sample < frame_size:
        return np.zeros((x.shape[0], frame_size, 0), dtype=x.dtype)
    n_frame = (n_sample - frame_size) // frame_shift + 1
    
    batch_size = x.shape[0]
    out_shape = (batch_size, frame_size, n_frame)
    framed_x = np.zeros(out_shape, dtype=x.dtype)
    
    for b in range(batch_size):
        for i in range(n_frame):
            start = i * frame_shift
            framed_x[b, :, i] = x[b, 0, start:start + frame_size]
    return framed_x


def np_unframe_signal(x, frame_shift):
    """
    NumPy implementation of torch.nn.functional.fold (overlap-add).
    x: (B, frame_size, n_frame)
    Returns: (B, 1, n_sample)
    """
    batch_size, frame_size, n_frame = x.shape
    if n_frame == 0:
        return np.zeros((batch_size, 1, frame_size), dtype=x.dtype)
    n_sample = frame_size + (n_frame - 1) * frame_shift
    
    output = np.zeros((batch_size, 1, n_sample), dtype=x.dtype)
    
    for b in range(batch_size):
        for i in range(n_frame):
            start = i * frame_shift
            output[b, 0, start:start + frame_size] += x[b, :, i]
    return output


def np_time_corr_framewise(framed_x, framed_y):
    """
    Computes framewise correlation.
    framed_x: (B, n_frame, nx)
    framed_y: (B, n_frame, ny)
    Returns: (B, n_frame, nx + ny - 1)
    """
    batch_size, n_frame, nx = framed_x.shape
    _, _, ny = framed_y.shape
    out_len = nx + ny - 1
    
    output = np.zeros((batch_size, n_frame, out_len), dtype=framed_x.dtype)
    
    # Iterate over batch and frames to apply correlation
    for b in range(batch_size):
        for f in range(n_frame):
            output[b, f, :] = np.correlate(framed_x[b, f, :], framed_y[b, f, :], mode='full')
    return output


def np_ltv_fir(x, filters, frame_size):
    """
    NumPy implementation of Linear Time-Varying FIR filter based on dsp.py
    x: (B, 1, n_sample)
    filters: (B, n_frame, filter_size)
    frame_size: int
    """
    filter_size = filters.shape[-1]
    n_sample = x.shape[-1]
    
    # 1. Frame the input signal (no overlap, as in original ltv_fir)
    framed_x = np_frame_signal(x, frame_size, frame_size).transpose(0, 2, 1)
    
    # 2. Shift the filters (mimics torch.fft.fftshift)
    filters_shifted = np.fft.fftshift(filters, axes=-1)
    
    # 3. Framewise correlation
    if framed_x.shape[1] == 0: # No frames to process
        return np.zeros((x.shape[0], 1, n_sample), dtype=x.dtype)
    framed_y = np_time_corr_framewise(framed_x, filters_shifted)
    
    # 4. Overlap-add (using frame_size as stride, as in original)
    y = np_unframe_signal(framed_y.transpose(0, 2, 1), frame_size)
    
    # 5. Strip padding to align signal, compensating for filter delay
    start_idx = filter_size // 2
    end_idx = n_sample + filter_size // 2
    striped_y = y[..., start_idx:end_idx]
    
    return striped_y

# --- Main Inference Class ---

class NHVSing_with_ONNX:
    def __init__(self, onnx_model_path: str, vocoder_cfg: dict, ltv_filter_cfg: dict):
        self.session = ort.InferenceSession(onnx_model_path)
        
        # Store necessary parameters
        self.fs = vocoder_cfg['sample_rate']
        self.hop_size = vocoder_cfg['hop_size']
        self.fft_size = ltv_filter_cfg['fft_size']

    def inference(self, z: np.ndarray, x: np.ndarray, cf0: np.ndarray):
        """
        Args:
            z (np.ndarray): Noise signal (B, 1, T * hop_size)
            x (np.ndarray): Mel-cepstrum (B, T, D)
            cf0 (np.ndarray): Continuous F0 (B, 1, T)
        """
        # 1. Run convolutions using ONNX model
        input_name = self.session.get_inputs()[0].name
        output_names = [o.name for o in self.session.get_outputs()]
        
        ccep_harm, ccep_noise = self.session.run(output_names, {input_name: x.astype(np.float32)})

        # 2. Generate harmonic source
        cf0_resampled = np_repeat_interpolate(cf0, self.hop_size)
        harmonic_source = np_generate_impulse_train(cf0_resampled, 200, float(self.fs))

        # 3. Synthesize harmonic part
        imp_harm = np_complex_cepstrum_to_imp(ccep_harm, self.fft_size)
        sig_harm = np_ltv_fir(harmonic_source, imp_harm, self.hop_size)

        # 4. Synthesize noise part
        imp_noise = np_complex_cepstrum_to_imp(ccep_noise, self.fft_size)
        sig_noise = np_ltv_fir(z, imp_noise, self.hop_size)

        # 5. Combine and clamp
        y = sig_harm + sig_noise
        y = np.clip(y, -1, 1)
        return y.reshape(x.shape[0], -1)