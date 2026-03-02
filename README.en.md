[日本語](./readme.md) | [English](./README.en.md)

# NHV-Sing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is a vocoder model based on the paper [Neural Homomorphic Vocoder](https://www.isca-archive.org/interspeech_2020/liu20_interspeech.pdf), **tuned for singing voice synthesis**. It is implemented in PyTorch and supports JIT compilation and single-file ONNX export. While following the structure proposed in the original paper, it includes modifications for application to singing voice synthesis.

***

## Audio Samples

**Ground Truth:**
<audio controls src="sample_wav/ground_truth.wav"></audio>

**Synthesized Voice:**
<audio controls src="sample_wav/inference_wav.wav"></audio>

## Features

### Performance

*   **Lightweight & Fast:** Achieves very fast inference on a typical PC's CPU, despite its small model size of about 4MB.
*   **High Fidelity:** Faithfully reproduces the speaker's voice quality.
*   **Stable Quality:** Synthesizes stably without artifacts, even for long tones, staying true to the fundamental frequency (F0).

The Real-Time Factor (RTF) and average inference time of this vocoder were measured under the following environment and conditions.

- **Measurement Environment:** Apple M-series CPU (MacBook)
- **Measurement Conditions:**
    - Input Audio Sampling Rate: 44.1kHz
    - Input Audio Length: Approx. 26 seconds
    - Batch Size: 1

| Model Type     | Device | Avg. Inference Time | RTF      |
|----------------|--------|---------------------|----------|
| Native Python  | CPU    | 1.923 sec           | 0.0740   |
| JIT Script     | CPU    | 1.935 sec           | 0.0745   |
| Unified ONNX   | CPU    | 4.586 sec           | 0.1765   |

### Differences from the Original Implementation
The following points have been changed from the implementation in the original paper (some parameters can be edited in `config.yaml`):

*   **Sampling Rate**: Supports **44.1kHz**.
*   **Complex Cepstrum**: The number of dimensions has been expanded to **512**.
*   **Removal of FIR (postfilter)**: Although it contributes to reducing STFT loss, it was removed because it slows down processing and was judged not to contribute to learning the waveform, which is intuitively important.
*   **Discriminator**: Uses Multi-Scale Waveform Discriminator + Multi-Scale Complex STFT Discriminator. An adversarial loss warmup period is provided to prevent collapse of existing training, allowing fine-tuning from mid-training (`adversarial_warmup_epochs`).
*   **Additional Loss Functions**:
    *   **Envelope loss**: Extracts upper and lower envelopes via 1D max-pooling and computes MAE (RefineGAN §2.5.1). Suppresses jaggedness in the amplitude envelope (`envelope_scale`).
    *   **Harmonic penalty loss**: L1 penalty for voiced components (`sig_harm`) being output in unvoiced regions (frames where F0=0). Suppresses breath-like sounds generated in unvoiced regions (`harmonic_penalty_scale`).
*   **quef_norm**: Off by default (`use_quef_norm: false`). Experiments confirmed that enabling it inhibits learning of high-frequency bands.
*   **Input Features**:
    *   **log Mel Spectrogram**: Takes a log Mel spectrogram from **40Hz to 22050Hz** as input. While the paper cuts off high-frequency bands, it was determined that the reproducibility of high-frequency bands is necessary for improving intuitive quality.
    *   **F0**: Takes an F0 with the **unvoiced sections linearly interpolated** as input, making the Unvoiced/Voiced flag unnecessary. In singing voice synthesis, it is not only important to be able to draw the F0 curve including unvoiced sections, but also, when changing behavior with a UV flag as in the paper, a smooth transition from unvoiced to voiced sections cannot be reproduced.

### Export Formats

*  **PyTorch Native** (`model.pth`): Same as the model used during training.
*  **TorchScript** (`model_jit.pt`): Becomes executable from other languages through JIT compilation.
*  **Unified ONNX** (`full_vocoder.onnx`): Exports the entire vocoder (NN + DSP) as a single ONNX file. Inference is possible with ONNXRuntime only.

***

## Environment

*   Verified on Python 3.10

```bash
pip install -r requirements.txt
```

## Usage

### 1. Preprocessing

Extracts the features (in npz format) required for model training from WAV files.

```bash
# Run WAV trimming → F0/mel extraction → train/test split all at once
python preprocess.py --step all
```

Each step can also be run individually.

```bash
python preprocess.py --step resample  # Resampling
python preprocess.py --step cut      # Silence trimming
python preprocess.py --step npz      # Create npz (F0 · mel spectrogram)
python preprocess.py --step split    # Train/test split
```

The train/test ratio can be configured with `preprocess.train_test_split` in `config.yaml` (default: 100:1).

### 2. Training

Starts model training. TensorBoard logs are saved to `training.log_dir` and snapshots to `training.snapshot_dir` in `config.yaml`.

```bash
python train.py
```

Gradient accumulation and AMP (mixed precision) are supported. Configure with `gradient_accumulation_steps` and `use_amp` in `config.yaml`.

### 3. Exporting the Model

Exports the trained model (snapshot) into formats for inference.

```bash
# Export PyTorch native + JIT + Unified ONNX all at once
python export.py \
  --checkpoint snapshots/000990epoch.pth \
  --config config.yaml \
  --output_dir exported_models \
  --pytorch \
  --jit \
  --full_onnx
```

| Option         | Output File               | Description                        |
|----------------|---------------------------|------------------------------------|
| `--pytorch`    | `model.pth`               | state_dict (for retraining / fine-tuning) |
| `--jit`        | `model_jit.pt`            | TorchScript (for other language integration) |
| `--full_onnx`  | `full_vocoder.onnx`       | Single ONNX file                   |

### 4. Inference

Generates audio from WAV or NPZ files using the exported model.

```bash
# Inference with PyTorch model (also runs Native vs JIT speed comparison)
python inference.py input.wav \
  --snapshot exported_models/model.pth \
  --config config.yaml \
  --output_dir output

# Also benchmark Unified ONNX at the same time
python inference.py input.wav \
  --snapshot exported_models/model.pth \
  --config config.yaml \
  --output_dir output \
  --onnx exported_models/full_vocoder.onnx
```

***

## Known Issues

*   **Training Process:** Multi-GPU training is not supported.
*   **Other Frame Sizes:** Only the default frame size (hop_size=256) has been verified. Operation with other frame sizes is not guaranteed.
*   **Speaker Dependency:** The generated waveform reflects the characteristics of the speaker it was trained on, making it difficult to reproduce the voices of multiple people. Instead, it can produce a realistic-sounding voice even from ambiguous acoustic features like those from FastSpeech2.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

This repository is based on the following papers and repositories published by Liu, et al.

*   Z. Liu, Y. Wang, K. Chen and Y. Jia, "Neural Homomorphic Vocoder," *Proc. Interspeech 2020*, pp. 3500-3504, doi: 10.21437/Interspeech.2020-2325.
*   [https://www.isca-archive.org/interspeech_2020/liu20_interspeech.pdf](https://www.isca-archive.org/interspeech_2020/liu20_interspeech.pdf)
*   [https://github.com/xcmyz/FastVocoder/tree/main](https://github.com/xcmyz/FastVocoder/tree/main)
*   [https://github.com/zjlww/dsp](https://github.com/zjlww/dsp)
*   [https://pypi.org/project/neural-homomorphic-vocoder/](https://pypi.org/project/neural-homomorphic-vocoder/)
