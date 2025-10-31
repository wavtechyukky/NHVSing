[日本語](./readme.md) | [English](./README.en.md)

# NHV-Sing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is a vocoder model based on the paper [Neural Homomorphic Vocoder](https://www.isca-archive.org/interspeech_2020/liu20_interspeech.pdf), **tuned for singing voice synthesis**. It is implemented in PyTorch and supports JIT compilation. While following the structure proposed in the original paper, it includes modifications for application to singing voice synthesis.

***

## Audio Samples

**Ground Truth:**  
[▶️ Listen](sample_wav/ground_truth.wav)

**生成された音声:**  
[▶️ Listen](sample_wav/inference_wav.wav)

## Features

### Performance

*   **Lightweight & Fast:** Achieves very fast inference on a typical PC's CPU, despite its small model size of about 4MB.
*   **High Fidelity:** Faithfully reproduces the speaker's voice quality.
*   **Stable Quality:** Synthesizes stably without artifacts, even for long tones, staying true to the fundamental frequency (F0).

### Differences from the Original Implementation
The following points have been changed from the implementation in the original paper (some parameters can be edited in `config.yaml`):

*   **Sampling Rate**: Supports **44.1kHz**.
*   **Complex Cepstrum**: The number of dimensions has been expanded to **444**.
*   **Removal of FIR (postfilter)**: Although it contributes to reducing STFT loss, it was removed because it slows down processing and was judged not to contribute to learning the waveform, which is intuitively important.
*   **Discriminator**: Uses the one from HiFi-GAN.
*   **Input Features**:
    *   **log Mel Spectrogram**: Takes a log Mel spectrogram from **40Hz to 22050Hz** as input. While the paper cuts off high-frequency bands, it was determined that the reproducibility of high-frequency bands is necessary for improving intuitive quality. If high-frequency bands are not input as in the paper, frequency band control by FIR would be important.
    *   **F0**: Takes an F0 with the **unvoiced sections linearly interpolated** as input, making the Unvoiced/Voiced flag unnecessary. In singing voice synthesis, it is not only important to be able to draw the F0 curve including unvoiced sections, but also, when changing behavior with a UV flag as in the paper, a smooth transition from unvoiced to voiced sections cannot be reproduced.

### Export Formats

*   **PyTorch Native**: Same as the model used during training.
*   **TorchScript**: Becomes executable from other languages through JIT compilation.
*   **ONNX+PyTorch**: Implements the neural network part with ONNX and the DSP part with PyTorch. Used to verify if ONNX works correctly.
*   **ONNX+NumPy (Prototype)**: Implements DSP with NumPy. While not noticeable in short audio, long audio breaks down and generation is slow.

The reason for exporting in ONNX format is to enable execution in other languages and calculation libraries.

***

## Environment

*   Verified on Python 3.10.18

```bash
pip install -r requirements.txt
```

## Usage

### 1. Preprocessing

Extracts the features (in npz format) required for model training from WAV files.

```bash
python preprocess.py --step all
```

After preprocessing is complete, manually distribute the npz files generated in the `dataset/npz` directory into the `dataset/training_normal` (for training) and `dataset/inference` (for validation) folders.

### 2. Training

Starts model training. With the default `config.yaml`, the training progress and various logs are saved in `logs_normal`, and model snapshots are saved in `snapshots_normal`.

```bash
python train.py
```

### 3. Exporting the Model

Exports the trained model (snapshot) into formats that can be used for inference (`.pth`, `.pt`, `.onnx`).

```bash
# Example: Exporting the snapshot from epoch 1000
python export.py --checkpoint snapshots_normal/001000epoch.pth --config config.yaml
```

### 4. Inference

Generates audio from npz or WAV files using the exported model.

**Using the PyTorch native model (.pth):**

```bash
python inference.py <input.wav_or_npz> --snapshot <path/to/model.pth>
```

**To check the operation of various exported models at once:**

```bash
python -m debug.inference_checker <path/to/wav> <path/to/output> --config config.yaml --pth_path exported_models/model.pth --pt_path exported_models/model_jit.pt --onnx_path exported_models/core_model.onnx
```

***

## Known Issues

*   **Training Process:** It does not currently support multi-GPU training or batch sizes greater than 2. (Training with a batch size of 1 also requires a large amount of memory).
*   **ONNX Export:** Exporting the entire vocoder in ONNX format is not supported. The process gets stuck on `torch.fft` related code and the LTV Filter processing.
*   **Low-Pitch Quality:** As far as tested with the described settings, there is a tendency for sound quality to degrade when synthesizing particularly low-pitched male vocals. In principle, it should be possible to reproduce male voices, so it is thought that the quality could be improved by changing hyperparameters or the frequency band of the input Mel spectrogram.
*   **Speaker Dependency:** The generated waveform reflects the characteristics of the speaker it was trained on, making it difficult to reproduce the voices of multiple people. Instead, it can produce a realistic-sounding voice even from ambiguous acoustic features like those from FastSpeech2.

## License

This project is licensed under the [MIT License](LICENCE).

## Acknowledgements

This repository is largely based on the following papers and repositories published by Liu, et al.

*   Z. Liu, Y. Wang, K. Chen and Y. Jia, "Neural Homomorphic Vocoder," *Proc. Interspeech 2020*, pp. 3500-3504, doi: 10.21437/Interspeech.2020-2325.
*   [https://www.isca-archive.org/interspeech_2020/liu20_interspeech.pdf](https://www.isca-archive.org/interspeech_2020/liu20_interspeech.pdf)
*   [https://github.com/xcmyz/FastVocoder/tree/main](https://github.com/xcmyz/FastVocoder/tree/main)
*   [https://github.com/zjlww/dsp](https://github.com/zjlww/dsp)
*   [https://pypi.org/project/neural-homomorphic-vocoder/](https://pypi.org/project/neural-homomorphic-vocoder/)
