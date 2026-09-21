[日本語](./README.md) | [English](./README.en.md)

# NHV-Sing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A [Neural Homomorphic Vocoder](https://www.isca-archive.org/interspeech_2020/liu20_interspeech.pdf) model **tuned for singing voice synthesis**. Implemented in PyTorch with support for JIT compilation and single-file ONNX export.

This repository contains the latest **NHVSing V3 / V3X** (the quality-improved **V3.2 / V3.2X** are recommended), plus the legacy **NHVSing** (V1, single-speaker) and **NHVSingV2** (multi-speaker). V3.1 is deprecated: at certain frequencies it exhibited a per-frame waveform phase flip in which a frame's waveform was cancelled by the neighbouring frame.

🎧 **Listen → [NHVSing V3.2 Demo Page](https://wavtechyukky.github.io/NHVSing/v3_2.html)** (the latest released weights — copy-synthesis vs NSF-HiFiGAN)
&nbsp;·&nbsp; [V3 Demo Page](https://wavtechyukky.github.io/NHVSing/v3.html) (RTF & model-size figures, singing synthesis from DiffSinger acoustic models)

***

## NHVSing V3 / V3X (latest, recommended)

**V3** is the latest model, refined for singing (44.1kHz / hop256 / 128-mel [40–16000Hz, **ln**]). Three essential improvements over V2:

- **Multi-Resolution Discriminator (MRD)** (+ MPD): the single biggest factor in quality. UnivNet-style multi-resolution spectrogram-magnitude discrimination.
- **Faster impulse-response synthesis (`fft_corr`)**: the LTV-FIR uses FFT — bit-equivalent to the time-domain version, ~8× faster on CPU.
- **Training-data curation & augmentation**: gathered high-quality 44.1 kHz audio and, during training, fed it while randomly varying volume and pitch. Although this feeds the model waveforms that deviate from ordinary ones, it significantly improved the vocoder's generalization.
- **Cleaner excitation/conditioning**: many measures were tried to raise quality, but the two above had the largest effect, so the configuration was reverted to follow the original paper for now (a major restructuring would likely be V4 or later). The excitation impulse sums sine waves up to the 200th harmonic / quef_norm α=1.0 / **mel-only input** (F0 embedding removed) / linear F0 interpolation (a return to the original NHV).

**V3X** lets V3 run on **hop512 input**: it takes hop512 mel/F0, interpolates internally to the hop256 grid, then runs V3 (weights/state_dict shared with V3). This **resolves the "hop512 severely degrades quality" limitation** of V1/V2.

| Class | Purpose | config |
|---|---|---|
| `NHVSing` | V1 legacy (single-speaker) | `config.yaml` |
| `NHVSingV2` | V2 legacy (multi-speaker) | `config_v2.yaml` |
| **`NHVSingV3`** | **final model (hop256 native)** | **`config_v3.yaml`** |
| **`NHVSingV3X`** | **hop512-input variant of V3** | `config_v3.yaml` + `ltv_filter.use_v3x: true` |

> V3.2 uses the same `NHVSingV3` / `NHVSingV3X` classes; only the config changes to **`config_v3_2.yaml`** (`ltv_filter.ola_mode: hann`).

### V3.2 — latest released weights (2026-09)

**V3.2** is the successor to V3 (V3.1 is deprecated). The architecture is unchanged (same classes, same model size, same RTF); what changed is a few fixes and how it was trained:

- **Hann window on the LTV-filter output (Hann WOLA, 50% overlap)**: the main cause of the per-frame sudden waveform attenuation was that the time-varying FIR was overlap-added with a non-overlapping rectangular window, so a frame's convolved waveform spilled into the neighbouring frame at large amplitude and the neighbour cancelled it 180° out of phase. Switching to a Hann window of length 2×frame_size tapers the spillover at the boundaries and reduces the effective waveform overlap from four frames down to the strength of two. It also lessens the mel/waveform mismatch when adjacent frames' waveforms overlap under a steep mel change. The released ONNX (`export.py` → `LTVFirONNX`) is Hann-windowed too, so training and deployment compute identically.
- **float64 excitation phase**: the impulse-train phase used to be accumulated in float32, which loses precision on long inputs and smears the harmonics. The phase is now accumulated in float64 and folded with mod 1.0 before being handed to a float32 cos (PyTorch and ONNX compute nearly identically).
- **Continuous-ratio pitch augmentation**: `torchaudio.resample` only supports rational ratios, so the old augmentation quantized the pitch-shift ratio to 22 discrete steps. Replaced with a continuous-ratio linear-interpolation resample.
- **Randomised excitation start phase during training**: because V3.1 momentarily flipped the impulse-response phase and cancelled the neighbouring frame's waveform, this was added so the model does not learn to lock onto a specific local phase.
- **An extra short-window resolution in the multi-resolution STFT loss**: likewise a countermeasure for the neighbour-cancellation seen in V3.1 — the original resolutions could not detect a single frame's waveform attenuation.
- **A low-learning-rate finishing pass**: added to raise mel fidelity while still stopping training early enough to avoid over-fitting.

> Note: "float64 excitation phase" and "continuous-ratio pitch augmentation" were originally introduced in **V3.1** and carried over into V3.2. What is new in V3.2 is the Hann windowing, the randomised excitation start phase, the extra short STFT window, and the low-LR finishing pass.

**Memory fix in the released ONNX (2026-09-19; weights unchanged)**

The released ONNX allocated memory proportional to the input length and **died past 3.4 GB on a single 24-second phrase** (on a 16 GB machine this fills swap and drags the whole OS down). Three places each expanded the full length at once:

| Where | Intermediate | Fix |
|---|---|---|
| Excitation (`impulse_train_onnx`) | `cos` of `[B, 200, 256T]` | Sum the 200 harmonics in a **loop** into a `[B,1,n]` accumulator |
| Time-varying FIR FFT (`ltv_fir_onnx`) | `[B, T, 2048, 2]` ×10 | **Block over frames** (the FFT is within a frame, so per-frame results are unchanged) |
| Cepstrum→IR (`complex_cepstrum_to_imp_onnx`) | `[B, T, 1024, 2]` ×6 | Same |

Both use `torch._higher_order_ops.scan` (ONNX `Scan`) for a dynamic trip count. Memory went from **~125 MB/s to ~8 MB/s** of audio: 0.33 GB at 24 s, 2.05 GB even at 240 s. Against the previously released V3.2 the deterministic `harmonic` output matches to **within −142 dB** (float32 rounding; 46 dB below 16-bit quantization noise).

**What it cost in speed** — the released ONNX before and after the fix, measured back to back on one machine (3 s input, median of 9, one measurement per process):

| CPU threads | Before | After | |
|---|---|---|---|
| 1 | 0.071 | 0.073 | +4% |
| 2 | 0.063 | 0.065 | +4% |
| 4 | 0.059 | 0.063 | +7% |
| 8 | 0.061 | 0.081 | **+33%** |

The looped part cannot be split across threads, so the penalty grows with core count. A few percent at 1–4 threads is a fair price for not dying on a long phrase; if you want the fastest many-core path, the torch route (RTF tables below) does not use `Scan` and is faster there.

> The released `v3` ONNX had been exported before the "float64 excitation phase" fix above and was never re-exported. It is re-exported here too (the error grows with length: −27.8 dB at 24 s). `v3_1` / `v3_2` already had the fix and are unaffected.

**The torch route is now memory-lean by default as well (2026-09-21; weights unchanged).** The torch side had the same problem: the `dsp.py` eager implementation that `NHVVocoder` (`nhv_vocoder.py`, see Usage (V3) below) runs through peaked at +5–8 GB on a 37-second input, and `export.py`'s `FullVocoderV3` (the ONNX-facing parts run eagerly) at +4.2 GB. Two changes, both **bit-identical** to the one-shot computation:

- Excitation impulse train: only the phase cumsum is taken over the full length; the cosines and the harmonic sum are built block by block in a plain Python loop (the default of `generate_impulse_train`, fixed 32768-sample blocks; the eager execution of `impulse_train_onnx` does the same with 16384-sample blocks, environment variable `NHV_IMP_BLOCK`).
- Time-varying FIR and cepstrum→IR: frames are processed 256 at a time (`ltv_filter.frame_block` or the environment variable `NHV_FRAME_BLOCK`), and the OLA runs once over the full length after concatenation, so the summation order is unchanged and the output is the same.

`NHVVocoder.infer` returns exactly what it did before (max difference 0 at 7 s and 37 s). Full-length buffers (the framed signal, the impulse responses and so on) remain, so the peak still grows with the input, but at about 9 MB/s instead of 150–280 MB/s: +0.5 GB at 37 s and +2.3 GB even at 240 s. Dropping the full-length expansion also makes it faster (37 s: 1.8–3.1 s → 1.2–1.3 s; `FullVocoderV3` before/after in one session: −11% / −18% / −26% / −41% at 1/2/4/8 threads). **Training does not change by a single bit**: a crop of at most 32768 samples is one block for the excitation, and a frame count at or below the frame block goes through the same one-shot FIR, so the computation is the one from before (the V3 recipe's 372 ms crop grows to at most 18176 samples / 71 frames under pitch augmentation; verified that `forward_train` outputs and gradients match before and after at 64 and 71 frames). `vocoder.harm_block` is no longer needed but is kept for compatibility. The ONNX export path (`Scan`) is untouched: an ONNX exported after the change is identical to the released one in both graph and output.

Released files in `exported_models/v3_2/` (standard `export.py` outputs, renamed):

- **`nhv_v3_2.pth`** — V3/V3X shared weights
- **`nhv_v3_2.onnx` / `nhv_v3_2x.onnx`** — self-contained single-file ONNX, same I/O contract as `nhv_v3.onnx` / `nhv_v3x.onnx`

The exact training recipe is **`config_v3_2.yaml`**.

### Performance (V3)

**Model size**: `nhv_v3.onnx` is about **7.8 MB** (no quantization) — roughly **1/7** the size of NSF-HiFiGAN (pc-nsf-hifigan, 56.7 MB), the reference we compare against.

> It was about 2.2 MB up to V3.2. Fixing the long-input memory blow-up (above) turned the excitation and the time-varying FIR into `Scan` loops, which bakes the `scatter_add` index tables into the graph as constants (+5.6 MB). **The weights are still 0.478 M parameters** — only indices were added.

**RTF** (Real-Time Factor = seconds of compute per second of audio; lower is faster, and < 1 means faster than real time). Measured on an M4 MacBook Air 10-core CPU (4 performance + 6 efficiency) / ~5 s input / batch 1 / median of 9 runs. **Each table comes from one interleaved session**, one measurement per process (putting several ORT sessions in one process shifts the numbers by up to 1.4×). The NSF-HiFiGAN comparison was measured on 2026-09-19, the torch comparison on 2026-09-21 after the eager fix above.

Under ONNX Runtime (CPU), side by side with NSF-HiFiGAN:

| CPU threads | NHVSing V3 | NSF-HiFiGAN | NHVSing speed-up |
|---|---|---|---|
| 1 | 0.073 (14×) | 0.589 (2×) | **8.0×** |
| 2 | 0.061 (16×) | 0.314 (3×) | 5.2× |
| 4 | 0.058 (17×) | 0.196 (5×) | 3.4× |
| 8 | 0.075 (13×) | 0.199 (5×) | 2.6× |

**The two scale very differently with core count.** NHVSing V3's per-frame impulse-response generation is fully independent (*embarrassingly parallel*), but **ONNX Runtime barely exploits this**, so V3 is essentially single-core-bound (14×→17× and then flat). NSF-HiFiGAN's large transposed convolutions parallelize well, so it keeps speeding up with more cores (2×→5×). As a result, **NHVSing's speed lead is largest on low-core devices (~8×) and narrows to ~3× on many cores**, but it stays ahead throughout.

Both models do worse at 8 threads than at 4, but **not for the same reason**. NSF-HiFiGAN loses 1.5% because 8 threads reaches into the efficiency cores; NHVSing loses 29%, and most of that is the **sequential loop** introduced by the memory fix described above, which extra threads cannot split.

**Native PyTorch does realize the parallelism.** Running the same V3 in torch scales with core count as the per-frame independence allows:

| CPU threads | ONNX Runtime | PyTorch |
|---|---|---|
| 1 | 0.067 (15×) | 0.059 (17×) |
| 2 | 0.058 (17×) | 0.039 (25×) |
| 4 | 0.059 (17×) | **0.032 (32×)** |
| 8 | 0.073 (14×) | **0.029 (34×)** |

On multiple cores, **torch actually beats our own ONNX export** (~1.9× at 4 threads, ~2.5× at 8): for such a tiny, FFT-dominated model, torch's batched-FFT parallelism helps more than ORT's graph optimizations. So "NHVSing is fast/slow" cannot be captured by a single number — it depends on the **runtime × core-count** combination.

> That torch column is `export.py`'s `FullVocoderV3` (the ONNX-facing parts run eagerly), not the training-side `model.py::NHVSingV3`. The ORT column was re-measured on the same day in the same session, so it differs by a few percent from the ORT column of the NSF-HiFiGAN table above, which comes from a different day.

> RTF depends only on the amount of compute, not on the weight values (it is the same for any checkpoint).

The ONNX graph pads the `LTVFirONNX` FFT length to a **power of two** for speed (ONNX Runtime's DFT is fast only for power-of-two sizes); V3X is comparable. The LTV-FIR time-correlation itself is also FFT-based via `fft_corr` (bit-equivalent to the time-domain version, ~8× faster on CPU — see "Key changes" above).

### Usage (V3)

**Preprocess** (F0 = RMVPE alone, plus a post-process that rejects per-frame estimation errors; `rmvpe.pt` auto-downloads on first run):
```bash
python preprocess.py --indir <dir_of_singing_wavs> --out <npz_dir> --config config_v3.yaml
```

> **Splitting train / test**: `preprocess.py` just writes all shards to `--out`; it does **not** auto-split into train/test. **Run it twice on separate wav sets** (train vs. eval) and point each to its own directory (a few held-out songs are enough for test):
> ```bash
> python preprocess.py --indir wavs/train --out dataset/train --config config_v3.yaml
> python preprocess.py --indir wavs/eval  --out dataset/test  --config config_v3.yaml
> ```
> Set `training.train_dir` / `test_dir` in `config_v3.yaml` accordingly. `VocoderDataset` recursively reads both shard npz (`<sid>|f0` / `|log_melspc` / `|wav`) and single-segment npz, so either layout works.

**Train** (MRD + MPD GAN; set `training.train_dir` / `test_dir` / `snapshot_dir` in `config_v3.yaml`):
```bash
python train_v3.py --config config_v3.yaml
```

**ONNX export** (both V3 and V3X; 3 outputs: `waveform` / `harmonic` / `noise`):
```bash
python export.py --config config_v3.yaml --ckpt <weights.ckpt> --out exported_models
```
By default this writes to `exported_models/v3/`:

- **`nhv_v3.pth`** — shared weights for V3/V3X (load with `NHVSingV3(vc, lc).load_state_dict(torch.load('nhv_v3.pth'))`). **V3X shares these weights, so it has no `.pth`** (ONNX only).
- **`nhv_v3.onnx` / `nhv_v3x.onnx`** — each a **single self-contained ONNX** (NN + DSP, weights embedded — no external `.onnx.data`; runnable with ONNX Runtime alone, same format as V1/V2's `full_vocoder.onnx`). Inputs `mel` / `f0` / `uv` → 3 outputs `waveform` / `harmonic` / `noise`, with `clamp(harmonic + noise) == waveform`. The time length T is dynamic (any length).

> To export **V3.2**, pass `--config config_v3_2.yaml` and rename the resulting `nhv_v3.*` to `nhv_v3_2.*` (the released `exported_models/v3_2/` was built this way).

**Inference (Python)**:
```python
from nhv_vocoder import NHVVocoder
voc = NHVVocoder('weights.ckpt', 'config_v3.yaml')      # V3/V3X auto-selected via config use_v3x
cf0, uv = NHVVocoder.prep_f0(f0_hz)                       # raw F0 (0=unvoiced) → continuous F0 + uv
wav = voc.infer(mel, cf0, uv)                            # mel: [T, 128] ln-mel
```

### F0 extraction

Preprocessing F0 uses **RMVPE only + jump-cleaning post-processing** (`tools/f0`). The RMVPE weights `rmvpe.pt` (~173MB) are **not** bundled — they **auto-download from HuggingFace on first run**.

### Weight licensing

Distributed trained weights are **non-commercial** (due to the training data).

***

# Everything below is about the legacy versions (NHVSing V1 / NHVSingV2)

> ⚠️ **The sections below (Audio Samples, Architecture, Performance, Usage, etc.) all describe the older V1 / V2 models.** For new work, use **V3 / V3X (recommended)** above. V1 / V2 are kept for compatibility.

## Audio Samples

→ **[NHVSingV2 Demo Page](https://wavtechyukky.github.io/NHVSing/v2.html)**

→ [NHVSing (V1) Demo Page](https://wavtechyukky.github.io/NHVSing/)

Compare synthesized audio from Kiritan & Natsume Yuri (NHVSing) and M4Singer / GTSinger evaluation data (NHVSingV2).

***

## NHVSing vs NHVSingV2

| | NHVSing | NHVSingV2 |
|---|---|---|
| CNN backbone | Dual-branch (Harmonic/Noise independent) | Shared trunk CNN |
| F0 input | Mel spectrogram only | F0 embedder (256 bins, log₂ scale, 128-dim) concatenated with mel |
| quef_norm | Off (was found to inhibit high-freq learning in V1) | Soft scaling with alpha=0.3 (stabilizes training without sacrificing high frequencies) |
| Amplitude augmentation | None | 0.5–2.0× (log-uniform) random scale |
| Speaker generalization | **Single-speaker specialized.** Artifacts occur easily for speakers outside training data | **Multi-speaker capable.** Can be trained on multi-speaker corpora such as M4Singer and ACE-Opencpop |
| Config file | `config.yaml` | `config_v2.yaml` |

NHVSing excels at faithfully reproducing the voice quality of a single speaker trained on a speaker-specific dataset. NHVSingV2 can vocoder acoustic features of various speakers at high quality by training on multi-speaker corpora.

***

## Performance (V1 / V2)

Measured under the following environment and conditions.

- **Measurement Environment:** Apple M-series CPU (MacBook)
- **Measurement Conditions:** 44.1kHz input, approx. 26 seconds, batch size 1

### NHVSing (V1)

| Model Type     | Device | Avg. Inference Time | RTF      |
|----------------|--------|---------------------|----------|
| Native Python  | CPU    | 2.048 sec           | 0.0788   |
| JIT Script     | CPU    | 2.145 sec           | 0.0825   |
| Unified ONNX   | CPU    | 4.275 sec           | 0.1645   |

### NHVSingV2

| Model Type     | Device | Avg. Inference Time | RTF      |
|----------------|--------|---------------------|----------|
| Native Python  | CPU    | 1.920 sec           | 0.0739   |
| JIT Script     | CPU    | 1.991 sec           | 0.0766   |
| Unified ONNX   | CPU    | 4.043 sec           | 0.1556   |

***

## Architecture (V1 / V2)

The following changes have been made from the original paper's implementation. (V3's discriminator is MRD + MPD, which differs from the MSD + complex-STFT setup described below.)

* **Sampling Rate**: Supports **44.1kHz**.
* **Complex Cepstrum**: Dimensions expanded to **512**.
* **Removal of FIR (postfilter)**: Although it reduces STFT loss, it was judged not to contribute to waveform learning.
* **Discriminator**: Multi-Scale Waveform Discriminator + Multi-Scale Complex STFT Discriminator. An adversarial loss warmup period allows fine-tuning from mid-training (`adversarial_warmup_epochs`).
* **Additional Loss Functions**:
    * **Envelope loss**: Extracts upper and lower envelopes via 1D max-pooling and computes MAE (RefineGAN §2.5.1). Suppresses amplitude envelope instability (`envelope_scale`).
    * **Harmonic penalty loss**: L1 penalty when voiced components (`sig_harm`) are output in unvoiced regions (frames where F0=0). Suppresses buzzing in unvoiced regions (`harmonic_penalty_scale`).
* **F0 Input**: Takes **linearly interpolated F0** in unvoiced regions as input, eliminating the need for an Unvoiced/Voiced flag. In singing voice synthesis, drawing the F0 curve through unvoiced regions is important, and UV flag-based switching cannot reproduce smooth unvoiced→voiced transitions.
* **Log Mel Spectrogram**: Takes the full band from **40Hz to 22050Hz** as input. High-frequency reproduction was judged to contribute to intuitive quality improvement.

### NHVSingV2-Specific Changes

* **Shared trunk CNN** (`use_shared_trunk: true`): Both harmonic and noise branches share a common trunk CNN before splitting into their respective heads. This eliminates the need for each branch to independently learn how to decompose the input acoustic features into harmonic and noise components.
* **F0 Embedder** (`use_f0_embed: true`): Continuous F0 is discretized into 256 bins on a log₂ scale and embedded into 128 dimensions, then concatenated with the mel spectrogram. Providing explicit pitch information gives the network additional cues about the waveform shape it needs to generate per period.
* **quef_norm** (`use_quef_norm: true`, `quef_norm_alpha: 0.3`): Applies gentle 1/|n|^α scaling to quefrency components, stabilizing training without over-suppressing high harmonics. In V1, enabling this inhibited high-frequency learning, but in V2 a small alpha of 0.3 allows stabilization without sacrificing high frequencies.

### Export Formats

*  **PyTorch Native** (`model.pth`): Same as the model used during training.
*  **TorchScript** (`model_jit.pt`): Executable from other languages via JIT compilation.
*  **Unified ONNX** (`full_vocoder.onnx`): Exports the entire vocoder (NN + DSP) as a single ONNX file. Inference possible with ONNXRuntime only.

***

## Environment

* Verified on Python 3.10

```bash
pip install -r requirements.txt
```

***

## Usage (V1 / V2)

### NHVSing (V1)

#### 1. Preprocessing

```bash
# Run WAV trimming → F0/mel extraction → train/test split all at once
python preprocess.py --config config.yaml --step all
```

#### 2. Training

```bash
python train.py --config config.yaml
```

#### 3. Fine-tuning (Transfer Learning to a New Speaker)

Inherits only the model weights from a trained model and re-trains on a different speaker's dataset. The Discriminator and Optimizers are freshly initialized.

```bash
python prepare_finetune.py \
  --weights exported_models/natsume/model.pth \
  --config config.yaml \
  --output snapshots_kiritan/000000epoch.pth

python train.py --resume_path snapshots_kiritan/000000epoch.pth --config config_fine_tuning.yaml
```

#### 4. Exporting

```bash
python export.py \
  --checkpoint snapshots/000990epoch.pth \
  --config config.yaml \
  --output_dir exported_models/kiritan \
  --all
```

#### 5. Inference

```bash
python inference.py input.wav \
  --snapshot exported_models/kiritan/model.pth \
  --config config.yaml \
  --output_dir output \
  --onnx exported_models/kiritan/full_vocoder.onnx
```

---

### NHVSingV2

Use `config_v2.yaml` as the starting point. Edit paths, speaker prefixes, and training parameters to match your environment.

#### 1. Preprocessing

F0 extraction uses RMVPE (model is auto-downloaded on first run).

```bash
python preprocess.py --config config_v2.yaml --step all
```

#### 2. Training

```bash
python train.py --config config_v2.yaml
```

#### 3. Fine-tuning (Transfer Learning to a New Speaker / Corpus)

```bash
python prepare_finetune.py \
  --weights exported_models/v2/model.pth \
  --config config_v2.yaml \
  --output snapshots_finetune/000000epoch.pth

python train.py --resume_path snapshots_finetune/000000epoch.pth --config config_v2.yaml
```

#### 4. Exporting

```bash
python export.py \
  --checkpoint snapshots_v2/000900epoch.pth \
  --config config_v2.yaml \
  --output_dir exported_models/v2 \
  --all
```

#### 5. Inference

```bash
python inference.py input.wav \
  --snapshot exported_models/v2/model.pth \
  --config config_v2.yaml \
  --output_dir output \
  --onnx exported_models/v2/full_vocoder.onnx
```

For WAV input, if `target_rms: 0.083` is set in `config_v2.yaml`, the RMS of voiced regions is automatically normalized to the median of M4Singer training data.

***

## Training Best Practices

### Amplitude Augmentation (`amp_augment`)

NHVSingV2 introduces amplitude augmentation during training, randomly scaling volume by 0.5–2.0× (log-uniform) (`amp_augment: true`, `amp_aug_range: [0.5, 2.0]`). This makes the model robust to input volume variation, so quality does not degrade even if the volume is slightly off during inference.

### Quefrency Norm Scale (`quef_norm_alpha`)

Setting `use_quef_norm: true` normalizes quefrency components and improves training stability. However, too large an alpha degrades reproducibility of high-frequency bands (consonants, fricatives). `quef_norm_alpha: 0.3` is a reasonable value that stabilizes training without sacrificing high frequencies.

### Harmonic Penalty Loss Strength (`harmonic_penalty_scale`)

This penalty suppresses harmonic components from leaking in unvoiced regions (frames where F0=0). This value has a large impact on quality.

* Too small (0–10): Buzzing sounds tend to occur in unvoiced regions.
* Too large (1000+): When given acoustic features that are ambiguous between voiced and unvoiced, the model may attempt to reproduce the mel spectrogram using unvoiced components even in regions that should be voiced.
* **Recommended: `harmonic_penalty_scale: 100`**

***

## Known Issues

*   **Training Process:** Multi-GPU training is not supported.
*   **Frame size:** V1/V2 degrade significantly at hop_size=512. **V3X handles hop512 input** (interpolating internally to hop256), resolving this.
*   **NHVSing (V1) Speaker Dependency:** The generated waveform strongly reflects the characteristics of the trained speaker. Use NHVSingV2 if multi-speaker support is required.

***

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

This repository is based on the following papers and repositories published by Liu, et al.

*   Z. Liu, Y. Wang, K. Chen and Y. Jia, "Neural Homomorphic Vocoder," *Proc. Interspeech 2020*, pp. 3500-3504, doi: 10.21437/Interspeech.2020-2325.
*   [https://www.isca-archive.org/interspeech_2020/liu20_interspeech.pdf](https://www.isca-archive.org/interspeech_2020/liu20_interspeech.pdf)
*   [https://github.com/xcmyz/FastVocoder/tree/main](https://github.com/xcmyz/FastVocoder/tree/main)
*   [https://github.com/zjlww/dsp](https://github.com/zjlww/dsp)
*   [https://pypi.org/project/neural-homomorphic-vocoder/](https://pypi.org/project/neural-homomorphic-vocoder/)

## Singing Voice Databases Used

### V3

The distributed V3 weights are trained on the following non-commercial data. Commercial use is not permitted. **Please check each dataset's license, availability and terms directly at the primary sources linked below.**

*   Tohoku Kiritan — [Zunko Project](https://zunko.jp/kiridev/login.php)
*   Natsume Yuri — [NJKS Official](https://ksdcm1ng.wixsite.com/njksofficial)
*   Namine Ritsu (Ritsu Singing DB Ver2.0-2.2 / Soft) — [Canon Voice](https://www.canon-voice.com/voicebanks/)
*   Children's Song Dataset (CSD) — [Zenodo](https://zenodo.org/records/4916302)
*   NUS-48E — [Zenodo](https://zenodo.org/records/19595152)
*   ONIKU_KURUMI Utagoe DB — [Onikuru](https://onikuru.info/db-download/)
*   Opencpop — [WeNet Opencpop](https://wenet-e2e.github.io/opencpop/download/)
*   No.7 — [VOICE SEVEN](https://voiceseven.com/7dev/login.php)
*   VocalSet — [Zenodo](https://zenodo.org/records/1193957)
*   ccmusic-database / acapella — [HuggingFace](https://huggingface.co/datasets/ccmusic-database/acapella)

### V1 / V2 (legacy)

*   Tohoku Kiritan — [Zunko Project](https://zunko.jp/kiridev/login.php)
*   Natsume Yuri — [NJKS Official](https://ksdcm1ng.wixsite.com/njksofficial)
*   M4Singer (CC BY-NC-SA 4.0) — [M4Singer GitHub](https://github.com/M4Singer/M4Singer)
    *   Zhang et al., "M4Singer: a Multi-Style, Multi-Singer and Musical Score Provided Mandarin Singing Corpus," *NeurIPS 2022*.
*   ACE-Opencpop — [HuggingFace](https://huggingface.co/datasets/espnet/ace-opencpop-segments)
