"""Add the hnsep harmonic component to every segment of a preprocessed NHVSing dataset.

Each input npz is copied to --out_dir (same relative path) with one extra key per segment:
    segment npz : 'harmonic'
    shard npz   : '<sid>|harmonic'
The harmonic has the same length and dtype (float32) as the segment's wav; the noise target is
wav - harmonic, so it is not stored. All existing keys are written back unchanged.

The separation runs once on the whole (un-cropped, un-augmented) segment. Pitch augmentation
(linear-interpolation resampling) and amplitude augmentation are linear, so applying them to
wav and harmonic separately gives the exact decomposition of the augmented wav.

Meant for Colab (GPU, onnxruntime-gpu):
    pip install onnxruntime-gpu
    python tools/add_hnsep_harmonic.py \
        --model /content/drive/MyDrive/.../hnsep_VR_44.1k_hop512_2024.05.onnx \
        --in_dir  /content/drive/MyDrive/.../NHVSing/dataset/training/eval_44k \
        --out_dir /content/drive/MyDrive/.../NHVSing_hnsep/dataset/training/eval_44k
"""
import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tqdm import tqdm

SR = 44100


def overlapped_fold(y, chunk_size, overlap_size):
    hop_size = chunk_size - overlap_size
    if y.shape[0] < chunk_size:
        n_frames = 1
    else:
        n_frames = (y.shape[0] - chunk_size + hop_size - 1) // hop_size + 1
    y = np.pad(y, (0, (n_frames - 1) * hop_size + chunk_size - y.shape[0]))
    indices = np.arange(chunk_size)[None, :] + (np.arange(n_frames) * hop_size)[:, None]
    return y[indices]


def overlapped_unfold(y, chunk_size, overlap_size):
    n_frames = y.shape[0]
    if n_frames == 1:
        return y[0]
    hop_size = chunk_size - overlap_size
    indices = np.arange(chunk_size)[None, :] + (np.arange(n_frames) * hop_size)[:, None]
    ramp = np.arange(1, overlap_size + 1) / overlap_size
    weights = np.minimum(
        np.concatenate([ramp, np.ones(chunk_size - overlap_size)]),
        np.concatenate([np.ones(chunk_size - overlap_size), ramp[::-1]]),
    )[None, :]
    out = np.zeros((n_frames - 1) * hop_size + chunk_size, dtype=y.dtype)
    out_w = np.zeros_like(out)
    np.add.at(out, indices, y * weights)
    np.add.at(out_w, indices, np.broadcast_to(weights, y.shape))
    return out / out_w


def separate_harmonic(session, wav, chunk_sec, batch_size):
    """Same chunking as HarmonicNoiseSeparationGUI (inference.py / main.py defaults)."""
    wav = wav.astype(np.float32)
    chunk = int(chunk_sec * SR)
    overlap = max(chunk // 4, 16384)
    cut = max(chunk // 16, 4096)
    if len(wav) <= chunk:
        harmonic, _ = session.run(['harmonic', 'noise'], {'waveform': wav[None]})
        return harmonic[0]
    fold = overlapped_fold(np.pad(wav, (cut, cut)), chunk, overlap)
    out = np.zeros_like(fold)
    for s in range(0, len(fold), batch_size):
        out[s:s + batch_size], _ = session.run(['harmonic', 'noise'], {'waveform': fold[s:s + batch_size]})
    return overlapped_unfold(out[:, cut:-cut], chunk - 2 * cut, overlap - 2 * cut)[:len(wav)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='hnsep ONNX (waveform -> harmonic, noise), 44.1 kHz')
    ap.add_argument('--in_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--chunk_sec', type=float, default=10.0)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--overwrite', action='store_true', help='redo npz that already exist in out_dir')
    args = ap.parse_args()

    providers = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider') if p in ort.get_available_providers()]
    session = ort.InferenceSession(args.model, providers=providers)
    print('providers:', session.get_providers())
    if 'CUDAExecutionProvider' not in session.get_providers():
        print('WARNING: running on CPU (install onnxruntime-gpu on a GPU runtime)')

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    paths = sorted(in_dir.rglob('*.npz'))
    print(f'{len(paths)} npz in {in_dir}')
    total_sec, t0 = 0.0, time.time()
    files_bar = tqdm(paths, desc='npz', unit='file', position=0)
    for p in files_bar:
        dst = out_dir / p.relative_to(in_dir)
        if dst.exists() and not args.overwrite:
            files_bar.set_postfix_str(f'skip {p.name}')
            continue
        with np.load(p) as f:
            data = {k: f[k] for k in f.files}
        wav_keys = [k for k in data if k == 'wav' or k.endswith('|wav')]
        seg_bar = tqdm(wav_keys, desc=p.name, unit='seg', position=1, leave=False)
        for wk in seg_bar:
            hk = wk[:-len('wav')] + 'harmonic'
            data[hk] = separate_harmonic(session, data[wk], args.chunk_sec, args.batch_size).astype(np.float32)
            assert data[hk].shape == data[wk].shape
            total_sec += len(data[wk]) / SR
            files_bar.set_postfix_str(f'{total_sec / 3600:.2f} h audio, RTF {(time.time() - t0) / total_sec:.4f}')
        seg_bar.close()
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.stem + '.tmp.npz')
        np.savez(tmp, **data)
        tmp.replace(dst)
    elapsed = time.time() - t0
    print(f'done: {total_sec / 3600:.2f} h of audio in {elapsed / 60:.1f} min'
          + (f' (RTF {elapsed / total_sec:.4f})' if total_sec else ''))


if __name__ == '__main__':
    main()
