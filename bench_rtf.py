"""NHVSing V3 / V3X の RTF(real-time factor)とパラメータ数を計測し、Markdown 表を出力。

RTF = 推論時間 / 音声長。×realtime = 1/RTF。torch(CPU)と ONNX Runtime(CPU)の両方。
ONNX は export.py と同じ FullVocoder(LTV FIR を 2の冪 FFT で高速化済)を使う。

Usage:
    python bench_rtf.py --config config_v3.yaml --ckpt <weights.ckpt> [--seconds 3 --trials 9]
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import yaml

from model import NHVSingV3
from export import FullVocoderV3, FullVocoderV3X, load_core


def _rtf_torch(mod, inputs, dur, n):
    with torch.no_grad():
        for _ in range(2): mod(*inputs)
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); mod(*inputs); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) / dur


def _rtf_ort(path, feeds, dur, n):
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    for _ in range(2): s.run(['waveform'], feeds)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); s.run(['waveform'], feeds); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) / dur


def bench(tag, FullCls, vc, lc, sd, T, out_dir, trials):
    hop = vc['hop_size']
    torch.manual_seed(0)
    mel = torch.randn(1, T, vc['in_channels'])
    f0  = torch.rand(1, 1, T) * 300 + 120
    uv  = (torch.rand(1, 1, T) > 0.2).float()
    n_out = (T if tag == 'V3' else 2 * T - 1) * hop
    dur = n_out / vc['sample_rate']
    full = FullCls(vc, lc, noise_on=True); load_core(full, sd, vc, lc); full.eval()
    path = os.path.join(out_dir, f'_bench_{tag}.onnx')
    import export as _ex
    _ex._export(full, (mel, f0, uv), path)
    feeds = {'mel': mel.numpy(), 'f0': f0.numpy(), 'uv': uv.numpy()}
    rt_torch = _rtf_torch(full, (mel, f0, uv), dur, trials)
    rt_ort = _rtf_ort(path, feeds, dur, trials)
    os.remove(path)
    return dur, rt_torch, rt_ort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config_v3.yaml')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--seconds', type=float, default=3.0)
    ap.add_argument('--trials', type=int, default=9)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config)); vc, lc = cfg['model']['vocoder'], cfg['model']['ltv_filter']
    sd = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = sd['model'] if isinstance(sd, dict) and 'model' in sd else sd

    n_params = sum(p.numel() for p in NHVSingV3(vc, lc).parameters())
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exported_models')
    os.makedirs(out_dir, exist_ok=True)

    T_v3 = int(round(args.seconds * vc['sample_rate'] / vc['hop_size']))          # hop256 frames
    T_v3x = int(round(args.seconds * vc['sample_rate'] / (vc['hop_size'] * 2)))    # hop512 frames

    rows = []
    for tag, Cls, T in [('V3', FullVocoderV3, T_v3), ('V3X', FullVocoderV3X, T_v3x)]:
        dur, rtt, rto = bench(tag, Cls, vc, lc, sd, T, out_dir, args.trials)
        rows.append((tag, dur, rtt, rto))

    print(f'\nNHVSing params: {n_params/1e6:.3f} M   (CPU, ~{args.seconds:.0f}s input, batch 1)\n')
    print('| Model | Backend | RTF | ×realtime |')
    print('|---|---|---|---|')
    for tag, dur, rtt, rto in rows:
        print(f'| {tag} | PyTorch (CPU)      | {rtt:.4f} | {1/rtt:.0f}× |')
        print(f'| {tag} | ONNX Runtime (CPU) | {rto:.4f} | {1/rto:.0f}× |')


if __name__ == '__main__':
    main()
