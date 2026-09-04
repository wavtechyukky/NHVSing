"""Export an OpenUtau vocoder package for NHVSing V3.1X.

OpenUtau's DiffSinger renderer drives the vocoder with a different contract than
our standard ``nhv_v3x.onnx``:

    OpenUtau : mel [B, T, 128], f0 [B, T] (2-D, gap-less)      -> waveform [B, N]
    ours     : mel [B, T, 128], f0 [B, 1, T], uv [B, 1, T]     -> waveform / harmonic / noise

This wrapper adapts everything inside the graph:

  * ``f0`` is unsqueezed to [B, 1, T]
  * ``uv`` is fixed to all-voiced (harmonic gate off). V3 is trained with
    ``vuv_dropout_prob: 1.0`` — the gate off and the mel suppressing unvoiced
    energy by itself — and OpenUtau's F0 is gap-less, so no external uv exists.
  * ``--mel-base 10`` bakes a x ln(10) rescale into the graph, for classic
    DiffSinger voicebanks whose acoustic model emits log10 mel (NHVSing wants ln).
    Use ``--mel-base e`` (default) for current-generation ln-mel voicebanks.
  * single output ``waveform`` [B, N]

Writes ``<out>/<name>/{<name>.onnx, vocoder.yaml}``. Install: copy that folder
into OpenUtau's ``Dependencies/`` and set ``vocoder: <name>`` in the voicebank's
``dsconfig.yaml``.

Usage:
    python export_openutau.py --config config_v3.yaml \
        --ckpt exported_models/v3_1/nhv_v3_1.pth --mel-base 10 --name nhvsing_v3_1x
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
import yaml

from export import FullVocoderV3X, load_core


class OpenUtauVocoderV3X(nn.Module):
    """OpenUtau-contract wrapper around FullVocoderV3X (hop512 input grid)."""

    def __init__(self, vc, lc, sd, noise_on=True, mel_scale=1.0):
        super().__init__()
        self.core = FullVocoderV3X(vc, lc, noise_on=noise_on)
        load_core(self.core, sd, vc, lc)
        self.mel_scale = float(mel_scale)

    def forward(self, mel, f0):
        mel = mel * self.mel_scale
        f0_3 = f0.unsqueeze(1)
        uv = torch.zeros_like(f0_3)              # all voiced = gate off (see module doc)
        waveform, _harm, _noise = self.core(mel, f0_3, uv)
        return waveform.reshape(waveform.shape[0], -1)   # [B, N]


def _export_onnx(mod, inputs, path):
    torch.onnx.export(mod, inputs, path, input_names=['mel', 'f0'],
                      output_names=['waveform'],
                      dynamic_axes={'mel': {0: 'B', 1: 'T'}, 'f0': {0: 'B', 1: 'T'},
                                    'waveform': {0: 'B', 1: 'N'}},
                      opset_version=18, do_constant_folding=True)
    import onnx
    m = onnx.load(path, load_external_data=True)
    # プライバシー: dynamo エクスポータの stack_trace メタデータ(ローカル絶対パス)を除去
    # (export.py の _export と同じ処理。重みも単一ファイルへ埋め込む)
    m.doc_string = ''
    del m.metadata_props[:]
    for g in [m.graph] + [f for f in m.functions]:
        for n in g.node:
            n.doc_string = ''
            del n.metadata_props[:]
    m.graph.doc_string = ''
    del m.graph.metadata_props[:]
    # OpenUtau 同梱の ONNX Runtime は IR version 9 までしか読めない(実機 0.1.565 で確認)。
    # torch の dynamo エクスポータは IR 10 で吐くが、IR10 固有要素(node metadata_props /
    # overload)は上で全て除去済みなので、9 へ書き下げれば合法な IR9 モデルになる。
    m.ir_version = 9
    for g in [m.graph] + [f for f in m.functions]:
        for n in g.node:
            if n.overload:
                n.overload = ''
    for f in m.functions:
        if f.overload:
            f.overload = ''
    onnx.checker.check_model(m)
    onnx.save_model(m, path, save_as_external_data=False)
    if os.path.exists(path + '.data'):
        os.remove(path + '.data')


def main():
    ap = argparse.ArgumentParser(description='OpenUtau vocoder package export (V3.1X)')
    ap.add_argument('--config', default='config_v3.yaml')
    ap.add_argument('--ckpt', default='exported_models/v3_1/nhv_v3_1.pth')
    ap.add_argument('--name', default='nhvsing_v3_1x')
    ap.add_argument('--mel-base', choices=['e', '10'], default='e',
                    help="what the acoustic model emits: 'e' = ln mel (current gen), "
                         "'10' = log10 mel (classic DiffSinger voicebanks)")
    ap.add_argument('--out', default='exported_models/openutau')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    vc, lc = cfg['model']['vocoder'], cfg['model']['ltv_filter']
    sd = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = sd['model'] if 'model' in sd else sd
    mel_scale = math.log(10.0) if args.mel_base == '10' else 1.0
    print(f'mel_base={args.mel_base} -> baked mel scale = {mel_scale:.6f}')

    out_dir = os.path.join(args.out, args.name)
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(0)
    T = 256                                       # hop512 frames (~3 s)
    mel = torch.randn(1, T, vc['in_channels']) - 4.0
    if args.mel_base == '10':
        mel = mel / math.log(10.0)                # pretend log10 input for the checks
    f0 = torch.rand(1, T) * 300 + 120

    # 1) torch-level parity: wrapper (det) vs inner path fed manually (det)
    det = OpenUtauVocoderV3X(vc, lc, sd, noise_on=False, mel_scale=mel_scale).eval()
    ref = FullVocoderV3X(vc, lc, noise_on=False)
    load_core(ref, sd, vc, lc)
    ref.eval()
    with torch.no_grad():
        a = det(mel, f0)
        b = ref(mel * mel_scale, f0.unsqueeze(1), torch.zeros(1, 1, T))[0].reshape(1, -1)
    print(f'wrapper vs inner (torch, det): max|diff| = {(a - b).abs().max().item():.3e}')

    # 2) export the deployable graph (noise on) + a det twin for strict ORT verification
    full = OpenUtauVocoderV3X(vc, lc, sd, noise_on=True, mel_scale=mel_scale).eval()
    p_full = os.path.join(out_dir, f'{args.name}.onnx')
    p_det = os.path.join(out_dir, f'_verify_{args.name}_det.onnx')
    _export_onnx(full, (mel, f0), p_full)
    _export_onnx(det, (mel, f0), p_det)

    import onnxruntime as ort
    feeds = {'mel': mel.numpy(), 'f0': f0.numpy()}
    y_ort = ort.InferenceSession(p_det, providers=['CPUExecutionProvider']).run(
        ['waveform'], feeds)[0]
    fid = float(np.abs(y_ort - a.numpy()).max())
    sess = ort.InferenceSession(p_full, providers=['CPUExecutionProvider'])
    wav = sess.run(['waveform'], feeds)[0]
    n_expect = (2 * T - 1) * vc['hop_size'] // 2 * 2      # V3X: hop512 grid -> ~2T-1 hop256 frames
    print(f'onnx-vs-torch (det) = {fid:.2e}   full out shape = {wav.shape}  '
          f'finite = {bool(np.isfinite(wav).all())}')
    os.remove(p_det)

    # 3) privacy scan
    blob = open(p_full, 'rb').read()
    hits = [s for s in (b'/Users/', b'/home/', b'stack_trace') if s in blob]
    print('privacy scan:', hits or 'CLEAN', f'  ({os.path.getsize(p_full)/1e6:.2f} MB)')

    # 4) vocoder.yaml — the hop-512 input grid the package presents to OpenUtau
    voc_yaml = {
        'name': args.name,
        'model': f'{args.name}.onnx',
        'sample_rate': int(vc['sample_rate']),
        'hop_size': 512,
        'win_size': 2048,
        'fft_size': 2048,
        'num_mel_bins': int(vc['in_channels']),
        'mel_fmin': float(cfg['preprocess']['mel_min']),
        'mel_fmax': float(cfg['preprocess']['mel_max']),
    }
    with open(os.path.join(out_dir, 'vocoder.yaml'), 'w') as f:
        yaml.safe_dump(voc_yaml, f, sort_keys=False)
    print('wrote', out_dir, '->', sorted(os.listdir(out_dir)))


if __name__ == '__main__':
    main()
