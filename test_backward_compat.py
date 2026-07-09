"""回帰テスト(pytest 不要・plain python)。

公開する4クラス NHVSing(V1) / NHVSingV2 / NHVSingV3 / NHVSingV3X について:
  - config から build できる
  - forward が有限値・想定 shape を返す(V3=hop256, V3X=hop512入力)
  - state_dict の round-trip(strict=True)が通る
  - select_model_class の flag → クラス対応
オプション: --ckpt を渡すと V3 に strict ロードして shape 検証。

Usage: python test_backward_compat.py [--ckpt path/to/weights.ckpt]
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import yaml

from model import NHVSing, NHVSingV2, NHVSingV3, NHVSingV3X, select_model_class

HERE = os.path.dirname(os.path.abspath(__file__))
B, T = 2, 64
_pass = _fail = 0


def _ok(name, cond, extra=''):
    global _pass, _fail
    print(f'  [{"PASS" if cond else "FAIL"}] {name} {extra}')
    _pass += int(bool(cond)); _fail += int(not cond)


def _cfg(fname):
    p = os.path.join(HERE, fname)
    return yaml.safe_load(open(p)) if os.path.exists(p) else None


def _rand_inputs(mel_dim, t):
    x  = torch.randn(B, t, mel_dim)
    f0 = torch.rand(B, 1, t) * 300 + 120
    uv = (torch.rand(B, 1, t) > 0.5).float()
    return x, f0, uv


def test_dispatch():
    print('[dispatch]')
    base = {'use_shared_trunk': False}
    _ok('V1 (既定)', select_model_class({}, {}) is NHVSing)
    _ok('V2 (use_shared_trunk)', select_model_class({}, {'use_shared_trunk': True}) is NHVSingV2)
    _ok('V3 (use_v3)', select_model_class({}, {'use_v3': True}) is NHVSingV3)
    _ok('V3 (use_v6 互換)', select_model_class({}, {'use_v6': True}) is NHVSingV3)
    _ok('V3X (use_v3x)', select_model_class({}, {'use_v3x': True}) is NHVSingV3X)


def _build_forward_roundtrip(Cls, cfg, tag, hop512=False, uv=True):
    vc, lc = cfg['model']['vocoder'], cfg['model']['ltv_filter']
    m = Cls(vc, lc); m.eval()
    # forward の入力はメル(V2 は内部で f0_embed を concat するので入力自体はメル次元)
    mel_dim = cfg.get('preprocess', {}).get('mel_dim', 128)
    x, f0, uvt = _rand_inputs(mel_dim, T)
    with torch.no_grad():
        if uv:
            y = m(x, f0, uvt, noise_std=0.0)
        else:
            y = m(x, f0, noise_std=0.0)
    hop = vc['hop_size']
    exp = ((2 * T - 1) if hop512 else T) * hop
    _ok(f'{tag} forward', tuple(y.shape) == (B, exp) and bool(torch.isfinite(y).all()),
        f'shape={tuple(y.shape)} exp=({B},{exp})')
    # round-trip strict
    m2 = Cls(vc, lc)
    miss, unexp = m2.load_state_dict(m.state_dict(), strict=True)
    _ok(f'{tag} round-trip strict', not miss and not unexp)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None)
    args = ap.parse_args()

    test_dispatch()

    c3 = _cfg('config_v3.yaml')
    print('[V3 / V3X]')
    _build_forward_roundtrip(NHVSingV3, c3, 'V3', hop512=False, uv=True)
    _build_forward_roundtrip(NHVSingV3X, c3, 'V3X', hop512=True, uv=True)

    for fname, Cls, tag, uv in [('config.yaml', NHVSing, 'V1', False),
                                ('config_v2.yaml', NHVSingV2, 'V2', False)]:
        c = _cfg(fname)
        print(f'[{tag}]')
        if c is None:
            _ok(f'{tag} config 存在', False, f'({fname} が無い)'); continue
        try:
            _build_forward_roundtrip(Cls, c, tag, hop512=False, uv=uv)
        except Exception as e:
            _ok(f'{tag} build/forward', False, f'{type(e).__name__}: {e}')

    if args.ckpt and os.path.exists(args.ckpt):
        print('[ckpt strict load → V3]')
        sd = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        sd = sd['model'] if isinstance(sd, dict) and 'model' in sd else sd
        vc, lc = c3['model']['vocoder'], c3['model']['ltv_filter']
        m = NHVSingV3(vc, lc)
        miss, unexp = m.load_state_dict(sd, strict=True)
        _ok('V3 strict load ckpt', not miss and not unexp)

    print(f'\n=== {_pass} passed, {_fail} failed ===')
    sys.exit(1 if _fail else 0)


if __name__ == '__main__':
    main()
