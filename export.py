"""NHVSing V3 / V3X の ONNX エクスポート + 検証 + RTF 計測。

V3  … hop256 native。入力 mel[B,T,mel_dim] / f0[B,1,T] / uv[B,1,T]。
V3X … hop512 入力を内部で hop256 へ整列補間してから V3。OpenUtau 等の hop512 mel をそのまま食う。

ONNX グラフは dsp_rebuild の ONNX 互換部品(GenerateImpulseTrainONNX / ComplexCepstrumToImpONNX /
LTVFirONNX)で組む。LTVFirONNX は FFT 長を 2 の冪へ pad 済み(ORT DFT が非冪で ~4-8x 遅い対策・bit 等価)。
noise z は内部生成(RandomNormal)。uv は f0(=0 で無声)から導出すべきだが RTF 計測のため外部入力にしている。

Usage:
    python export.py --config config_v3.yaml --ckpt path/to/weights.ckpt --out exported_models
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
import yaml

from model import NHVSingV3
from onnx_model import NHVConvsShared
from dsp_rebuild.impulse_train_onnx import GenerateImpulseTrainONNX
from dsp_rebuild.complex_cepstrum_to_imp_onnx import ComplexCepstrumToImpONNX
from dsp_rebuild.ltv_fir_onnx import LTVFirONNX


def _up_lin_t(x):
    mid = 0.5 * (x[:, :-1] + x[:, 1:])
    out = torch.stack([x[:, :-1], mid], dim=2).reshape(x.shape[0], -1, x.shape[2])
    return torch.cat([out, x[:, -1:]], dim=1)

def _up_lin_c(x):
    mid = 0.5 * (x[..., :-1] + x[..., 1:])
    out = torch.stack([x[..., :-1], mid], dim=-1).reshape(x.shape[0], x.shape[1], -1)
    return torch.cat([out, x[..., -1:]], dim=-1)

def _up_hold_c(x):
    return torch.repeat_interleave(x, 2, dim=-1)


class FullVocoderV3(nn.Module):
    """V3 の全パイプラインを ONNX 互換部品で(hop256 native)。noise_on=False で harmonic のみ(決定的)。"""
    def __init__(self, vocoder_cfg, ltv_filter_cfg, noise_on=True):
        super().__init__()
        self.hop_size    = vocoder_cfg['hop_size']
        self.noise_std   = float(vocoder_cfg.get('noise_std', 0.03))
        self.noise_on    = noise_on
        self.nn_core     = NHVConvsShared(dict(ltv_filter_cfg))
        self.impulse_train = GenerateImpulseTrainONNX(int(vocoder_cfg.get('n_harmonic', 200)),
                                                      vocoder_cfg['sample_rate'])
        self.ccep_to_imp = ComplexCepstrumToImpONNX(ltv_filter_cfg['fft_size'], use_float64=False)
        self.ltv_fir     = LTVFirONNX(self.hop_size, filter_size=ltv_filter_cfg['fft_size'])  # pow2 pad 済

    def forward(self, mel, f0, uv):
        ccep_harm, ccep_noise = self.nn_core(mel)
        cf0 = torch.nn.functional.interpolate(f0, scale_factor=self.hop_size,
                                              mode='linear', align_corners=False)  # V3=linear
        harmonic_source = self.impulse_train(cf0)
        voiced = torch.repeat_interleave(1.0 - uv, self.hop_size, dim=-1)
        harmonic_source = harmonic_source * voiced
        sig_harm = self.ltv_fir(harmonic_source, self.ccep_to_imp(ccep_harm))
        if self.noise_on:
            z = torch.randn_like(harmonic_source) * self.noise_std
            sig_noise = self.ltv_fir(z, self.ccep_to_imp(ccep_noise))
        else:
            sig_noise = torch.zeros_like(sig_harm)
        waveform = torch.clamp(sig_harm + sig_noise, -1.0, 1.0)
        # 3 出力: 合成波形 + harmonic / noise(生・pre-clamp)。消費側で成分ゲイン等に使える。
        return waveform, sig_harm, sig_noise


class FullVocoderV3X(FullVocoderV3):
    """hop512 入力 → 内部整列補間 → V3(ONNX 部品)。"""
    def forward(self, mel512, f0_512, uv_512):
        mel = _up_lin_t(mel512)
        f0  = _up_lin_c(f0_512)
        uv  = _up_hold_c(uv_512)[..., :f0.shape[-1]]
        return super().forward(mel, f0, uv)


def load_core(full, ckpt_sd, vc, lc):
    """ckpt(weight_norm 済)を V3 に load→除去→convs_onnx を full.nn_core へコピー。"""
    v3 = NHVSingV3(vc, lc); v3.load_state_dict(ckpt_sd); v3.remove_weight_norm()
    core = {k.replace('convs_onnx.', '', 1): v
            for k, v in v3.state_dict().items() if k.startswith('convs_onnx.')}
    m, u = full.nn_core.load_state_dict(core, strict=False)
    return len(core), len(m), len(u)


def _export(mod, inputs, path):
    torch.onnx.export(mod, inputs, path, input_names=['mel', 'f0', 'uv'],
                      output_names=['waveform', 'harmonic', 'noise'],
                      dynamic_axes={'mel': {0: 'B', 1: 'T'}, 'f0': {0: 'B', 2: 'T'}, 'uv': {0: 'B', 2: 'T'},
                                    'waveform': {0: 'B', 2: 'N'}, 'harmonic': {0: 'B', 2: 'N'},
                                    'noise': {0: 'B', 2: 'N'}},
                      opset_version=18, do_constant_folding=True)
    # 新エクスポータは重みを path+'.data' に外部化することがある。V1/V2 の full_vocoder.onnx と同じ
    # 「重み内蔵の単一ファイル」にするため、外部データを読み込んで埋め込み再保存し .data を削除。
    import onnx
    m = onnx.load(path, load_external_data=True)
    onnx.save_model(m, path, save_as_external_data=False)
    if os.path.exists(path + '.data'):
        os.remove(path + '.data')


def _rtf_ort(session, feeds, dur, n=7):
    for _ in range(2): session.run(None, feeds)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); session.run(None, feeds); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) / dur


def _rtf_torch(mod, inputs, dur, n=7):
    with torch.no_grad():
        for _ in range(2): mod(*inputs)
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); mod(*inputs); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) / dur


def export_variant(tag, FullCls, vc, lc, sd, T, out_dir):
    import onnxruntime as ort
    hop = vc['hop_size']
    torch.manual_seed(0)
    mel = torch.randn(1, T, vc['in_channels'])
    f0  = torch.rand(1, 1, T) * 300 + 120
    uv  = (torch.rand(1, 1, T) > 0.2).float()
    n_out = (T if tag == 'v3' else 2 * T - 1) * hop
    dur = n_out / vc['sample_rate']

    full = FullCls(vc, lc, noise_on=True);  n, mm, uu = load_core(full, sd, vc, lc); full.eval()
    det  = FullCls(vc, lc, noise_on=False); load_core(det, sd, vc, lc); det.eval()
    p_full = os.path.join(out_dir, f'nhv_{tag}.onnx')          # ← デプロイ用の単一 ONNX(V1/V2 の full_vocoder 相当)
    p_det = os.path.join(out_dir, f'_verify_{tag}_det.onnx')   # 検証専用(noise off で厳密照合)→ 後で削除
    _export(full, (mel, f0, uv), p_full)
    _export(det,  (mel, f0, uv), p_det)

    feeds = {'mel': mel.numpy(), 'f0': f0.numpy(), 'uv': uv.numpy()}
    y_ort = ort.InferenceSession(p_det, providers=['CPUExecutionProvider']).run(['waveform'], feeds)[0].reshape(1, -1)
    with torch.no_grad():
        y_det = det(mel, f0, uv)[0].reshape(1, -1).numpy()   # 出力[0]=waveform
    fid = float(np.abs(y_ort - y_det).max())
    # 3出力の整合: clamp(harmonic + noise) == waveform か(成分だけ取っても足せば波形に戻る)
    sess = ort.InferenceSession(p_full, providers=['CPUExecutionProvider'])
    wf, hm, ns = sess.run(['waveform', 'harmonic', 'noise'], feeds)
    comp = float(np.abs(np.clip(hm + ns, -1.0, 1.0) - wf).max())
    rt_torch = _rtf_torch(full, (mel, f0, uv), dur)
    rt_ort   = _rtf_ort(sess, feeds, dur)
    del sess
    for _p in (p_det, p_det + '.data'):                       # 検証用は削除 → 出力は nhv_{tag}.onnx の単一ファイル
        if os.path.exists(_p):
            os.remove(_p)
    print(f'[{tag}] -> {os.path.basename(p_full)} ({os.path.getsize(p_full)/1024/1024:.2f}MB, 単一)  '
          f'copied {n}(miss {mm}/unexp {uu})  onnx-vs-torch={fid:.2e}  clamp(harm+noise)-wave={comp:.2e}  '
          f'RTF torch={rt_torch:.4f}({1/rt_torch:.0f}x)  ORT={rt_ort:.4f}({1/rt_ort:.0f}x)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config_v3.yaml')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default='exported_models/v3')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config)); vc, lc = cfg['model']['vocoder'], cfg['model']['ltv_filter']
    sd = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = sd['model'] if 'model' in sd else sd
    os.makedirs(args.out, exist_ok=True)
    p_pth = os.path.join(args.out, 'nhv_v3.pth')             # V3/V3X 共有の重み(NHVSingV3 の state_dict)。V3X は重み共有なので onnx のみ
    torch.save(sd, p_pth)
    print(f'saved nhv_v3.pth ({os.path.getsize(p_pth)/1024/1024:.2f}MB, V3/V3X 共有重み)')
    export_variant('v3',  FullVocoderV3,  vc, lc, sd, T=512, out_dir=args.out)
    export_variant('v3x', FullVocoderV3X, vc, lc, sd, T=256, out_dir=args.out)
    print('done.')


if __name__ == '__main__':
    main()
