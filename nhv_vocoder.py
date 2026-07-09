"""NHVSing 推論ラッパ — ckpt + config から mel/f0/uv → wav。

    from nhv_vocoder import NHVVocoder
    voc = NHVVocoder('weights.ckpt', 'config_v3.yaml', device='cpu')
    cf0, uv = NHVVocoder.prep_f0(f0_raw_hz)     # 0=無声 の生 F0 → 連続 F0 + uv
    wav = voc.infer(mel, cf0, uv)               # mel:[T, mel_dim] ln-mel

config の use_v3x=true なら V3X(mel/f0/uv は hop512 グリッド)、それ以外は V3(hop256)。
モデルクラスは select_model_class が config フラグから自動選択する。
"""
import numpy as np
import torch
import yaml

from model import select_model_class


class NHVVocoder:
    def __init__(self, ckpt_path: str, config_path: str, device: str = 'cpu'):
        cfg = yaml.safe_load(open(config_path))
        vc, lc = cfg['model']['vocoder'], cfg['model']['ltv_filter']
        self.device = torch.device(device)
        self.sample_rate = int(vc['sample_rate'])
        self.hop_size = int(vc['hop_size'])
        ModelCls = select_model_class(vc, lc)
        self.name = ModelCls.__name__
        self.model = ModelCls(vc, lc)
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = sd['model'] if isinstance(sd, dict) and 'model' in sd else sd
        self.model.load_state_dict(sd)
        self.model.remove_weight_norm()
        self.model = self.model.to(self.device).eval()

    @staticmethod
    def prep_f0(f0_hz):
        """0=無声 の生 F0[Hz] → (連続 F0[Hz], uv[1=無声])。無声フレームは線形補間。
        (dataset.norm_interp_f0 と同一規約)"""
        f0 = np.asarray(f0_hz, dtype=np.float32).copy()
        uv = f0 == 0
        if 0 < uv.sum() < len(f0):
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        return f0, (1 * uv).astype(np.float32)

    @torch.no_grad()
    def infer(self, mel, cf0, uv, seed: int | None = None, return_components: bool = False):
        """mel:[T, mel_dim] ln-mel / cf0:[T] 連続F0[Hz] / uv:[T] 1=無声 → wav[N] float32。
        V3X の場合、mel/cf0/uv は hop512 グリッド(内部で hop256 へ補間)。
        return_components=True で (wav, harmonic, noise) を返す(harmonic/noise は生・pre-clamp)。"""
        if seed is not None:
            torch.manual_seed(int(seed))
        mel_t = torch.as_tensor(np.asarray(mel), dtype=torch.float32, device=self.device).unsqueeze(0)  # [1,T,C]
        cf0_t = torch.as_tensor(np.asarray(cf0), dtype=torch.float32, device=self.device).reshape(1, 1, -1)
        uv_t  = torch.as_tensor(np.asarray(uv),  dtype=torch.float32, device=self.device).reshape(1, 1, -1)
        if return_components:
            # forward_train は (wav[B,N], harmonic[B,1,N], noise[B,1,N])。wav は既に clamp 済み。
            wav, harm, noise = self.model.forward_train(mel_t, cf0_t, uv_t)
            to_np = lambda t: t.detach().cpu().reshape(-1).numpy().astype(np.float32)
            return to_np(wav), to_np(harm), to_np(noise)
        wav = self.model(mel_t, cf0_t, uv_t)
        return wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
