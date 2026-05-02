[日本語](./README.md) | [English](./README.en.md)

# NHV-Sing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Neural Homomorphic Vocoder](https://www.isca-archive.org/interspeech_2020/liu20_interspeech.pdf)を、**歌声合成向けにチューニングした**モデルです。PyTorchで実装されており、JITコンパイルおよび単一ファイルONNXエクスポートに対応しています。

本リポジトリには **NHVSing**（単一話者特化モデル）と **NHVSingV2**（多話者汎用モデル）の2つのモデルクラスが含まれます。

***

## 音声サンプル

→ **[NHVSingV2 デモページで試聴する](https://wavtechyukky.github.io/NHVSing/v2.html)**

→ [NHVSing (V1) デモページ](https://wavtechyukky.github.io/NHVSing/)

きりたん・夏目悠李（NHVSing）、およびM4Singer・GTSinger評価データ（NHVSingV2）による生成音声を聴き比べできます。

***

## NHVSing と NHVSingV2 の違い

| | NHVSing | NHVSingV2 |
|---|---|---|
| CNN backbone | Dual-branch（Harmonic/Noise独立） | Shared trunk（CNNを共有） |
| F0入力 | メルスペクトログラムのみ | F0 embedder（256 bins, log₂スケール, 128-dim）を追加結合 |
| quef_norm | オフ（V1では高周波学習が阻害されたため） | alpha=0.3のソフトスケーリング（高周波を犠牲にせず安定化） |
| 学習時の振幅拡張 | なし | 0.5〜2.0×（log-uniform）でランダムスケール |
| 話者汎用性 | **単一話者特化**。学習話者以外ではアーティファクトが出やすい | **多話者対応**。M4Singer・ACE-Opencpop等の多話者コーパスで学習可能 |
| 設定ファイル | `config.yaml` | `config_v2.yaml` |

NHVSingは単一話者のデータセットで学習することで、その話者の声質を忠実に再現することに優れています。一方NHVSingV2は多話者コーパスで学習することにより、様々な話者の音響特徴量を高品質にボコードすることが可能です。

***

## 性能

以下の環境・条件で測定しました。

- **測定環境:** Apple M-series CPU（MacBook）
- **測定条件:** 入力44.1kHz・約26秒・バッチサイズ1

### NHVSing（V1）

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

## アーキテクチャ

元の論文の実装から以下の点を変更しています。

* **サンプリング周波数**: **44.1kHz**に対応
* **複素ケプストラム**: 次元数を**512次元**に拡張
* **FIR（postfilter）の削除**: STFT損失の低下には寄与するものの、波形の学習に寄与しないと判断
* **Discriminator**: Multi-Scale Waveform Discriminator + Multi-Scale Complex STFT Discriminatorを使用。Adversarial lossにwarmup期間を設けており、学習途中からのFine-tuningにも対応（`adversarial_warmup_epochs`）
* **損失関数の追加**:
    * **Envelope loss**: 1D max-poolingで上下包絡を抽出しMAEを計算（RefineGAN §2.5.1）。振幅包絡の不安定化を抑制する（`envelope_scale`）
    * **Harmonic penalty loss**: 無声区間（F0=0のフレーム）で有声成分（`sig_harm`）が出力されることへのL1ペナルティ。無声区間でのブザー音を抑制する（`harmonic_penalty_scale`）
* **F0入力**: **無声区間を線形補完**したF0を入力とし、Unvoiced/Voicedフラグを不要にした。歌声合成では無声区間も含めてF0カーブを描けることが重要で、UVフラグによる挙動切り替えでは無声→有声のなめらかな推移が再現できない
* **logメルスペクトログラム**: **40Hz〜22050Hz**の全帯域を入力とする。高周波数帯の再現度が直感的な品質向上に寄与すると判断した

### NHVSingV2 固有の変更点

* **Shared trunk CNN** (`use_shared_trunk: true`): HarmonicとNoiseの両ブランチが共有の幹CNNを通ってからそれぞれのヘッドへ分岐する。これによって入力された音響特徴量を、HarmonicとNoiseのどちらを支配的にして生成すべきかを、両者が独立して学習する必要がなくなった。
* **F0 Embedder** (`use_f0_embed: true`): 連続F0を256ビンのlog₂スケールで離散化し、128次元に埋め込んでメルスペクトログラムと結合する。ネットワークに明示的な音高情報を与えることで、生成すべき一周期分の波形の手がかりが増える。
* **quef_norm** (`use_quef_norm: true`, `quef_norm_alpha: 0.3`): ケフレンシー成分に1/|n|^αの緩やかなスケーリングをかけ、高次倍音を抑制しすぎずに学習を安定化させる。V1ではオンにすると高周波数帯の阻害されたが、V2ではalphaを0.3と小さく設定することで高周波を犠牲にせず安定化できることが確認された

### エクスポート形式

*  **PyTorchネイティブ** (`model.pth`): 学習時に使うモデルと同じ。
*  **TorchScript** (`model_jit.pt`): JITコンパイルにより他の言語から実行できるようになる。
*  **Unified ONNX** (`full_vocoder.onnx`): ボコーダー全体（NN + DSP）を単一のONNXファイルにエクスポート。ONNXRuntimeのみで推論可能。

***

## 環境

* Python 3.10で検証

```bash
pip install -r requirements.txt
```

***

## 使い方

### NHVSing（V1）の場合

#### 1. 前処理

```bash
# WAVカット → F0/メル抽出 → train/test振り分けをまとめて実行
python preprocess.py --config config.yaml --step all
```

#### 2. 学習

```bash
python train.py --config config.yaml
```

#### 3. Fine-tuning（別話者への転移学習）

学習済みモデルのウェイトのみを引き継ぎ、別話者のデータセットで再学習します。DiscriminatorとOptimizerは新規初期化されます。

```bash
python prepare_finetune.py \
  --weights exported_models/natsume/model.pth \
  --config config.yaml \
  --output snapshots_kiritan/000000epoch.pth

python train.py --resume_path snapshots_kiritan/000000epoch.pth --config config_fine_tuning.yaml
```

#### 4. エクスポート

```bash
python export.py \
  --checkpoint snapshots/000990epoch.pth \
  --config config.yaml \
  --output_dir exported_models/kiritan \
  --all
```

#### 5. 推論

```bash
python inference.py input.wav \
  --snapshot exported_models/kiritan/model.pth \
  --config config.yaml \
  --output_dir output \
  --onnx exported_models/kiritan/full_vocoder.onnx
```

---

### NHVSingV2 の場合

`config_v2.yaml` を起点として使います。パス・話者プレフィックス・学習パラメータを環境に合わせて編集してください。

#### 1. 前処理

F0抽出にRMVPEを使用します（初回実行時にモデルを自動ダウンロード）。

```bash
python preprocess.py --config config_v2.yaml --step all
```

#### 2. 学習

```bash
python train.py --config config_v2.yaml
```

#### 3. Fine-tuning（別話者・別コーパスへの転移学習）

```bash
python prepare_finetune.py \
  --weights exported_models/v2/model.pth \
  --config config_v2.yaml \
  --output snapshots_finetune/000000epoch.pth

python train.py --resume_path snapshots_finetune/000000epoch.pth --config config_v2.yaml
```

#### 4. エクスポート

```bash
python export.py \
  --checkpoint snapshots_v2/000900epoch.pth \
  --config config_v2.yaml \
  --output_dir exported_models/v2 \
  --all
```

#### 5. 推論

```bash
python inference.py input.wav \
  --snapshot exported_models/v2/model.pth \
  --config config_v2.yaml \
  --output_dir output \
  --onnx exported_models/v2/full_vocoder.onnx
```

WAV入力の場合、`config_v2.yaml` の `target_rms: 0.083` が設定されていれば有声区間のRMSをM4Singer学習データの中央値に自動正規化します。

***

## 学習のベストプラクティス

### 振幅拡張（`amp_augment`）

NHVSingV2では学習時に音量を0.5〜2.0倍（log-uniform）でランダムスケールする振幅拡張を導入しています（`amp_augment: true`, `amp_aug_range: [0.5, 2.0]`）。これにより、入力音量の変動に対してモデルが頑健になり、推論時の音量合わせが多少ずれても品質が劣化しにくくなります。

### quef_norm のスケール（`quef_norm_alpha`）

`use_quef_norm: true` にすることでケフレンシー成分を正規化し、学習の安定性が向上します。ただしalphaを大きくしすぎると高周波数帯（子音・摩擦音）の再現度が低下します。`quef_norm_alpha: 0.3` が高周波を犠牲にせずに安定化できる妥当な値です。

### Harmonic penalty loss の強さ（`harmonic_penalty_scale`）

無声区間（F0=0フレーム）でハーモニック成分が漏れ出すことを抑制するペナルティです。この値は品質に大きく影響します。

* 小さすぎる（0〜10程度）: 無声区間でブザーのような音が発生しやすい
* 大きすぎる（1000以上）: 有声/無声か怪しい音響特徴量が与えられた時に、本来は有声区間であっても、無声音でメルスペクトログラムを再現しようとしてしまう。
* **推奨値: `harmonic_penalty_scale: 100`**

***

## 課題点

*   **学習プロセス:** 複数のGPUの使用を想定していません。
*   **他のフレームサイズへの対応:** hop_size=512にすると非常に品質が落ちます。
*   **NHVSing（V1）の話者依存性:** 生成される波形には学習させた話者の特徴が強く反映されます。多話者対応が必要な場合はNHVSingV2を使用してください。

***

## ライセンス

このプロジェクトは [MIT License](LICENSE) のもとで公開されています。

## 謝辞

このリポジトリは、Liu, et al.によって発表された以下の論文・リポジトリに基づいています。

*   Z. Liu, Y. Wang, K. Chen and Y. Jia, "Neural Homomorphic Vocoder," *Proc. Interspeech 2020*, pp. 3500-3504, doi: 10.21437/Interspeech.2020-2325.
*   [https://www.isca-archive.org/interspeech_2020/liu20_interspeech.pdf](https://www.isca-archive.org/interspeech_2020/liu20_interspeech.pdf)
*   [https://github.com/xcmyz/FastVocoder/tree/main](https://github.com/xcmyz/FastVocoder/tree/main)
*   [https://github.com/zjlww/dsp](https://github.com/zjlww/dsp)
*   [https://pypi.org/project/neural-homomorphic-vocoder/](https://pypi.org/project/neural-homomorphic-vocoder/)

## 使用した歌声データベース

*   東北きりたん — [Zunko Project](https://zunko.jp/kiridev/login.php)
*   夏目悠李 — [NJKS Official](https://ksdcm1ng.wixsite.com/njksofficial)
*   M4Singer (CC BY-NC-SA 4.0) — [M4Singer GitHub](https://github.com/M4Singer/M4Singer)
    *   Zhang et al., "M4Singer: a Multi-Style, Multi-Singer and Musical Score Provided Mandarin Singing Corpus," *NeurIPS 2022*.
*   ACE-Opencpop — [HuggingFace](https://huggingface.co/datasets/espnet/ace-opencpop-segments)
