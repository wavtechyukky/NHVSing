[日本語](./README.md) | [English](./README.en.md)

# NHV-Sing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Neural Homomorphic Vocoder](https://www.isca-archive.org/interspeech_2020/liu20_interspeech.pdf)を、**歌声合成向けにチューニングした**モデルです。PyTorchで実装されており、JITコンパイルおよび単一ファイルONNXエクスポートに対応しています。元の論文で提案された構造を踏襲しつつ、歌声合成への適用を考慮した変更を加えています。

***

## 音声サンプル

### 東北きりたんサンプル

**Ground Truth:** [ground_truth_kiritan.wav](sample_wav/ground_truth_kiritan.wav)

**生成された音声:** [output_onnx.wav](output_kiritan/output_onnx.wav)

### 夏目悠李サンプル

**Ground Truth:** [ground_truth_natsume.wav](sample_wav/ground_truth_natsume.wav)

**生成された音声:** [output_onnx.wav](output_natsume/output_onnx.wav)

## 特徴

### 性能

*   **軽量・高速:** 4MB程度のモデルサイズでありながら、一般的なPCのCPU環境でも非常に高速な推論を実現します。
*   **高い再現性:** 話者の声質を忠実に再現します。
*   **安定した品質:** F0（基本周波数）に忠実で、ロングトーンも破綻なく安定して合成できます。

本ボコーダーのRTF（Real-Time Factor）および平均推論時間は、以下の環境と条件で測定されました。

- **測定環境:** Apple M-series CPU（MacBook）
- **測定条件:**
    - 入力音声サンプリングレート: 44.1kHz
    - 入力音声の長さ: 約26秒
    - バッチサイズ: 1

| Model Type     | Device | Avg. Inference Time | RTF      |
|----------------|--------|---------------------|----------|
| Native Python  | CPU    | 1.923 sec           | 0.0740   |
| JIT Script     | CPU    | 1.935 sec           | 0.0745   |
| Unified ONNX   | CPU    | 4.586 sec           | 0.1765   |

### オリジナル実装との差異
元の論文の実装から以下の点を変更しています。（一部のパラメータはconfig.yamlから編集可能）

* **サンプリング周波数**: **44.1kHz**に対応
* **複素ケプストラム**: 次元数を**512次元**に拡張
* **FIR（postfilter）の削除**: STFT損失の低下には寄与するものの、処理が遅くなり、直感的に重要となる波形の学習に寄与しないと判断
* **Discriminator**: Multi-Scale Waveform Discriminator + Multi-Scale Complex STFT Discriminatorを使用。既存学習の崩壊を防ぐため、Adversarial lossにwarmup期間を設けており、学習途中からのFine-tuningにも対応する（`adversarial_warmup_epochs`）
* **損失関数の追加**:
    * **Envelope loss**: 1D max-poolingで上下包絡を抽出しMAEを計算（RefineGAN §2.5.1）。振幅包絡の不安定化を抑制する（`envelope_scale`）
    * **Harmonic penalty loss**: 無声区間（F0=0のフレーム）で有声成分（`sig_harm`）が出力されることへのL1ペナルティ。無声区間でブザーのような音が生成されることを抑制する（`harmonic_penalty_scale`）
* **quef_norm**: デフォルトでオフ（`use_quef_norm: false`）。オンにすると高周波数帯の学習が阻害されることが実験で確認されたため
* **入力特徴量**:
    * **logメルスペクトログラム**: **40Hz〜22050Hz**のlogメルスペクトログラムを入力とする。論文では高周波数帯をカットしているが、高周波数帯の再現度が直感的な品質の向上に必要であると判断した。
    * **F0**: **無声区間を線形補完**したF0を入力とし、Unvoiced/Voicedフラグを不要にした。歌声合成では無声区間も含めてF0カーブを描けることが重視されるだけでなく、論文のようにUVフラグによって挙動を変える場合、なめらかな無声区間→有声区間の推移が再現できない。

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

## 使い方

### 1. 前処理

WAVファイルからモデルの学習に必要な特徴量（npz形式）を抽出します。

```bash
# WAVカット → F0/メル抽出 → train/test振り分けをまとめて実行
python preprocess.py --step all
```

各ステップを個別に実行することも可能です。

```bash
python preprocess.py --step resample  # リサンプリング
python preprocess.py --step cut       # 無音カット
python preprocess.py --step npz       # npz作成（F0・メルスペクトログラム）
python preprocess.py --step split     # train/test振り分け
```

train/test比率は `config.yaml` の `preprocess.train_test_split` で設定できます（デフォルト: 100:1）。

### 2. 学習

モデルの学習を開始します。`config.yaml` の `training.log_dir` にTensorboardログが、`training.snapshot_dir` にスナップショットが保存されます。

```bash
python train.py
```

Gradient accumulationおよびAMP（混合精度）に対応しています。`config.yaml` の `gradient_accumulation_steps` と `use_amp` で設定できます。

### 3. Fine-tuning（新しい話者への転移学習）

学習済みモデルのウェイトのみを引き継ぎ、別の話者のデータセットで再学習する場合に使います。DiscriminatorとOptimizerは新規初期化されます。

```bash
# 例: 夏目悠李ベースモデル → きりたんFine-tuning
python prepare_finetune.py \
  --weights exported_natsume/model.pth \
  --config config.yaml \
  --output snapshots_kiritan/000000epoch.pth

python train.py --resume_path snapshots_kiritan/000000epoch.pth --config config_fine_tuning.yaml
```

`--weights` には、エクスポートした `model.pth`（state_dict）と学習スナップショットのどちらも指定可能です。

### 4. モデルのエクスポート

学習済みモデル（スナップショット）を推論用の形式にエクスポートします。

```bash
# PyTorchネイティブ + JIT + Unified ONNXをまとめてエクスポート
python export.py \
  --checkpoint snapshots/000990epoch.pth \
  --config config.yaml \
  --output_dir exported_models \
  --pytorch \
  --jit \
  --full_onnx
```

| オプション     | 出力ファイル              | 説明                       |
|----------------|---------------------------|----------------------------|
| `--pytorch`    | `model.pth`               | state_dict（再学習・Fine-tuning用）|
| `--jit`        | `model_jit.pt`            | TorchScript（他言語連携用）|
| `--full_onnx`  | `full_vocoder.onnx`       | 単一ONNXファイル           |

### 5. 推論

エクスポートしたモデルを使って、WAVまたはNPZファイルから音声を生成します。

```bash
# PyTorchモデルで推論（Native + JITの速度比較も実施）
python inference.py input.wav \
  --snapshot exported_models/model.pth \
  --config config.yaml \
  --output_dir output

# Unified ONNXも同時に計測する場合
python inference.py input.wav \
  --snapshot exported_models/model.pth \
  --config config.yaml \
  --output_dir output \
  --onnx exported_models/full_vocoder.onnx
```

***

## 課題点

*   **学習プロセス:** 複数のGPUの使用を想定しておりません。
*   **他のフレームサイズへの対応:** デフォルトのフレームサイズ（hop_size=256）のみで動作確認しています。他のフレームサイズでの動作は保証していません。
*   **話者依存性:** 生成される波形には学習させた話者の特徴が反映され、多人数の声を再現することは難しいです。その代わり、FastSpeech2のような曖昧な音響特徴量を入れてもリアルな質感の声にしてくれます。

## ライセンス

このプロジェクトは [MIT License](LICENSE) のもとで公開されています。

## 謝辞

このリポジトリは、Liu, et al.によって発表された以下の論文やリポジトリに基づいています。

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
