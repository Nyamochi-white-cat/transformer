# Transformer Lab

PyTorchでTransformerを実装し、原論文の機構から新しい設計へ段階的に学ぶプロジェクト。

「原論文の小型モデルを完成させる → Decoder-onlyへ進む → 機構を一つずつ交換する → 高速化・新しい設計へ進む」という順序で取り組む。論文ごとにコードをすべて複製するより、比較できる部品を少しずつ増やす。

## ロードマップ

| 段階 | 実装するもの | 次へ進む目安 |
| --- | --- | --- |
| 1. Attentionの基礎 | Scaled Dot-Product Attention、padding mask、causal mask、Multi-Head Attention | 手計算できる小さな入力で結果が合い、未来のトークンを参照しない |
| 2. 原論文のモデル | sinusoidal位置符号化、ReLU FFN、Post-LayerNorm、残差接続、Encoder、Decoder、cross-attention | 系列のコピー・反転タスクを学習し、自己回帰生成できる |
| 3. 原論文の学習手順 | teacher forcing、ターゲットのシフト、paddingを除外した損失、label smoothing、Adamとwarmupを含む学習率スケジュール | 小規模な翻訳データで学習・評価・チェックポイント再開が動く |
| 4. Decoder-only LM | cross-attentionを持たないブロック、次トークン予測、学習可能な位置埋め込み、GELU、Pre-LayerNorm | 小さな文字単位LMで検証損失が下がり、文章を生成できる |
| 5. Llama系の基本機構 | RMSNorm、RoPE、SwiGLU、GQAを順番に導入 | 同じ学習条件で各変更の影響を比較できる |
| 6. 推論・学習の高速化 | KV cache、PyTorch SDPA、混合精度、必要に応じて `torch.compile` | cache有無の出力が数値誤差の範囲で一致し、速度・メモリの変化を測定できる |
| 7. 発展テーマ | MoE、MLA、sliding-window attention、linear attentionなど | 選んだ機構を小型モデルで検証し、利点と制約を説明できる |

段階2では、原論文のEncoder–Decoder、Post-LayerNorm、ReLU、sinusoidal位置符号化を維持する。まず機構を再現し、学習規模は小さくする。原論文と同規模の訓練や評価値の再現は、別の目標として扱う。

段階5の具体的な到達点にはLlama系を使う。ただしRMSNorm・RoPE・SwiGLU・GQAを一度に導入せず、途中の構成も設定として残す。

最新の設計を追う部分は、テーマ別に分岐させる。例えばDeepSeek-V3のMLA・MoEや、Qwen3のdense／MoE構成を題材にする。その先は読みたい新論文を選び、ロードマップに追加する。

## 最初の到達目標

小型の原論文Transformerでコピータスクを学習し、BOSからEOSまで自己回帰生成できる状態を目指す。

- `d_model = 128`
- Attention heads: 4
- Encoder／Decoder: 各2層
- FFNの中間次元: 512
- データ: 特殊トークンを除いたランダムなトークン列と、そのコピー

Decoderには右にシフトした正解系列を入力する。損失計算からpaddingを除外し、生成時はBOSから始めてEOSまたは最大生成長で終了する。小さな固定バッチを過学習できることを確認してから、未学習の系列で評価する。

## ディレクトリ構成

以下は今後の構成案。必要になったディレクトリ・ファイルだけを作っていく。

```text
transformer/
├── pyproject.toml
├── README.md
├── src/
│   └── transformer_lab/
│       ├── __init__.py
│       ├── attention/
│       │   ├── scaled_dot_product.py  # 素朴な参照実装
│       │   ├── multi_head.py          # Q/K/V投影・head分割
│       │   ├── masks.py
│       │   ├── grouped_query.py       # 後から追加
│       │   └── backends.py            # manual / PyTorch SDPA
│       ├── layers/
│       │   ├── positional.py          # sinusoidal / learned / RoPE
│       │   ├── normalization.py       # LayerNorm / RMSNorm
│       │   └── feed_forward.py        # ReLU / GELU / SwiGLU
│       ├── blocks/
│       │   ├── encoder.py
│       │   └── decoder.py
│       ├── models/
│       │   ├── config.py
│       │   ├── seq2seq.py
│       │   └── causal_lm.py
│       ├── data/
│       │   ├── synthetic.py           # コピー・反転タスク
│       │   ├── tokenization.py
│       │   └── collate.py
│       ├── training/
│       │   ├── trainer.py
│       │   ├── losses.py
│       │   └── schedules.py
│       └── generation/
│           ├── sampling.py
│           └── kv_cache.py
├── configs/
│   ├── 01_original_tiny.yaml
│   ├── 02_decoder_only_tiny.yaml
│   └── 03_llama_tiny.yaml
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── generate.py
├── tests/
│   ├── test_attention.py
│   ├── test_masks.py
│   └── test_kv_cache.py
├── benchmarks/
│   └── attention.py
├── notebooks/
│   └── attention_visualization.ipynb
├── notes/
│   └── roadmap.md
└── runs/                             # Git管理対象外にする
```

| ディレクトリ | 役割 |
| --- | --- |
| `attention/` | トークン間の集約、Q/K/V投影、マスク、実行バックエンド |
| `layers/` | 位置符号化、正規化、FFN |
| `blocks/` | 残差接続を含むEncoder／Decoderの1層 |
| `models/` | 層の積み重ね、埋め込み、出力ヘッド、モデル設定 |
| `data/` | データ生成、トークン化、バッチ化 |
| `training/` | 学習ループ、損失、学習率スケジュール |
| `generation/` | 自己回帰生成、サンプリング、KV cache |
| `configs/` | 各段階のモデル・学習条件 |
| `tests/` | 機構の正しさを確認するテスト |
| `benchmarks/` | 速度・メモリ使用量の比較 |
| `notebooks/` | `src/` の実装を読み込んだ可視化・観察 |
| `notes/` | 論文メモ、数式、実験結果の考察 |
| `runs/` | ログ、チェックポイント、実験時の設定 |

## 実装と比較の方針

### テンソル形状とマスク

- 通常の隠れ状態は `[batch, seq, dim]` に統一する。
- Attention内部は `[batch, heads, seq, head_dim]` を基本にする。GQAではqueryとkey/valueでhead数が異なる。
- 自作コードのboolean maskは `True = 参照可能` に統一する。PyTorch SDPAもこの意味だが、`nn.MultiheadAttention` など別のAPIへ渡す際は、そのAPIの規約に変換する。
- Self-attentionとcross-attentionを、queryとkey/valueの入力元の違いとして整理する。cross-attentionではqueryとkeyの系列長が異なっても動くようにする。

### 部品の交換と共通化

- 最初は `nn.Linear`、行列積、softmaxなどを使って実装し、Attentionの計算を追えるようにする。
- 素朴なAttentionを参照実装として残し、後からSDPAバックエンドと比較する。
- FlashAttention系の高速化は段階6で扱い、出力の整合性を確認してから性能を測る。
- 二つ目の実装ができて共通部分が見えてから抽出する。最初からすべてのモデルを設定だけで表現する巨大なクラスを作らない。
- 原論文、Decoder-only、Llama系の設定を残し、後の変更で以前の構成が壊れないようにする。

### 検証と実験記録

- 小さな入力でAttentionの計算結果を確認する。
- 未来のトークンを変更しても、それ以前の出力が変わらないことを確認する。
- paddingされたkey/valueを変更しても、有効な位置の出力が変わらないことを確認する。
- KV cacheの比較はdropoutを無効にした評価モードで行い、全系列の計算と逐次計算の出力を比較する。
- 設定、seed、コードのcommit、検証損失、速度、メモリ使用量、実行環境を記録する。
- 機構は一つずつ変更し、同じデータと学習予算で比較する。SwiGLUなどではパラメータ数や計算量の差も記録する。
- 実験成果物は `runs/` に置き、Git管理対象外にする。

## 参考資料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762): 原論文のモデルと学習手順。
- [The Llama 3 Herd of Models](https://arxiv.org/html/2407.21783v3): Llama系の構成を学ぶ題材。
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245): GQAの機構。
- [PyTorch SDPA API](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html): Attention計算、マスク、dropoutの仕様。
- [PyTorch SDPA Tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial): SDPAによる高速化。
- [FlashAttention-2](https://arxiv.org/abs/2307.08691): Attentionの実行方法とメモリ効率。
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437): MLA・MoEなどの発展テーマ。
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388): dense／MoEモデルの設計と学習。
