# Irodori-TTS LoRA拡張実装仕様書 v2

> 作成日: 2026-03-02  
> 対象リポジトリ: Irodori-TTS  
> ステータス: 確認中（実装前）

---

## 1. 実装概要

既存の Irodori-TTS GUI に以下の機能を追加する。

| # | 機能 | 対象ファイル |
|---|------|------------|
| ① | SageAttention / FlashAttention オプション追加（通常学習） | `train.py` |
| ② | 通常学習タブに Attention Backend プルダウン追加 | `gradio_app.py` |
| ③ | LoRA 学習スクリプト新規作成 | `lora_train.py`（新規） |
| ④ | GUI に LoRA 学習タブ追加（タブ4として割り込み） | `gradio_app.py` |
| ⑤ | 推論タブの LoRA 対応（方針B: PeftModel保持） | `inference_runtime.py`・`gradio_app.py` |
| ⑥ | LoRA Full → EMA 変換スクリプト新規作成 | `convert_lora_checkpoint.py`（新規） |
| ⑦ | 変換タブを内部タブ分けして既存機能とLoRA変換を整理 | `gradio_app.py` |

---

## 2. タブ構成（変更後）

```
タブ1: 🔊 推論                    ← LoRA対応追加（手順⑤）
タブ2: 📂 Manifest前処理           ← 変更なし
タブ3: 🏋️ 学習                    ← Attention Backend追加（手順②）
タブ4: 🧬 LoRA学習                 ← 新規追加（手順④）※割り込み
タブ5: 🎙️ Dataset作成              ← 変更なし
タブ6: 🔄 チェックポイント変換      ← 内部タブ分け追加（手順⑦）
タブ7: 🔀 モデルマージ              ← 変更なし
```

タブ順の変更・入れ替えは `gr.Tab()` ブロックの並べ替えのみ対応可能。
タブ間でコンポーネントの参照は存在せず副作用ゼロ。

---

## 3. フォルダ構成（変更後）

### 新規追加フォルダ

```
プロジェクトルート/
├── checkpoints/        ← 既存（ベースモデル・推論用モデル置き場）
├── outputs/            ← 既存（通常学習チェックポイント出力先）
│   └── irodori_tts/
│       ├── checkpoint_0000100_ema.pt
│       └── checkpoint_0000100_full.pt
│
├── lora/               ← 新規追加（LoRA専用フォルダ）
│   └── run_name/
│       ├── lora_checkpoint_0000100_ema/
│       ├── lora_checkpoint_0000100_full/
│       └── lora_checkpoint_final_ema/
│
├── configs/            ← 既存
└── logs/               ← 既存
```

### LoRAフォルダの命名規則

通常学習との見分けがつくよう **ファイル名・フォルダ名に `lora_` プレフィックスを付与** する。

```
lora/
└── {run_name}/
    ├── lora_checkpoint_{step:07d}_ema/       ← 定期保存・推論用
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors
    │
    ├── lora_checkpoint_{step:07d}_full/      ← --save-full 時のみ・Resume用
    │   ├── adapter_config.json
    │   ├── adapter_model.safetensors         ← EMA適用前の生重み
    │   ├── ema_shadow.pt                     ← EMA shadow重み（変換ツール用）
    │   ├── optimizer.pt
    │   ├── scheduler.pt
    │   └── train_state.json
    │
    ├── lora_checkpoint_best_val_loss_{step:07d}_{loss:.6f}_ema/
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors
    │
    └── lora_checkpoint_final_ema/            ← 学習完了時に必ず保存
        ├── adapter_config.json
        └── adapter_model.safetensors
```

### 通常学習との命名比較

| | 通常学習 | LoRA学習 |
|--|---------|---------|
| 定期保存（推論用） | `checkpoint_0000100_ema.pt` | `lora_checkpoint_0000100_ema/` |
| 定期保存（再開用） | `checkpoint_0000100_full.pt` | `lora_checkpoint_0000100_full/` |
| 最良バリデーション | `checkpoint_best_val_loss_...pt` | `lora_checkpoint_best_val_loss_..._ema/` |
| 最終保存 | `checkpoint_final.pt` | `lora_checkpoint_final_ema/` |

---

## 4. gradio_app.py への定数追加

```python
# 既存
CHECKPOINTS_DIR  = BASE_DIR / "checkpoints"
OUTPUTS_DIR      = BASE_DIR / "gradio_outputs"

# 追加
LORA_DIR         = BASE_DIR / "lora"          ← LoRA保存デフォルト先
```

### スキャン関数の追加

```python
def _scan_lora_adapters() -> list[str]:
    """lora/ 配下の adapter_config.json を持つフォルダを列挙。
    _ema/ フォルダを優先表示（推論・Resume選択用の共通関数）。"""
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    result = []
    for p in sorted(LORA_DIR.rglob("adapter_config.json")):
        result.append(str(p.parent))
    return result

def _scan_lora_full_adapters() -> list[str]:
    """lora/ 配下の _full フォルダのみを列挙（変換・Resume用）。"""
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    result = []
    for p in sorted(LORA_DIR.rglob("adapter_config.json")):
        if p.parent.name.endswith("_full"):
            result.append(str(p.parent))
    return result
```

---

## 5. 手順① - SageAttention / FlashAttention 追加（train.py）

### 変更箇所

`apply_attention_backend()` 関数（L96〜118）に `sage` 分岐を追加する。

```python
elif backend == "sage":
    try:
        import sageattention
        print("Attention backend: SageAttention")
    except ImportError:
        print("warning: sageattention not installed, falling back to sdpa.")
        backend = "sdpa"
```

### argparse 変更

```python
parser.add_argument(
    "--attention-backend",
    choices=["sdpa", "flash2", "sage", "eager"],  # sage を追加
    default="sdpa",
)
```

`lora_train.py` にも同一の `apply_attention_backend()` を引き継ぐ。

---

## 6. 手順② - 通常学習タブへの Attention Backend プルダウン追加（gradio_app.py）

学習タブ（タブ3）の GPU数・保存モード行付近に追加。

```python
attention_backend = gr.Dropdown(
    label="Attention Backend",
    choices=["sdpa", "flash2", "sage", "eager"],
    value="sdpa",
    info="sdpa=推奨 / flash2=FlashAttention2要インストール / sage=SageAttention要インストール",
)
```

`_build_train_command()` に `--attention-backend` を追加する。

---

## 7. 手順③ - lora_train.py 新規作成

### 基本設計方針

- `train.py` をベースとして LoRA 差分学習に特化
- `peft` ライブラリを使用
- 操作面・引数仕様を `train.py` に最大限準拠
- デフォルト出力先は `lora/{run_name}/`

### LoRA ターゲットモジュール（model.py から確定済み）

```python
DEFAULT_TARGET_MODULES  = ["wq", "wk", "wv", "wo"]   # デフォルト推奨

EXTENDED_TARGET_MODULES = [
    "wq", "wk", "wv", "wo",        # JointAttention コア
    "wk_text", "wv_text",           # テキスト KV 投影
    "wk_speaker", "wv_speaker",     # 話者 KV 投影
    "w1", "w2", "w3",               # SwiGLU MLP
]
```

### LoRA 固有の追加引数

```
--base-model        ベースモデルパス（.pt または .safetensors）※必須
--lora-rank         LoRA ランク（デフォルト: 16）
--lora-alpha        LoRA スケール（デフォルト: 32）
--lora-dropout      LoRA ドロップアウト率（デフォルト: 0.05）
--target-modules    LoRA 適用モジュール カンマ区切り（デフォルト: wq,wk,wv,wo）
--run-name          出力サブフォルダ名（デフォルト: タイムスタンプ自動生成）
--resume-lora       既存 _full フォルダパスを指定して Resume
```

### train.py と共通の引数（準拠）

```
--manifest / --output-dir / --config
--batch-size / --gradient-accumulation-steps
--learning-rate / --optimizer / --lr-scheduler
--warmup-steps / --max-steps / --save-every / --log-every
--ema-decay
--early-stopping / --early-stopping-patience / --early-stopping-min-delta
--attention-backend（sage含む）
--grad-checkpoint
--wandb 系
--precision / --device / --num-workers / --seed
```

### 省略する引数（LoRA では不要・簡略化）

```
--num-gpus（DDP）  ← 初期実装は単一GPU前提
「Fullのみ」保存モード  ← EMAのみ / EMA+Full両方 の2択に簡略化
```

### 保存モード仕様（train.py 準拠・2択）

| モード | 保存内容 | 用途 |
|--------|---------|------|
| **EMAのみ**（デフォルト） | `lora_checkpoint_XXX_ema/` のみ | 推論専用 |
| **EMA + Full両方** | `_ema/` と `_full/` 両方 | Resume前提の学習時 |

### デフォルト出力先

```python
# lora_train.py
_DEFAULT_LORA_DIR = Path(__file__).resolve().parent / "lora"

# --output-dir 未指定時
output_dir = _DEFAULT_LORA_DIR / run_name  # run_nameは--run-nameまたはタイムスタンプ
```

### train_state.json の保存内容

```json
{
  "step": 500,
  "base_model_path": "checkpoints/Aratako_Irodori-TTS-500M/model.safetensors",
  "base_model_sha256": "a3f2c1...",
  "base_model_config": { "model_dim": 512, "..." : "..." },
  "lora_config": {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["wq", "wk", "wv", "wo"]
  },
  "train_config": { "max_steps": 1000, "...": "..." },
  "ema_decay": 0.9999
}
```

※ EMA shadow 重みは `ema_shadow.pt` として独立保存（JSON肥大化防止）

### optimizer の構築方針

```python
# ベースモデルは凍結、LoRAパラメータのみ渡す
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = build_optimizer_extended(trainable_params, train_cfg, ...)
```

### Resume の動作

```python
if args.resume_lora:
    # _full フォルダから再開
    model = PeftModel.from_pretrained(base_model, args.resume_lora)
    # train_state.json から step 復元
    # optimizer.pt / scheduler.pt から状態復元
else:
    # 新規 LoRA 学習
    lora_config = LoraConfig(r=rank, lora_alpha=alpha, ...)
    model = get_peft_model(base_model, lora_config)
```

---

## 8. 手順④ - LoRA 学習タブ追加（gradio_app.py）

### タブ位置

タブ3（学習）の直後に `with gr.Tab("🧬 LoRA学習"):` を挿入する。

### UI構成

```
🧬 LoRA学習タブ
│
├── ベースモデル選択
│     _scan_checkpoints() でプルダウン
│     デフォルト: checkpoints/Aratako_Irodori-TTS-500M/model.safetensors
│
├── マニフェストファイル選択（_scan_manifests() 流用）
│
├── LoRA出力フォルダ
│     デフォルト: lora/{run_name}/
│
├── 実行名（run_name）テキストボックス
│     空欄 = タイムスタンプ自動生成
│
├── 保存モード（EMAのみ / EMA+Full両方）
│
├── Attention Backend プルダウン（sdpa/flash2/sage/eager）
│
├── [Accordion] 🔧 LoRA設定
│     LoRAランク（デフォルト: 16）
│     lora_alpha（デフォルト: 32）
│     lora_dropout（デフォルト: 0.05）
│     ターゲットモジュール テキスト（デフォルト: wq,wk,wv,wo）
│
├── [Accordion] 🔁 Resume設定
│     Resume有効チェック
│     既存LoRAフォルダ選択
│       _scan_lora_adapters()（_ema + _full 両方）でプルダウン + 更新ボタン
│       選択中フォルダ名に "_ema" が含まれる場合、以下の注意文を動的表示：
│       ⚠️ _ema フォルダを選択しています。
│          EMA版にはoptimizer状態・step数が含まれないため、
│          学習は step=0 から再スタートします。
│          学習率ウォームアップが再度かかり、学習曲線が不連続になります。
│          中断した学習を完全に再開する場合は _full フォルダを選択してください。
│
├── [Accordion] ⚙️ 学習パラメータ（train.py タブと同一構成）
│     バッチサイズ / 学習率 / オプティマイザ / スケジューラ等
│
├── [Accordion] 📊 EMA設定
│     EMA有効チェック / EMA減衰率（デフォルト: 0.9999）
│
├── [Accordion] ✅ バリデーション設定
│     valid_ratio / valid_every
│
├── [Accordion] 🛑 Early Stopping設定
│
├── [Accordion] 📊 W&B設定
│
├── コマンドプレビュー
├── [▶️ 学習開始] [⏹️ 停止]
└── ログ・グラフ（train.py タブと同一実装）
```

### プロセス管理（既存と独立）

```python
_LORA_TRAIN_PROC: subprocess.Popen | None = None
_LORA_TRAIN_LOG_PATH: Path | None = None
_LORA_TRAIN_LOG_LOCK = threading.Lock()
```

---

## 9. 手順⑤ - 推論 LoRA 対応（inference_runtime.py / gradio_app.py）

### 方針B：PeftModel をマージせずそのまま保持

`merge_and_unload()` を使わないことで以下の将来拡張が可能。
- 特定層だけ LoRA を有効/無効にする
- LoRA スケールの動的変更
- LoRA なし推論との即時切り替え

### inference_runtime.py の変更

#### RuntimeKey への追加フィールド

```python
@dataclass(frozen=True)
class RuntimeKey:
    # 既存フィールド（変更なし）
    lora_path: str | None = None   # 追加。None = 通常推論と完全互換
```

`frozen=True` によりキャッシュキーとして自動機能する。

#### SamplingRequest への追加フィールド

スケール・有効層はリクエストごとに動的変更するため RuntimeKey には含めない。

```python
@dataclass
class SamplingRequest:
    # 既存フィールド（変更なし）
    lora_scale: float = 1.0                      # 追加
    lora_disabled_modules: tuple[str,...] = ()    # 追加（例: ("text_encoder",)）
```

#### InferenceRuntime.from_key() の変更

```python
model = TextToLatentRFDiT(model_cfg).to(model_device)
model.load_state_dict(model_state)
model = model.to(dtype=model_dtype)
model.eval()

# 追加（lora_path がある場合のみ）
if key.lora_path:
    from peft import PeftModel
    model = PeftModel.from_pretrained(
        model,
        key.lora_path,
        is_trainable=False,
    )
    model = model.to(dtype=model_dtype)
    model.eval()
```

#### InferenceRuntime.synthesize() の変更

```python
# 推論開始時にLoRA設定を適用 → finally で元に戻す
_lora_active = key.lora_path is not None and hasattr(self.model, 'set_adapter')
if _lora_active:
    _apply_lora_settings(self.model, req.lora_scale, req.lora_disabled_modules)
try:
    # 既存の推論処理（変更なし）
    ...
finally:
    if _lora_active:
        _restore_lora_defaults(self.model)
```

`self._infer_lock` 既存のためスレッドセーフ保証済み。

### gradio_app.py 推論タブの変更

```
🔊 推論タブ
├── チェックポイント選択（既存）
├── LoRAアダプタ選択（省略可）    ← 追加
│     _scan_lora_adapters() でプルダウン（_ema フォルダを列挙）
│     更新ボタン
├── LoRAスケール スライダー       ← 追加（0.0〜1.0、デフォルト1.0）
│     LoRAアダプタ未選択時は非表示
└── 以降既存のまま
```

---

## 10. 手順⑥ - convert_lora_checkpoint.py 新規作成

### 目的

LoRA Full版（`lora_checkpoint_XXX_full/`）から
EMA平滑化済み重みを取り出して EMA版（`lora_checkpoint_XXX_ema/`）を生成する。

### ファイル構造と処理フロー

```
入力: lora/run_name/lora_checkpoint_0000500_full/
  ├── adapter_config.json
  ├── adapter_model.safetensors    ← EMA適用前の生重み
  ├── ema_shadow.pt                ← EMA shadow重み
  └── train_state.json

        ↓ ema_shadow.pt の重みで adapter_model.safetensors を上書き

出力: lora/run_name/lora_checkpoint_0000500_ema/（新規）
  ├── adapter_config.json          ← コピー
  └── adapter_model.safetensors    ← EMA平滑化済み重み
```

### 既存 convert_checkpoint_to_safetensors.py との比較

| | 通常版（既存） | LoRA版（新規） |
|--|--------------|--------------|
| 入力 | `_ema.pt`（単体ファイル） | `_full/`（フォルダ） |
| EMA取り出し | `payload["model"]` に含まれる | `ema_shadow.pt` から取り出し |
| ベースモデル関与 | なし | なし（LoRA差分のみ） |
| 出力 | `.safetensors`（フルモデル） | `_ema/`（LoRA差分アダプタフォルダ） |
| 設計流用 | ─ | `_load_checkpoint()` パターンを流用 |

### CLI仕様

```
python convert_lora_checkpoint.py <input_full_dir>
  --output <出力先ディレクトリ>
    （デフォルト: input_full_dir の _full を _ema に置換した同階層フォルダ）
  --force  既存出力の上書き
```

---

## 11. 手順⑦ - 変換タブの内部タブ分け（gradio_app.py）

既存の「🔄 チェックポイント変換」タブ（タブ6）を内部タブで2分割する。
独立タブにはせず統合することでタブ数の増加を抑える。

### 変換タブの構成

```
🔄 チェックポイント変換タブ
│
├── [内部タブA] 📦 通常チェックポイント変換（既存機能・変更なし）
│     説明: 学習チェックポイント（.pt）を推論用 .safetensors に変換
│     変換対象: _scan_train_checkpoints() で .pt ファイルをプルダウン
│     変換実行 → convert_checkpoint_to_safetensors.py を呼び出し
│     出力: 入力 .pt と同フォルダに .safetensors を生成
│
└── [内部タブB] 🧬 LoRA変換（新規）
      説明: LoRA Full版から EMA版（推論用）を生成
      変換対象: _scan_lora_full_adapters() で _full フォルダをプルダウン
      出力先フォルダ（省略可、デフォルト自動）
      変換実行 → convert_lora_checkpoint.py を呼び出し
      変換結果テキスト
```

### 内部タブの実装

```python
with gr.Tab("🔄 チェックポイント変換"):
    with gr.Tabs():
        with gr.Tab("📦 通常チェックポイント変換"):
            # 既存コードをそのまま移動（変更なし）
            ...

        with gr.Tab("🧬 LoRA変換"):
            # 新規追加
            ...
```

---

## 12. 将来拡張として保留する機能

| 機能 | 保留理由 |
|------|---------|
| LoRA Stacking（積み重ね追加学習） | Resume実装後に低コストで追加可能 |
| convert_lora_to_safetensors.py（LoRA+ベースモデル→単体safetensors） | 配布・共有ニーズ発生時に追加 |
| DDP マルチGPU LoRA 学習 | 初期は単一GPU前提で十分 |
| 推論タブでの LoRA 無効化モジュール指定UI | 方針B実装後に拡張可能 |
| LoRA同士のマージ機能（_ema版同士） | 現状設計で障害なし。peft の `add_weighted_adapter()` を使用。モデルマージタブへの内部タブ追加として実装予定。互換性検証は train_state.json の base_model_sha256 と target_modules を照合する |

---

## 13. 依存ライブラリ

| ライブラリ | 用途 | 必須/任意 |
|-----------|------|---------|
| `peft` | LoRA学習・推論 | **必須**（LoRA機能全体） |
| `flash-attn` | FlashAttention2 | 任意（既存） |
| `sageattention` | SageAttention | 任意（新規追加） |

---

## 14. ファイル変更・新規作成一覧

| ファイル | 種別 | 変更規模 |
|---------|------|---------|
| `train.py` | 変更 | 小（sage分岐数行） |
| `lora_train.py` | 新規 | 大 |
| `convert_lora_checkpoint.py` | 新規 | 中 |
| `irodori_tts/inference_runtime.py` | 変更 | 小〜中 |
| `gradio_app.py` | 変更 | 中〜大 |

---

## 15. 実装順序

```
① train.py に sage 分岐追加          （小・リスク最小）
② gradio_app.py 通常学習タブ更新     （小）
③ lora_train.py 新規作成             （大・核心）
④ gradio_app.py LoRAタブ追加         （中）
⑤ inference_runtime.py + 推論タブ    （小〜中）
⑥ convert_lora_checkpoint.py 新規    （中）
⑦ gradio_app.py 変換タブ内部タブ分け （小）
```

各手順は独立してテスト可能。
