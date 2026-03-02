import gradio as gr
import subprocess
from pathlib import Path
import glob

# ここでローカルのモデルファイルを読み込み
def list_local_models(folder="models"):
    paths = glob.glob(f"{folder}/*")
    # 各ファイル名だけ返す
    return [Path(p).name for p in paths]

def generate_manifest(
    local_model,
    dataset, split, audio_col, text_col,
    speaker_col, output_manifest, latent_dir, device
):
    Path(latent_dir).mkdir(parents=True, exist_ok=True)
    Path(Path(output_manifest).parent).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "prepare_manifest.py",
        "--dataset", dataset,
        "--split", split,
        "--audio-column", audio_col,
        "--text-column", text_col,
        "--output-manifest", output_manifest,
        "--latent-dir", latent_dir,
        "--device", device,
        "--model", local_model # スクリプト側で model を使う想定
    ]

    if speaker_col:
        cmd += ["--speaker-column", speaker_col]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.stdout if proc.returncode == 0 else f"Error:\n{proc.stderr}"

with gr.Blocks(title="DACVAE Manifest Generator") as demo:

    gr.Markdown("## ローカルモデル選択および manifest 生成")

    model_choices = list_local_models("models")  # models/ フォルダ内モデルを列挙
    local_model = gr.Dropdown(
        choices=model_choices,
        label="ローカルモデル選択",
        info="使用するモデルファイルを選んでください（models フォルダ内）"
    )

    dataset = gr.Textbox(label="Dataset（Hugging Face データセットID）",
                         info="例: mozilla/common_voice")
    split = gr.Dropdown(choices=["train","validation","test"], label="Split（分割）",
                        info="manifest を作るデータ分割を選択")
    audio_col = gr.Textbox(label="音声カラム名",
                           info="音声データが入っているカラム名")
    text_col = gr.Textbox(label="テキストカラム名",
                          info="発話テキストが入っているカラム名")
    speaker_col = gr.Textbox(label="スピーカーカラム（任意）",
                             info="複数話者データセットの場合はスピーカーIDが入っているカラム名")
    output_manifest = gr.Textbox(label="出力 manifest パス",
                                 info="生成する JSONL の保存ファイルパス")
    latent_dir = gr.Textbox(label="latent 保存ディレクトリ",
                            info="特徴量（latent）ファイルを保存するフォルダ")
    device = gr.Dropdown(choices=["cpu","cuda"], label="Device（計算デバイス）",
                         info="latent 計算に使用するデバイス")

    run_btn = gr.Button("manifest 生成開始")
    result_box = gr.Textbox(label="実行ログ / 結果")

    run_btn.click(
        generate_manifest,
        inputs=[local_model, dataset, split, audio_col, text_col,
                speaker_col, output_manifest, latent_dir, device],
        outputs=result_box
    )

demo.launch()