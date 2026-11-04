import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download
from model_validation import MODEL_FILES

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODELS = (
    (
        "Systran/faster-whisper-base.en",
        "whisper_base_en",
        ["*.h5", "*.msgpack", "*.ot", "*.safetensors", "*.onnx", "*.pb", "*.index"],
    ),
    (
        "Helsinki-NLP/opus-mt-en-zh",
        "opus_mt_en_zh",
        ["*.h5", "*.msgpack", "*.ot", "*.safetensors", "*.onnx", "*.pb", "*.index"],
    ),
)

for repo, local, ignore_patterns in MODELS:
    target = os.path.join(MODELS_DIR, local)
    if all(
        os.path.isfile(os.path.join(target, filename))
        and os.path.getsize(os.path.join(target, filename)) > 0
        for filename in MODEL_FILES[local]
    ):
        print(f"[skip] {repo}: {target}")
        continue
    print(f"[download] {repo} -> {target}")
    snapshot_download(
        repo_id=repo,
        local_dir=target,
        endpoint=os.environ["HF_ENDPOINT"],
        ignore_patterns=ignore_patterns,
    )

print(f"Models saved to {MODELS_DIR}")

ct2_opus_dir = os.path.join(MODELS_DIR, "opus_mt_en_zh_ct2")
if not os.path.isfile(os.path.join(ct2_opus_dir, "model.bin")):
    try:
        from convert_marian_to_ct2 import convert
        convert()
    except Exception as exc:
        print(f"[warning] CTranslate2 自动转换跳过: {exc}")
