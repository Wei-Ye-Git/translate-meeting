import os
import shutil
import sys
import ctranslate2

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
SOURCE_DIR = os.path.join(MODELS_DIR, "opus_mt_en_zh")
OUTPUT_DIR = os.path.join(MODELS_DIR, "opus_mt_en_zh_ct2")


def convert(source_dir=SOURCE_DIR, output_dir=OUTPUT_DIR, quantization="int8"):
    print(f"[ct2] Converting {source_dir} -> {output_dir} (quantization={quantization})...")
    converter = ctranslate2.converters.TransformersConverter(source_dir)
    converter.convert(output_dir, quantization=quantization, force=True)
    for filename in ("source.spm", "target.spm", "vocab.json", "tokenizer_config.json"):
        src = os.path.join(source_dir, filename)
        dst = os.path.join(output_dir, filename)
        if os.path.isfile(src) and not os.path.isfile(dst):
            shutil.copy2(src, dst)
    print(f"[ct2] Completed: {output_dir}")


if __name__ == "__main__":
    convert()
