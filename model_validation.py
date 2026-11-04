import os


MODEL_FILES = {
    "whisper_base_en": (
        "model.bin",
        "config.json",
        "tokenizer.json",
        "vocabulary.txt",
    ),
    "opus_mt_en_zh": (
        "config.json",
        "generation_config.json",
        "pytorch_model.bin",
        "source.spm",
        "target.spm",
        "tokenizer_config.json",
        "vocab.json",
    ),
}


def missing_model_files(models_dir):
    missing = []
    for model_dir, files in MODEL_FILES.items():
        missing.extend(
            os.path.join(model_dir, filename)
            for filename in files
            if not os.path.isfile(os.path.join(models_dir, model_dir, filename))
            or os.path.getsize(os.path.join(models_dir, model_dir, filename)) == 0
        )
    return missing
