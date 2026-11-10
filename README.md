# Meeting Translate / 会议实时字幕翻译

**English ↔ Chinese project overview is below.**
**项目中文说明见下文；English documentation follows the Chinese section.**

---

## 中文

Meeting Translate 是一个面向 Windows 的本地实时字幕翻译工具。它从电脑扬声器回环设备采集音频，使用本地 Whisper 模型识别英文，并使用本地 MarianMT 模型翻译成中文，在桌面浮窗中显示双语字幕和历史记录。

### 特性

- 运行时离线推理，不向云端发送音频或文本。
- 支持完整模式和低延迟模式。
- 支持通用会议与软件术语表、实时预览、延迟转写修订和双语历史记录。
- 术语表通过本地 JSON 文件配置；可在本地 `terminology.json` 中加入自己的词条，运行期间修改后会在后续音频块生效。词表数据需自行维护，不随代码仓库分发。
- UI、运行状态、CLI、异常提示和翻译规则中的中文资源集中在 `locales/zh-CN.json`，按 `messages` 与 `data` 分类。
- 按会议保存 WAV 录音、转写文本和诊断日志；这些文件仅保存在本机，不纳入源码仓库。

### 环境要求

- Windows 10/11，64 位。
- Python 3.14（推荐）或 3.11。
- 可用的 Windows 扬声器回环录音设备。
- 安装与首次模型下载需要网络；下载完成后，程序推理可离线运行。
- 建议预留至少 4 GiB 磁盘空间。模型约 1.5 GiB，Python 依赖另占空间。

程序默认使用 CPU 和 INT8 模型，不要求 NVIDIA GPU。若没有可用的扬声器回环设备，请先在 Windows 声音设置或声卡驱动中启用 Stereo Mix/立体声混音等回环输入；程序不会擅自切换到麦克风。

### 安装

1. 安装 64 位 Python 3.14（推荐）或 3.11，并确保 Python Launcher (`py`) 可用。
2. 克隆或下载本仓库。
3. 双击 `setup.bat`。脚本会创建 `.venv`、安装 `requirements.txt` 中锁定的依赖，并下载/准备本地模型。
4. 首次安装时需要网络和足够磁盘空间；模型不会包含在 GitHub 源码中。

依赖包括 `faster-whisper`、`ctranslate2`、`transformers`、`torch`、`sentencepiece`、`numpy`、`huggingface_hub`、`sacremoses` 和 `soundcard`。具体版本以 `requirements.txt` 为准。PyTorch 安装包因平台而异；若默认软件源无法提供锁定版本，请使用 PyTorch 官方安装选择器为当前 Python 和平台安装兼容的 PyTorch 2.9.0，然后重新运行 `setup.bat`。

### 启动

| 文件 | 说明 |
| --- | --- |
| `run-complete.bat` | 保留完整待处理音频，优先追求内容完整；负载过高时字幕可能延迟。 |
| `run-low-latency.bat` | 优先跟上实时语音；持续过载时可能跳过较旧的待识别音频，并提示间隙。 |
| `run.bat` | 使用默认完整模式启动。 |

关闭字幕窗口后，程序会停止采集、处理已排队音频和当前尾句、完成翻译并保存文件；等待后台任务结束后退出。

### 模型

首次运行 `setup.bat` 时，`download_models.py` 会下载或准备模型。程序可加载默认、公司和本地术语表；词表文件由用户自行提供，本地 `terminology.json` 的同名术语会覆盖其他来源：

程序按音频块/句段边界检查词表文件更新，并发布带版本号的识别快照。单次 ASR 调用只使用一个术语版本；诊断日志中的 `glossary_version` 用于追踪对应配置。无效 JSON 会被拒绝，继续使用最后一个有效版本。

| 用途 | Hugging Face 模型 | 本地目录 |
| --- | --- | --- |
| 英文语音识别 | `Systran/faster-whisper-base.en` | `models/whisper_base_en/` |
| 英译中翻译 | `Helsinki-NLP/opus-mt-en-zh` | `models/opus_mt_en_zh/` |
| 可选的翻译加速格式 | 由 Marian 模型转换为 CTranslate2 INT8 | `models/opus_mt_en_zh_ct2/` |

缺少 CTranslate2 翻译模型时，转换失败会回退到 PyTorch/Transformers 翻译后端。模型文件较大并受各自上游模型许可约束，请在使用或再分发前查看对应模型卡及许可证。

本地 `terminology.json` 不包含在仓库中。可从以下通用示例开始创建：

```json
{
  "AI": {
    "target": "AI",
    "aliases": ["人工智能"],
    "asr_always": true
  },
  "product owner": {
    "target": "产品负责人",
    "aliases": ["产品拥有者"]
  }
}
```

### 输出、诊断与隐私

每次运行在 `sessions/meeting_<timestamp>/` 下创建独立目录，包括录音 WAV、转写文本、修订记录和 `diagnostics.jsonl`。运行日志写到本地 `run.log`。这些输出可能含会议敏感信息，**不要上传到公共仓库或未经授权的服务**。

诊断日志包含 ASR/翻译耗时、队列积压、识别到的音频间隙、设备重连、采集层间断警告，以及采集和 WAV 的帧数/时长。设备重连时，估算的时间缺口会在 WAV 中以静音填充；估算值不等同于声卡报告的精确丢失量。

### 项目结构

```text
main.py                 实时处理和线程编排
asr_pipeline.py         ASR 音频窗口构造、解码和重试
audio_capture.py        扬声器回环采集、队列和设备重连
asr.py                  faster-whisper 识别
translator.py           MarianMT / CTranslate2 翻译
overlay.py              Tkinter 字幕窗口
tests/                  按音频、ASR、翻译、会话和 UI 分组的测试
worker_runtime.py       后台 worker 生命周期与异常边界
translation_segmentation.py 长输入的 token 预算与断句分段
terminology.json        可选的本地术语表（不纳入仓库，支持运行时热加载）
locales/zh-CN.json      中文界面、状态、日志和算法文本资源
i18n.py                 分类资源访问层
requirements.txt        固定版本的 Python 依赖
setup.bat               环境安装和模型准备
run*.bat                运行入口
models/                 本地模型（不纳入仓库）
sessions/               本地会议录音、文本和诊断（不纳入仓库）
```

### 测试

```bat
.venv\Scripts\python.exe -B -m unittest discover
```

测试按 `tests/audio`、`tests/asr`、`tests/translation`、`tests/session` 和 `tests/ui` 分组。
测试使用内存中的示例术语，运行测试无需取得任何私有词表。

The test suite is grouped under `tests/audio`, `tests/asr`, `tests/translation`,
`tests/session`, and `tests/ui`. Glossary files are user-managed local configuration
and are not distributed with the source repository. Local `terminology.json` entries
override matching terms from other configured glossary files. Ambiguous ASR variants
can require contextual evidence so unrelated everyday words are not rewritten as
technical terms.

Replay a session recording offline to compare recognition, translation, and latency:

```bat
.venv\Scripts\python.exe replay_audio.py "sessions\<session>\meeting_audio.wav" --output "sessions\<session>\quality_replay.jsonl" --start 0 --duration 90 --beam-size 2
```

Long inputs are segmented at sentence/clause boundaries under the model token budget,
and tiny segments are merged. Context groups that exceed the budget fall back to
sentence-level translation instead of one large context batch.

### 许可证

本项目采用 MIT 许可证，详见 `LICENSE`。模型权重和第三方依赖有各自独立的许可条款；使用或再分发前请查看上游许可证。

---

## English

Meeting Translate is a local, real-time subtitle translator for Windows. It captures computer audio through a speaker loopback device, transcribes English with a local Whisper model, translates it into Chinese with a local MarianMT model, and displays bilingual captions and history in a desktop overlay.

### Features

- Offline inference at runtime; audio and text are not sent to a cloud service.
- Complete and low-latency operating modes.
- General meeting/software terminology, live preview, delayed transcript correction, and bilingual history.
- Glossary files are local configuration and are not distributed with the source repository. Add private vocabulary in `terminology.json`; changes are picked up at subsequent audio-chunk boundaries.
- Per-session WAV recordings, transcripts, and diagnostic logs are stored locally and are not part of the source repository.

### Requirements

- 64-bit Windows 10/11.
- Python 3.14 (recommended) or 3.11.
- An available Windows speaker loopback recording device.
- Network access is required for installation and the initial model download. Inference works offline after the models are available.
- Allow at least 4 GiB of disk space. The models use about 1.5 GiB, in addition to Python dependencies.

The default configuration uses CPU inference and INT8 models; an NVIDIA GPU is not required. If no loopback device is available, enable Stereo Mix or another driver-provided loopback input in Windows sound settings. The application does not silently switch to a microphone.

### Installation

1. Install 64-bit Python 3.14 (recommended) or 3.11, and make sure the Python Launcher (`py`) is available.
2. Clone or download this repository.
3. Run `setup.bat`. It creates `.venv`, installs the pinned dependencies from `requirements.txt`, and downloads/prepares the local models.
4. The initial setup requires network access and sufficient disk space. Model files are not included in the GitHub source repository.

Dependencies include `faster-whisper`, `ctranslate2`, `transformers`, `torch`, `sentencepiece`, `numpy`, `huggingface_hub`, `sacremoses`, and `soundcard`. See `requirements.txt` for exact pinned versions. PyTorch wheels are platform-specific. If the default package index does not provide the pinned version, install a compatible PyTorch 2.9.0 wheel for your Python version and platform using the official PyTorch installation selector, then rerun `setup.bat`.

### Run

| File | Description |
| --- | --- |
| `run-complete.bat` | Keeps queued audio and prioritizes completeness; captions may fall behind under heavy load. |
| `run-low-latency.bat` | Prioritizes following live speech; sustained overload may skip older pending audio and report gaps. |
| `run.bat` | Starts the default complete mode. |

When the overlay is closed, the application stops capture, processes queued audio and the current tail utterance, finishes translations, and saves the session. It exits after background work has completed.

### Models

On the first `setup.bat` run, `download_models.py` downloads or prepares the models. Glossary files are user-managed local configuration; a local `terminology.json` in the project root overrides entries with matching terms:

Chinese UI, runtime-status, CLI, error, and translation-rule resources are categorized in `locales/zh-CN.json` under `messages` and `data`. Glossary files are checked at audio-chunk/segment boundaries and published as versioned snapshots; an ASR call uses one consistent glossary version, recorded as `glossary_version` in diagnostics. Invalid JSON is rejected while the last valid snapshot remains active.

| Purpose | Hugging Face model | Local directory |
| --- | --- | --- |
| English speech recognition | `Systran/faster-whisper-base.en` | `models/whisper_base_en/` |
| English-to-Chinese translation | `Helsinki-NLP/opus-mt-en-zh` | `models/opus_mt_en_zh/` |
| Optional accelerated translation format | Marian model converted to CTranslate2 INT8 | `models/opus_mt_en_zh_ct2/` |

If the CTranslate2 translation model is unavailable, conversion failures fall back to the PyTorch/Transformers translation backend. Model files are large and subject to their respective upstream licenses; review the model cards and licenses before use or redistribution.

Local `terminology.json` files are not included in the repository. You can create one using this general-purpose example:

```json
{
  "AI": {
    "target": "AI",
    "aliases": ["人工智能"],
    "asr_always": true
  },
  "product owner": {
    "target": "产品负责人",
    "aliases": ["产品拥有者"]
  }
}
```

### Output, diagnostics, and privacy

Each run creates a separate `sessions/meeting_<timestamp>/` directory containing the WAV recording, transcript, revision history, and `diagnostics.jsonl`. Runtime output is written to the local `run.log`. These files may contain sensitive meeting information. **Do not upload them to a public repository or an unauthorized service.**

Diagnostics include ASR/translation latency, queue backlog, detected audio gaps, device reconnects, capture-layer discontinuity warnings, and capture/WAV frame counts and durations. Estimated reconnect gaps are padded with silence in the WAV to preserve its timeline; the estimate is not an exact device-reported loss measurement.

### Project layout

```text
main.py                 Real-time pipeline and thread orchestration
asr_pipeline.py         ASR window construction, decoding, and retry
audio_capture.py        Speaker loopback capture, queue, and reconnects
asr.py                  faster-whisper transcription
translator.py           MarianMT / CTranslate2 translation
overlay.py              Tkinter subtitle window
tests/                  Tests grouped by audio, ASR, translation, session, and UI
worker_runtime.py       Worker lifecycle and exception boundaries
translation_segmentation.py Token-budgeted long-input segmentation
terminology.json        Optional local glossary (excluded from repository; hot-loaded)
locales/zh-CN.json      Chinese UI, status, log, and algorithm text resources
i18n.py                 Categorized resource access layer
requirements.txt        Pinned Python dependencies
setup.bat               Environment setup and model preparation
run*.bat                Launch scripts
models/                 Local models (excluded from the repository)
sessions/               Local recordings, transcripts, and diagnostics (excluded)
```

### Tests

```bat
.venv\Scripts\python.exe -B -m unittest discover
```

Tests are grouped under `tests/audio`, `tests/asr`, `tests/translation`,
`tests/session`, and `tests/ui`. They use in-memory glossary examples and do not
require access to private glossary files.

Replay a session recording offline to compare recognition, translation, and latency:

```bat
.venv\Scripts\python.exe replay_audio.py "sessions\<session>\meeting_audio.wav" --output "sessions\<session>\quality_replay.jsonl" --start 0 --duration 90 --beam-size 2
```

Long inputs are segmented at sentence/clause boundaries under the model token budget,
and tiny segments are merged. Context groups that exceed the budget fall back to
sentence-level translation instead of one large context batch.

### License

This project is distributed under the MIT License; see `LICENSE`. Model weights and third-party dependencies have separate license terms. Review their upstream licenses before use or redistribution.
