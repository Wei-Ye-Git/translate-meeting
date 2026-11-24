# Meeting Translate / 会议实时字幕翻译

**English ↔ Chinese project overview is below.**
**项目中文说明见下文；English documentation follows the Chinese section.**

---

## 中文

Meeting Translate 是一个面向 Windows 的实时字幕翻译工具。它从电脑扬声器回环设备采集音频，使用本地 Whisper 模型识别英文，并使用本地 MarianMT 模型翻译成中文，在桌面浮窗中显示双语字幕和历史记录；可选通过已授权的 OpenCode/GitHub Copilot 按上下文修订单句并翻译。

### 特性

- 默认由本地模型处理音频与文本；实时 OpenCode 历史复核默认开启，可通过 `MEETING_COPILOT_ENABLED=0` 关闭。只发送已收录的文本和有限上下文。
- 支持完整模式和低延迟模式。
- 支持通用会议与软件术语表、实时预览、延迟转写修订和双语历史记录。
- 术语表通过本地 JSON 文件配置；可在本地 `terminology.json` 中加入自己的词条，运行期间修改后会在后续音频块生效。词表数据需自行维护，不随代码仓库分发。
- UI、运行状态、CLI、异常提示和翻译规则中的中文资源集中在 `locales/zh-CN.json`，按 `messages` 与 `data` 分类。
- 按会议保存 WAV 录音、转写文本和诊断日志；这些文件仅保存在本机，不纳入源码仓库。
- 诊断日志记录 ASR/翻译质量信号、低置信词、疑似快速语速/稀疏识别、口头填充词与长停顿、断句边界、纠错前后文本及采集缺口；这些是复核线索，不等于人工确认的错误或漏音。

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

关闭字幕窗口后，程序停止采集并尝试收尾、补齐翻译和保存会话。后台任务超过退出等待预算时，会记录未结束的 worker 和待处理翻译数。定稿翻译以记录 ID 去重，不阻塞英文识别；积压时优先处理已定稿的句子。

#### 实时 AI CLI 增强

可选 AI CLI 扩展、agent、用量工具与扩展测试可放在本地 `local_ai/`；该目录和 `.opencode/` 被 Git 默认忽略，因为其内容可能依赖本机授权或配置。扩展默认关闭；仅在本机已安装并获准使用时，设置 `MEETING_COPILOT_ENABLED=1` 启用：

```powershell
$env:MEETING_COPILOT_ENABLED = '1' # 启用可选扩展
$env:MEETING_AI_CLI = 'opencode' # 默认选择 OpenCode
# $env:MEETING_AI_MODEL = 'provider/model' # 可选
.\run-complete.bat
```

若扩展目录、所选 CLI 或它所需的 agent 文件缺失，则继续运行本地识别和翻译并记录原因。`MEETING_AI_CLI` 也可明确选用 `claude`、`gemini`、`codex` 或 `copilot`；不会因为 OpenCode 不可用而自动切换服务。默认启动约 20 秒后允许首次请求，此后每 30 秒串行复核，最多一次 60 秒，符合条件的失败最多重试一次。本地翻译不等待 AI；实时预览仅走本地翻译。虽然 CLI 在本机执行，配置的上游模型仍可能接收会议文字；音频不会交给 CLI。

AI 可根据已出现的会议内容提出术语候选，记录在会话 `ai_term_candidates.json`。候选不自动加入共享术语表；只有后续上下文出现匹配线索时才用于 ASR 提示。

AI 修订属于文本上下文推断，不是音频确认；修订及超时/错误跳过会记录在会话 `diagnostics.jsonl`，采纳的文本变更记录在 `meeting_revisions.jsonl`。诊断日志还会逐请求保存实际发送的 prompt/文本窗口、所选 CLI 返回内容、结果、模型（若可确定）、耗时，以及 CLI 明确返回的 token/cost 元数据。日志是本地文件，但包含完整会议文字，按敏感数据保护。

GitHub Copilot credits 不会通过 CLI 调用接口逐请求返回，因此项目不会估算或伪报 credits；汇总中会显示 `copilot_github_credits: null` 和原因。若 CLI 明确提供 token/cost，则另外汇总这些原始 provider usage 数据，它们不等同于 GitHub 账户 credits。查看单场频率与耗时：

```bat
.venv\Scripts\python.exe summarize_diagnostics.py "sessions\meeting_<timestamp>\diagnostics.jsonl"
```

汇总所有历史会话的项目使用频率（默认读取 `sessions/`）：

```bat
.venv\Scripts\python.exe -m local_ai.summarize_copilot_usage
.venv\Scripts\python.exe -m local_ai.summarize_copilot_usage "sessions"
```

#### 术语表维护

`glossaries/general.json` 收录可复用的通用术语，本地 `terminology.json` 用于会议或用户私有词条。复盘时从已核实的修订中持续补充：先人工确认 ASR 变体、标准写法和中文译法，再更新目标术语对应条目；不将 AI 自动输出、未核实的人名或低置信 ASR 猜测直接加入术语表。修改 JSON 后会热加载，新增术语只在证据确认后纳入。

### 模型

首次运行 `setup.bat` 时，`download_models.py` 会下载或准备模型。程序可加载通用和本地术语表；词表文件由用户自行提供，本地 `terminology.json` 的同名术语会覆盖通用词表：

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

诊断日志采用 UTF-8 JSON Lines。每条事件统一带有 `session_id`、单调递增的 `event_sequence`、相对 `elapsed_seconds` 和工作线程名，便于按发生顺序分析并关联并发阶段。

转写链路按 `window_id` → `segment_id` → `record_index` 关联：`asr_window_input` 记录送入 Whisper 的边界、配置、上下文/词汇表规模；`asr_window` 记录音频 RMS/峰值/削波比例和 ASR 统计；`asr_word_timestamps` 记录词级文本、起止时间及置信度；`asr_segmentation` 记录句子边界、保留尾句和触发原因；`transcript_segment` 记录原始识别、清理/规范化后的文本和复核判断。无识别结果、被过滤或去重的片段会记录对应原因。

翻译链路按同一个窗口/片段/记录索引关联，记录选择的上下文或单句翻译模式、模型后端/设备、输入规模、质量检查与重试结果。采集阶段定期记录音量和队列状态；设备运行警告会进入 `diagnostics.jsonl`，并在会话结束汇总。UI 队列接近拥塞或临时事件被舍弃也会记录，历史记录事件保持可靠入队。

ASR/翻译正文、逐词时间戳和音频质量指标可能包含或关联敏感会议内容。运行日志仅保存在本机；分享前请审查并脱敏，尤其不要直接上传完整 `diagnostics.jsonl` 或 WAV。

完成会话后可汇总质量问题与延迟（仅读取日志，`--output` 目标必须不存在）：

```bat
.venv\Scripts\python.exe summarize_diagnostics.py "sessions\meeting_<timestamp>\diagnostics.jsonl"
.venv\Scripts\python.exe summarize_diagnostics.py "sessions\meeting_<timestamp>\diagnostics.jsonl" --output "sessions\meeting_<timestamp>\quality_summary.json"
```

报告统计每类质量 issue 次数、ASR/纠错/翻译耗时 P50/P95、定稿翻译排队时间 P50/P95，以及英文预览、实时中文、定稿英文/中文的音频到 UI 应用延迟和 UI 队列耗时。还分别报告采集设备重连时间线缺口、低延迟队列跳过块、ASR 检测到的帧间隙及过期音频帧；Soundcard discontinuity/underflow/overflow 告警是独立的设备信号，不能直接当作丢帧计数。旧诊断没有采集时钟和上屏时间，新增端到端分位数只会出现在新会话中。快速语速、低置信词、长停顿或 ASR 稀疏都是规则触发的候选信号，需结合对应音频和人工校订稿判断；不能据此自动宣称漏音或改变识别文本。

诊断日志包含 ASR/翻译耗时、队列积压、识别到的音频间隙、设备重连、采集层间断警告，以及采集和 WAV 的帧数/时长。设备重连时，估算的时间缺口会在 WAV 中以静音填充；估算值不等同于声卡报告的精确丢失量。

转写清理事件会同时保存 ASR 候选与清理后的候选文本。当前流程会清理重复/填充语，但日志可用于评估是否清理过度；无法从文本规则区分有语义作用的 “like / actually / you know” 与纯填充用法，需在人工确认后再调整词表或清理策略。

### 实时性能与质量改进顺序

1. **把延迟测到上屏。** 给采集块打单调时钟，在预览、定稿和 UI 应用事件上关联窗口 ID、记录 ID 与版本。分别统计首条英文、首条中文和定稿中文的端到端 P50/P95，同时报告采集告警、录音缺口和 ASR 跳块。先用固定的本地测试音频建立基线。
2. **按压力分配解码预算。** 以待处理音频秒数和最近 ASR 实时因子为输入；压力升高时先降低重复预览频率，再推迟音频复核，最后才调整定稿解码参数。压力恢复后再逐步恢复质量。通过积压秒数、定稿覆盖率和丢块率验证两种运行模式。
3. **让低延迟模式可追回。** 为跳块记录起止帧，空闲或会后从会话 WAV 补转写，通过时间重叠去重、按版本合并。录音本身缺失与实时识别跳过需要分开显示；前者无法用补转写恢复。
4. **提高术语、关键事实与长会稳定性。** 用人工参考稿比较 WER、术语准确率以及数字、日期、单位和否定遗漏率。将推测性的 AI 修订保留为候选，以音频复核或人工确认提升可信度；长会逐步采用增量持久化和按记录 ID 管理上下文，避免修订时扫描全量历史。

### 项目结构

```text
main.py                 实时处理和线程编排
asr_pipeline.py         ASR 音频窗口构造、解码和重试
speech_quality.py       ASR、语速、填充词、停顿与边界质量指标
diagnostics.py          会话 JSONL 事件、线程/时间关联与质量计数
summarize_diagnostics.py 汇总单场会话的质量 issue 和 P50/P95 延迟
pipeline_metrics.py     按音频采集到 UI 应用计算字幕端到端延迟
audio_capture.py        扬声器回环采集、队列和设备重连
asr.py                  faster-whisper 识别
translator.py           MarianMT / CTranslate2 翻译
overlay.py              Tkinter 字幕窗口
tests/                  按音频、ASR、翻译、会话和 UI 分组的测试
worker_runtime.py       后台 worker 生命周期与异常边界
review_extension.py     可选文本复核加载入口
local_ai/               本地 CLI 扩展、agent 和用量工具（默认不纳入 Git）
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

本机保留 `local_ai/` 时，可再运行 `.venv\Scripts\python.exe -B -m unittest discover -s local_ai`。

测试按 `tests/audio`、`tests/asr`、`tests/translation`、`tests/session` 和 `tests/ui` 分组。

从人工校订参考稿中只提取可复用的确定术语补充到 `glossaries/general.json`；人名/模型名/不确定 ASR 猜测不自动进入热词表。建议复盘步骤：对照 `asr_correction_candidate` 与应用/拒绝结果，定位发生变化的词和时间范围；以修订前后文本及对应音频人工确认；确认后的通用术语再补入通用词表，会议特有词条放入本地 `terminology.json`。不要直接用未核实的会议转写或自动译文训练模型。
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

The optional AI CLI extension can be installed locally in the Git-ignored `local_ai/` directory, which may contain machine-specific authorization or configuration. It is disabled by default; set `MEETING_COPILOT_ENABLED=1` to opt in when the extension is installed and approved. OpenCode is the default provider; other providers require explicit selection via `MEETING_AI_CLI`. If the extension or selected CLI is unavailable, the application continues locally. The provider may receive committed meeting text and bounded context; the CLI does not receive audio. Reviews start after 20 seconds, repeat every 30 seconds, and have a 60-second deadline.

### Features

- Transcription and translation run locally by default. Committed translation work is deduplicated and queued independently of transcription.
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

When the overlay is closed, the application stops capture, attempts to drain queued work, and saves the session. Diagnostics report workers and queued translations left over when the shutdown wait expires.

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

ASR and translation quality events also record review signals such as low-confidence words, possible fast speech or sparse recognition, filler-heavy speech, long pauses, candidate sentence-boundary issues, rejected/applied corrections, and English/Chinese pairs for translation issues. These signals are heuristics, not human-confirmed transcription errors or proof of omitted speech. Diagnostic transcript text and timestamps can contain sensitive meeting content; review and redact them before sharing.

Summarize one session's issue counts and latency percentiles with:

```bat
.venv\Scripts\python.exe summarize_diagnostics.py "sessions\meeting_<timestamp>\diagnostics.jsonl"
.venv\Scripts\python.exe summarize_diagnostics.py "sessions\meeting_<timestamp>\diagnostics.jsonl" --output "sessions\meeting_<timestamp>\quality_summary.json"
```

The summary reports issue counts, ASR/translation P50/P95 latency, and ASR real-time factor. An output path must not already exist.

### Project layout

```text
main.py                 Real-time pipeline and thread orchestration
asr_pipeline.py         ASR window construction, decoding, and retry
speech_quality.py       Explainable ASR, speaking-rate, filler, pause, and boundary metrics
summarize_diagnostics.py Summarize per-session quality issues and latency percentiles
audio_capture.py        Speaker loopback capture, queue, and reconnects
asr.py                  faster-whisper transcription
translator.py           MarianMT / CTranslate2 translation
overlay.py              Tkinter subtitle window
tests/                  Tests grouped by audio, ASR, translation, session, and UI
worker_runtime.py       Worker lifecycle and exception boundaries
review_extension.py     Optional transcript-review loader
local_ai/               Local CLI extension, agent, and usage reporting (Git-ignored)
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
