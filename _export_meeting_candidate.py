import re
import sys
from datetime import datetime
from pathlib import Path

session = Path(sys.argv[1])
source = session / "meeting_transcript_20261021_150056_368230.txt"
duration = 1549.5
pattern = re.compile(r"^\[(\d{2}:\d{2}\.\d{3})-(\d{2}:\d{2}\.\d{3})\]\s+(?:(.*?)\s+)?\[EN\]\s+(.*)$")

def to_seconds(value):
    m, s = value.split(":")
    return int(m) * 60 + float(s)

def timecode(value):
    ms = round(value * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

lines = source.read_text(encoding="utf-8-sig").splitlines()
cues, current = [], None
for line in lines:
    match = pattern.match(line)
    if match:
        if current and current["zh"]:
            cues.append(current)
        start, end, tags, en = match.groups()
        current = {"start": to_seconds(start), "end": to_seconds(end), "tags": tags or "", "en": en.strip(), "zh": ""}
    elif current and line.startswith("[ZH]"):
        current["zh"] = line[4:].strip()
if current and current["zh"]:
    cues.append(current)
cues.sort(key=lambda c: (c["start"], c["end"]))
for cue in cues:
    if not (0 <= cue["start"] < cue["end"] <= duration):
        raise ValueError(f"Invalid candidate cue: {cue}")

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
out = session / f"output_{stamp}"
out.mkdir()
stem = "meeting_20261021_150056_368230"
en_srt, bi_srt, table = [], [], []
for n, cue in enumerate(cues, 1):
    tc = f"{timecode(cue['start'])} --> {timecode(cue['end'])}"
    en_srt.append(f"{n}\n{tc}\n{cue['en']}")
    bi_srt.append(f"{n}\n{tc}\n{cue['en']}\n{cue['zh']}")
    table.append(f"| {n} | {tc} | 未确认 | {cue['en'].replace('|', '\\|')} | {cue['zh'].replace('|', '\\|')} | 沿用源转写边界；未经音频核准 | {cue['tags'] or '待核'} |")

def save(name, value):
    (out / f"{stem}{name}").write_text(value.rstrip() + "\n", encoding="utf-8")

save(".en.srt", "\n\n".join(en_srt))
save("_en+zh.srt", "\n\n".join(bi_srt))
save("_split.md", "# Cue 对照与复核台账\n\n此文件仅将既有带时间戳转写整理为 Cue；中文行沿用源文件候选译文，未在 Python 中生成或翻译。尚未对照 WAV 回听。\n\n| Cue | 候选时间码 | 说话轮次 | English | 中文候选 | 断句依据 | 待核项 |\n|---:|---|---|---|---|---|---|\n" + "\n".join(table))

# The report prose below is editorial synthesis, distinct from the source transcript.
report = """# 会议报告（候选稿）

**状态：候选 / 待人工回听。** 下列稿件只涵盖现有稀疏转写中出现的语音片段，不代表整场会议完整内容。Cue 内中文沿用已有译文候选，需逐条校译。

## 1. 英文修正整理稿（基于现有转写片段）

The meeting opened with a short wait for more participants. The facilitator welcomed everyone to the third workshop and said the group would first review the second workshop, focusing on what plan mode had changed and what participants had experienced. The workshop series was organized around planning, prompting, and guardrails. Before code is written, the team should explore the ticket, clarify the expected outcome, create a written plan, read it, and iterate on it. Keeping the work in manageable slices was presented as useful both for controlling token use and for making the intended result clear.

The facilitator said the team had held several one-to-one conversations, including with some level-two participants, and thanked attendees for sharing their experiences. Participants were asked to complete a feedback form with free-text answers so the trainers could better understand what they wanted, what they needed, and where they encountered pain points. The group then reviewed responses and asked how well planning before coding fit participants’ day-to-day work.

One participant described planning as a way to discover behavior changes and details that were not explicit in the ticket. The plan helped expose edge cases and compatibility checks involving old user data. A step-by-step guide was described as a useful outcome: it can make the AI’s proposed work easier to understand and can give both AI and human reviewers clearer context in a pull request. The discussion also connected plans with verification: a plan can include unit tests, a use case, or an integration test, and can help the work reach a testable state.

The facilitator encouraged participants not to fall behind on the lecture series and invited anyone who needed help to arrange a one-to-one session. Participants noted that splitting a large ticket into smaller steps can make it more manageable and easier to follow when reviewing one’s own changes. The meeting also acknowledged a tradeoff: planning and discussing a task with AI can feel like extra work when a small prompt might seem sufficient. Plan mode, and a “Grill Me” add-on mentioned in the source transcript, were discussed as ways to iteratively clarify requirements and discover issues, including unexpected legacy-code problems.

The discussion cautioned that a one-line change is not necessarily a simple change: even a tiny code edit can have a substantial impact. A suggested practice was to ask AI to assess the risk and impact of a specific line change, with or without a full plan, to help avoid rework. The available transcript ends with thanks for the review and a request to join another meeting shortly, with a brief break in between.

## 2. 中文译稿（对应上节的整理稿）

会议开始时，主持人先等待了片刻，希望有更多参与者加入。随后，主持人欢迎大家参加第三次研讨会，并表示将先回顾第二次研讨会，重点了解计划模式带来了哪些变化、参与者有哪些体验。系列研讨会围绕规划、提示和护栏展开。编写代码之前，应先探索工单、明确预期结果，形成书面计划，通读计划并持续迭代。将工作切分为适当的小块，既有助于控制 token 使用量，也有助于明确真正想要的结果。

主持人提到，团队上周进行了多次一对一交流，其中包括与部分二级参与者的交流，并感谢大家分享经验。主持人请参与者填写带有自由文本回答的反馈表，以便培训人员更好地了解大家想要什么、需要什么，以及遇到哪些痛点。随后，大家查看反馈，并讨论编码前进行规划与日常工作的契合程度。

一位参与者谈到，规划有助于发现行为变化，以及工单中没有明确写出的细节。计划也帮助他们发现边界情况，以及涉及旧用户数据的兼容性检查。逐步指南被认为是一个有用的结果：它能让 AI 提出的工作内容更容易理解，也能为拉取请求中的 AI 审查和人工审查提供更清晰的背景。讨论还将计划与验证联系起来：计划可以列入单元测试、用例或集成测试，并帮助工作达到可测试的状态。

主持人鼓励大家不要落下系列讲座，并邀请需要帮助的人预约一对一交流。参与者提到，把大型工单拆成更小的步骤，有助于降低处理难度，也更容易在回顾时跟上自己所做的改动。讨论同时承认其中存在取舍：当一个小提示看起来就能完成任务时，先制定计划并与 AI 讨论，可能会让人觉得增加了工作量。根据现有转写，大家还讨论了计划模式以及名为“Grill Me”的附加工具，认为它们可以通过与 AI 迭代交流来澄清需求和发现问题，包括意料之外的遗留代码问题。

讨论提醒大家，改动只有一行代码，并不意味着它一定简单；很小的代码改动也可能造成重大影响。一个建议的做法是，针对某一行代码的改动，请 AI 评估其风险和影响；无论是否制定完整计划，这都可能帮助避免返工。现有转写的结尾部分是对本次回顾的致谢，并请大家稍后加入另一场会议，中间短暂休息。

## 3. 会议内容总结（限现有转写片段）

- 回顾计划模式的使用体验，并收集参与者的自由文本反馈。
- 讨论编码前探索工单、明确结果、撰写并迭代计划的价值。
- 计划可能帮助发现工单遗漏的边界情况、旧数据兼容性要求及验证步骤，也能改善拉取请求审查所需的上下文。
- 把大型任务拆成小步骤有助于理解、跟踪和验证；但规划可能增加小任务的前期工作量，仍需结合风险判断。
- 现有片段提及一对一辅导和后续会议安排；无法据此确认正式作业或完整行动项。

## 4. 技术细节与延伸思考

### 会议片段中明确提及

计划模式、工单探索、工作切片、低 token 使用量、边界情况、旧用户数据兼容性、单元测试、用例、集成测试、可测试状态、拉取请求中的 AI 与人工审查、遗留代码问题，以及对代码改动进行风险和影响评估。以上技术点来自现有转写候选，专名及个别措辞仍待音频确认。

### 补充分析（非会议原文）

实际使用时，可要求计划明确验收条件、受影响的数据与兼容性、边界场景、验证方法及回滚/风险点；小型改动也应按影响面而非代码行数判断风险。此段是整理者的建议，不代表会议原话。

## 5. 课后作业与实践练习

### 录音片段中实际布置

现有转写可见主持人鼓励大家继续使用计划模式，并在需要赶进度或深入讨论时联系培训人员安排一对一交流。是否构成正式作业、提交形式与截止时间，现有材料无法确认。

### 补充练习

选择一个实际工单，先让计划模式列出步骤、边界情况、旧数据兼容性和验证方式；人工审阅后再实施，并记录计划中遗漏或不准确的部分。此为补充练习，不是录音确认的作业。

## 制作与覆盖说明

- 音频：`meeting_audio.wav`，FFmpeg 4.4.1 检查为 PCM S16LE、16 kHz、mono，时长约 00:25:49.50。
- 文字底稿：session 中既有带时间戳转写。现有时间码和中文译文均未通过本轮逐条回听核准。
- 本轮没有成功运行 Whisper：模型文件存在，但 faster-whisper 依赖安装时下载 PyAV wheel 多次超时；没有生成词级 ASR JSONL。
- 没有播放器可用于人工听校。中文整理稿由本轮直接翻译现有可见片段；字幕 Cue 中保留源文件原有中文候选，尚待逐条校译。
- 可解析源片段映射到本轮字幕共 202 条。该数字只表示材料间的条目映射，不是音频覆盖率。英文、中文及字幕的音频覆盖率均无法计算。
- 约 05:19.670–11:19.650 有长时间无源转写条目；其他空档也未判定为静音或语音。全片覆盖和所有缺段仍待音频核验。
- 不得将本候选稿标记为 final 或称为全程人工听校。
"""
save("_report.md", report)
save("_en_transcript.md", "# 英文口语整理稿（部分候选）\n\n" + report.split("## 1. 英文修正整理稿（基于现有转写片段）\n\n", 1)[1].split("\n\n## 2.", 1)[0])
save("_review.md", "# 总复核报告\n\n状态：候选 / 待人工回听。\n\n- 音频：既有 WAV，FFmpeg 4.4.1 检查通过，PCM S16LE、16 kHz、mono、约 00:25:49.50。\n- 英文/中文底稿：session 中现有带时间戳转写；共整理出 202 条含 EN 与 ZH 的候选。\n- Whisper 模型文件存在；faster-whisper 1.2.1 安装因 PyPI 上 PyAV wheel 下载超时而未完成，未加载模型/运行 ASR。\n- 本机没有可用播放器，未进行人工回听；Cue 时间未核准。\n- 双语 SRT 的中文沿用原转写候选；报告中文整理稿由本轮直接翻译可见内容，仍需对音频校核。\n- 输出：`*.en.srt`、`*_en+zh.srt`、`*_split.md`、`*_report.md`、`*_en_transcript.md`、`*_review.md`、`*_segmentation_review.md`、`*_coverage_ledger.md`、`*_text_alignment_report.md`。\n- 音频覆盖率不可计算；缺少 ASR 全程覆盖、逐 Cue 人工回听、术语裁决。")
save("_segmentation_review.md", "# 英文拆分复核记录\n\nCue 依据现有转写的时间段；未通过音频回听。各 Cue 的文本、断句、入点、尾音、短间隔及重叠均待人工核实。逐 Cue 候选及源转写标记见 `_split.md`。")

ledger = [f"| {n} | {timecode(c['start'])}–{timecode(c['end'])} | 既有转写 | 未核准 | {n} | {n} | {n} | {n} |" for n, c in enumerate(cues, 1)]
save("_coverage_ledger.md", "# 覆盖台账（候选映射）\n\n此处一段对应一个既有转写条目；这是文档映射而非经音频确认的有声段。\n\n| 段 | 源时间 | 来源 | 回听核准 | 英文稿段 | 中文稿段 | EN Cue | 双语 Cue |\n|---:|---|---|---|---:|---:|---:|---:|\n" + "\n".join(ledger) + "\n\n源材料大段时间空档包括 00:05:19.670–00:11:19.650；此处未判断为空白静音。音频可辨段总数未知，覆盖率不可计算。")
save("_text_alignment_report.md", f"# 文本对齐检查\n\n从辅助转写解析到 {len(cues)} 条含英文和中文的候选；每条映射至两个 SRT 和 split 表。该校验仅为源文档间的文本映射，不证明与音频对齐或全场覆盖。音频未映射的可辨语音数未知，长空档需要回听；未生成 ASR JSONL。")
print(f"{out}\nCue count: {len(cues)}")
