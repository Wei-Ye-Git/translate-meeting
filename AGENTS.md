# MeetingTwo：本地音视频与字幕工具

## Skills

| Skill | 触发场景 | 入口 |
|---|---|---|
| `bilingual-media-subtitles` | “音视频双语字幕制作”“制作英中字幕和完整原话报告” | [.agents/skills/bilingual-media-subtitles/SKILL.md](.agents/skills/bilingual-media-subtitles/SKILL.md) |
| `bilingual-media-subtitles-en` | “Make English-Chinese subtitles and a full verbatim report” | [.agents/skills/bilingual-media-subtitles-en/SKILL.md](.agents/skills/bilingual-media-subtitles-en/SKILL.md) |

本 Skill 优先复用现有 Python 环境、本地 Whisper 模型与 FFmpeg。
完整规范和跨机器部署步骤随 Skill 提供；源提示词保留在 `prompts/音视频双语字幕制作.md`。
