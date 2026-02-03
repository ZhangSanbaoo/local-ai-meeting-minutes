# Meeting AI 🎙️

本地离线会议纪要工具 - 语音转写、说话人分离、智能命名、会议总结

## 功能特性

- 🎤 **语音转写** - 使用 faster-whisper 将音频转换为文字
- 👥 **说话人分离** - 使用 pyannote-audio 3.1 识别不同说话人
- 🏷️ **智能命名** - 使用本地 LLM (Qwen2.5-7B) 推断说话人身份
- 🚻 **性别检测** - 基于基频分析判断说话人性别
- ✏️ **错别字校正** - LLM 修复常见转写错误
- 📝 **会议总结** - 自动生成会议摘要和要点
- 🖥️ **桌面 GUI** - Flet 0.80+ 图形界面，支持音频播放和实时高亮
- 🔒 **完全离线** - 所有处理都在本地完成，保护隐私

## 安装

### 前置要求

- Python 3.10+
- ffmpeg（音频处理）
- HuggingFace 账号（用于下载 pyannote 模型）

### 安装步骤

```bash
# 克隆项目
git clone https://github.com/yourusername/meeting-ai.git
cd meeting-ai

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖（包含 GUI 和音频增强）
pip install -e ".[dev,gui,enhance]"
```

### 下载模型（重要！）

本项目完全离线运行，需要提前下载模型。

**步骤 1：获取 HuggingFace Token**

1. 注册 HuggingFace：https://huggingface.co/join
2. 同意模型使用协议（必须！）：
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0  
   - https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
3. 获取 Token：https://huggingface.co/settings/tokens

**步骤 2：下载模型**

```bash
# 设置 Token
export HF_TOKEN="你的token"

# 运行下载脚本
python scripts/download_models.py

# 或手动下载（如果脚本有问题）
pip install huggingface_hub
huggingface-cli download pyannote/speaker-diarization-3.1 \
    --local-dir ./models/pyannote/speaker-diarization-3.1 \
    --token $HF_TOKEN
```

下载完成后，`models/` 目录结构应该是：
```
models/
├── pyannote/
│   ├── speaker-diarization-3.1/
│   ├── segmentation-3.0/
│   └── wespeaker-voxceleb-resnet34-LM/
└── whisper/
    └── faster-whisper-small/
```

## 快速开始

### CLI 命令

```bash
# 查看帮助
meeting-ai --help

# 查看系统信息
meeting-ai info

# 完整处理音频文件
meeting-ai process meeting.mp3

# 带选项处理
meeting-ai process meeting.mp3 --no-summary --enhance

# 仅转写
meeting-ai transcribe meeting.mp3

# 仅说话人分离
meeting-ai diarize meeting.mp3
```

### GUI 运行

```bash
# 运行图形界面（推荐，支持热重载）
flet run src/meeting_ai/gui.py

# 或直接运行
python src/meeting_ai/gui.py
```

GUI 功能：
- 选择音频文件或历史记录
- 实时显示处理进度
- 音频播放器（带进度条和片段高亮）
- 编辑说话人名字和对话内容
- 查看会议总结
- 导出为 TXT/JSON/Markdown

## 配置

可以通过环境变量或 `.env` 文件配置：

```bash
# .env 文件示例

# 路径配置
MEETING_AI_DATA_DIR=./data
MEETING_AI_MODELS_DIR=./models
MEETING_AI_OUTPUT_DIR=./outputs

# ASR 配置
MEETING_AI_ASR__MODEL_NAME=medium      # tiny/base/small/medium/large-v3
MEETING_AI_ASR__DEVICE=auto            # cpu/cuda/auto
MEETING_AI_ASR__COMPUTE_TYPE=int8      # int8/float16/float32
MEETING_AI_ASR__LANGUAGE=zh

# 说话人分离配置
HF_TOKEN=your_huggingface_token
MEETING_AI_DIAR__MODEL_DIR=models/pyannote/speaker-diarization-3.1

# LLM 配置
MEETING_AI_LLM__ENABLED=true
MEETING_AI_LLM__MODEL_PATH=models/llm/qwen2.5-7b-instruct-q4_k_m.gguf
MEETING_AI_LLM__N_CTX=6144
```

## 项目结构

```
meeting-ai/
├── src/meeting_ai/
│   ├── __init__.py          # 包入口
│   ├── cli.py               # 命令行接口 (typer)
│   ├── gui.py               # 桌面 GUI (Flet 0.80+)
│   ├── config.py            # 配置管理 (pydantic-settings)
│   ├── logger.py            # 日志系统 (rich)
│   ├── models.py            # 数据模型 (Pydantic)
│   ├── services/
│   │   ├── diarization.py   # 说话人分离
│   │   ├── asr.py           # 语音转写
│   │   ├── alignment.py     # 时间对齐
│   │   ├── gender.py        # 性别检测
│   │   ├── naming.py        # 智能命名
│   │   ├── correction.py    # 错别字校正
│   │   └── summary.py       # 会议总结
│   └── utils/
│       ├── audio.py         # 音频格式转换
│       └── enhance.py       # 音频增强
├── models/                   # 本地模型目录
├── outputs/                  # 处理结果输出
├── tests/                    # 测试
├── scripts/                  # 脚本
├── pyproject.toml           # 项目配置
└── README.md
```

## 开发路线图

- [x] **阶段 0** - 项目骨架（CLI, config, models）
- [x] **阶段 1** - 说话人分离（pyannote-audio 3.1）
- [x] **阶段 2** - 语音转写 + 时间对齐（faster-whisper）
- [x] **阶段 3** - 智能命名 + 性别检测（LLM + 基频分析）
- [x] **阶段 4** - 会议总结 + 音频增强（LLM + noisereduce）
- [x] **阶段 5** - 桌面 GUI（Flet 0.80+）
- [ ] **阶段 6** - 实时流式录音（sounddevice + webrtcvad）
- [ ] **阶段 7** - Tauri 打包（Rust + 前端）

## 许可证

MIT License
