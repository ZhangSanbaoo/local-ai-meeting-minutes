# Meeting AI - 本地离线会议纪要工具

完全离线运行的 AI 会议纪要工具，自动将音频转换为带说话人标识的结构化会议纪要。

## 功能特性

- **实时录音转写** - 边录边转，流式 ASR + VAD 自动分段
- **音频文件处理** - 上传音频文件进行完整处理
- **说话人分离** - pyannote-audio 3.1 识别不同说话人
- **智能命名** - 本地 LLM 推断说话人身份（"张教授"、"小柔"）
- **性别检测** - 4 引擎可选（f0 基频 / ECAPA-TDNN / Wav2Vec2 / Audeering），最高 98% 准确率
- **错别字校正** - LLM 修复常见转写错误
- **会议总结** - 自动生成会议摘要和要点
- **历史记录** - 浏览、编辑、导出历史处理结果
- **完全离线** - 所有处理都在本地完成，保护隐私
- **RTX 5090 支持** - 支持最新 Blackwell 架构 GPU (sm_120)

## 架构

```
┌──────────────────┐         ┌──────────────────┐
│   React 前端     │  HTTP   │  FastAPI 后端     │
│  TypeScript      │ ◄─────► │  Python 3.13      │
│  Vite + Tailwind │   WS    │  WebSocket        │
└──────────────────┘         └──────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
             ┌───────────┐  ┌──────────────┐  ┌───────────┐
             │ pyannote   │  │ FunASR /     │  │ Qwen2.5   │
             │ 说话人分离 │  │ sherpa-onnx  │  │ 7B LLM    │
             └───────────┘  │ + fsmn-vad   │  └───────────┘
                            └──────────────┘
```

## 项目结构

```
meeting-ai/
├── backend/                    # Python 后端 (FastAPI)
│   ├── src/meeting_ai/
│   │   ├── api/               # REST API + WebSocket 端点
│   │   │   └── routes/        # process, history, models, realtime
│   │   ├── services/          # 核心 AI 服务
│   │   │   ├── streaming_asr.py   # 流式 ASR (FunASR + sherpa-onnx + fsmn-vad)
│   │   │   ├── diarization.py     # 说话人分离
│   │   │   ├── asr.py             # 离线转写 (faster-whisper)
│   │   │   ├── gender.py          # 性别检测 (f0 / ECAPA / Wav2Vec2 / Audeering)
│   │   │   ├── naming.py          # 智能命名
│   │   │   └── summary.py         # 会议总结
│   │   ├── utils/             # 工具函数
│   │   └── config.py          # 配置管理
│   ├── tests/
│   ├── pyproject.toml
│   └── .env
│
├── frontend/                   # React 前端
│   ├── src/
│   │   ├── pages/             # FilePage, RealtimePage, SettingsPage
│   │   ├── components/        # AudioPlayer, SegmentCard, SummaryPanel
│   │   ├── hooks/             # useAudioCapture, useRealtimeWebSocket
│   │   ├── stores/            # Zustand 状态管理
│   │   └── api/               # API 客户端
│   ├── public/audio-worklet/  # AudioWorklet PCM 处理器
│   └── package.json
│
├── models/                     # 本地 AI 模型
│   ├── pyannote/              # 说话人分离模型
│   ├── whisper/               # Whisper ASR 模型
│   ├── gender/                # 性别检测模型 (ecapa-gender / wav2vec2-gender / audeering-gender)
│   ├── llm/                   # Qwen2.5-7B (GGUF)
│   └── streaming/             # 流式 ASR 模型
│       ├── funasr/            # paraformer-zh-streaming + ct-punc + fsmn-vad
│       └── sherpa-onnx/       # paraformer-trilingual (zh/粤/en)
│
├── outputs/                    # 处理结果
└── scripts/                    # 工具脚本
```

## 快速开始

### 前置要求

- Python 3.13+ (RTX 5090) 或 3.11+ (旧显卡)
- Node.js 18+
- ffmpeg
- NVIDIA GPU (推荐)

### 1. 安装后端

```bash
cd backend

# 创建 Python 环境
mamba create -n meeting-ai python=3.13 -y
conda activate meeting-ai

# 安装 PyTorch（根据显卡选择）
# RTX 5090 (Blackwell):
pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
# RTX 4090 及更早:
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装系统依赖和项目
mamba install ffmpeg cmake -c conda-forge -y
pip install -e ".[stream,enhance]"
```

### 2. 安装前端

```bash
cd frontend
npm install
```

### 3. 配置

复制并编辑后端配置:
```bash
cp backend/.env.example backend/.env
# 编辑 .env 设置模型路径和 GPU 参数
```

### 4. 启动服务

```bash
# 终端 1: 后端
cd backend
uvicorn meeting_ai.api.main:app --reload --host 0.0.0.0 --port 8000

# 终端 2: 前端
cd frontend
npm run dev
```

打开浏览器访问 http://localhost:5173

## 使用方式

### Web 界面

- **实时录音**: 选择 ASR 引擎 → 加载模型 → 开始录音 → 自动转写 → 停止后生成完整纪要
- **音频文件**: 上传文件 / 选择历史记录 → 配置选项 → 开始处理 → 查看/编辑/导出结果

### CLI

```bash
cd backend
meeting-ai process audio.mp3
meeting-ai process audio.mp3 --no-summary --enhance
meeting-ai diarize audio.mp3
meeting-ai transcribe audio.mp3
```

## API 文档

后端启动后访问:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 主要接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/process` | 上传音频并处理 |
| GET | `/api/jobs/{id}` | 查询任务状态 |
| GET | `/api/jobs/{id}/result` | 获取处理结果 |
| GET | `/api/history` | 历史记录列表 |
| GET | `/api/history/{id}` | 历史记录详情 |
| GET | `/api/history/{id}/export/{format}` | 导出 (txt/json/md) |
| GET | `/api/models` | 可用模型列表 |
| GET | `/api/streaming-engines` | 流式 ASR 引擎列表 |
| WebSocket | `/api/ws/realtime` | 实时流式 ASR |

## 配置

### 后端配置 (backend/.env)

```bash
# 路径（相对于 backend/ 目录）
MEETING_AI_MODELS_DIR=../models
MEETING_AI_OUTPUT_DIR=../outputs

# ASR
MEETING_AI_ASR__MODEL_NAME=medium       # tiny/base/small/medium/large-v3
MEETING_AI_ASR__DEVICE=cuda             # cpu/cuda/auto
MEETING_AI_ASR__COMPUTE_TYPE=float16    # int8/float16/float32

# LLM
MEETING_AI_LLM__ENABLED=true
MEETING_AI_LLM__MODEL_PATH=llm/Qwen2.5-7B-Instruct-Q4_K_M.gguf
MEETING_AI_LLM__N_CTX=6144
```

## 模型

所有模型存放在 `models/` 目录下:

| 模型 | 用途 | 大小 | 准确率/性能 |
|------|------|------|------------|
| pyannote speaker-diarization-3.1 | 说话人分离 | ~50MB | DER ~11% |
| faster-whisper-medium | 离线语音转写 | ~1.5GB | CER 5.1% |
| ecapa-gender | 性别检测 | ~60MB | ~97% |
| wav2vec2-gender | 性别检测 | ~1.2GB | ~95% |
| audeering-gender | 性别检测（年龄+性别） | ~1.2GB | ~98% ⭐ |
| Qwen2.5-7B-Instruct Q4_K_M | 命名/校正/总结 | ~4.4GB | 7B 参数 |
| FunASR paraformer-zh-streaming | 中文流式 ASR | ~220MB | 实时 |
| FunASR ct-punc | 标点恢复 | ~300MB | - |
| FunASR fsmn-vad | 语音活动检测 | ~1.6MB | - |
| sherpa-onnx paraformer-trilingual | 三语流式 ASR | ~220MB | zh/粤/en |

## 技术栈

**后端:** Python 3.13, FastAPI, pyannote-audio 3.1, faster-whisper, FunASR 1.3.1, sherpa-onnx 1.12.23, llama-cpp-python, Qwen2.5-7B

**前端:** React 18, TypeScript, Vite, TailwindCSS, Zustand

**开发环境:** Windows 原生, RTX 5090 (sm_120), PyTorch nightly cu128

## 开发路线图

- [x] 核心功能 (说话人分离 + ASR + 命名 + 总结)
- [x] CLI + Flet GUI (MVP)
- [x] 前后端分离 (FastAPI + React)
- [x] 实时流式 ASR (FunASR + sherpa-onnx 双引擎)
- [x] fsmn-vad 流式语音活动检测
- [x] 多引擎性别检测 (4 引擎，最高 98% 准确率)
- [x] Windows 文件锁处理 (删除重试机制)
- [ ] 集成测试
- [ ] 系统音频采集
- [ ] Tauri 桌面应用打包

## 许可证

MIT License
