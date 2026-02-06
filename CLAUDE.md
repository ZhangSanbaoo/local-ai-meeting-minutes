# Meeting AI 项目 - Claude Code 指南

## 项目概述

**项目名称**: 会议纪要 AI (meeting-ai)

**一句话描述**: 完全离线运行的本地 AI 软件，自动将音频转换为带说话人标识的会议纪要

**架构**: 前后端分离 — FastAPI 后端 + React 前端（最终目标: Tauri 桌面应用）

**用途**: 个人项目，用于简历展示

**开发环境**: Windows 原生 (不使用 WSL)

---

## Snapshot 快照规则 (重要！必须遵守)

**在进行以下操作之前，必须先运行快照脚本：**
- 架构重构（如前后端分离、目录大调整）
- 大规模文件修改（影响 5 个以上文件的改动）
- 技术栈变更（如更换框架、引入新依赖）
- 删除或重写核心模块

### 使用方式

```powershell
powershell scripts/snapshot.ps1 -Desc "改动描述"
# 例: powershell scripts/snapshot.ps1 -Desc "before realtime ASR rewrite"
```

### 快照包含
1. **Git commit + tag** — 用于代码回滚 (`git reset --soft snapshot/<timestamp>`)
2. **状态记录 MD 文件** — 保存在 `snapshots/` 目录，记录当时的文件结构、配置、依赖、功能状态

### 回滚命令

```bash
# 查看快照后的改动
git diff snapshot/<timestamp>..HEAD

# 软回滚（保留改动为未提交状态）
git reset --soft snapshot/<timestamp>

# 硬回滚（丢弃所有改动）
git reset --hard snapshot/<timestamp>

# 恢复单个文件
git checkout snapshot/<timestamp> -- path/to/file
```

---

## 核心功能

1. **说话人分离** - pyannote-audio 3.1，识别音频中不同说话人
2. **语音转写** - faster-whisper，音频文件离线转写
3. **实时流式转写** - FunASR Paraformer / sherpa-onnx 双引擎，边录边转
4. **流式 VAD** - fsmn-vad 语音活动检测，自动分段
5. **智能命名** - LLM 推断说话人身份（"张教授"、"小柔"）
6. **性别检测** - librosa 基频分析
7. **错别字校正** - LLM 修复转写错误
8. **会议总结** - LLM 自动生成会议摘要
9. **音频增强** - 降噪、去混响（可选）

---

## 技术栈

| 模块 | 技术 | 说明 |
|------|------|------|
| 后端框架 | FastAPI | REST API + WebSocket 实时通信 |
| 前端框架 | React 18 + TypeScript | Vite + Tailwind CSS + Zustand |
| 说话人分离 | pyannote-audio 3.1 | 本地离线 |
| 实时流式 ASR | FunASR 1.3.1 + sherpa-onnx 1.12.23 | 双引擎可选 |
| 流式 VAD | fsmn-vad | FunASR 官方 0.4M 参数 VAD |
| 离线 ASR | faster-whisper | CTranslate2，后处理和音频文件 |
| LLM | llama-cpp-python + Qwen2.5-7B | 命名/校正/总结 |
| 性别检测 | librosa 基频分析 | 男性 < 165Hz < 女性 |
| 音频处理 | ffmpeg + librosa | 格式转换 |
| 配置 | pydantic-settings | 支持 .env 文件 |

---

## 项目结构

```
meeting-ai/
├── backend/                         # Python 后端 (FastAPI)
│   ├── src/meeting_ai/
│   │   ├── api/
│   │   │   ├── main.py              # FastAPI 应用入口
│   │   │   ├── schemas.py           # API 数据模型
│   │   │   └── routes/
│   │   │       ├── process.py       # 音频上传处理 + 任务轮询
│   │   │       ├── history.py       # 历史记录 CRUD + 导出
│   │   │       ├── models.py        # 模型管理 + 系统信息
│   │   │       └── realtime.py      # WebSocket 实时流式 ASR
│   │   ├── services/
│   │   │   ├── streaming_asr.py     # 流式 ASR 引擎抽象 (FunASR + sherpa-onnx + fsmn-vad)
│   │   │   ├── diarization.py       # 说话人分离 (pyannote)
│   │   │   ├── asr.py               # 离线转写 (faster-whisper)
│   │   │   ├── alignment.py         # 说话人-文本对齐
│   │   │   ├── gender.py            # 性别检测 (基频分析)
│   │   │   ├── naming.py            # 智能命名 (LLM + 正则)
│   │   │   ├── correction.py        # 错别字校正 (LLM)
│   │   │   ├── summary.py           # 会议总结 (LLM)
│   │   │   ├── llm.py               # LLM 服务
│   │   │   └── llm_postprocess.py   # LLM 后处理管线
│   │   ├── utils/
│   │   │   ├── audio.py             # 音频格式转换 (ffmpeg)
│   │   │   ├── enhance.py           # 音频增强 (noisereduce)
│   │   │   └── wav_writer.py        # 增量 WAV 写入器
│   │   ├── config.py                # 配置管理 (pydantic-settings)
│   │   ├── models.py                # 数据模型 (Segment, SpeakerInfo)
│   │   └── logger.py                # 日志配置
│   ├── tests/
│   ├── pyproject.toml
│   └── .env                         # 后端环境变量
│
├── frontend/                        # React 前端
│   ├── public/
│   │   └── audio-worklet/
│   │       └── pcm-processor.js     # AudioWorklet PCM 采集处理器
│   ├── src/
│   │   ├── api/client.ts            # Axios API 封装
│   │   ├── components/              # UI 组件 (AudioPlayer, SegmentCard, SummaryPanel, Dialog)
│   │   ├── hooks/
│   │   │   ├── useAudioCapture.ts   # 麦克风采集 (AudioWorklet)
│   │   │   ├── useRealtimeWebSocket.ts  # WebSocket 客户端
│   │   │   ├── useAudioPlayer.ts    # 音频播放器
│   │   │   └── useRecordingTimer.ts # 录音计时器
│   │   ├── pages/
│   │   │   ├── FilePage.tsx         # 音频文件处理页
│   │   │   ├── RealtimePage.tsx     # 实时录音页
│   │   │   └── SettingsPage.tsx     # 设置页
│   │   ├── stores/appStore.ts       # Zustand 全局状态
│   │   ├── types/index.ts           # TypeScript 类型定义
│   │   └── App.tsx                  # 根组件 (Tab 切换)
│   ├── package.json
│   └── vite.config.ts               # Vite 配置 (含 API 代理)
│
├── models/                          # 本地模型目录
│   ├── pyannote/                    # 说话人分离模型
│   ├── whisper/                     # Whisper ASR 模型
│   ├── llm/                         # LLM (Qwen2.5-7B GGUF)
│   └── streaming/                   # 流式 ASR 模型
│       ├── funasr/
│       │   ├── paraformer-zh-streaming/  # 流式中文 ASR
│       │   ├── ct-punc/                  # 标点恢复
│       │   └── fsmn-vad/                 # 语音活动检测
│       └── sherpa-onnx/             # 三语 ASR (zh/粤/en)
│
├── outputs/                         # 处理结果输出
├── scripts/
│   ├── snapshot.ps1                 # 快照脚本
│   └── snapshot-check.ps1           # 快照提醒 hook
└── docs/
```

---

## API 路由

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/process` | 上传音频并开始处理 |
| GET | `/api/jobs/{id}` | 查询任务状态 |
| GET | `/api/jobs/{id}/result` | 获取处理结果 |
| PUT | `/api/jobs/{id}/segments/{id}` | 编辑段落文本/说话人 |
| PUT | `/api/jobs/{id}/speakers` | 重命名说话人 |
| POST | `/api/jobs/{id}/segments/{id}/split` | 分割段落 |
| GET | `/api/history` | 历史记录列表 |
| GET | `/api/history/{id}` | 历史记录详情 |
| PUT | `/api/history/{id}/segments/{id}` | 编辑历史段落 |
| PUT | `/api/history/{id}/speakers` | 重命名历史说话人 |
| POST | `/api/history/{id}/segments/{id}/split` | 分割历史段落 |
| POST | `/api/history/{id}/segments/merge` | 合并历史段落 |
| POST | `/api/history/{id}/summary/regenerate` | 重新生成总结 |
| GET | `/api/history/{id}/export/{format}` | 导出 (txt/json/md) |
| GET | `/api/models` | 可用模型列表 |
| GET | `/api/streaming-engines` | 流式 ASR 引擎列表 |
| GET | `/api/system` | 系统信息 (CUDA/GPU) |
| GET | `/api/audio-devices` | 音频设备列表 |
| WebSocket | `/api/ws/realtime` | 实时流式 ASR |

### WebSocket 消息协议 (`/api/ws/realtime`)

**客户端 → 服务端:**
| type | 说明 |
|------|------|
| `preload_models` | 加载 ASR 引擎（手动触发） |
| `unload_models` | 释放 ASR 引擎 GPU 内存 |
| `start_recording` | 开始录音 |
| `stop_recording` | 停止录音 |
| (binary) | PCM 16kHz int16 音频数据 |

**服务端 → 客户端:**
| type | 说明 |
|------|------|
| `connected` | WebSocket 连接成功 |
| `models_ready` | 模型加载完成 |
| `models_unloaded` | 模型已释放 |
| `recording_started` | 录音开始，返回 session_id |
| `partial` | 流式转写结果（部分/最终） |
| `recording_stopped` | 录音停止，进入后处理 |
| `post_progress` | 后处理进度 |
| `final_result` | 最终处理结果 |
| `error` | 错误信息 |

---

## 实时流式架构

```
Browser Mic → AudioWorklet(PCM 16kHz) → WebSocket → FastAPI → ASR Engine → text
                                                                    ↕
                                                              fsmn-vad (并行)
                                                                    ↕
                                                              VAD 端点 → 自动分段

Recording stops → pyannote diarization → alignment → LLM pipeline → results
```

### Producer-Consumer 架构 (关键)
- **NEVER process ASR in the WebSocket receive loop** — 使用 asyncio.Queue + 后台任务
- Producer (receive loop): `ws.receive()` → `queue.put_nowait()` (纳秒级，不阻塞)
- Consumer (background task): `queue.get()` → drain & batch → `feed_chunk()` → send results

### ASR 引擎选择
- **FunASR Paraformer**: 中文流式 ASR，PyTorch，600ms 延迟，chunk_size=[1,10,5]
- **sherpa-onnx Paraformer**: 三语 (zh/粤/en)，ONNX Runtime，无 PyTorch 依赖
- 运行时切换: `get_streaming_asr_engine(engine_type)` 工厂函数
- **手动加载/释放**: 用户选择引擎后点击"加载"按钮，不自动预加载

### fsmn-vad 流式 VAD
- 与 ASR 并行运行，独立 cache
- `max_end_silence_time=800ms` 控制端点灵敏度
- VAD 检测到 speech_end → 确认段落（加标点）
- Fallback: VAD 未触发但静默超时 → 时间基准分段（3s）

### 模型加载/释放
- **手动控制**: 用户点击"加载" → `preload_models` WS 消息 → `models_ready`
- **释放**: 用户点击"释放" → `unload_models` WS 消息 → `models_unloaded`
- 录音按钮在 `modelsReady=true` 前禁用
- 后处理结束后自动卸载 ASR → 发送 `models_unloaded`

---

## 数据处理流程

### 音频文件处理 (FilePage)
```
上传音频 → 音频转换(16kHz WAV) → [音频增强] → 说话人分离 → 语音转写
→ 时间对齐 → [错别字校正] → 性别检测 → 智能命名 → [会议总结] → 输出
```

### 实时录音处理 (RealtimePage)
```
麦克风 → PCM 16kHz → WebSocket → ASR + VAD 并行 → 实时文字
→ 停止录音 → pyannote 分离 → 对齐 → [校正] → 性别 → 命名 → [总结]
```

---

## 配置说明

### 后端配置 (backend/.env)

```bash
# 路径配置（相对于 backend/ 目录）
MEETING_AI_DATA_DIR=../data
MEETING_AI_MODELS_DIR=../models
MEETING_AI_OUTPUT_DIR=../outputs

# ASR 配置
MEETING_AI_ASR__MODEL_NAME=medium     # tiny/base/small/medium/large-v3
MEETING_AI_ASR__DEVICE=cuda           # cpu/cuda/auto
MEETING_AI_ASR__COMPUTE_TYPE=float16  # RTX 5090 用 float16

# LLM 配置
MEETING_AI_LLM__ENABLED=true
MEETING_AI_LLM__MODEL_PATH=llm/Qwen2.5-7B-Instruct-Q4_K_M.gguf
MEETING_AI_LLM__N_CTX=6144
```

### 路径解析 (关键)
- `root_dir` 解析到 `backend/`（不是项目根！）
- `backend/.env` 设置 `MEETING_AI_MODELS_DIR=../models` 补偿
- `_resolve_relative_model_path()` helper 剥离 `models/` 前缀
- 始终使用此 helper 或引擎内部解析，不要直接拼 `models_dir`

---

## 环境安装

### 开发环境: Windows 原生

| 组件 | 要求 | 说明 |
|------|------|------|
| Python | 3.13 | RTX 5090 (sm_120) |
| PyTorch | nightly cu128 | 稳定版不支持 Blackwell |
| CUDA | 12.8+ | 驱动需支持 |
| Node.js | 18+ | 前端构建 |

### 后端安装

```powershell
mamba create -n meeting-ai python=3.13 -y
conda activate meeting-ai

# RTX 5090:
pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
# 旧显卡:
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

mamba install ffmpeg cmake -c conda-forge -y
cd backend
pip install -e ".[stream,enhance]"
```

### 前端安装

```bash
cd frontend
npm install
```

### 启动服务

```bash
# 终端 1: 后端
cd backend
uvicorn meeting_ai.api.main:app --reload --host 0.0.0.0 --port 8000

# 终端 2: 前端
cd frontend
npm run dev
```

打开浏览器访问 http://localhost:5173

---

## 智能命名核心逻辑 (naming.py)

### 命名优先级（从高到低）

1. **正则提取的真实名字** - 从对话中直接提取（如"小柔，我想问你"→"小柔"）
2. **LLM 识别的真实名字** - LLM 返回 kind="name" 且在对话中出现
3. **主持人判断** - 问句比例 >= 30% 的说话人
4. **LLM 推断的角色** - LLM 返回 kind="role"（如"组长"、"汇报人"）
5. **性别兜底** - "男性01"、"女性01"、"说话人01"

### 关键函数

- `extract_names_from_text(text)` - 正则提取候选名字
- `is_name_in_text(name, text)` - 验证名字在对话中出现
- `_validate_names_with_llm(names, context)` - LLM 验证是否是人名
- `_llm_name_speakers(segments, speakers)` - LLM 推断名字/角色
- `name_speakers(segments, gender_map)` - 主入口函数

---

## 开发阶段

| 阶段 | 功能 | 状态 |
|------|------|------|
| 0-5 | 核心功能 + CLI + Flet GUI | ✅ 完成 |
| 6 | 实时流式 ASR (双引擎 + fsmn-vad) | ✅ 代码完成，待集成测试 |
| 7 | 前后端分离 (FastAPI + React) | ✅ 完成 |
| 8 | 功能完善和优化 | 🔄 进行中 |
| 9 | Tauri 桌面应用打包 | 📅 待做 |

---

## 开发注意事项

### Pydantic v2 迭代陷阱 (关键)
- 迭代 Pydantic v2 BaseModel 产生 `(field_name, value)` 元组，不是字段值
- 始终用 `model.field_name` 或 `model.segments` 访问字段

### 流式 API 契约
- `detect_all_genders(wav_path, segments: list[Segment])` — 传 `diar_result.segments`，不是 `diar_result`
- `naming_service.name_speakers(segments, gender_map)` → 返回 `dict[str, SpeakerInfo]`
- `feed_chunk()` 返回 `list[tuple[StreamingSegment, bool]]` — bool=True 表示段落完成

### 静默检测必须基于时间
- **绝不用 feed_chunk() 调用次数做静默检测** — 批处理下 1 次调用可覆盖多秒
- 用 `session.last_text_time` 和 `chunk_end_time` 比较实际静默时长

### 总结格式
- `summarize_meeting()` 返回 `MeetingSummary` Pydantic 对象，不是字符串
- 必须用 `format_summary_markdown(summary, speakers, duration)` 转 Markdown

### AudioContext (关键)
- 在 `getUserMedia()` 之前创建 AudioContext — Chrome autoplay 策略
- `GainNode(gain=0)` 是正确模式 — 保持渲染器活跃
- 不要强制 `new AudioContext({ sampleRate: 16000 })` — 某些浏览器返回全零缓冲

### WebSocket 断连处理
- 主循环检查 `message.get("type") == "websocket.disconnect"` 防止 RuntimeError

### 前端注意
- Tab 切换用 CSS 隐藏 (`className="hidden"`) 保持组件状态，不要条件渲染
- 历史记录编辑需区分 `sourceType === 'history'` 调用正确 API 路径
- `regenerateSummary` 超时设置 600s（长会议 LLM 生成慢）

---

## 已知问题与修复

- Windows tempfile: 用 `delete=False` + 手动 `os.unlink`
- RTX 5090: 必须用 PyTorch nightly cu128, float16 compute type
- PowerShell git: 用 `$ErrorActionPreference = "Continue"`
- Python: 用完整路径 `C:\ProgramData\miniforge3\envs\meeting-ai\python.exe`
- Terminal 编码: 中文输出在 git bash 中乱码，写文件验证

---

## 代码风格

- Python 3.13 / TypeScript
- 类型注解
- Docstring (Google 风格)
- Ruff 格式化
- 行长度 100

---

*最后更新: 2026-02-06 (前后端分离完成，实时流式 ASR + fsmn-vad 代码完成，待集成测试)*
