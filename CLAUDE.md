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

1. **说话人辨识** - pyannote-audio 3.1 / DiariZen-large，识别"谁在什么时候说话"
2. **多引擎语音转写** - faster-whisper / FunASR (SenseVoice, Paraformer) / FireRedASR，可选切换
3. **VAD 预分段** - fsmn-vad 切短音频再逐段转写，所有引擎获得精确时间戳
4. **字级对齐** - 逐字/词时间戳 + diarization → 精确说话人切分（见下文）
5. **实时流式转写** - FunASR Paraformer / sherpa-onnx 双引擎，边录边转
6. **智能命名** - LLM + 正则推断说话人身份（"张教授"、"小柔"），去重保护
7. **性别检测** - f0 基频 / ECAPA-TDNN / Wav2Vec2，多引擎可选（默认始终运行作为兜底命名）
8. **错别字校正** - LLM 修复转写错误
9. **会议总结** - LLM 自动生成会议摘要
10. **音频增强** - Demucs 人声分离 + DeepFilterNet3 降噪 + Resemble Enhance（可选）

---

## 技术栈

| 模块 | 技术 | 说明 |
|------|------|------|
| 后端框架 | FastAPI | REST API + WebSocket 实时通信 |
| 前端框架 | React 18 + TypeScript | Vite + Tailwind CSS + Zustand |
| 说话人辨识 | pyannote-audio 3.1 / DiariZen-large | 多引擎可选（A层）|
| 实时流式 ASR | FunASR 1.3.1 + sherpa-onnx 1.12.23 | 双引擎可选（A层）|
| **VAD** | **Silero VAD** ← fsmn-vad | **✅ 已测试**：59 语音段检测成功，更可靠、多语言（B层）|
| 文件 ASR | faster-whisper / FunASR / FireRedASR | 3 引擎可选（A层）|
| **强制对齐** | **Whisper** ← Paraformer | **✅ 已测试**：字级时间戳生成成功，word timestamps + LCS（B层）|
| 标点恢复 | ct-punc | FunASR 生态标准（B层）✅ |
| LLM | llama-cpp-python + Qwen2.5-7B | 用户可配置（A层）|
| 性别检测 | f0 / ECAPA-TDNN / Wav2Vec2 | 3 引擎可选（A层）|
| 音频增强 | Demucs + DeepFilterNet3 + Resemble | 业界顶级（B层）✅ |
| 音频处理 | ffmpeg | 格式转换 |
| 配置 | pydantic-settings | 支持 .env 文件 |

---

## 架构哲学（2026-02-15 重要决策）

### **双层模块设计原则**

本项目采用"用户可配置层"和"内部实现层"分离的架构，确保在保持接口稳定的前提下使用业界最佳实践。

#### **A层：用户可配置模块**（保持多引擎灵活性）
- **ASR 引擎** - Whisper / FunASR / FireRedASR（用户按任务选择）
- **LLM** - Qwen / 其他 GGUF 模型（用户可配置）
- **说话人分离** - pyannote-3.1 / DiariZen-large（用户选择）
- **性别检测** - f0 / ECAPA-TDNN / Wav2Vec2 / Audeering（用户选择）

#### **B层：内部实现模块**（使用业界标准，对用户透明）
- **VAD** - **Silero VAD**（whisperX 标准）→ 替换 fsmn-vad，更可靠
- **强制对齐** - **wav2vec2-based**（whisperX 核心）→ 替换 Paraformer，10-50ms 精度
- **音频增强** - Demucs + DeepFilterNet3 + Resemble Enhance（已是业界顶级 ✅）
- **标点恢复** - ct-punc（FunASR 生态标准 ✅）

### **核心原则**

> **"保持接口稳定，升级内部实现到最佳方案"**

- 用户面向的 API 遵循 OpenAPI 3.0 规范（对标 OpenAI Whisper / AssemblyAI）
- 内部实现使用经过同行评审的学术界/工业界标准
- 升级内部组件时黑盒输入输出保持不变

### **竞争优势**

与商业 API 对比，我们在以下方面超越或持平：

| 特性 | 商业API（OpenAI/AssemblyAI/Google） | 我们 |
|------|-------------------------------------|------|
| 时间戳粒度 | 词级（word-level） | **字级（char-level）** ✅ |
| 性别检测 | ❌ 不提供 | ✅ 三引擎可选 |
| 智能命名 | ❌ 只有 SPEAKER_00 | ✅ LLM自动推断（全球独创）|
| 多引擎支持 | ❌ 锁定单一引擎 | ✅ 三引擎可选 |
| 离线运行 | ❌ 必须联网 | ✅ 完全离线 |
| 接口规范 | OpenAPI 3.0 | ✅ 完全一致 |

### **接口标准验证**

我们的核心接口完全符合业界标准：

```python
# ASR 接口 - 对标 OpenAI Whisper API
class ASREngine:
    def transcribe(audio_path, language) -> TranscriptResult
        # segments + char_timestamps (我们甚至更细粒度)

# 说话人分离接口 - 对标 pyannote.audio
def diarize(audio_path) -> DiarizationResult
    # speakers + segments (完全一致)

# 对齐接口 - 对标 whisperX
def align_transcript_with_speakers(
    transcript: TranscriptResult,
    diarization: DiarizationResult
) -> list[Segment]
    # 与 whisperX 的 assign_word_speakers() 思路一致
```

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
│   │   │   ├── diarization.py       # 说话人辨识 (pyannote / DiariZen)
│   │   │   ├── asr.py               # 多引擎 ASR + VAD 预分段 + 强制对齐
│   │   │   ├── alignment.py         # 说话人-文本对齐 (字级/中点/句级)
│   │   │   ├── gender.py            # 性别检测 (f0 / ECAPA-TDNN / Wav2Vec2 / Audeering)
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
│   ├── pyannote/                    # pyannote 共享子模型 (wespeaker, segmentation)
│   ├── diarization/                 # 说话人辨识模型 (pyannote-3.1/, reverb-diarization-v2/)
│   ├── whisper/                     # Whisper ASR 模型 (faster-whisper-*)
│   ├── asr/                         # 非 Whisper ASR 模型 (sensevoice-small/, paraformer-large/, fireredasr-aed/)
│   ├── gender/                      # 性别检测模型 (ecapa-gender/, wav2vec2-gender/)
│   ├── llm/                         # LLM (Qwen2.5-7B GGUF)
│   ├── deepfilter/                  # DeepFilterNet3 ONNX 降噪模型
│   └── streaming/                   # 流式 ASR + 共享辅助模型
│       ├── funasr/
│       │   ├── paraformer-zh-streaming/  # 流式中文 ASR
│       │   ├── ct-punc/                  # 标点恢复 (FireRedASR 后处理 + 流式)
│       │   └── fsmn-vad/                 # VAD (流式 + 文件转写预分段)
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
上传音频 → 音频转换(16kHz WAV) → [音频增强]
    ├→ 说话人辨识 (pyannote/DiariZen) → 说话人时间线 (谁在什么时候说话)
    └→ VAD 预分段 (fsmn-vad) → 逐段 ASR 转写 → [ct-punc 标点] → 字级时间戳
         ↓
    字级对齐 (逐字查说话人) → [错别字校正] → 性别检测 → 智能命名 → [会议总结] → 输出
```

### 实时录音处理 (RealtimePage)
```
麦克风 → PCM 16kHz → WebSocket → ASR + VAD 并行 → 实时文字
→ 停止录音 → pyannote 分离 → 对齐 → [校正] → 性别 → 命名 → [总结]
```

---

## 多引擎 ASR 架构 (asr.py)

### 引擎

| 引擎 | 模型目录 | 特点 |
|------|---------|------|
| FasterWhisperEngine | `models/whisper/faster-whisper-*` | 99 语言，词级时间戳 |
| FunASRFileEngine | `models/asr/sensevoice-*`, `paraformer-*` | 中文最优，字级时间戳 |
| FireRedASREngine | `models/asr/fireredasr-*` | 中文 SOTA (CER 0.6%)，60s 限制 |

### 处理管线

```
1. Silero VAD 预分段 → 2-15s 语音段列表（✅ 2026-02-15 更新：替换 fsmn-vad，更可靠）
2. 逐段转写 (任意 ASR 引擎)
3. 提取字级时间戳:
   - Whisper: word_timestamps=True (原生词级)
   - FunASR: timestamp 字段 (原生字级)
   - FireRedASR: Whisper 强制对齐 + LCS 映射 (间接字级) ← 更新
4. FireRedASR: ct-punc 标点恢复
```

### 强制对齐 (FireRedASR 专用)

**✅ 2026-02-15 更新**：从 Paraformer 迁移到 Whisper（whisperX 标准，10-50ms 精度）

FireRedASR 不返回时间戳。用 Whisper 做强制对齐：
1. 用 Whisper 对同一段音频转写 → 得到参考文本 + 词级时间戳
2. 词级时间戳转字级（均匀分配每个词的时长）
3. 用 LCS (最长公共子序列) DP 对齐 FireRedASR 文本 ↔ Whisper 文本
4. 匹配字复用时间戳，不匹配字线性插值
5. 对齐完成后释放 Whisper 回收显存

### 对齐策略 (alignment.py)

| 优先级 | 策略 | 条件 | 精度 |
|--------|------|------|------|
| 1 | **字级对齐** | 有 char_timestamps | 每个字独立查 diarization |
| 2 | 中点匹配 | 片段 < 5s | 用中点时间查说话人 |
| 3 | 句级分割 | 片段 >= 5s | 按标点拆句再分配 |

### 工厂函数

```python
engine = get_asr_engine("sensevoice-small")  # 自动检测引擎类型
result = engine.transcribe(audio_path)        # 返回 TranscriptResult (含 char_timestamps)
```

### 辅助模型 (懒加载单例)

- `_get_vad_model()` — fsmn-vad，文件转写 VAD 预分段
- `_get_punc_model()` — ct-punc，FireRedASR 标点恢复
- `_get_fa_model()` — Paraformer-Large，FireRedASR 强制对齐

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
5. **性别兜底** - "男性01"、"女性01"、"说话人01"（性别检测默认始终运行，不依赖智能命名开关）

### 关键函数

- `extract_names_from_text(text)` - 正则提取候选名字
- `is_name_in_text(name, text)` - 验证名字在对话中出现
- `_validate_names_with_llm(names, context)` - LLM 验证是否是人名
- `_llm_name_speakers(segments, speakers)` - LLM 推断名字/角色
- `name_speakers(segments, gender_map)` - 主入口函数

---

## 说话人辨识引擎 (diarization.py)

### 支持的引擎 (2026-02-15 更新)

| 引擎 | 模型目录 | 性能 (DER) | VRAM | 特点 |
|------|---------|-----------|------|------|
| **pyannote-3.1** | `models/diarization/pyannote-3.1/` | **11%** | ≈2G | **推荐**，工业标准，通用性好 |
| DiariZen-large | `models/diarization/reverb-diarization-v2/` | **13% (AMI)** | ≈2.5G | Rev.com SOTA，比 pyannote 3.0 提升 22% |
| pyannote-community-1 | `models/diarization/pyannote-community-1/` | 15% | ≈1.5G | 社区版本，轻量化 |

### 架构决策 (2026-02-15)

**移除 CAM++ (3D-Speaker)：**
- 原因：测试中发现 CAM++ 在说话人切换检测上准确率不足（未能检测到 12s 处的切换点）
- pyannote-3.1 在同一测试中表现优异
- **结论**：专注于 pyannote 系列 + DiariZen（均基于 pyannote Pipeline API，whisperX 标准架构）

### DiariZen-large (2026-02-15 新增)

**模型信息：**
- **来源**：Rev.com 开源（HuggingFace: `Revai/reverb-diarization-v2`）
- **架构**：基于 WavLM-Large + pyannote Pipeline
- **性能**：DER 13.3% (AMI)，比 pyannote 3.0 相对改进 22.25%
- **许可**：查看 LICENSE 文件（需 HF_TOKEN）

**使用方式：**
```python
# 与 pyannote 完全兼容
service = DiarizationService("reverb-diarization-v2")
result = service.diarize("audio.wav")
```

**下载：**
```bash
# 设置 HF_TOKEN
export HF_TOKEN=your_token_here  # Linux/Mac
# $env:HF_TOKEN="your_token_here"  # Windows PowerShell

# 运行下载脚本
python backend/scripts/download_all_models.py
```

### 统一接口

所有引擎均通过 pyannote Pipeline API 实现：
- 输入：`diarize(audio_path, min_speakers, max_speakers)`
- 输出：`DiarizationResult{speakers: dict[str, SpeakerInfo], segments: list[Segment]}`
- 自动加载：检测 `config.yaml` (pyannote 格式)

---

## 开发阶段

| 阶段 | 功能 | 状态 |
|------|------|------|
| 0-5 | 核心功能 + CLI + Flet GUI | ✅ 完成 |
| 6 | 实时流式 ASR (双引擎 + fsmn-vad) | ✅ 代码完成，待集成测试 |
| 7 | 前后端分离 (FastAPI + React) | ✅ 完成 |
| 8 | 多引擎 ASR + VAD 预分段 + 字级对齐 | ✅ 完成 |
| 9 | 多引擎说话人辨识 + 性别检测 | ✅ 完成 |
| 10 | 音频增强专业管线 | ✅ 完成 |
| 10.5 | 对话编辑增强 (删除/分割/合并/说话人重分配/新建) | ✅ 完成 |
| 10.6 | Qwen3-ASR 集成 (批量推理 + 上下文链接 + 非阻塞实时转写) | ✅ 完成 |
| 10.7 | 实时在线说话人辨识 (diart，实时颜色标注) | ✅ 完成 |
| 10.8 | 暗黑模式 (圆心扩散弹性动效，View Transitions API) | ✅ 完成 |
| 11 | Tauri 桌面应用打包 | 📅 待做 |

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
- 编辑对话框: 点击说话人名字 → 全局重命名; 点击编辑图标 → 单条片段说话人重分配 + 文本编辑
- 说话人重分配: `updateSegmentSpeaker()` 只改单条片段, `updateSpeakerName()` 改所有同说话人片段
- 新建说话人: 编辑对话框下拉菜单底部 "+ 新建说话人" → 自动生成 SPEAKER_XX ID

---

## 已知问题与修复

### **环境相关**
- Windows tempfile: 用 `delete=False` + 手动 `os.unlink`
- RTX 5090: 必须用 PyTorch nightly cu128, float16 compute type
- PowerShell git: 用 `$ErrorActionPreference = "Continue"`
- Python: 用完整路径 `C:\ProgramData\miniforge3\envs\meeting-ai\python.exe`
- Terminal 编码: 中文输出在 git bash 中乱码，写文件验证
- librosa.resample: Python 3.13 下会挂死，用 `scipy.signal.resample_poly` 替代

### **模型相关**
- ecapa-gender: 不是 transformers 模型，用 PyTorchModelHubMixin + 内联架构
- FunASR merge_vad=True: 会合并为单段输出，VAD 预分段绕过此问题
- FireRedASR 无标点: ct-punc 后处理恢复
- Naming dedup: `used_names` set 防止多个说话人被分配相同名字

### **Windows 文件锁相关**
- **历史记录删除失败**: 浏览器播放音频时文件被占用 → 添加重试机制（3次，0.5s间隔）+ 友好错误提示（409 Conflict）
- **解决方案**: `shutil.rmtree()` 捕获 `PermissionError` → 自动重试 → 失败后提示"请先停止音频播放"

### **已修复的问题（2026-02-25）**
- ✅ **CUDA 库缺失**: PyTorch CUDA 版本修复，faster-whisper 正常运行
- ✅ **DeepFilterNet ONNX CUDA**: FusedConv+Sigmoid 不支持 → 强制 CPUExecutionProvider
- ✅ **Resemble Enhance SSL**: 本地已下载时跳过 git pull
- ✅ **torchcodec DLL**: ABI 不兼容 → 已卸载
- ✅ **音频切换播放器不重置**: src 变更时重置 isPlaying 状态
- ✅ **LLM 命名偏差**: 移除 prompt 中"组长"等有偏见的示例
- ✅ **audeering-gender 移除**: 准确率不佳，保留 f0 / ECAPA / Wav2Vec2 三引擎
- ✅ **性别兜底默认运行**: 不再依赖智能命名开关

### **已修复的问题（2026-03-06）**
- ✅ **Qwen3-ASR 串行推理慢**: 改为批量推理，动态 batch_size 基于空闲显存，3-5x 加速
- ✅ **Qwen3-ASR 标点跨段断裂**: 上下文链接（上一批末 80 字 → context），改善标点连贯
- ✅ **torchcodec/torchaudio 挂死**: Qwen3-ASR 不再传文件路径，改用 soundfile+numpy 数组
- ✅ **实时段级转写阻塞 VAD 循环**: `await _segment_asr_transcribe()` → `asyncio.create_task()` + `asyncio.Semaphore(1)` + `asyncio.Lock()` 保护 WebSocket send
- ✅ **暗黑模式直接切换**: 改为 View Transitions API 圆心扩散弹性动效（cubic-bezier 0.34,1.56,0.64,1）

---

## 代码风格

- Python 3.13 / TypeScript
- 类型注解
- Docstring (Google 风格)
- Ruff 格式化
- 行长度 100

---

### **Qwen3-ASR 关键注意事项**
- `model.transcribe()` 接受 `List[(numpy_array, sr)]` 或 `List[str]`，返回 `List[ASRTranscription]`（始终是列表！）
- 传文件路径会触发 torchcodec（ABI 不兼容挂死）→ 始终用 `sf.read()` 返回的 `(numpy_array, sr)` 元组
- `return_time_stamps=True` 仅在 `_has_aligner=True`（已加载 qwen3-forced-aligner-0.6B）时传入
- `_compute_batch_size(chunks)`: `torch.cuda.mem_get_info()` 获取空闲显存，减 1.5GB 安全余量，除以每段估算显存（0.15+时长×0.05 GB），上限 32
- 上下文链接：`context_tail = batch_texts[-1][-80:]`，在批次间传递

*最后更新: 2026-03-06 (✅ Qwen3-ASR 批量推理+上下文链接 + 实时非阻塞转写 + 暗黑模式动效 + 实时说话人辨识)*
