# 实时录音：diart 级说话人追踪（帧级检测 + 在线聚类）

## Context

当前实时录音的分段完全依赖 VAD（静音检测）。如果说话人 A 和 B 之间没有明显停顿，会被合并成同一段。用户希望加入专业级实时声纹检测，在说话人切换时自动切段。

**核心需求**：效果对标 [diart](https://github.com/juanmc2005/diart)（业界最专业的实时 diarization 框架），允许 0.5-1.5s 延迟，输出丝滑无跳变。

### 与旧方案的关键区别

| | 旧方案（已废弃） | 新方案（diart 级） |
|---|---|---|
| 说话人检测 | VAD speech_end → 提 embedding | **segmentation-3.0 帧级检测**（~17ms 分辨率） |
| 检测能力 | 只能检测有静音间隔的切换 | **能检测无间隔的说话人切换** |
| 检测精度 | 段级（整段一个 speaker） | **帧级（每帧知道谁在说话）** |
| 核心模型 | 仅 WeSpeaker | **segmentation-3.0 + WeSpeaker** |
| 架构参考 | 简单的 embedding 比对 | **diart 滑动窗口 + 增量聚类** |

---

## 已有模型（无需额外下载）

| 模型 | 路径 | 大小 | 用途 |
|------|------|------|------|
| segmentation-3.0 | `models/pyannote/segmentation-3.0/` | 5.7MB | 帧级说话人活动检测（powerset 7类输出） |
| WeSpeaker ResNet34-LM | `models/pyannote/wespeaker-voxceleb-resnet34-LM/` | 26MB | 说话人 embedding（256维） |

### segmentation-3.0 技术细节

- 架构：PyanNet（SincNet + 4层 BiLSTM + 2层 Linear）
- 输入：10s 音频 @ 16kHz（我们用 5s 窗口也可以）
- 输出：`(num_frames, 7)` powerset 矩阵
  - 7 类 = {无语音, S1, S2, S3, S1+S2, S1+S3, S2+S3}
  - 帧分辨率：~17ms/帧（5s 输入 → 293 帧）
- 已验证：`model(waveform)` 输入 `(1, 1, 80000)` → 输出 `(1, 293, 7)`

### WeSpeaker 技术细节

- 输入：`(batch, 1, num_samples)` — 注意需要 channel 维度
- 输出：`(batch, 256)` embedding 向量
- 已在 `gender.py` 中有加载参考代码

---

## 核心架构：SpeakerTracker

### 数据流（diart 风格）

```
PCM 音频流（每 200-300ms 一个 chunk）
       │
       ▼
┌─────────────────────────┐
│  滚动音频缓冲区（5s）    │ ← 新 chunk 推入右端，左端滑出
└─────────┬───────────────┘
          │ 每 500ms 触发一次处理
          ▼
┌─────────────────────────┐
│  segmentation-3.0       │ → (293, 7) powerset → (293, 3) 多标签
│  帧级说话人活动检测       │   每帧知道 3 个局部说话人各自的活动概率
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  提取 embedding         │ 对每个活跃说话人的语音区域（≥0.5s）
│  WeSpeaker(256维)       │ 提取声纹向量
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  在线余弦聚类            │ 局部说话人 → 全局说话人
│  阈值=0.70, EMA=0.10   │ 质心滑动平均更新
└─────────┬───────────────┘
          │
          ▼
   SpeakerTurn 列表（时间 + 全局 speaker_id）
          │
          ▼
   与 ASR 文本合并 → 前端显示
```

### 关键参数

| 参数 | 值 | 来源 |
|------|-----|------|
| 滑动窗口 | 5.0s | diart 默认（seg 模型原生 10s，5s 足够） |
| 步进 | 0.5s（8000 样本） | diart 默认，500ms 延迟可接受 |
| 帧分辨率 | ~17ms/帧 | segmentation-3.0 实测 |
| Embedding 维度 | 256 | WeSpeaker 实测 |
| 活动阈值 | 0.5 | softmax 后二值化标准阈值 |
| 最短语音 | 0.5s | embedding 最低可靠长度 |
| 聚类阈值 | 0.70 余弦相似度 | diart 默认 / pyannote config |
| 质心 EMA | 0.10 | 旧质心权重 0.90，稳定更新 |
| 最大说话人 | 8 | 会议场景合理上限 |

### GPU 占用

| 模型 | VRAM (float16) |
|------|---------------|
| segmentation-3.0 | ~12 MB |
| WeSpeaker ResNet34-LM | ~53 MB |
| **合计** | **~65 MB** |

远在 0.5GB 预算内。推理耗时：seg ~5ms + emb ~10ms ≈ **15ms/步**（RTX 5090）。

---

## 实现步骤

### 1. 新建 `backend/src/meeting_ai/services/speaker_tracker.py`

```python
class SpeakerTracker:
    """
    diart 级实时说话人追踪：
    segmentation-3.0（帧级检测）+ WeSpeaker（embedding）+ 在线余弦聚类
    """

    WINDOW_DURATION = 5.0      # 滑动窗口（秒）
    STEP_DURATION = 0.5        # 步进（秒）
    WINDOW_SAMPLES = 80000     # 5s * 16000
    STEP_SAMPLES = 8000        # 0.5s * 16000
    ACTIVITY_THRESHOLD = 0.5   # 说话人活动阈值
    MIN_SPEECH_SAMPLES = 8000  # 最短 0.5s 才提 embedding
    COSINE_THRESHOLD = 0.70    # 余弦相似度聚类阈值
    CENTROID_EMA = 0.10        # 质心 EMA 更新系数
    MAX_SPEAKERS = 8

    def load(self) -> None:
        """加载 segmentation-3.0 + WeSpeaker 到 GPU"""

    def unload(self) -> None:
        """释放 GPU 显存"""

    def create_state(self) -> TrackerState:
        """创建录音会话状态（滚动缓冲区 + 聚类状态）"""

    def process_chunk(self, state: TrackerState, pcm_int16: bytes) -> list[SpeakerTurn]:
        """
        核心处理：
        1. 音频追加到滚动缓冲区
        2. 每 500ms 触发：
           a. segmentation → (num_frames, 7) → powerset 解码 → (num_frames, 3) 多标签
           b. 对每个活跃局部说话人提取 WeSpeaker embedding
           c. 在线聚类：局部 → 全局说话人映射
           d. 只输出最新 STEP 区域（窗口右端 0.5s）的 speaker turns
        """


@dataclass
class TrackerState:
    """可变状态，每个录音会话独立"""
    audio_buffer: np.ndarray          # float32, shape=(WINDOW_SAMPLES,)
    buffer_start_time: float = 0.0    # 缓冲区左端的绝对时间
    total_samples_received: int = 0
    last_processed_step: int = -1     # 步进计数器

    # 在线聚类
    centroids: list[np.ndarray]       # 每个全局说话人的质心 (256,)
    centroid_counts: list[int]        # 观测次数
    next_speaker_id: int = 0

    # 输出
    current_speaker: str = "SPEAKER_00"
    recent_turns: list[SpeakerTurn]   # 最近的说话人轮次（用于 ASR 段匹配）


@dataclass
class SpeakerTurn:
    """一个说话人活动区间"""
    start: float       # 绝对时间（秒）
    end: float
    speaker: str       # "SPEAKER_00" 等
    confidence: float
```

#### 核心算法细节

**Powerset 解码**：使用 `pyannote.audio.utils.powerset.Powerset` 类，将 7 类 softmax 输出转换为 3 说话人的多标签（0/1）矩阵。

**活跃区域检测**：扫描每个局部说话人的活动向量，找连续 ≥3 帧（~51ms）的活跃段。

**在线聚类**：
- 计算当前 embedding 与所有质心的余弦相似度
- 相似度 ≥ 0.70 → 匹配已知说话人，用 EMA 更新质心
- 相似度 < 0.70 → 新说话人（如未达上限）
- 质心归一化后存储

**步进区域输出**：只对窗口最右端 0.5s（最新 step）生成 SpeakerTurn，避免重复输出历史帧。

---

### 2. 修改 `backend/src/meeting_ai/services/streaming_asr.py`

#### 2a. `StreamingSession` 新增字段

```python
@dataclass
class StreamingSession:
    # ... 现有字段 ...
    tracker_state: Any | None = None          # SpeakerTracker 的 TrackerState
    current_segment_speaker: str = "SPEAKER_00"  # 当前段的说话人
```

#### 2b. `feed_chunk()` 集成说话人切段

在两个引擎的 `feed_chunk()` 中，新增说话人变化触发的切段逻辑：

```python
# 现有切段触发条件：
# 1. VAD speech_end → 终结段
# 2. 3s 静默 fallback → 终结段
# 3. is_final → 终结段

# 新增条件（插入到现有判断之后）：
# 4. speaker_tracker 检测到说话人变化 → 终结当前段，新段用新 speaker

if session.tracker_state and session.current_text:
    split_time = _check_speaker_change_split(session, chunk_end_time)
    if split_time is not None:
        # 终结当前段（到 split_time）
        emit_final_segment(session, end_time=split_time)
        # 新段用新 speaker
        session.current_segment_speaker = new_speaker
```

**_check_speaker_change_split 逻辑**：
- 检查 `tracker_state.recent_turns` 中最新 step 的说话人
- 如果与 `session.current_segment_speaker` 不同
- 且当前段已累积 ≥1.0s 文本（避免微切段）
- → 返回切分时间点

#### 2c. `StreamingSegment` 输出包含 speaker

`_emit_final_segment()` 中，`temp_speaker` 使用 `session.current_segment_speaker`（由 tracker 赋值）。

---

### 3. 修改 `backend/src/meeting_ai/api/routes/realtime.py`

#### 3a. SpeakerTracker 加载/卸载

```python
# preload_models 消息处理中：
speaker_tracker = get_speaker_tracker()  # 单例
await loop.run_in_executor(None, speaker_tracker.load)

# unload_models / 录音结束后：
speaker_tracker.unload()
```

#### 3b. 消费者任务集成

三个消费者函数（`_audio_processor` / `_segment_processor` / `_hybrid_processor`）增加 `speaker_tracker` 参数：

```python
async def _audio_processor(ws, queue, engine, session, segments,
                           speaker_tracker=None):
    while True:
        chunk = await queue.get()
        if chunk is None:
            break

        # 1. ASR 处理（现有）
        results = await loop.run_in_executor(
            None, engine.feed_chunk, session, batched, False
        )

        # 2. Speaker 追踪（新增，同步调用）
        if speaker_tracker and session.tracker_state:
            turns = await loop.run_in_executor(
                None, speaker_tracker.process_chunk,
                session.tracker_state, batched
            )
            # 更新当前说话人
            if turns:
                latest = turns[-1]
                session.current_segment_speaker = latest.speaker

        # 3. 发送结果（temp_speaker 已由 feed_chunk 内部赋值）
        for seg, is_final in results:
            await send_partial(ws, seg, is_final)
```

#### 3c. WebSocket 消息协议

`partial` 消息已有 `temp_speaker` 字段，无需新增协议。但增加可选的 `speaker_update` 消息用于前端更细粒度显示：

```json
{
  "type": "speaker_update",
  "turns": [
    {"start": 5.2, "end": 5.7, "speaker": "SPEAKER_01", "confidence": 0.85}
  ]
}
```

#### 3d. start_recording 时创建 TrackerState

```python
if speaker_tracker and speaker_tracker.is_loaded():
    session.tracker_state = speaker_tracker.create_state()
```

---

### 4. 前端修改

#### 4a. `frontend/src/hooks/useRealtimeWebSocket.ts`

`PartialSegment` 已有 `tempSpeaker` 或类似字段。确保从 WS 消息中提取 `temp_speaker`。

新增 `speaker_update` 消息处理（可选，用于实时颜色更新）。

#### 4b. `frontend/src/pages/RealtimePage.tsx`

实时转写列表中，根据 `temp_speaker` 显示颜色标签：

```tsx
const SPEAKER_COLORS = [
  'text-blue-600',    // SPEAKER_00
  'text-green-600',   // SPEAKER_01
  'text-orange-500',  // SPEAKER_02
  'text-red-500',     // SPEAKER_03
  'text-purple-600',  // SPEAKER_04
  'text-pink-500',    // SPEAKER_05
]

// 每个段前显示说话人标签
{seg.tempSpeaker && (
  <span className={`text-xs font-medium mr-1 ${getSpeakerColor(seg.tempSpeaker)}`}>
    [{seg.tempSpeaker}]
  </span>
)}
```

#### 4c. `frontend/src/types/index.ts`

确保 `RealtimeSegment` / `PartialSegment` 包含 `tempSpeaker?: string`。

---

## 关键文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `backend/src/meeting_ai/services/speaker_tracker.py` | **新建** | 核心：segmentation + embedding + 在线聚类 |
| `backend/src/meeting_ai/services/streaming_asr.py` | 修改 | Session 增加 tracker_state；feed_chunk 增加说话人切段 |
| `backend/src/meeting_ai/api/routes/realtime.py` | 修改 | 加载/卸载 tracker；消费者传递 tracker；创建 state |
| `frontend/src/types/index.ts` | 修改 | PartialSegment 加 tempSpeaker |
| `frontend/src/hooks/useRealtimeWebSocket.ts` | 修改 | 解析 temp_speaker + speaker_update |
| `frontend/src/pages/RealtimePage.tsx` | 修改 | 说话人颜色标签显示 |

## 复用的现有代码

| 代码 | 位置 | 复用方式 |
|------|------|----------|
| WeSpeaker 加载模式 | `gender.py:165-196` `_get_embedding_model()` | 相同 `Model.from_pretrained()` + `.to(device)` |
| `StreamingSegment.embedding` 字段 | `models.py:80` | 已预留，直接使用 |
| `StreamingSegment.temp_speaker` 字段 | `models.py:68` | 已有，tracker 赋值 |
| Powerset 解码 | `pyannote.audio.utils.powerset.Powerset` | 直接 import 使用 |
| 聚类阈值 0.70 | `models/diarization/pyannote-3.1/config.yaml` | 参考值 |

## 三种模式的数据流

```
=== 段级模式 ===
PCM → VAD → speech_end → [audio segment]
                              ├→ tracker.process_chunk() → speaker_id
                              └→ File ASR transcribe() → text
                          → send_partial(text, temp_speaker=speaker_id)

=== 字级模式 ===
PCM ──┬→ ASR feed_chunk() → streaming text (实时, 无延迟)
      └→ tracker.process_chunk() → 每 500ms 输出 speaker turns
              → if speaker changed & segment ≥ 1s: 强制切段

=== 混合-段级模式 ===
PCM ──┬→ Streaming ASR → preview text (实时)
      └→ VAD → speech_end → [audio segment]
                    ├→ tracker.process_chunk() → speaker_id
                    └→ File ASR upgrade → final text
                → send_partial(text, temp_speaker=speaker_id)
```

## 验证

1. **两人对话** → 段级模式 → 不同说话人自动分配 SPEAKER_00 / SPEAKER_01
2. **无间隔切换** → 说话人 A→B 无停顿 → segmentation 检测到帧级切换 → 自动切段
3. **字级模式** → 说话人切换时强制切段，新段显示不同颜色标签
4. **混合-段级** → 同段级
5. **单人独白** → 始终 SPEAKER_00，不误切
6. **停止录音** → 后处理 pyannote 完整 diarization 覆盖实时标签
7. **GPU 显存增加 < 100MB**
8. **延迟** ≤ 1s（500ms 步进 + ~15ms 推理）

## 技术决策

- **segmentation-3.0 而非纯 VAD**：帧级检测能捕获无间隔的说话人切换，这是专业级方案的核心
- **5s 窗口 + 500ms 步进**：diart 默认配置，平衡延迟与准确率
- **Powerset 解码**：直接用 pyannote 内置的 `Powerset` 类，无需手写
- **EMA α=0.10**（旧权重 0.90）：比之前 α=0.30 更保守，质心更稳定
- **实时标签是"预览"**：录音结束后完整 pyannote pipeline 做最终分配
- **不引入 diart 依赖**：diart 引入 RxPY + 额外框架。我们只复用核心算法思路
