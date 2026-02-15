# FunASR 字级时间戳丢失 Bug 修复报告

**日期**: 2026-02-15
**严重性**: 🔴 Critical - 导致说话人对齐严重错误

## 问题现象

同一音频文件多次处理，产生截然不同的分段结果：
- 最差：**3 个片段** - 多个说话人的对话被错误合并
- 最好：**30 个片段** - 说话人切分精确
- 其他：4, 5, 6 个片段 - 质量不稳定

### 典型错误案例

**测试**: `汉语水平考试听力_trimmed_20260215_012531`
- Segment 0 (37s, SPEAKER_00):
  ```
  今天我们大家将跟张教授一起讨论...请问张教授，青少年的性格有什么特点？
  青少年的性格特点和成人有些不同，特别是外向型...
  ```
  ❌ **包含主持人问题 + 张教授回答，错误合并！**

## 根本原因

### 问题链条

1. **VAD 预分段失败**（原因待查）:
   ```
   01:26:08 INFO Step 1/4: VAD 语音活动检测... (3D-Speaker diarization)
   01:26:08 INFO VAD 检测到 54 个语音片段
   01:26:17 INFO [SenseVoice] VAD 不可用，使用整文件转写 (ASR)
   ```
   - 3D-Speaker 成功使用 VAD，但随后 ASR 使用 VAD 时失败
   - 可能的原因：模型资源冲突、单例状态问题

2. **整文件转写不生成字级时间戳**:
   ```python
   # 旧代码 - backend/src/meeting_ai/services/asr.py
   def _transcribe_whole_file(...) -> list[Segment]:
       res = self._model.generate(**generate_kwargs)
       return self._parse_funasr_output(res, duration)  # ❌ 只返回 segments
   ```
   - `_parse_funasr_output()` 只提取文本和片段时间戳
   - **没有调用 `_extract_text_and_timestamps()` 提取字级时间戳！**

3. **字级对齐无法执行**:
   ```
   01:26:18 INFO 对齐: 1 个转写片段, 17 个说话人片段, 字级时间戳=无
   ```
   - `char_timestamps=None` → `has_char_ts=False`
   - 回退到句级分割（`_split_by_speakers`）
   - 对于长片段（37s），句级分割不准确，多个说话人被合并

### 对比：Whisper 为何正常

**测试**: `汉语水平考试听力_trimmed_20260215_012850`
```
01:29:36 INFO [faster-whisper] 开始转写
01:30:01 INFO 片段数=39, 词级时间戳=39/39
01:30:01 INFO 对齐: 39 个转写片段, 17 个说话人片段, 字级时间戳=有
```
- Whisper 内置 VAD 过滤器，无需外部 VAD
- `word_timestamps=True` 强制开启词级时间戳
- 字级对齐正常执行 → 30 个精确片段

## 修复方案

### 1. 修改 `_transcribe_whole_file()` 返回类型

```python
def _transcribe_whole_file(
    self,
    audio_path: Path,
    duration: float,
    language: str | None = None,
) -> tuple[list[Segment], list[list[CharTimestamp]]]:  # ✅ 新增 char_timestamps
    """
    整文件转写（作为 VAD 回退）

    返回: (segments, char_timestamps)
    """
    generate_kwargs = {...}
    res = self._model.generate(**generate_kwargs)
    segments, char_timestamps = self._parse_funasr_output_with_timestamps(res, duration)  # ✅ 调用新方法
    return segments, char_timestamps
```

### 2. 创建 `_parse_funasr_output_with_timestamps()`

```python
def _parse_funasr_output_with_timestamps(
    self, res: list, duration: float,
) -> tuple[list[Segment], list[list[CharTimestamp]]]:
    """
    解析 FunASR model.generate() 的输出，同时提取字级时间戳。

    返回: (segments, char_timestamps)
    """
    segments: list[Segment] = []
    all_char_ts: list[list[CharTimestamp]] = []

    for item in res:
        # ✅ 使用 _extract_text_and_timestamps 提取字级时间戳
        text, char_ts = self._extract_text_and_timestamps([item], time_offset=0.0)
        if not text:
            continue

        # 提取片段时间戳
        timestamp = item.get("timestamp", [])
        if timestamp:
            start = timestamp[0][0] / 1000.0
            end = timestamp[-1][1] / 1000.0
        else:
            estimated_dur = len(text) * 0.3
            start = last_end
            end = min(start + estimated_dur, duration)

        segments.append(Segment(...))
        all_char_ts.append(char_ts)  # ✅ 收集字级时间戳

    return segments, all_char_ts
```

### 3. 更新调用处

```python
if result is None:
    logger.info(f"[{engine_name}] VAD 不可用，使用整文件转写")
    segments, char_timestamps = self._transcribe_whole_file(audio_path, duration, language)  # ✅ 接收字级时间戳
    has_ts = sum(1 for ts in char_timestamps if ts) if char_timestamps else 0
    result = TranscriptResult(
        language=language or "zh",
        language_probability=1.0,
        duration=duration,
        segments=segments,
        char_timestamps=char_timestamps if has_ts > 0 else None,  # ✅ 传递字级时间戳
    )
```

## 预期效果

- ✅ 即使 VAD 预分段失败，整文件转写也会提取字级时间戳
- ✅ 对齐时可以使用字级对齐（`_align_by_char_timestamps`）
- ✅ 说话人切分精确，避免多人对话被合并到同一段
- ✅ 结果稳定性提升，不再出现 3 段 vs 30 段的巨大差异

## 验证步骤

1. 重新运行之前失败的测试文件（`汉语水平考试听力_trimmed`）
2. 检查日志：
   ```
   INFO 对齐: N 个转写片段, 17 个说话人片段, 字级时间戳=有  # ✅ 应该是"有"
   ```
3. 检查分段结果：
   - 段落数应该在 20-40 之间（与 Whisper 接近）
   - 不应出现多个说话人混合的情况
   - 主持人的问题和张教授的回答应该分开

## 待解决问题

**VAD 预分段失败的根本原因**（需进一步调查）:
- 为什么 3D-Speaker diarization 可以使用 VAD，但随后的 ASR 就失败了？
- 可能的原因：
  1. `_get_vad_model()` 单例状态问题
  2. FunASR 模型资源冲突
  3. GPU 内存不足
- 建议：添加详细日志，捕获 VAD 失败的具体异常信息

## 文件修改

- `backend/src/meeting_ai/services/asr.py`:
  - `_transcribe_whole_file()` - 修改返回类型
  - `_parse_funasr_output_with_timestamps()` - 新增方法
  - `transcribe()` (FunASRFileEngine) - 更新调用处
