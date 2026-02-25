# Meeting AI Backend

本地离线会议纪要 AI - FastAPI 后端服务

## 功能

- **多引擎语音识别**: faster-whisper / FunASR (SenseVoice, Paraformer) / FireRedASR
- **说话人辨识**: pyannote-audio 3.1，自动识别"谁在什么时候说话"
- **字级时间戳对齐**: 每个字独立对应说话人，精确到 10-50ms
- **性别检测**: f0 基频分析 / ECAPA-TDNN / Wav2Vec2（默认兜底命名）
- **智能命名**: LLM 自动推断说话人身份（正则 + LLM + 性别兜底）
- **对话编辑**: 删除 / 分割 / 合并 / 说话人重分配 / 新建说话人
- **音频增强**: Demucs 人声分离 + DeepFilterNet3 降噪 + Resemble Enhance
- **实时流式**: FunASR + sherpa-onnx 双引擎 + fsmn-vad
- **完全离线**: 所有模型本地运行，无需联网

## 启动

```bash
conda activate meeting-ai
uvicorn meeting_ai.api.main:app --reload --port 8000
```

## API 文档

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 主要 API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/process` | 上传音频并处理 |
| GET | `/api/jobs/{id}` | 查询任务状态 |
| GET | `/api/jobs/{id}/result` | 获取处理结果 |
| PUT | `/api/jobs/{id}/segments/{id}` | 编辑片段（文本/说话人重分配） |
| DELETE | `/api/jobs/{id}/segments/{id}` | 删除片段 |
| GET | `/api/history` | 历史记录列表 |
| GET | `/api/history/{id}` | 历史记录详情 |
| GET | `/api/models` | 可用模型列表 |
| WebSocket | `/api/ws/realtime` | 实时流式 ASR |

*最后更新: 2026-02-25*
