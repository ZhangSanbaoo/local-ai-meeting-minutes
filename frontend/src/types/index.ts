// API 类型定义

export type JobStatus = 'pending' | 'processing' | 'completed' | 'failed'

export type EnhanceMode = 'none' | 'denoise' | 'enhance' | 'vocal' | 'full'

export interface Segment {
  id: number
  start: number
  end: number
  text: string
  speaker: string
  speaker_name?: string
}

export interface Speaker {
  id: string
  display_name: string
  gender?: string
  total_duration: number
  segment_count: number
}

export interface ProcessResult {
  job_id: string
  status: JobStatus
  segments: Segment[]
  speakers: Record<string, Speaker>
  summary: string
  audio_url?: string
  audio_original_url?: string
  duration: number
  output_dir: string
}

export interface JobResponse {
  job_id: string
  status: JobStatus
  progress: number
  message: string
  created_at: string
  completed_at?: string
}

export interface HistoryItem {
  id: string
  name: string
  created_at: string
  duration: number
  segment_count: number
  has_summary: boolean
}

export interface ModelInfo {
  name: string
  display_name: string
  path: string
  size_mb?: number
  engine?: string  // ASR 引擎类型: faster-whisper / funasr / fireredasr
}

export interface ModelsResponse {
  asr_models: ModelInfo[]
  whisper_models: ModelInfo[]  // 向后兼容
  llm_models: ModelInfo[]
  diarization_models: ModelInfo[]
  gender_models: ModelInfo[]
}

export interface SystemInfo {
  version: string
  cuda_available: boolean
  cuda_version?: string
  gpu_name?: string
  models_dir: string
  output_dir: string
}

// 处理选项
export interface ProcessOptions {
  asr_model: string
  llm_model?: string
  diarization_model?: string
  gender_model?: string
  enable_naming: boolean
  enable_correction: boolean
  enable_summary: boolean
  enhance_mode: EnhanceMode
}

// WebSocket 消息类型
export type WSMessageType =
  | 'connected'
  | 'recording_started'
  | 'partial'
  | 'recording_stopped'
  | 'post_progress'
  | 'final_result'
  | 'status'
  | 'error'

export interface WSMessage {
  type: WSMessageType
  [key: string]: unknown
}

export interface WSPartialMessage extends WSMessage {
  type: 'partial'
  text: string
  is_final: boolean
  segment_id: number
  start_time: number
  end_time: number
}

export interface WSPostProgressMessage extends WSMessage {
  type: 'post_progress'
  step: string
  progress: number
  overall_progress: number
  message: string
}

export interface WSFinalResultMessage extends WSMessage {
  type: 'final_result'
  result: unknown
  history_id: string | null
}

export interface WSStatusMessage extends WSMessage {
  type: 'status'
  message?: string
}

export interface WSErrorMessage extends WSMessage {
  type: 'error'
  message: string
  recoverable: boolean
}

// Realtime segment (for display during recording)
export interface RealtimeSegment {
  id: number
  text: string
  isFinal: boolean
  startTime: number
  endTime: number
}

export type RealtimeState =
  | 'idle'
  | 'connecting'
  | 'connected'
  | 'recording'
  | 'post_processing'
  | 'done'
  | 'error'

// 流式 ASR 引擎
export interface StreamingEngine {
  id: string
  name: string
  description: string
  installed: boolean
  model_dir: string
}

export interface StreamingEnginesResponse {
  engines: StreamingEngine[]
  current: string
}
