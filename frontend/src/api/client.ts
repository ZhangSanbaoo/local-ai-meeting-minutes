import axios from 'axios'
import type {
  HistoryItem,
  JobResponse,
  ModelsResponse,
  ProcessResult,
  StreamingEnginesResponse,
  SystemInfo,
} from '../types'

const API_BASE = '/api'

const client = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
})

// 模型相关
export async function getModels(): Promise<ModelsResponse> {
  const { data } = await client.get<ModelsResponse>('/models')
  return data
}

export async function uploadWhisperModel(file: File): Promise<{ status: string; model_name: string; path: string }> {
  const formData = new FormData()
  formData.append('file', file)
  const { data } = await client.post('/models/upload/whisper', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 300000, // 5 minutes for large models
  })
  return data
}

export async function uploadLlmModel(file: File): Promise<{ status: string; model_name: string; path: string }> {
  const formData = new FormData()
  formData.append('file', file)
  const { data } = await client.post('/models/upload/llm', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 600000, // 10 minutes for large LLM models
  })
  return data
}

export async function deleteWhisperModel(modelName: string): Promise<void> {
  await client.delete(`/models/whisper/${modelName}`)
}

export async function deleteLlmModel(modelName: string): Promise<void> {
  await client.delete(`/models/llm/${modelName}`)
}

export async function getSystemInfo(): Promise<SystemInfo> {
  const { data } = await client.get<SystemInfo>('/system')
  return data
}

// 流式 ASR 引擎
export async function getStreamingEngines(): Promise<StreamingEnginesResponse> {
  const { data } = await client.get<StreamingEnginesResponse>('/streaming-engines')
  return data
}

// 音频设备相关
export interface AudioDevice {
  id: number
  name: string
  channels: number
  sample_rate: number
  is_loopback: boolean
}

export interface AudioDevicesResponse {
  input_devices: AudioDevice[]
  loopback_devices: AudioDevice[]
  default_input: string | null
  default_output: string | null
  loopback_available: boolean
  error?: string | null
}

export async function getAudioDevices(): Promise<AudioDevicesResponse> {
  const { data } = await client.get<AudioDevicesResponse>('/audio-devices')
  return data
}

// 处理相关
export async function uploadDiarizationModel(file: File): Promise<{ status: string; model_name: string; path: string }> {
  const formData = new FormData()
  formData.append('file', file)
  const { data } = await client.post('/models/upload/diarization', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 300000,
  })
  return data
}

export async function uploadGenderModel(file: File): Promise<{ status: string; model_name: string; path: string }> {
  const formData = new FormData()
  formData.append('file', file)
  const { data } = await client.post('/models/upload/gender', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 300000,
  })
  return data
}

export async function deleteDiarizationModel(modelName: string): Promise<void> {
  await client.delete(`/models/diarization/${modelName}`)
}

export async function deleteGenderModel(modelName: string): Promise<void> {
  await client.delete(`/models/gender/${modelName}`)
}

export async function uploadAndProcess(
  file: File,
  options: {
    name?: string  // 自定义会议名称
    whisper_model: string
    llm_model?: string
    diarization_model?: string
    gender_model?: string
    enable_naming: boolean
    enable_correction: boolean
    enable_summary: boolean
    enhance_mode: string
  }
): Promise<JobResponse> {
  const formData = new FormData()
  formData.append('file', file)
  if (options.name) {
    formData.append('name', options.name)
  }
  formData.append('whisper_model', options.whisper_model)
  if (options.llm_model) {
    formData.append('llm_model', options.llm_model)
  }
  if (options.diarization_model) {
    formData.append('diarization_model', options.diarization_model)
  }
  if (options.gender_model) {
    formData.append('gender_model', options.gender_model)
  }
  formData.append('enable_naming', String(options.enable_naming))
  formData.append('enable_correction', String(options.enable_correction))
  formData.append('enable_summary', String(options.enable_summary))
  formData.append('enhance_mode', options.enhance_mode)

  const { data } = await client.post<JobResponse>('/process', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 60000, // 上传可能需要更长时间
  })
  return data
}

export async function getJobStatus(jobId: string): Promise<JobResponse> {
  const { data } = await client.get<JobResponse>(`/jobs/${jobId}`)
  return data
}

export async function getJobResult(jobId: string): Promise<ProcessResult> {
  const { data } = await client.get<ProcessResult>(`/jobs/${jobId}/result`)
  return data
}

export async function updateSegment(
  jobId: string,
  segmentId: number,
  updates: { text?: string; speaker_name?: string }
): Promise<void> {
  await client.put(`/jobs/${jobId}/segments/${segmentId}`, updates)
}

export async function renameSpeaker(
  jobId: string,
  speakerId: string,
  newName: string
): Promise<void> {
  await client.put(`/jobs/${jobId}/speakers`, {
    speaker_id: speakerId,
    new_name: newName,
  })
}

export async function updateSummary(
  jobId: string,
  summary: string
): Promise<void> {
  await client.put(`/jobs/${jobId}/summary`, { summary })
}

export async function splitSegment(
  jobId: string,
  segmentId: number,
  splitPosition: number,
  newSpeaker?: string
): Promise<{ status: string; segments: Array<{ id: number; start: number; end: number; text: string; speaker: string }> }> {
  const { data } = await client.post(`/jobs/${jobId}/segments/${segmentId}/split`, {
    split_position: splitPosition,
    new_speaker: newSpeaker,
  })
  return data
}

export async function changeSegmentSpeaker(
  jobId: string,
  segmentId: number,
  speakerId: string
): Promise<void> {
  await client.put(`/jobs/${jobId}/segments/${segmentId}/speaker`, null, {
    params: { speaker_id: speakerId }
  })
}

// 历史记录相关
export async function getHistory(
  limit = 50,
  offset = 0
): Promise<{ items: HistoryItem[]; total: number }> {
  const { data } = await client.get<{ items: HistoryItem[]; total: number }>(
    '/history',
    { params: { limit, offset } }
  )
  return data
}

export async function getHistoryItem(historyId: string): Promise<ProcessResult> {
  const { data } = await client.get<ProcessResult>(`/history/${historyId}`)
  return data
}

export async function deleteHistoryItem(historyId: string): Promise<void> {
  await client.delete(`/history/${historyId}`)
}

export async function updateHistorySegment(
  historyId: string,
  segmentId: number,
  updates: { text?: string; speaker_name?: string }
): Promise<void> {
  await client.put(`/history/${historyId}/segments/${segmentId}`, updates)
}

export async function splitHistorySegment(
  historyId: string,
  segmentId: number,
  splitPosition: number,
  newSpeaker?: string
): Promise<{ status: string; segments: Array<{ id: number; start: number; end: number; text: string; speaker: string }> }> {
  const { data } = await client.post(`/history/${historyId}/segments/${segmentId}/split`, {
    split_position: splitPosition,
    new_speaker: newSpeaker,
  })
  return data
}

export async function mergeHistorySegments(
  historyId: string,
  segmentIds: number[]
): Promise<{ status: string; merged_segment: { id: number; start: number; end: number; text: string; speaker: string }; new_segment_count: number }> {
  const { data } = await client.post(`/history/${historyId}/segments/merge`, {
    segment_ids: segmentIds,
  })
  return data
}

export async function changeHistorySegmentSpeaker(
  historyId: string,
  segmentId: number,
  speakerId: string
): Promise<void> {
  await client.put(`/history/${historyId}/segments/${segmentId}/speaker`, null, {
    params: { speaker_id: speakerId }
  })
}

export async function renameHistorySpeaker(
  historyId: string,
  speakerId: string,
  newName: string
): Promise<void> {
  await client.put(`/history/${historyId}/speakers`, {
    speaker_id: speakerId,
    new_name: newName,
  })
}

export async function updateHistorySummary(
  historyId: string,
  summary: string
): Promise<void> {
  await client.put(`/history/${historyId}/summary`, { summary })
}

export async function regenerateSummary(
  historyId: string,
  llmModel?: string
): Promise<{ status: string; summary: string }> {
  const { data } = await client.post(`/history/${historyId}/summary/regenerate`, {
    llm_model: llmModel,
  }, {
    timeout: 600000, // LLM 生成长会议总结可能需要数分钟
  })
  return data
}

export async function renameHistoryItem(historyId: string, newName: string): Promise<{ new_id: string }> {
  const { data } = await client.put(`/history/${historyId}/rename`, null, {
    params: { new_name: newName }
  })
  return data
}

export function getExportUrl(historyId: string, format: 'txt' | 'json' | 'md'): string {
  return `${API_BASE}/history/${historyId}/export/${format}`
}
