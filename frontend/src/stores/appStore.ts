import { create } from 'zustand'
import type {
  EnhanceMode,
  HistoryItem,
  ModelInfo,
  ProcessResult,
  RealtimeSegment,
  RealtimeState,
  Segment,
  Speaker,
} from '../types'

export type TabType = 'realtime' | 'file' | 'settings'

interface AppState {
  // Tab 状态
  activeTab: TabType

  // 模型
  asrModels: ModelInfo[]
  llmModels: ModelInfo[]
  diarizationModels: ModelInfo[]
  genderModels: ModelInfo[]
  selectedAsrModel: string
  selectedLlmModel: string
  selectedDiarizationModel: string
  selectedGenderModel: string

  // 处理选项
  enableNaming: boolean
  enableCorrection: boolean
  enableSummary: boolean
  enhanceMode: EnhanceMode

  // 当前任务
  currentJobId: string | null
  isProcessing: boolean
  progress: number
  progressMessage: string

  // 结果
  result: ProcessResult | null
  segments: Segment[]
  speakers: Record<string, Speaker>
  summary: string
  audioUrl: string | null
  audioOriginalUrl: string | null
  useOriginalAudio: boolean
  duration: number

  // 播放状态
  isPlaying: boolean
  currentTime: number
  currentSegmentId: number

  // 历史记录
  historyItems: HistoryItem[]
  pendingHistoryId: string | null  // 从实时录音跳转后要自动选中的历史记录

  // 实时录音
  isRecording: boolean
  recordingDuration: number
  recordingVolume: number
  isSpeech: boolean
  realtimeSegments: RealtimeSegment[]
  realtimeState: RealtimeState
  realtimeSessionId: string | null
  postProcessProgress: number
  postProcessStep: string

  // Actions
  setModels: (asr: ModelInfo[], llm: ModelInfo[], diarization?: ModelInfo[], gender?: ModelInfo[]) => void
  setSelectedAsrModel: (model: string) => void
  setSelectedLlmModel: (model: string) => void
  setSelectedDiarizationModel: (model: string) => void
  setSelectedGenderModel: (model: string) => void
  setProcessOptions: (options: Partial<{
    enableNaming: boolean
    enableCorrection: boolean
    enableSummary: boolean
    enhanceMode: EnhanceMode
  }>) => void
  setCurrentJob: (jobId: string | null) => void
  setProcessing: (processing: boolean) => void
  setProgress: (progress: number, message: string) => void
  setResult: (result: ProcessResult | null) => void
  toggleAudioSource: () => void
  updateSegmentText: (segmentId: number, text: string) => void
  updateSpeakerName: (speakerId: string, name: string) => void
  updateSummary: (summary: string) => void
  setPlayback: (state: Partial<{
    isPlaying: boolean
    currentTime: number
    currentSegmentId: number
  }>) => void
  setHistoryItems: (items: HistoryItem[]) => void
  setPendingHistoryId: (id: string | null) => void
  setRecording: (state: Partial<{
    isRecording: boolean
    recordingDuration: number
    recordingVolume: number
    isSpeech: boolean
  }>) => void
  addRealtimeSegment: (segment: RealtimeSegment) => void
  updateRealtimeSegment: (segment: RealtimeSegment) => void
  clearRealtimeSegments: () => void
  setRealtimeState: (state: RealtimeState) => void
  setRealtimeSessionId: (id: string | null) => void
  setPostProcessProgress: (progress: number, step: string) => void
  setActiveTab: (tab: TabType) => void
  reset: () => void
}

const initialState = {
  activeTab: 'file' as TabType,
  asrModels: [],
  llmModels: [],
  diarizationModels: [],
  genderModels: [],
  selectedAsrModel: 'medium',
  selectedLlmModel: 'disabled',
  selectedDiarizationModel: '',
  selectedGenderModel: 'f0',
  enableNaming: true,
  enableCorrection: true,
  enableSummary: true,
  enhanceMode: 'none' as EnhanceMode,
  currentJobId: null,
  isProcessing: false,
  progress: 0,
  progressMessage: '',
  result: null,
  segments: [],
  speakers: {},
  summary: '',
  audioUrl: null,
  audioOriginalUrl: null,
  useOriginalAudio: false,
  duration: 0,
  isPlaying: false,
  currentTime: 0,
  currentSegmentId: -1,
  historyItems: [],
  pendingHistoryId: null,
  isRecording: false,
  recordingDuration: 0,
  recordingVolume: 0,
  isSpeech: false,
  realtimeSegments: [] as RealtimeSegment[],
  realtimeState: 'idle' as RealtimeState,
  realtimeSessionId: null,
  postProcessProgress: 0,
  postProcessStep: '',
}

export const useAppStore = create<AppState>((set) => ({
  ...initialState,

  setModels: (asr, llm, diarization, gender) =>
    set((state) => {
      // 自动选择第一个可用的说话人分离模型（当前值为空或无效时）
      let newDiarizationModel = state.selectedDiarizationModel
      if (diarization && diarization.length > 0) {
        // 如果当前选择为空字符串或不在新列表中，自动选择第一个
        const isCurrentValid = state.selectedDiarizationModel &&
          diarization.some(m => m.name === state.selectedDiarizationModel)
        if (!isCurrentValid) {
          newDiarizationModel = diarization[0].name
        }
      }

      // 同样处理性别检测模型
      let newGenderModel = state.selectedGenderModel
      if (gender && gender.length > 0) {
        const isCurrentValid = state.selectedGenderModel &&
          gender.some(m => m.name === state.selectedGenderModel)
        if (!isCurrentValid) {
          newGenderModel = gender[0].name
        }
      }

      return {
        asrModels: asr,
        llmModels: llm,
        diarizationModels: diarization ?? state.diarizationModels,
        genderModels: gender ?? state.genderModels,
        selectedDiarizationModel: newDiarizationModel,
        selectedGenderModel: newGenderModel,
      }
    }),

  setSelectedAsrModel: (model) =>
    set({ selectedAsrModel: model }),

  setSelectedLlmModel: (model) =>
    set({ selectedLlmModel: model }),

  setSelectedDiarizationModel: (model) =>
    set({ selectedDiarizationModel: model }),

  setSelectedGenderModel: (model) =>
    set({ selectedGenderModel: model }),

  setProcessOptions: (options) =>
    set((state) => ({ ...state, ...options })),

  setCurrentJob: (jobId) =>
    set({ currentJobId: jobId }),

  setProcessing: (processing) =>
    set({ isProcessing: processing }),

  setProgress: (progress, message) =>
    set({ progress, progressMessage: message }),

  setResult: (result) =>
    set({
      result,
      segments: result?.segments ?? [],
      speakers: result?.speakers ?? {},
      summary: result?.summary ?? '',
      audioUrl: result?.audio_url ?? null,
      audioOriginalUrl: result?.audio_original_url ?? null,
      useOriginalAudio: false,
      duration: result?.duration ?? 0,
    }),

  toggleAudioSource: () =>
    set((state) => ({ useOriginalAudio: !state.useOriginalAudio })),

  updateSegmentText: (segmentId, text) =>
    set((state) => ({
      segments: state.segments.map((seg) =>
        seg.id === segmentId ? { ...seg, text } : seg
      ),
    })),

  updateSpeakerName: (speakerId, name) =>
    set((state) => ({
      speakers: {
        ...state.speakers,
        [speakerId]: {
          ...state.speakers[speakerId],
          display_name: name,
        },
      },
      segments: state.segments.map((seg) =>
        seg.speaker === speakerId ? { ...seg, speaker_name: name } : seg
      ),
    })),

  updateSummary: (summary) =>
    set({ summary }),

  setPlayback: (state) =>
    set((prev) => ({ ...prev, ...state })),

  setHistoryItems: (items) =>
    set({ historyItems: items }),

  setPendingHistoryId: (id) =>
    set({ pendingHistoryId: id }),

  setRecording: (state) =>
    set((prev) => ({ ...prev, ...state })),

  addRealtimeSegment: (segment) =>
    set((state) => ({
      realtimeSegments: [...state.realtimeSegments, segment],
    })),

  updateRealtimeSegment: (segment) =>
    set((state) => {
      // 查找并更新现有片段，或添加新片段
      const index = state.realtimeSegments.findIndex((s) => s.id === segment.id)
      if (index >= 0) {
        const updated = [...state.realtimeSegments]
        updated[index] = segment
        return { realtimeSegments: updated }
      }
      return { realtimeSegments: [...state.realtimeSegments, segment] }
    }),

  clearRealtimeSegments: () =>
    set({ realtimeSegments: [] }),

  setRealtimeState: (realtimeState) =>
    set({ realtimeState }),

  setRealtimeSessionId: (realtimeSessionId) =>
    set({ realtimeSessionId }),

  setPostProcessProgress: (postProcessProgress, postProcessStep) =>
    set({ postProcessProgress, postProcessStep }),

  setActiveTab: (tab) =>
    set({ activeTab: tab }),

  reset: () =>
    set({
      currentJobId: null,
      isProcessing: false,
      progress: 0,
      progressMessage: '',
      result: null,
      segments: [],
      speakers: {},
      summary: '',
      audioUrl: null,
      audioOriginalUrl: null,
      useOriginalAudio: false,
      duration: 0,
      isPlaying: false,
      currentTime: 0,
      currentSegmentId: -1,
    }),
}))
