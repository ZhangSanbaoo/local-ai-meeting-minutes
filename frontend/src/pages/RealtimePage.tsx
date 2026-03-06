import { useCallback, useEffect, useRef, useState } from 'react'
import { Mic, Square, Loader2, Monitor, Power, PowerOff } from 'lucide-react'
import { clsx } from 'clsx'
import { formatTime } from '../utils/format'
import { useAppStore } from '../stores/appStore'
import { useAudioCapture, type AudioSourceType } from '../hooks/useAudioCapture'
import {
  useRealtimeWebSocket,
  type PartialSegment,
  type PostProgress,
  type FinalResult,
} from '../hooks/useRealtimeWebSocket'
import { useRecordingTimer } from '../hooks/useRecordingTimer'
import { ProgressBar } from '../components/ProgressBar'
import { getStreamingEngines, getModels, getAudioDevices, unloadLlm } from '../api/client'
import type { ModelInfo, RecordingMode, RealtimeSegment, StreamingEngine, HybridUpgradeMode } from '../types'

export function RealtimePage() {
  // ── Local state ──
  const [meetingName, setMeetingName] = useState('')
  const [engines, setEngines] = useState<StreamingEngine[]>([])
  const [selectedEngine, setSelectedEngine] = useState('')
  const [isStartPending, setIsStartPending] = useState(false)

  // Microphone devices (from browser API)
  const [micDevices, setMicDevices] = useState<MediaDeviceInfo[]>([])
  const [selectedMicId, setSelectedMicId] = useState('default')
  const [defaultMicName, setDefaultMicName] = useState<string | null>(null)

  // System audio capture
  const [enableSystemAudio, setEnableSystemAudio] = useState(false)
  const [enableMicrophone, setEnableMicrophone] = useState(true)

  // Input validation error
  const [inputError, setInputError] = useState<string | null>(null)

  // LLM models
  const [llmModels, setLlmModels] = useState<ModelInfo[]>([])
  const [selectedLlm, setSelectedLlm] = useState('')

  // Gender models
  const [genderModels, setGenderModels] = useState<ModelInfo[]>([])
  const [selectedGenderModel, setSelectedGenderModel] = useState('f0')

  // Recording mode
  const [recordingMode, setRecordingMode] = useState<RecordingMode>('streaming')

  // Hybrid upgrade granularity
  const [hybridUpgrade, setHybridUpgrade] = useState<HybridUpgradeMode>('segment')

  // File ASR models (for segment/hybrid modes)
  const [fileAsrModels, setFileAsrModels] = useState<ModelInfo[]>([])
  const [selectedFileAsr, setSelectedFileAsr] = useState('')

  // Realtime denoise
  const [enableDenoise, setEnableDenoise] = useState(false)

  // Upgrade animation tracking
  const [upgradedSegmentIds, setUpgradedSegmentIds] = useState<Set<number>>(new Set())

  // ── Load available engines, models, mic devices ──
  useEffect(() => {
    getStreamingEngines()
      .then((res) => {
        setEngines(res.engines)
        setSelectedEngine(res.current)
      })
      .catch((err) => console.error('Failed to load streaming engines:', err))

    getModels()
      .then((res) => {
        setLlmModels(res.llm_models)
        setSelectedLlm('disabled')

        setGenderModels(res.gender_models)
        if (res.gender_models.length > 0) {
          setSelectedGenderModel(res.gender_models[0].name)
        }

        // File ASR models for segment/hybrid modes
        setFileAsrModels(res.asr_models)
        if (res.asr_models.length > 0) {
          setSelectedFileAsr(res.asr_models[0].name)
        }
      })
      .catch((err) => console.error('Failed to load models:', err))

    // Enumerate browser microphone devices
    navigator.mediaDevices
      .enumerateDevices()
      .then((devices) => {
        const audioInputs = devices.filter((d) => d.kind === 'audioinput')
        setMicDevices(audioInputs)
        // Select first device (default) if available
        if (audioInputs.length > 0 && !selectedMicId) {
          setSelectedMicId(audioInputs[0].deviceId)
        }
      })
      .catch((err) => console.error('Failed to enumerate devices:', err))

    // Get default mic name from backend (sounddevice has better device info)
    getAudioDevices()
      .then((res) => {
        if (res.default_input) {
          setDefaultMicName(res.default_input)
        }
      })
      .catch(() => {})
  }, [])

  // ── Store ──
  const {
    enableNaming,
    enableCorrection,
    enableSummary,
    realtimeSegments,
    realtimeState,
    postProcessProgress,
    postProcessStep,
    setRecording,
    addRealtimeSegment,
    updateRealtimeSegment,
    clearRealtimeSegments,
    setRealtimeState,
    setPostProcessProgress,
    setPendingHistoryId,
    setActiveTab,
    setProcessOptions,
  } = useAppStore()

  // ── Timer ──
  const timer = useRecordingTimer()

  // ── Scroll ref for auto-scroll ──
  const scrollRef = useRef<HTMLDivElement>(null)

  // ── WebSocket ──
  const wsCallbacks = {
    onPartial: useCallback(
      (seg: PartialSegment) => {
        const realtimeSeg: RealtimeSegment = {
          id: seg.segmentId,
          text: seg.text,
          isFinal: seg.isFinal,
          startTime: seg.startTime,
          endTime: seg.endTime,
          isPlaceholder: seg.isPlaceholder,
          isUpgraded: seg.isUpgrade,
        }

        const existing = useAppStore
          .getState()
          .realtimeSegments.find((s) => s.id === seg.segmentId)

        if (existing) {
          updateRealtimeSegment(realtimeSeg)
        } else {
          addRealtimeSegment(realtimeSeg)
        }

        // 混合模式升级动画
        if (seg.isUpgrade) {
          setUpgradedSegmentIds((prev) => new Set(prev).add(seg.segmentId))
          setTimeout(() => {
            setUpgradedSegmentIds((prev) => {
              const next = new Set(prev)
              next.delete(seg.segmentId)
              return next
            })
          }, 800)
        }
      },
      [addRealtimeSegment, updateRealtimeSegment]
    ),

    onPostProgress: useCallback(
      (progress: PostProgress) => {
        setPostProcessProgress(progress.overallProgress, progress.message)
      },
      [setPostProcessProgress]
    ),

    onFinalResult: useCallback(
      (result: FinalResult) => {
        setRealtimeState('done')
        if (result.historyId) {
          setPendingHistoryId(result.historyId)
          setActiveTab('file')
        }
      },
      [setRealtimeState, setPendingHistoryId, setActiveTab]
    ),

    onError: useCallback(
      (message: string, _recoverable: boolean) => {
        console.error('[Realtime] Error:', message)
        setRealtimeState('error')
        setIsStartPending(false)
      },
      [setRealtimeState]
    ),

    onStateChange: useCallback(
      (state: string) => {
        setRealtimeState(
          state as
            | 'idle'
            | 'connecting'
            | 'connected'
            | 'recording'
            | 'post_processing'
            | 'done'
            | 'error'
        )
        if (state === 'recording') {
          setIsStartPending(false)
          timer.reset()
          timer.start()
        }
      },
      [setRealtimeState, timer]
    ),
  }

  const ws = useRealtimeWebSocket(wsCallbacks)

  // ── Audio capture ──
  // Ref to allow onSystemAudioEnded callback to call handleStopRecording
  const stopRecordingRef = useRef<(() => void) | null>(null)

  // Only send audio chunks after backend confirms recording has started.
  // This prevents blank audio at the beginning during model loading.
  const audioCallbacks = {
    onPCMChunk: useCallback(
      (buffer: ArrayBuffer) => {
        const state = useAppStore.getState().realtimeState
        if (state === 'recording') {
          ws.sendAudioChunk(buffer)
        }
      },
      [ws]
    ),
    onSystemAudioEnded: useCallback(() => {
      // User stopped screen sharing — auto-stop recording
      console.log('[RealtimePage] System audio ended, stopping recording')
      stopRecordingRef.current?.()
    }, []),
  }

  const audio = useAudioCapture(audioCallbacks)

  // ── Load / Unload models ──
  const handleLoadModels = useCallback(() => {
    // 先卸载聊天加载的 LLM，释放显存给 ASR 模型
    unloadLlm().catch(() => {})

    const doLoad = () => {
      if (recordingMode === 'segment') {
        ws.loadModels(undefined, { mode: 'segment', sentenceAsrModel: selectedFileAsr || undefined })
      } else if (recordingMode === 'hybrid') {
        ws.loadModels(selectedEngine || undefined, {
          mode: 'hybrid',
          sentenceAsrModel: selectedFileAsr || undefined,
          hybridUpgrade,
        })
      } else {
        ws.loadModels(selectedEngine || undefined)
      }
    }
    // If not connected, connect first, then load
    if (ws.state === 'disconnected' || ws.state === 'error') {
      ws.connect()
      // Wait for connection, then load
      const check = setInterval(() => {
        const state = useAppStore.getState().realtimeState
        if (state === 'connected' || state === 'done') {
          clearInterval(check)
          doLoad()
        }
      }, 100)
      // Timeout after 5s
      setTimeout(() => clearInterval(check), 5000)
    } else {
      doLoad()
    }
  }, [ws, selectedEngine, recordingMode, selectedFileAsr, hybridUpgrade])

  const handleUnloadModels = useCallback(() => {
    if (ws.state !== 'disconnected') {
      if (recordingMode === 'segment') {
        ws.unloadModels(undefined, { mode: 'segment' })
      } else if (recordingMode === 'hybrid') {
        ws.unloadModels(selectedEngine || undefined, { mode: 'hybrid' })
      } else {
        ws.unloadModels(selectedEngine || undefined)
      }
    }
  }, [ws, selectedEngine, recordingMode])

  // ── Auto-scroll transcript ──
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [realtimeSegments])

  // ── Start recording ──
  const handleStartRecording = useCallback(async () => {
    // Determine source type from UI selections
    if (!enableMicrophone && !enableSystemAudio) {
      setInputError('请至少选择一个输入源（麦克风或系统音频）')
      return
    }
    let sourceType: AudioSourceType
    if (enableMicrophone && enableSystemAudio) {
      sourceType = 'mixed'
    } else if (enableSystemAudio) {
      sourceType = 'system_audio'
    } else {
      sourceType = 'microphone'
    }

    // 勾选了需要 LLM 的功能但没选 LLM
    const needsLlm = enableNaming || enableCorrection || enableSummary
    if (needsLlm && (!selectedLlm || selectedLlm === 'disabled')) {
      setInputError('已勾选智能命名/错别字校正/会议总结，请先选择一个 LLM 模型')
      return
    }

    setInputError(null)
    clearRealtimeSegments()
    setPostProcessProgress(0, '')
    setIsStartPending(true)

    // 卸载聊天加载的 LLM，释放显存给 ASR/diarization
    try { await unloadLlm() } catch { /* 忽略 */ }

    // CRITICAL: Start audio capture FIRST while in user gesture context.
    // AudioContext.resume() requires a user gesture. If we await
    // other async operations first, the gesture context expires.
    try {
      await audio.start({
        sourceType,
        deviceId: enableMicrophone ? (selectedMicId || 'default') : undefined,
        targetSampleRate: 16000,
        chunkDurationMs: 200,
      })
    } catch (err) {
      console.error('Failed to start audio capture:', err)
      setRealtimeState('error')
      setIsStartPending(false)
      return
    }

    // WebSocket should already be connected (auto-connect on mount).
    // If not (e.g. disconnected due to error), reconnect.
    if (ws.state === 'disconnected' || ws.state === 'error') {
      ws.connect()

      // Wait for connection (max 5s)
      const waitForConnection = () =>
        new Promise<boolean>((resolve) => {
          let attempts = 0
          const check = setInterval(() => {
            const state = useAppStore.getState().realtimeState
            if (state === 'connected' || state === 'recording') {
              clearInterval(check)
              resolve(true)
            }
            if (++attempts > 50) {
              clearInterval(check)
              resolve(false)
            }
          }, 100)
        })

      const connected = await waitForConnection()
      if (!connected) {
        audio.stop()
        setRealtimeState('error')
        setIsStartPending(false)
        return
      }
    }

    ws.startRecording({
      meetingName: meetingName || 'realtime',
      enableNaming,
      enableCorrection,
      enableSummary,
      asrEngine: selectedEngine || undefined,
      genderModel: selectedGenderModel || undefined,
      llmModel: selectedLlm || undefined,
      mode: recordingMode,
      hybridUpgrade: recordingMode === 'hybrid' ? hybridUpgrade : undefined,
      sentenceAsrModel: recordingMode !== 'streaming' ? selectedFileAsr : undefined,
      enableDenoise,
    })

    setRecording({ isRecording: true })
  }, [
    ws,
    audio,
    meetingName,
    selectedMicId,
    selectedEngine,
    enableMicrophone,
    enableSystemAudio,
    enableNaming,
    enableCorrection,
    enableSummary,
    enableDenoise,
    selectedLlm,
    recordingMode,
    hybridUpgrade,
    selectedFileAsr,
    clearRealtimeSegments,
    setPostProcessProgress,
    setRealtimeState,
    setRecording,
  ])

  // ── Stop recording ──
  const handleStopRecording = useCallback(() => {
    audio.stop()
    ws.stopRecording()
    timer.pause()
    setRecording({ isRecording: false })
    setIsStartPending(false)
  }, [ws, audio, timer, setRecording])

  // Keep ref in sync so onSystemAudioEnded can call handleStopRecording
  stopRecordingRef.current = handleStopRecording

  // ── Derived state ──
  const isRecording = realtimeState === 'recording'
  const isPostProcessing = realtimeState === 'post_processing'
  const isConnecting = realtimeState === 'connecting'
  const isLoading = isStartPending && !isRecording
  const isModelsLoading = ws.modelsLoading
  const canRecord =
    (realtimeState === 'idle' ||
    realtimeState === 'connected' ||
    realtimeState === 'done' ||
    realtimeState === 'error') && ws.modelsReady
  const controlsDisabled = !(
    realtimeState === 'idle' ||
    realtimeState === 'connected' ||
    realtimeState === 'done' ||
    realtimeState === 'error'
  )

  // Volume display
  const volumePercent = Math.min(audio.volume * 5, 1) * 100
  const volumeColor =
    audio.volume > 0.14
      ? 'bg-red-500'
      : audio.volume > 0.08
        ? 'bg-yellow-500'
        : 'bg-green-500'

  return (
    <div className="flex flex-col h-full">
      {/* ── Row 1: 模型选择 ── */}
      <div className="flex items-center justify-center gap-3 py-3 bg-gray-100 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-600 flex-wrap px-4">
        {/* 会议名称 */}
        <input
          type="text"
          value={meetingName}
          onChange={(e) => setMeetingName(e.target.value)}
          placeholder="会议名称（可选）"
          disabled={controlsDisabled}
          className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded text-sm w-36 disabled:bg-gray-200 dark:disabled:bg-gray-700 disabled:cursor-not-allowed dark:bg-gray-700 dark:text-gray-200 dark:placeholder-gray-500"
        />

        {/* 流式引擎选择 (streaming/hybrid) */}
        {recordingMode !== 'segment' && (
          <div className="flex items-center gap-1">
            <select
              value={selectedEngine}
              onChange={(e) => setSelectedEngine(e.target.value)}
              disabled={controlsDisabled || engines.length <= 1}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-l text-sm disabled:bg-gray-200 dark:disabled:bg-gray-700 disabled:cursor-not-allowed dark:bg-gray-700 dark:text-gray-200"
              title="流式 ASR 引擎"
            >
              {engines.filter((e) => e.installed).map((engine) => (
                <option key={engine.id} value={engine.id}>
                  {engine.name}
                </option>
              ))}
              {engines.length === 0 && <option>加载中...</option>}
            </select>
          </div>
        )}

        {/* 文件 ASR 模型选择 (segment/hybrid) */}
        {recordingMode !== 'streaming' && (
          <div className="flex items-center gap-1">
            <span className="text-xs text-gray-500 dark:text-gray-400 whitespace-nowrap">
              {recordingMode === 'hybrid' ? '升级ASR' : '文件ASR'}
            </span>
            <select
              value={selectedFileAsr}
              onChange={(e) => setSelectedFileAsr(e.target.value)}
              disabled={controlsDisabled}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded text-sm disabled:bg-gray-200 dark:disabled:bg-gray-700 disabled:cursor-not-allowed max-w-[220px] dark:bg-gray-700 dark:text-gray-200"
              title="文件 ASR 模型（用于段级/混合模式的高精度转写）"
            >
              {fileAsrModels.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.display_name}{m.engine ? ` [${m.engine}]` : ''}
                </option>
              ))}
              {fileAsrModels.length === 0 && <option>无可用模型</option>}
            </select>
          </div>
        )}

        {/* 模型加载/释放 */}
        <div className="flex items-center gap-0">
          <button
            onClick={handleLoadModels}
            disabled={controlsDisabled || ws.modelsReady || isModelsLoading}
            className={clsx(
              'flex items-center gap-1 px-2.5 py-2 border text-sm font-medium transition-colors rounded-l',
              ws.modelsReady
                ? 'bg-green-50 dark:bg-green-900/30 border-green-300 dark:border-green-600 text-green-700 dark:text-green-400 cursor-default'
                : isModelsLoading
                  ? 'bg-yellow-50 dark:bg-yellow-900/30 border-yellow-300 dark:border-yellow-600 text-yellow-700 dark:text-yellow-400 cursor-wait'
                  : 'bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-200 hover:bg-blue-50 dark:hover:bg-blue-900/30 hover:border-blue-300 hover:text-blue-700 dark:hover:text-blue-400',
              (controlsDisabled && !ws.modelsReady) && 'opacity-60 cursor-not-allowed'
            )}
            title={ws.modelsReady ? '模型已就绪' : isModelsLoading ? '加载中...' : '加载模型到 GPU'}
          >
            {isModelsLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Power className="w-4 h-4" />
            )}
            {ws.modelsReady ? '已就绪' : isModelsLoading ? '加载中' : '加载'}
          </button>
          <button
            onClick={handleUnloadModels}
            disabled={controlsDisabled || !ws.modelsReady || isModelsLoading}
            className={clsx(
              'flex items-center gap-1 px-2.5 py-2 border border-gray-300 dark:border-gray-600 rounded-r text-sm font-medium transition-colors',
              ws.modelsReady && !controlsDisabled
                ? 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-red-50 dark:hover:bg-red-900/30 hover:border-red-300 hover:text-red-700 dark:hover:text-red-400'
                : 'bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-500 cursor-not-allowed'
            )}
            title="释放模型显存"
          >
            <PowerOff className="w-4 h-4" />
            释放
          </button>
        </div>

        {/* LLM 模型选择 */}
        <div className="flex items-center gap-1">
          <span className="text-xs text-gray-500 dark:text-gray-400 whitespace-nowrap">LLM</span>
          <select
            value={selectedLlm}
            onChange={(e) => setSelectedLlm(e.target.value)}
            disabled={controlsDisabled}
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded text-sm disabled:bg-gray-200 dark:disabled:bg-gray-700 disabled:cursor-not-allowed max-w-[220px] dark:bg-gray-700 dark:text-gray-200"
            title="LLM 模型（智能命名/总结）"
          >
            {llmModels.map((m) => (
              <option key={m.name} value={m.name}>
                {m.display_name}{m.size_mb ? ` (${Math.round(m.size_mb / 1024 * 10) / 10}GB)` : ''}
              </option>
            ))}
            {llmModels.length === 0 && <option>加载中...</option>}
          </select>
        </div>

        {/* 性别检测模型 */}
        <div className="flex items-center gap-1">
          <span className="text-xs text-gray-500 dark:text-gray-400 whitespace-nowrap">性别</span>
          <select
            value={selectedGenderModel}
            onChange={(e) => setSelectedGenderModel(e.target.value)}
            disabled={controlsDisabled}
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded text-sm disabled:bg-gray-200 dark:disabled:bg-gray-700 disabled:cursor-not-allowed max-w-[240px] dark:bg-gray-700 dark:text-gray-200"
            title="性别检测模型"
          >
            {genderModels.map((m) => (
              <option key={m.name} value={m.name}>
                {m.display_name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* ── Row 2: 模式选择 + 录音控制 ── */}
      <div className="flex items-center justify-center gap-4 py-3 bg-gray-100 dark:bg-gray-800">
        {/* 模式选择器 */}
        <div className="flex rounded-lg overflow-hidden border border-gray-300 dark:border-gray-600">
          {([
            { mode: 'streaming' as RecordingMode, label: '字级流式', tip: '逐字实时输出，~600ms延迟' },
            { mode: 'segment' as RecordingMode, label: '段级转写', tip: 'VAD 检测到语音段结束后高精度转写，~1-2s延迟' },
            { mode: 'hybrid' as RecordingMode, label: '混合模式', tip: '流式先出字，段完成后升级替换' },
          ]).map(({ mode, label, tip }) => (
            <button
              key={mode}
              onClick={() => setRecordingMode(mode)}
              disabled={controlsDisabled}
              title={tip}
              className={clsx(
                'px-3 py-2 text-sm font-medium transition-colors border-r last:border-r-0 border-gray-300 dark:border-gray-600',
                recordingMode === mode
                  ? 'bg-blue-600 text-white'
                  : controlsDisabled
                    ? 'bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-500 cursor-not-allowed'
                    : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-blue-50 dark:hover:bg-blue-900/30 hover:text-blue-700 dark:hover:text-blue-400'
              )}
            >
              {label}
            </button>
          ))}
        </div>

        {/* 混合模式升级粒度子选择器 */}
        {recordingMode === 'hybrid' && (
          <div className="flex rounded overflow-hidden border border-blue-300 dark:border-blue-600">
            {([
              { mode: 'segment' as HybridUpgradeMode, label: '段级', tip: '流式引擎完成一段后用文件 ASR 升级，反馈最快' },
              { mode: 'full' as HybridUpgradeMode, label: '整体', tip: '录音结束后用文件 ASR 重新转写，精度最好' },
            ]).map(({ mode, label, tip }) => (
              <button
                key={mode}
                onClick={() => setHybridUpgrade(mode)}
                disabled={controlsDisabled}
                title={tip}
                className={clsx(
                  'px-2 py-1 text-xs font-medium transition-colors border-r last:border-r-0 border-blue-300 dark:border-blue-600',
                  hybridUpgrade === mode
                    ? 'bg-blue-500 text-white'
                    : controlsDisabled
                      ? 'bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-500 cursor-not-allowed'
                      : 'bg-white dark:bg-gray-800 text-blue-700 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/30'
                )}
              >
                {label}
              </button>
            ))}
          </div>
        )}

        {/* 系统音频选择 */}
        <div className="flex items-center gap-1">
          <Monitor className={clsx('w-4 h-4', enableSystemAudio ? 'text-blue-600 dark:text-blue-400' : 'text-gray-500 dark:text-gray-400')} />
          <select
            value={enableSystemAudio ? 'enabled' : 'disabled'}
            onChange={(e) => setEnableSystemAudio(e.target.value === 'enabled')}
            disabled={controlsDisabled}
            className={clsx(
              'px-2 py-2 border rounded text-sm max-w-[200px]',
              controlsDisabled
                ? 'bg-gray-200 dark:bg-gray-700 cursor-not-allowed border-gray-300 dark:border-gray-600 dark:text-gray-400'
                : enableSystemAudio
                  ? 'border-blue-400 dark:border-blue-500 bg-blue-50 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300'
                  : 'border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-200'
            )}
            title="系统音频（通过屏幕共享捕获）"
          >
            <option value="disabled">不启用</option>
            <option value="enabled">系统音频（屏幕共享）</option>
          </select>
        </div>

        {/* 麦克风选择 */}
        <div className="flex items-center gap-1">
          <Mic className={clsx('w-4 h-4', enableMicrophone ? 'text-blue-600 dark:text-blue-400' : 'text-gray-500 dark:text-gray-400')} />
          <select
            value={enableMicrophone ? selectedMicId : 'disabled'}
            onChange={(e) => {
              if (e.target.value === 'disabled') {
                setEnableMicrophone(false)
              } else {
                setEnableMicrophone(true)
                setSelectedMicId(e.target.value)
              }
            }}
            disabled={controlsDisabled}
            className={clsx(
              'px-2 py-2 border rounded text-sm max-w-[280px]',
              controlsDisabled
                ? 'bg-gray-200 dark:bg-gray-700 cursor-not-allowed border-gray-300 dark:border-gray-600 dark:text-gray-400'
                : enableMicrophone
                  ? 'border-blue-400 dark:border-blue-500 bg-blue-50 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300'
                  : 'border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-200'
            )}
          >
            <option value="disabled">不启用</option>
            <option value="default">
              {defaultMicName ? `默认 (${defaultMicName})` : '默认麦克风'}
            </option>
            {micDevices.filter(d => d.deviceId !== 'default').map((d) => (
              <option key={d.deviceId} value={d.deviceId}>
                {d.label || `麦克风 ${d.deviceId.slice(0, 8)}`}
              </option>
            ))}
          </select>
        </div>

        {/* 录音按钮 */}
        <button
          onClick={isRecording ? handleStopRecording : handleStartRecording}
          disabled={!canRecord && !isRecording}
          className={clsx(
            'flex items-center gap-2 px-6 py-3 rounded text-white font-medium transition-colors',
            isRecording
              ? 'bg-gray-700 hover:bg-gray-800'
              : isLoading
                ? 'bg-gray-400 cursor-wait'
                : canRecord
                  ? 'bg-red-500 hover:bg-red-600'
                  : 'bg-gray-400 cursor-not-allowed opacity-60',
          )}
        >
          {isLoading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              {isConnecting ? '连接中...' : '准备中...'}
            </>
          ) : isRecording ? (
            <>
              <Square className="w-5 h-5" />
              停止录音
            </>
          ) : (
            <>
              <Mic className="w-5 h-5" />
              开始录音
            </>
          )}
        </button>

        {/* 时长显示 */}
        <span className="text-3xl font-bold text-gray-700 dark:text-gray-200 font-mono w-24 text-center">
          {formatTime(timer.elapsedTime)}
        </span>

        {/* 音量条 */}
        <div className="flex flex-col gap-1">
          <span className="text-xs text-gray-500 dark:text-gray-400">音量</span>
          <div className="w-32 h-2.5 bg-gray-300 dark:bg-gray-600 rounded-full overflow-hidden">
            <div
              className={clsx(
                'h-full transition-all duration-75 ease-out',
                volumeColor
              )}
              style={{ width: `${volumePercent}%` }}
            />
          </div>
        </div>

        {/* VAD / 录音状态指示器 */}
        <div
          className={clsx(
            'w-10 h-10 rounded-full flex items-center justify-center transition-colors',
            isRecording
              ? 'bg-red-500 animate-pulse'
              : 'bg-gray-300 dark:bg-gray-600'
          )}
          title={isRecording ? '录音中' : '语音活动检测'}
        >
          <Mic className={clsx(
            'w-5 h-5',
            isRecording ? 'text-white' : 'text-gray-500 dark:text-gray-400'
          )} />
        </div>
      </div>

      {/* ── 后处理进度条 ── */}
      {isPostProcessing && (
        <div className="px-4 py-3 bg-blue-50 dark:bg-blue-900/30 border-b border-blue-200 dark:border-blue-700">
          <div className="flex items-center gap-2 mb-1">
            <Loader2 className="w-4 h-4 animate-spin text-blue-600 dark:text-blue-400" />
            <span className="text-sm font-medium text-blue-800 dark:text-blue-300">
              后处理中...
            </span>
          </div>
          <ProgressBar
            progress={postProcessProgress}
            message={postProcessStep}
          />
        </div>
      )}

      {/* ── 状态栏 ── */}
      {realtimeState === 'error' && (
        <div className="px-4 py-2 bg-red-50 dark:bg-red-900/30 text-sm text-red-700 dark:text-red-400 border-b border-red-200 dark:border-red-700">
          连接错误。请检查后端是否运行，然后重试。
        </div>
      )}
      {realtimeState === 'done' && (
        <div className="px-4 py-2 bg-green-50 dark:bg-green-900/30 text-sm text-green-700 dark:text-green-400 border-b border-green-200 dark:border-green-700">
          处理完成！结果已保存到历史记录。
        </div>
      )}

      {/* ── 实时转写列表 ── */}
      <div
        ref={scrollRef}
        className="flex-1 mx-4 my-2 border border-gray-300 dark:border-gray-600 rounded-lg overflow-auto dark:bg-gray-900"
      >
        {realtimeSegments.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-400 dark:text-gray-500 text-sm">
            {isModelsLoading
              ? '正在加载 ASR 模型...'
              : isLoading
                ? '正在准备录音...'
                : isRecording
                  ? '等待语音输入...'
                  : !ws.modelsReady
                    ? '请先选择引擎/模型并点击「加载」按钮'
                    : canRecord
                      ? '模型已就绪，点击「开始录音」开始实时转写'
                      : '处理中...'}
          </div>
        ) : (
          <div className="p-3 space-y-2">
            {realtimeSegments.map((seg) => (
              <div
                key={seg.id}
                className={clsx(
                  'px-3 py-2 rounded-lg text-sm transition-colors duration-500',
                  seg.isPlaceholder
                    ? 'bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-600 italic'
                    : upgradedSegmentIds.has(seg.id)
                      ? 'bg-green-50 dark:bg-green-900/30 border border-green-300 dark:border-green-600'
                      : seg.isFinal
                        ? 'bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600'
                        : 'bg-yellow-50 dark:bg-yellow-900/30 border border-yellow-200 dark:border-yellow-600 italic'
                )}
              >
                <span className="text-xs text-gray-400 dark:text-gray-500 mr-2 font-mono">
                  [{formatTime(seg.startTime)}]
                </span>
                {seg.isUpgraded && !upgradedSegmentIds.has(seg.id) && (
                  <span className="text-xs text-green-600 dark:text-green-400 mr-1" title="已由高精度模型升级">
                    [升级]
                  </span>
                )}
                <span
                  className={clsx(
                    seg.isPlaceholder
                      ? 'text-gray-400 dark:text-gray-500'
                      : upgradedSegmentIds.has(seg.id)
                        ? 'text-green-800 dark:text-green-300'
                        : seg.isFinal
                          ? 'text-gray-800 dark:text-gray-100'
                          : 'text-gray-600 dark:text-gray-300'
                  )}
                >
                  {seg.text}
                </span>
                {!seg.isFinal && !seg.isPlaceholder && (
                  <span className="inline-block w-0.5 h-4 bg-yellow-500 ml-0.5 animate-pulse align-middle" />
                )}
                {seg.isPlaceholder && (
                  <Loader2 className="inline-block w-3 h-3 ml-1 animate-spin text-gray-400 dark:text-gray-500 align-middle" />
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── 底部选项 ── */}
      <div className="flex items-center gap-4 px-4 py-3 bg-gray-100 dark:bg-gray-800 flex-wrap">
        <span className="text-sm text-gray-600 dark:text-gray-300">录音结束后处理:</span>

        <label className="flex items-center gap-1.5 text-sm cursor-pointer dark:text-gray-200">
          <input
            type="checkbox"
            checked={enableNaming}
            onChange={(e) =>
              setProcessOptions({ enableNaming: e.target.checked })
            }
            disabled={controlsDisabled}
          />
          智能命名
        </label>
        <label className="flex items-center gap-1.5 text-sm cursor-pointer dark:text-gray-200">
          <input
            type="checkbox"
            checked={enableCorrection}
            onChange={(e) =>
              setProcessOptions({ enableCorrection: e.target.checked })
            }
            disabled={controlsDisabled}
          />
          错别字校正
        </label>
        <label className="flex items-center gap-1.5 text-sm cursor-pointer dark:text-gray-200">
          <input
            type="checkbox"
            checked={enableSummary}
            onChange={(e) =>
              setProcessOptions({ enableSummary: e.target.checked })
            }
            disabled={controlsDisabled}
          />
          会议总结
        </label>

        <span className="text-gray-300 dark:text-gray-600">|</span>

        <label className="flex items-center gap-1.5 text-sm cursor-pointer dark:text-gray-200" title="DeepFilterNet3 实时降噪，降低背景噪音对转写的干扰">
          <input
            type="checkbox"
            checked={enableDenoise}
            onChange={(e) => setEnableDenoise(e.target.checked)}
            disabled={controlsDisabled}
          />
          实时降噪
        </label>

        <div className="flex-1" />

        {(audio.error || inputError) && (
          <span className="text-xs text-red-500 dark:text-red-400">{audio.error || inputError}</span>
        )}

        <span className="text-xs text-gray-400 dark:text-gray-500">
          {recordingMode === 'streaming'
            ? engines.find((e) => e.id === selectedEngine)?.name || '流式识别'
            : recordingMode === 'segment'
              ? `段级转写: ${selectedFileAsr}`
              : `混合(${hybridUpgrade === 'segment' ? '段级' : '整体'}): ${engines.find((e) => e.id === selectedEngine)?.name || '流式'} + ${selectedFileAsr}`}
        </span>
      </div>
    </div>
  )
}
