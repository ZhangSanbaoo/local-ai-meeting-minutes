import { useCallback, useEffect, useRef, useState } from 'react'
import { Mic, Square, Loader2, Monitor, Power, PowerOff } from 'lucide-react'
import { clsx } from 'clsx'
import { formatTime } from '../utils/format'
import { useAppStore } from '../stores/appStore'
import { useAudioCapture } from '../hooks/useAudioCapture'
import {
  useRealtimeWebSocket,
  type PartialSegment,
  type PostProgress,
  type FinalResult,
} from '../hooks/useRealtimeWebSocket'
import { useRecordingTimer } from '../hooks/useRecordingTimer'
import { ProgressBar } from '../components/ProgressBar'
import { getStreamingEngines, getModels, getAudioDevices } from '../api/client'
import type { ModelInfo, RealtimeSegment, StreamingEngine } from '../types'

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

  // LLM models
  const [llmModels, setLlmModels] = useState<ModelInfo[]>([])
  const [selectedLlm, setSelectedLlm] = useState('')

  // Diarization + Gender models
  const [diarizationModels, setDiarizationModels] = useState<ModelInfo[]>([])
  const [selectedDiarModel, setSelectedDiarModel] = useState('')
  const [genderModels, setGenderModels] = useState<ModelInfo[]>([])
  const [selectedGenderModel, setSelectedGenderModel] = useState('f0')

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
        const defaultLlm = res.llm_models.find((m) => m.name !== 'disabled')
        setSelectedLlm(defaultLlm?.name || 'disabled')

        setDiarizationModels(res.diarization_models)
        if (res.diarization_models.length > 0) {
          setSelectedDiarModel(res.diarization_models[0].name)
        }

        setGenderModels(res.gender_models)
        if (res.gender_models.length > 0) {
          setSelectedGenderModel(res.gender_models[0].name)
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
        }

        const existing = useAppStore
          .getState()
          .realtimeSegments.find((s) => s.id === seg.segmentId)

        if (existing) {
          updateRealtimeSegment(realtimeSeg)
        } else {
          addRealtimeSegment(realtimeSeg)
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
  }

  const audio = useAudioCapture(audioCallbacks)

  // ── Load / Unload models ──
  const handleLoadModels = useCallback(() => {
    // If not connected, connect first, then load
    if (ws.state === 'disconnected' || ws.state === 'error') {
      ws.connect()
      // Wait for connection, then load
      const check = setInterval(() => {
        const state = useAppStore.getState().realtimeState
        if (state === 'connected' || state === 'done') {
          clearInterval(check)
          ws.loadModels(selectedEngine || undefined)
        }
      }, 100)
      // Timeout after 5s
      setTimeout(() => clearInterval(check), 5000)
    } else {
      ws.loadModels(selectedEngine || undefined)
    }
  }, [ws, selectedEngine])

  const handleUnloadModels = useCallback(() => {
    if (ws.state !== 'disconnected') {
      ws.unloadModels(selectedEngine || undefined)
    }
  }, [ws, selectedEngine])

  // ── Auto-scroll transcript ──
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [realtimeSegments])

  // ── Start recording ──
  const handleStartRecording = useCallback(async () => {
    clearRealtimeSegments()
    setPostProcessProgress(0, '')
    setIsStartPending(true)

    // CRITICAL: Start audio capture FIRST while in user gesture context.
    // AudioContext.resume() requires a user gesture. If we await
    // other async operations first, the gesture context expires.
    try {
      await audio.start({
        sourceType: 'microphone',
        deviceId: selectedMicId || 'default',
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
      diarizationModel: selectedDiarModel || undefined,
      genderModel: selectedGenderModel || undefined,
    })

    setRecording({ isRecording: true })
  }, [
    ws,
    audio,
    meetingName,
    selectedMicId,
    selectedEngine,
    enableNaming,
    enableCorrection,
    enableSummary,
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
      {/* ── Row 1: 会议名称和模型选择 ── */}
      <div className="flex items-center justify-center gap-4 py-3 bg-gray-100 border-b border-gray-200">
        {/* 会议名称 */}
        <input
          type="text"
          value={meetingName}
          onChange={(e) => setMeetingName(e.target.value)}
          placeholder="会议名称（可选）"
          disabled={controlsDisabled}
          className="px-3 py-2 border border-gray-300 rounded text-sm w-40 disabled:bg-gray-200 disabled:cursor-not-allowed"
        />

        {/* ASR 引擎选择 + 加载/释放按钮 */}
        <div className="flex items-center gap-1">
          <select
            value={selectedEngine}
            onChange={(e) => setSelectedEngine(e.target.value)}
            disabled={controlsDisabled || engines.length <= 1}
            className="px-3 py-2 border border-gray-300 rounded-l text-sm disabled:bg-gray-200 disabled:cursor-not-allowed"
          >
            {engines.filter((e) => e.installed).map((engine) => (
              <option key={engine.id} value={engine.id}>
                {engine.name}
              </option>
            ))}
            {engines.length === 0 && <option>加载中...</option>}
          </select>
          <button
            onClick={handleLoadModels}
            disabled={controlsDisabled || ws.modelsReady || isModelsLoading}
            className={clsx(
              'flex items-center gap-1 px-2.5 py-2 border text-sm font-medium transition-colors',
              ws.modelsReady
                ? 'bg-green-50 border-green-300 text-green-700 cursor-default'
                : isModelsLoading
                  ? 'bg-yellow-50 border-yellow-300 text-yellow-700 cursor-wait'
                  : 'bg-white border-gray-300 text-gray-700 hover:bg-blue-50 hover:border-blue-300 hover:text-blue-700',
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
              'flex items-center gap-1 px-2.5 py-2 border border-gray-300 rounded-r text-sm font-medium transition-colors',
              ws.modelsReady && !controlsDisabled
                ? 'bg-white text-gray-700 hover:bg-red-50 hover:border-red-300 hover:text-red-700'
                : 'bg-gray-100 text-gray-400 cursor-not-allowed'
            )}
            title="释放模型显存"
          >
            <PowerOff className="w-4 h-4" />
            释放
          </button>
        </div>

        {/* LLM 模型选择 */}
        <select
          value={selectedLlm}
          onChange={(e) => setSelectedLlm(e.target.value)}
          disabled={controlsDisabled}
          className="px-3 py-2 border border-gray-300 rounded text-sm disabled:bg-gray-200 disabled:cursor-not-allowed max-w-[220px]"
        >
          {llmModels.map((m) => (
            <option key={m.name} value={m.name}>
              {m.display_name}{m.size_mb ? ` (${Math.round(m.size_mb / 1024 * 10) / 10}GB)` : ''}
            </option>
          ))}
          {llmModels.length === 0 && <option>加载中...</option>}
        </select>

        {/* 说话人分离模型 */}
        {diarizationModels.length > 0 && (
          <select
            value={selectedDiarModel}
            onChange={(e) => setSelectedDiarModel(e.target.value)}
            disabled={controlsDisabled}
            className="px-3 py-2 border border-gray-300 rounded text-sm disabled:bg-gray-200 disabled:cursor-not-allowed max-w-[180px]"
            title="说话人分离模型"
          >
            {diarizationModels.map((m) => (
              <option key={m.name} value={m.name}>
                {m.display_name}
              </option>
            ))}
          </select>
        )}

        {/* 性别检测模型 */}
        <select
          value={selectedGenderModel}
          onChange={(e) => setSelectedGenderModel(e.target.value)}
          disabled={controlsDisabled}
          className="px-3 py-2 border border-gray-300 rounded text-sm disabled:bg-gray-200 disabled:cursor-not-allowed max-w-[180px]"
          title="性别检测模型"
        >
          {genderModels.map((m) => (
            <option key={m.name} value={m.name}>
              {m.display_name}
            </option>
          ))}
        </select>

        {/* 系统音频（未实现占位） */}
        <div className="flex items-center gap-1 opacity-50">
          <Monitor className="w-4 h-4 text-gray-500" />
          <select
            disabled={true}
            className="px-2 py-2 border border-gray-300 rounded text-sm bg-gray-200 cursor-not-allowed max-w-[140px]"
            title="系统音频（待实现）"
          >
            <option>系统音频(待实现)</option>
          </select>
        </div>
      </div>

      {/* ── Row 2: 录音控制 ── */}
      <div className="flex items-center justify-center gap-4 py-3 bg-gray-100">
        {/* 麦克风选择 */}
        <div className="flex items-center gap-1">
          <Mic className="w-4 h-4 text-gray-500" />
          <select
            value={selectedMicId}
            onChange={(e) => setSelectedMicId(e.target.value)}
            disabled={controlsDisabled}
            className="px-2 py-2 border border-gray-300 rounded text-sm max-w-[280px] disabled:bg-gray-200 disabled:cursor-not-allowed"
          >
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
        <span className="text-3xl font-bold text-gray-700 font-mono w-24 text-center">
          {formatTime(timer.elapsedTime)}
        </span>

        {/* 音量条 */}
        <div className="flex flex-col gap-1">
          <span className="text-xs text-gray-500">音量</span>
          <div className="w-32 h-2.5 bg-gray-300 rounded-full overflow-hidden">
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
              : 'bg-gray-300'
          )}
          title={isRecording ? '录音中' : '语音活动检测'}
        >
          <Mic className={clsx(
            'w-5 h-5',
            isRecording ? 'text-white' : 'text-gray-500'
          )} />
        </div>
      </div>

      {/* ── 后处理进度条 ── */}
      {isPostProcessing && (
        <div className="px-4 py-3 bg-blue-50 border-b border-blue-200">
          <div className="flex items-center gap-2 mb-1">
            <Loader2 className="w-4 h-4 animate-spin text-blue-600" />
            <span className="text-sm font-medium text-blue-800">
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
        <div className="px-4 py-2 bg-red-50 text-sm text-red-700 border-b border-red-200">
          连接错误。请检查后端是否运行，然后重试。
        </div>
      )}
      {realtimeState === 'done' && (
        <div className="px-4 py-2 bg-green-50 text-sm text-green-700 border-b border-green-200">
          处理完成！结果已保存到历史记录。
        </div>
      )}

      {/* ── 实时转写列表 ── */}
      <div
        ref={scrollRef}
        className="flex-1 mx-4 my-2 border border-gray-300 rounded-lg overflow-auto"
      >
        {realtimeSegments.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-400 text-sm">
            {isModelsLoading
              ? '正在加载 ASR 模型...'
              : isLoading
                ? '正在准备录音...'
                : isRecording
                  ? '等待语音输入...'
                  : !ws.modelsReady
                    ? '请先选择 ASR 引擎并点击「加载」按钮'
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
                  'px-3 py-2 rounded-lg text-sm',
                  seg.isFinal
                    ? 'bg-white border border-gray-200'
                    : 'bg-yellow-50 border border-yellow-200 italic'
                )}
              >
                <span className="text-xs text-gray-400 mr-2 font-mono">
                  [{formatTime(seg.startTime)}]
                </span>
                <span
                  className={clsx(
                    seg.isFinal ? 'text-gray-800' : 'text-gray-600'
                  )}
                >
                  {seg.text}
                </span>
                {!seg.isFinal && (
                  <span className="inline-block w-0.5 h-4 bg-yellow-500 ml-0.5 animate-pulse align-middle" />
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── 底部选项 ── */}
      <div className="flex items-center gap-4 px-4 py-3 bg-gray-100 flex-wrap">
        <span className="text-sm text-gray-600">录音结束后处理:</span>

        <label className="flex items-center gap-1.5 text-sm cursor-pointer">
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
        <label className="flex items-center gap-1.5 text-sm cursor-pointer">
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
        <label className="flex items-center gap-1.5 text-sm cursor-pointer">
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

        <div className="flex-1" />

        {audio.error && (
          <span className="text-xs text-red-500">{audio.error}</span>
        )}

        <span className="text-xs text-gray-400">
          {engines.find((e) => e.id === selectedEngine)?.name || '流式识别'}
        </span>
      </div>
    </div>
  )
}
