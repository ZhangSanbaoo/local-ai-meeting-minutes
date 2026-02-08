/**
 * WebSocket hook for realtime streaming communication
 *
 * Handles:
 *   - Connection lifecycle (connect/disconnect)
 *   - Recording control (start/stop)
 *   - Binary PCM data transmission
 *   - JSON message reception and dispatch
 */

import { useCallback, useRef, useState } from 'react'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type RealtimeState =
  | 'disconnected'
  | 'connecting'
  | 'connected'
  | 'recording'
  | 'post_processing'
  | 'done'
  | 'error'

export interface PartialSegment {
  text: string
  isFinal: boolean
  segmentId: number
  startTime: number
  endTime: number
}

export interface PostProgress {
  step: string
  progress: number
  overallProgress: number
  message: string
}

export interface FinalResult {
  result: unknown
  historyId: string | null
  message?: string
}

export interface RecordingConfig {
  meetingName?: string
  enableNaming?: boolean
  enableCorrection?: boolean
  enableSummary?: boolean
  asrEngine?: string
  diarizationModel?: string
  genderModel?: string
}

interface UseRealtimeWebSocketCallbacks {
  onPartial?: (segment: PartialSegment) => void
  onPostProgress?: (progress: PostProgress) => void
  onFinalResult?: (result: FinalResult) => void
  onError?: (message: string, recoverable: boolean) => void
  onStateChange?: (state: RealtimeState) => void
  onModelsReady?: (engine: string, loadTime: number) => void
}

interface UseRealtimeWebSocketReturn {
  /** Connect to the WebSocket server */
  connect: () => void
  /** Disconnect from the server */
  disconnect: () => void
  /** Start recording (sends start_recording message) */
  startRecording: (config?: RecordingConfig) => void
  /** Stop recording (sends stop_recording message) */
  stopRecording: () => void
  /** Send binary PCM audio data */
  sendAudioChunk: (pcmBuffer: ArrayBuffer) => void
  /** Request model loading */
  loadModels: (asrEngine?: string) => void
  /** Request model unloading (free GPU memory) */
  unloadModels: (asrEngine?: string) => void
  /** Current connection state */
  state: RealtimeState
  /** Whether ASR models are loaded and ready */
  modelsReady: boolean
  /** Whether models are currently being loaded */
  modelsLoading: boolean
  /** Session ID from server */
  sessionId: string | null
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useRealtimeWebSocket(
  callbacks: UseRealtimeWebSocketCallbacks
): UseRealtimeWebSocketReturn {
  const [state, setState] = useState<RealtimeState>('disconnected')
  const [modelsReady, setModelsReady] = useState(false)
  const [modelsLoading, setModelsLoading] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const callbacksRef = useRef(callbacks)
  callbacksRef.current = callbacks

  const updateState = useCallback((newState: RealtimeState) => {
    setState(newState)
    callbacksRef.current.onStateChange?.(newState)
  }, [])

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    updateState('connecting')

    // Build WebSocket URL (works with Vite proxy in dev)
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const url = `${protocol}//${host}/api/ws/realtime`

    const ws = new WebSocket(url)
    ws.binaryType = 'arraybuffer'
    wsRef.current = ws

    ws.onopen = () => {
      console.log('[WS] Connected')
    }

    ws.onmessage = (event) => {
      // Binary frame - not expected from server, ignore
      if (event.data instanceof ArrayBuffer) return

      try {
        const data = JSON.parse(event.data)
        handleMessage(data)
      } catch (err) {
        console.error('[WS] Failed to parse message:', err)
      }
    }

    ws.onclose = (event) => {
      console.log('[WS] Disconnected:', event.code, event.reason)
      wsRef.current = null
      updateState('disconnected')
    }

    ws.onerror = (event) => {
      console.error('[WS] Error:', event)
      updateState('error')
      callbacksRef.current.onError?.('WebSocket 连接错误', true)
    }

    function handleMessage(data: Record<string, unknown>) {
      const type = data.type as string

      switch (type) {
        case 'connected':
          updateState('connected')
          break

        case 'models_ready':
          setModelsReady(true)
          setModelsLoading(false)
          callbacksRef.current.onModelsReady?.(
            data.engine as string,
            data.load_time as number
          )
          break

        case 'models_unloaded':
          setModelsReady(false)
          setModelsLoading(false)
          break

        case 'status':
          // Model loading status
          break

        case 'recording_started':
          setSessionId(data.session_id as string)
          updateState('recording')
          break

        case 'partial':
          callbacksRef.current.onPartial?.({
            text: data.text as string,
            isFinal: data.is_final as boolean,
            segmentId: data.segment_id as number,
            startTime: data.start_time as number,
            endTime: data.end_time as number,
          })
          break

        case 'recording_stopped':
          updateState('post_processing')
          break

        case 'post_progress':
          callbacksRef.current.onPostProgress?.({
            step: data.step as string,
            progress: data.progress as number,
            overallProgress: data.overall_progress as number,
            message: data.message as string,
          })
          break

        case 'final_result':
          callbacksRef.current.onFinalResult?.({
            result: data.result,
            historyId: (data.history_id as string) || null,
            message: data.message as string | undefined,
          })
          updateState('done')
          break

        case 'error':
          callbacksRef.current.onError?.(
            data.message as string,
            (data.recoverable as boolean) ?? true
          )
          break

        default:
          console.warn('[WS] Unknown message type:', type)
      }
    }
  }, [updateState])

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    updateState('disconnected')
    setModelsReady(false)
    setModelsLoading(false)
    setSessionId(null)
  }, [updateState])

  const startRecording = useCallback((config?: RecordingConfig) => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      callbacksRef.current.onError?.('WebSocket 未连接', true)
      return
    }

    ws.send(
      JSON.stringify({
        type: 'start_recording',
        config: {
          meeting_name: config?.meetingName || 'realtime',
          enable_naming: config?.enableNaming ?? true,
          enable_correction: config?.enableCorrection ?? true,
          enable_summary: config?.enableSummary ?? true,
          asr_engine: config?.asrEngine,
          diarization_model: config?.diarizationModel,
          gender_model: config?.genderModel,
        },
      })
    )
  }, [])

  const stopRecording = useCallback(() => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return

    ws.send(JSON.stringify({ type: 'stop_recording' }))
  }, [])

  const sendAudioChunk = useCallback((pcmBuffer: ArrayBuffer) => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return

    ws.send(pcmBuffer)
  }, [])

  const loadModels = useCallback((asrEngine?: string) => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return

    setModelsReady(false)
    setModelsLoading(true)
    ws.send(
      JSON.stringify({
        type: 'preload_models',
        config: { asr_engine: asrEngine },
      })
    )
  }, [])

  const unloadModels = useCallback((asrEngine?: string) => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return

    ws.send(
      JSON.stringify({
        type: 'unload_models',
        config: { asr_engine: asrEngine },
      })
    )
  }, [])

  return {
    connect,
    disconnect,
    startRecording,
    stopRecording,
    sendAudioChunk,
    loadModels,
    unloadModels,
    state,
    modelsReady,
    modelsLoading,
    sessionId,
  }
}
