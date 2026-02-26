/**
 * Audio capture hook - handles microphone, system audio, and mixed capture
 *
 * Architecture:
 *   getUserMedia → MediaStream → AudioContext → AudioWorklet → PCM chunks
 *   getDisplayMedia → system audio capture (screen sharing audio)
 *   mixed mode → ChannelMergerNode(mic→ch0, sys→ch1) → WorkletNode (stereo→mono)
 *
 * Audio graph (Google Chrome Labs standard pattern):
 *   source(s) → AudioWorkletNode → GainNode(0) → destination
 *
 *   The GainNode(gain=0) suppresses speaker output while keeping the
 *   pull-based audio renderer active. The Web Audio API uses a pull model
 *   where rendering starts at the destination node and walks backward
 *   through the graph — nodes not connected to destination may not receive
 *   input data.
 *
 * References:
 *   - https://github.com/GoogleChromeLabs/web-audio-samples (worklet-recorder)
 *   - https://developer.chrome.com/blog/audio-worklet-design-pattern
 *   - https://developer.chrome.com/blog/web-audio-autoplay
 */

import { useCallback, useRef, useState } from 'react'

export type AudioSourceType = 'microphone' | 'system_audio' | 'mixed'

interface AudioCaptureConfig {
  sourceType?: AudioSourceType
  deviceId?: string
  targetSampleRate?: number
  chunkDurationMs?: number
}

interface UseAudioCaptureReturn {
  /** Start capturing audio */
  start: (config?: AudioCaptureConfig) => Promise<void>
  /** Stop capturing and release resources */
  stop: () => void
  /** Current volume level (0-1 RMS) */
  volume: number
  /** Whether audio capture is active */
  isCapturing: boolean
  /** Error message if capture failed */
  error: string | null
}

interface UseAudioCaptureCallbacks {
  /** Called with each PCM chunk (Int16 ArrayBuffer) */
  onPCMChunk: (buffer: ArrayBuffer) => void
  /** Called when system audio track ends (user stopped sharing) */
  onSystemAudioEnded?: () => void
}

/**
 * Request system audio via getDisplayMedia.
 * Tries audio-only first, falls back to video+audio then stops the video track.
 */
async function getSystemAudioStream(): Promise<MediaStream> {
  if (!navigator.mediaDevices?.getDisplayMedia) {
    throw new Error('浏览器不支持系统音频捕获，请使用 Chrome 74+ 或 Edge')
  }

  let stream: MediaStream
  try {
    // Try audio-only first (Chrome 94+)
    stream = await navigator.mediaDevices.getDisplayMedia({
      audio: true,
      video: false,
    } as DisplayMediaStreamOptions)
  } catch {
    // Fallback: request with video, then discard the video track
    try {
      stream = await navigator.mediaDevices.getDisplayMedia({
        audio: true,
        video: true,
      })
      // Stop video track immediately — we only need audio
      stream.getVideoTracks().forEach((t) => t.stop())
    } catch (innerErr) {
      if (innerErr instanceof DOMException && innerErr.name === 'NotAllowedError') {
        throw new Error('屏幕共享权限被拒绝')
      }
      throw innerErr
    }
  }

  // Verify we actually got audio tracks
  if (stream.getAudioTracks().length === 0) {
    stream.getTracks().forEach((t) => t.stop())
    throw new Error('未获取到系统音频，请重试并勾选「共享音频」选项')
  }

  return stream
}

export function useAudioCapture(
  callbacks: UseAudioCaptureCallbacks
): UseAudioCaptureReturn {
  const [volume, setVolume] = useState(0)
  const [isCapturing, setIsCapturing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Refs for cleanup
  const audioContextRef = useRef<AudioContext | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const systemStreamRef = useRef<MediaStream | null>(null)
  const workletNodeRef = useRef<AudioWorkletNode | null>(null)
  const callbacksRef = useRef(callbacks)
  callbacksRef.current = callbacks

  const stop = useCallback(() => {
    // Stop worklet
    if (workletNodeRef.current) {
      workletNodeRef.current.port.postMessage({ type: 'stop' })
      workletNodeRef.current.disconnect()
      workletNodeRef.current = null
    }

    // Stop media stream tracks (microphone)
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop())
      mediaStreamRef.current = null
    }

    // Stop system audio stream tracks
    if (systemStreamRef.current) {
      systemStreamRef.current.getTracks().forEach((track) => track.stop())
      systemStreamRef.current = null
    }

    // Close audio context
    if (audioContextRef.current) {
      audioContextRef.current.close()
      audioContextRef.current = null
    }

    setIsCapturing(false)
    setVolume(0)
    console.log('[AudioCapture] Capture stopped')
  }, [])

  const start = useCallback(async (config?: AudioCaptureConfig) => {
    const {
      sourceType = 'microphone',
      deviceId,
      targetSampleRate = 16000,
      chunkDurationMs = 300,
    } = config || {}

    try {
      setError(null)

      // ──────────────────────────────────────────────────────────────
      // STEP 1: Create AudioContext FIRST, while still in user gesture
      // context. Chrome's autoplay policy requires AudioContext to be
      // created or resumed during a user gesture (click/tap handler).
      //
      // Reference: https://developer.chrome.com/blog/web-audio-autoplay
      // ──────────────────────────────────────────────────────────────
      const audioContext = new AudioContext()

      if (audioContext.state === 'suspended') {
        await audioContext.resume()
      }

      console.log('[AudioCapture] AudioContext created:', {
        state: audioContext.state,
        sampleRate: audioContext.sampleRate,
        baseLatency: audioContext.baseLatency,
      })

      if (audioContext.state !== 'running') {
        audioContext.close()
        throw new Error(
          `AudioContext failed to start (state: ${audioContext.state}). ` +
          'Check browser autoplay settings.'
        )
      }
      audioContextRef.current = audioContext

      // ──────────────────────────────────────────────────────────────
      // STEP 2: Acquire audio streams based on source type
      // ──────────────────────────────────────────────────────────────
      let micStream: MediaStream | null = null
      let systemStream: MediaStream | null = null

      const micConstraints: MediaStreamConstraints = {
        audio: {
          channelCount: 1,
          sampleRate: { ideal: targetSampleRate },
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          deviceId: deviceId ? { exact: deviceId } : undefined,
        },
      }

      if (sourceType === 'microphone') {
        micStream = await navigator.mediaDevices.getUserMedia(micConstraints)
      } else if (sourceType === 'system_audio') {
        systemStream = await getSystemAudioStream()
      } else if (sourceType === 'mixed') {
        // Get both streams. Start with mic (uses gesture context),
        // then system audio (triggers screen share dialog).
        micStream = await navigator.mediaDevices.getUserMedia(micConstraints)
        try {
          systemStream = await getSystemAudioStream()
        } catch (sysErr) {
          // System audio failed — fall back to mic-only with warning
          console.warn('[AudioCapture] System audio failed, mic-only:', sysErr)
          setError('系统音频获取失败，仅使用麦克风录音')
        }
      }

      // Store streams for cleanup
      if (micStream) mediaStreamRef.current = micStream
      if (systemStream) systemStreamRef.current = systemStream

      // Diagnostic logging
      const logTrack = (label: string, stream: MediaStream | null) => {
        if (!stream) return
        const track = stream.getAudioTracks()[0]
        console.log(`[AudioCapture] ${label} track:`, {
          label: track?.label,
          enabled: track?.enabled,
          muted: track?.muted,
          readyState: track?.readyState,
          settings: track?.getSettings?.(),
        })
      }
      logTrack('Microphone', micStream)
      logTrack('System audio', systemStream)

      // Listen for system audio track ending (user stops sharing)
      if (systemStream) {
        const sysTrack = systemStream.getAudioTracks()[0]
        if (sysTrack) {
          sysTrack.onended = () => {
            console.log('[AudioCapture] System audio track ended (user stopped sharing)')
            callbacksRef.current.onSystemAudioEnded?.()
          }
        }
      }

      // ──────────────────────────────────────────────────────────────
      // STEP 3: Load AudioWorklet processor
      // ──────────────────────────────────────────────────────────────
      await audioContext.audioWorklet.addModule('/audio-worklet/pcm-processor.js')

      // Determine channel count for the worklet: 2 for mixed (stereo merge), 1 otherwise
      const isMixed = micStream && systemStream
      const workletChannelCount = isMixed ? 2 : 1

      const workletNode = new AudioWorkletNode(audioContext, 'pcm-processor', {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        channelCount: workletChannelCount,
        channelCountMode: 'explicit',
        processorOptions: {
          targetSampleRate,
          chunkDurationMs,
          inputSampleRate: audioContext.sampleRate,
        },
      })
      workletNodeRef.current = workletNode

      // Handle messages from worklet
      workletNode.port.onmessage = (event) => {
        const { type } = event.data
        if (type === 'pcm-chunk') {
          callbacksRef.current.onPCMChunk(event.data.buffer)
        } else if (type === 'volume') {
          setVolume(event.data.volume)
        } else if (type === 'diagnostic') {
          console.log('[AudioWorklet Diag]', event.data)
        }
      }

      // ──────────────────────────────────────────────────────────────
      // STEP 4: Build audio graph
      //
      // Microphone only:
      //   micSource → workletNode → gainNode(0) → destination
      //
      // System audio only:
      //   sysSource → workletNode → gainNode(0) → destination
      //
      // Mixed mode:
      //   micSource → merger(ch0) ┐
      //   sysSource → merger(ch1) ┘→ workletNode → gainNode(0) → destination
      //   (WorkletNode processes 2-ch input, pcm-processor averages to mono)
      // ──────────────────────────────────────────────────────────────
      const gainNode = audioContext.createGain()
      gainNode.gain.value = 0 // mute speaker output, worklet still receives input

      if (isMixed) {
        // Mixed mode: merge mic (ch0) + system (ch1) into stereo
        const merger = audioContext.createChannelMerger(2)
        const micSource = audioContext.createMediaStreamSource(micStream!)
        const sysSource = audioContext.createMediaStreamSource(systemStream!)

        micSource.connect(merger, 0, 0)  // mic → channel 0
        sysSource.connect(merger, 0, 1)  // system → channel 1
        merger.connect(workletNode)
      } else {
        // Single source mode
        const activeStream = micStream || systemStream
        if (!activeStream) {
          audioContext.close()
          throw new Error('No audio stream available')
        }
        const source = audioContext.createMediaStreamSource(activeStream)
        source.connect(workletNode)
      }

      workletNode.connect(gainNode)
      gainNode.connect(audioContext.destination)

      setIsCapturing(true)
      console.log(`[AudioCapture] Capture started (mode: ${sourceType})`)
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      setError(message)
      console.error('[AudioCapture] Error:', err)
      // Clean up any partially acquired streams
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((t) => t.stop())
        mediaStreamRef.current = null
      }
      if (systemStreamRef.current) {
        systemStreamRef.current.getTracks().forEach((t) => t.stop())
        systemStreamRef.current = null
      }
      if (audioContextRef.current) {
        audioContextRef.current.close()
        audioContextRef.current = null
      }
    }
  }, [])

  return {
    start,
    stop,
    volume,
    isCapturing,
    error,
  }
}
