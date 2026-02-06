/**
 * Audio capture hook - handles microphone access and PCM extraction
 *
 * Architecture:
 *   getUserMedia → MediaStream → AudioContext → AudioWorklet → PCM chunks
 *
 * Audio graph (Google Chrome Labs standard pattern):
 *   MediaStreamSource → AudioWorkletNode → GainNode(0) → destination
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
 *
 * Future system audio support:
 *   getDisplayMedia → ChannelMergerNode → same AudioWorklet pipeline
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
  const workletNodeRef = useRef<AudioWorkletNode | null>(null)
  const callbacksRef = useRef(callbacks)
  callbacksRef.current = callbacks

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
      // Performing async operations like getUserMedia() before this
      // causes the gesture context to expire.
      //
      // Reference: https://developer.chrome.com/blog/web-audio-autoplay
      // ──────────────────────────────────────────────────────────────
      const audioContext = new AudioContext()

      // resume() synchronously initiated within user gesture
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
      // STEP 2: Get microphone stream (async, but AudioContext is
      // already running at this point)
      // ──────────────────────────────────────────────────────────────
      const constraints: MediaStreamConstraints = {
        audio: {
          channelCount: 1,
          sampleRate: { ideal: targetSampleRate },
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          deviceId: deviceId ? { exact: deviceId } : undefined,
        },
      }

      let stream: MediaStream
      if (sourceType === 'microphone') {
        stream = await navigator.mediaDevices.getUserMedia(constraints)
      } else {
        // Future: system_audio and mixed modes
        audioContext.close()
        throw new Error(`Audio source type '${sourceType}' not yet implemented`)
      }

      mediaStreamRef.current = stream

      // Diagnostic: log track info
      const audioTrack = stream.getAudioTracks()[0]
      console.log('[AudioCapture] Audio track:', {
        label: audioTrack?.label,
        enabled: audioTrack?.enabled,
        muted: audioTrack?.muted,
        readyState: audioTrack?.readyState,
        settings: audioTrack?.getSettings?.(),
      })

      // ──────────────────────────────────────────────────────────────
      // STEP 3: Load AudioWorklet processor
      // ──────────────────────────────────────────────────────────────
      await audioContext.audioWorklet.addModule('/audio-worklet/pcm-processor.js')

      // Create WorkletNode
      const workletNode = new AudioWorkletNode(audioContext, 'pcm-processor', {
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
      // STEP 4: Build audio graph (Google Chrome Labs standard pattern)
      //
      //   source → workletNode → gainNode(0) → destination
      //
      // The GainNode(gain=0) suppresses audible output while keeping
      // the pull-based audio renderer active. Without connection to
      // destination, the renderer may not call process() on the worklet.
      //
      // Reference: GoogleChromeLabs/web-audio-samples/worklet-recorder
      // ──────────────────────────────────────────────────────────────
      const source = audioContext.createMediaStreamSource(stream)
      const gainNode = audioContext.createGain()
      gainNode.gain.value = 0 // mute speaker output, worklet still receives input

      source.connect(workletNode)
      workletNode.connect(gainNode)
      gainNode.connect(audioContext.destination)

      setIsCapturing(true)
      console.log('[AudioCapture] Capture started successfully')
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      setError(message)
      console.error('[AudioCapture] Error:', err)
    }
  }, [])

  const stop = useCallback(() => {
    // Stop worklet
    if (workletNodeRef.current) {
      workletNodeRef.current.port.postMessage({ type: 'stop' })
      workletNodeRef.current.disconnect()
      workletNodeRef.current = null
    }

    // Stop media stream tracks
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop())
      mediaStreamRef.current = null
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

  return {
    start,
    stop,
    volume,
    isCapturing,
    error,
  }
}
