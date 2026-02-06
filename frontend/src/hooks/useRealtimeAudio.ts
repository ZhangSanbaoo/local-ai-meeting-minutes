import { useCallback, useEffect, useRef, useState } from 'react'

interface UseRealtimeAudioReturn {
  isActive: boolean
  volume: number  // 0-1 normalized volume
  start: () => Promise<void>
  stop: () => void
}

/**
 * 实时音频可视化 - 使用 Web Audio API
 * 提供 60fps 平滑音量更新，完全独立于后端
 */
export function useRealtimeAudio(): UseRealtimeAudioReturn {
  const [isActive, setIsActive] = useState(false)
  const [volume, setVolume] = useState(0)

  // Refs
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const rafIdRef = useRef<number | null>(null)
  const dataArrayRef = useRef<Float32Array | null>(null)

  // 动画循环 - 60fps 更新音量
  const updateVolume = useCallback(() => {
    if (!analyserRef.current || !dataArrayRef.current) {
      rafIdRef.current = requestAnimationFrame(updateVolume)
      return
    }

    // 获取时域数据（波形）- 这是测量音量的正确方法
    analyserRef.current.getFloatTimeDomainData(dataArrayRef.current)

    // 计算 RMS（均方根）
    let sum = 0
    const data = dataArrayRef.current
    for (let i = 0; i < data.length; i++) {
      sum += data[i] * data[i]
    }
    const rms = Math.sqrt(sum / data.length)

    // 转换为 0-1 范围（RMS 通常在 0-0.5 之间，放大一点）
    const normalizedVolume = Math.min(1, rms * 3)

    setVolume(normalizedVolume)

    // 继续动画循环
    rafIdRef.current = requestAnimationFrame(updateVolume)
  }, [])

  const start = useCallback(async () => {
    try {
      // 请求麦克风权限
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      })
      streamRef.current = stream

      // 创建音频上下文
      const audioContext = new AudioContext()
      audioContextRef.current = audioContext

      // 处理浏览器自动播放策略 - 需要用户交互后才能启动
      if (audioContext.state === 'suspended') {
        await audioContext.resume()
      }

      // 创建分析器节点
      const analyser = audioContext.createAnalyser()
      analyser.fftSize = 2048  // 更大的 FFT 以获得更准确的数据
      analyser.smoothingTimeConstant = 0.5
      analyserRef.current = analyser

      // 创建数据数组
      dataArrayRef.current = new Float32Array(analyser.fftSize)

      // 连接麦克风到分析器
      const source = audioContext.createMediaStreamSource(stream)
      source.connect(analyser)
      sourceRef.current = source

      // 启动动画循环
      setIsActive(true)
      rafIdRef.current = requestAnimationFrame(updateVolume)

      console.log('Audio visualization started')

    } catch (err) {
      console.error('启动音频可视化失败:', err)
      throw err
    }
  }, [updateVolume])

  const stop = useCallback(() => {
    // 停止动画循环
    if (rafIdRef.current !== null) {
      cancelAnimationFrame(rafIdRef.current)
      rafIdRef.current = null
    }

    // 断开音频节点
    if (sourceRef.current) {
      sourceRef.current.disconnect()
      sourceRef.current = null
    }

    // 关闭音频上下文
    if (audioContextRef.current) {
      audioContextRef.current.close()
      audioContextRef.current = null
    }

    // 停止媒体流
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }

    // 重置状态
    analyserRef.current = null
    dataArrayRef.current = null
    setIsActive(false)
    setVolume(0)

    console.log('Audio visualization stopped')
  }, [])

  // 卸载时清理
  useEffect(() => {
    return () => {
      stop()
    }
  }, [stop])

  return {
    isActive,
    volume,
    start,
    stop,
  }
}
