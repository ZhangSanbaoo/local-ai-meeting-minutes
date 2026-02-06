import { useCallback, useEffect, useRef, useState } from 'react'

interface UseAudioPlayerProps {
  src: string | null
  onTimeUpdate?: (currentTime: number) => void
  onPlayStateChange?: (isPlaying: boolean) => void
}

export function useAudioPlayer({ src, onTimeUpdate, onPlayStateChange }: UseAudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const onTimeUpdateRef = useRef(onTimeUpdate)
  const onPlayStateChangeRef = useRef(onPlayStateChange)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [isLoaded, setIsLoaded] = useState(false)

  // 保持回调引用最新
  useEffect(() => {
    onTimeUpdateRef.current = onTimeUpdate
  }, [onTimeUpdate])

  useEffect(() => {
    onPlayStateChangeRef.current = onPlayStateChange
  }, [onPlayStateChange])

  // 创建音频元素
  useEffect(() => {
    if (!src) {
      setIsLoaded(false)
      setDuration(0)
      setCurrentTime(0)
      return
    }

    const audio = new Audio(src)
    audioRef.current = audio

    const handleLoadedMetadata = () => {
      setDuration(audio.duration)
      setIsLoaded(true)
    }

    const handleTimeUpdate = () => {
      setCurrentTime(audio.currentTime)
      onTimeUpdateRef.current?.(audio.currentTime)
    }

    const handleEnded = () => {
      setIsPlaying(false)
      setCurrentTime(0)
      onPlayStateChangeRef.current?.(false)
    }

    const handlePlay = () => {
      setIsPlaying(true)
      onPlayStateChangeRef.current?.(true)
    }
    const handlePause = () => {
      setIsPlaying(false)
      onPlayStateChangeRef.current?.(false)
    }

    const handleError = (e: Event) => {
      console.error('Audio error:', e)
      setIsLoaded(false)
    }

    audio.addEventListener('loadedmetadata', handleLoadedMetadata)
    audio.addEventListener('timeupdate', handleTimeUpdate)
    audio.addEventListener('ended', handleEnded)
    audio.addEventListener('play', handlePlay)
    audio.addEventListener('pause', handlePause)
    audio.addEventListener('error', handleError)

    return () => {
      audio.pause()
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata)
      audio.removeEventListener('timeupdate', handleTimeUpdate)
      audio.removeEventListener('ended', handleEnded)
      audio.removeEventListener('play', handlePlay)
      audio.removeEventListener('pause', handlePause)
      audio.removeEventListener('error', handleError)
      audioRef.current = null
    }
  }, [src]) // 只依赖 src

  const play = useCallback(() => {
    audioRef.current?.play()
  }, [])

  const pause = useCallback(() => {
    audioRef.current?.pause()
  }, [])

  const toggle = useCallback(() => {
    if (isPlaying) {
      pause()
    } else {
      play()
    }
  }, [isPlaying, play, pause])

  const seek = useCallback((time: number) => {
    if (audioRef.current) {
      audioRef.current.currentTime = time
      setCurrentTime(time)
    }
  }, [])

  const seekAndPlay = useCallback((time: number) => {
    if (audioRef.current) {
      audioRef.current.currentTime = time
      setCurrentTime(time)
      audioRef.current.play()
    }
  }, [])

  return {
    isPlaying,
    currentTime,
    duration,
    isLoaded,
    play,
    pause,
    toggle,
    seek,
    seekAndPlay,
  }
}
