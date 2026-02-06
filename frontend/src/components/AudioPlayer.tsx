import { Play, Pause } from 'lucide-react'
import { useCallback, useEffect, useRef } from 'react'
import { useAudioPlayer } from '../hooks/useAudioPlayer'
import { formatTime } from '../utils/format'

interface AudioPlayerProps {
  src: string | null
  onTimeUpdate?: (currentTime: number) => void
  onSeek?: (time: number) => void
  onPlayStateChange?: (isPlaying: boolean) => void
  seekTo?: number | null  // 外部控制跳转的时间点
  seekId?: number         // 跳转请求的唯一 ID，变化时触发跳转
}

export function AudioPlayer({ src, onTimeUpdate, onSeek, onPlayStateChange, seekTo, seekId }: AudioPlayerProps) {
  const {
    isPlaying,
    currentTime,
    duration,
    isLoaded,
    toggle,
    seek,
    seekAndPlay,
  } = useAudioPlayer({ src, onTimeUpdate, onPlayStateChange })

  const lastSeekIdRef = useRef<number | undefined>(undefined)
  const pendingSeekRef = useRef<number | null>(null)

  // 响应外部跳转请求
  useEffect(() => {
    if (seekId !== undefined && seekId !== lastSeekIdRef.current && seekTo !== null && seekTo !== undefined) {
      lastSeekIdRef.current = seekId
      if (isLoaded) {
        seekAndPlay(seekTo)
      } else {
        // 音频未加载，暂存跳转请求
        pendingSeekRef.current = seekTo
      }
    }
  }, [seekId, seekTo, isLoaded, seekAndPlay])

  // 音频加载完成后执行暂存的跳转
  useEffect(() => {
    if (isLoaded && pendingSeekRef.current !== null) {
      seekAndPlay(pendingSeekRef.current)
      pendingSeekRef.current = null
    }
  }, [isLoaded, seekAndPlay])

  const handleSliderChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const time = parseFloat(e.target.value)
      seek(time)
      onSeek?.(time)
    },
    [seek, onSeek]
  )

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0

  return (
    <div className="flex items-center gap-3 px-3 py-2 border border-gray-300 rounded-lg bg-white">
      {/* 播放/暂停按钮 */}
      <button
        onClick={toggle}
        disabled={!isLoaded}
        className="flex items-center justify-center w-9 h-9 rounded-full bg-primary-600 text-white hover:bg-primary-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
      >
        {isPlaying ? (
          <Pause className="w-4 h-4" />
        ) : (
          <Play className="w-4 h-4 ml-0.5" />
        )}
      </button>

      {/* 进度条 */}
      <div className="flex-1 relative">
        <input
          type="range"
          min={0}
          max={duration || 100}
          step={0.1}
          value={currentTime}
          onChange={handleSliderChange}
          disabled={!isLoaded}
          className="audio-slider w-full disabled:cursor-not-allowed disabled:opacity-50"
          style={{
            background: isLoaded
              ? `linear-gradient(to right, #2563eb ${progress}%, #e5e7eb ${progress}%)`
              : '#e5e7eb',
          }}
        />
      </div>

      {/* 时间显示 */}
      <span className="text-xs text-gray-500 font-mono w-20 text-right">
        {formatTime(currentTime)} / {formatTime(duration)}
      </span>

      {/* 状态指示 */}
      <span
        className={`text-xs ${isLoaded ? 'text-green-500' : 'text-gray-400'}`}
      >
        {isLoaded ? '●' : '○'}
      </span>
    </div>
  )
}
