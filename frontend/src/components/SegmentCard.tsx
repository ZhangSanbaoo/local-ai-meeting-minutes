import { Play, Edit2, Scissors } from 'lucide-react'
import { clsx } from 'clsx'
import { formatTime } from '../utils/format'
import type { Segment, Speaker } from '../types'

interface SegmentCardProps {
  segment: Segment
  speaker?: Speaker
  isPlaying: boolean
  isSelected?: boolean  // 用于合并模式的选中状态
  onClick?: () => void
  onEdit?: () => void
  onSpeakerClick?: () => void
  onSplit?: () => void
}

export function SegmentCard({
  segment,
  speaker,
  isPlaying,
  isSelected = false,
  onClick,
  onEdit,
  onSpeakerClick,
  onSplit,
}: SegmentCardProps) {
  const displayName = speaker?.display_name || segment.speaker_name || segment.speaker

  return (
    <div
      className={clsx(
        'flex items-start gap-2.5 p-3 rounded-lg transition-colors group',
        onClick && 'cursor-pointer',
        isSelected
          ? 'bg-green-100 ring-2 ring-green-400'
          : isPlaying
            ? 'bg-blue-100'
            : onClick
              ? 'bg-gray-50 hover:bg-blue-50'
              : 'bg-gray-50'
      )}
      onClick={onClick}
    >
      {/* 播放指示器 */}
      <div className="w-4 h-4 mt-0.5 flex-shrink-0">
        {isPlaying && <Play className="w-4 h-4 text-primary-600" fill="currentColor" />}
      </div>

      {/* 时间 */}
      <span className="text-xs text-gray-500 w-12 flex-shrink-0 mt-0.5">
        {formatTime(segment.start)}
      </span>

      {/* 说话人 */}
      {onSpeakerClick ? (
        <button
          onClick={(e) => {
            e.stopPropagation()
            onSpeakerClick()
          }}
          className="text-xs font-semibold text-primary-600 hover:text-primary-800 w-20 text-left flex-shrink-0 truncate"
          title="点击重命名"
        >
          {displayName}
        </button>
      ) : (
        <span className="text-xs font-semibold text-primary-600 w-20 text-left flex-shrink-0 truncate">
          {displayName}
        </span>
      )}

      {/* 内容 */}
      <p className="flex-1 text-sm text-gray-800 leading-relaxed">
        {segment.text}
      </p>

      {/* 操作按钮 */}
      {(onSplit || onEdit) && (
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          {onSplit && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                onSplit()
              }}
              className="p-1 text-gray-400 hover:text-orange-600"
              title="分割片段"
            >
              <Scissors className="w-4 h-4" />
            </button>
          )}
          {onEdit && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                onEdit()
              }}
              className="p-1 text-gray-400 hover:text-gray-600"
              title="编辑"
            >
              <Edit2 className="w-4 h-4" />
            </button>
          )}
        </div>
      )}
    </div>
  )
}
