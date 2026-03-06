import { Play, Edit2, Scissors, Trash2 } from 'lucide-react'
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
  onDelete?: () => void
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
  onDelete,
}: SegmentCardProps) {
  const displayName = speaker?.display_name || segment.speaker_name || segment.speaker

  return (
    <div
      className={clsx(
        'flex items-start gap-2.5 p-3 rounded-lg transition-colors group',
        onClick && 'cursor-pointer',
        isSelected
          ? 'bg-green-100 dark:bg-green-900/30 ring-2 ring-green-400 dark:ring-green-600'
          : isPlaying
            ? 'bg-blue-100 dark:bg-blue-900/30'
            : onClick
              ? 'bg-gray-50 dark:bg-gray-800 hover:bg-blue-50 dark:hover:bg-blue-900/20'
              : 'bg-gray-50 dark:bg-gray-800'
      )}
      onClick={onClick}
    >
      {/* 播放指示器 */}
      <div className="w-4 h-4 mt-0.5 flex-shrink-0">
        {isPlaying && <Play className="w-4 h-4 text-primary-600" fill="currentColor" />}
      </div>

      {/* 时间 */}
      <span className="text-xs text-gray-500 dark:text-gray-400 w-12 flex-shrink-0 mt-0.5">
        {formatTime(segment.start)}
      </span>

      {/* 说话人 */}
      {onSpeakerClick ? (
        <button
          onClick={(e) => {
            e.stopPropagation()
            onSpeakerClick()
          }}
          className="text-xs font-semibold text-primary-600 hover:text-primary-800 dark:text-blue-400 dark:hover:text-blue-300 w-20 text-left flex-shrink-0 truncate"
          title="点击重命名"
        >
          {displayName}
        </button>
      ) : (
        <span className="text-xs font-semibold text-primary-600 dark:text-blue-400 w-20 text-left flex-shrink-0 truncate">
          {displayName}
        </span>
      )}

      {/* 内容 */}
      <p className="flex-1 text-sm text-gray-800 dark:text-gray-200 leading-relaxed">
        {segment.text}
      </p>

      {/* 操作按钮 */}
      {(onSplit || onEdit || onDelete) && (
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          {onSplit && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                onSplit()
              }}
              className="p-1 text-gray-400 dark:text-gray-500 hover:text-orange-600 dark:hover:text-orange-400"
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
              className="p-1 text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300"
              title="编辑"
            >
              <Edit2 className="w-4 h-4" />
            </button>
          )}
          {onDelete && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                onDelete()
              }}
              className="p-1 text-gray-400 dark:text-gray-500 hover:text-red-500 dark:hover:text-red-400"
              title="删除片段"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}
        </div>
      )}
    </div>
  )
}
