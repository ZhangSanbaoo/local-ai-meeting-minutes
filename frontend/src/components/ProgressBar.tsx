import { clsx } from 'clsx'

interface ProgressBarProps {
  progress: number // 0-1
  message?: string
  className?: string
}

export function ProgressBar({ progress, message, className }: ProgressBarProps) {
  const percentage = Math.round(progress * 100)

  return (
    <div className={clsx('flex items-center gap-3', className)}>
      <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
        <div
          className="h-full bg-primary-600 transition-all duration-300"
          style={{ width: `${percentage}%` }}
        />
      </div>
      <span className="text-xs text-gray-500 w-10 text-right">{percentage}%</span>
      {message && (
        <span className="text-xs text-gray-600 truncate max-w-[150px]">
          {message}
        </span>
      )}
    </div>
  )
}
