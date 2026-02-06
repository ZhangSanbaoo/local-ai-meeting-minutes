import { Edit2, RefreshCw } from 'lucide-react'
import ReactMarkdown from 'react-markdown'

interface SummaryPanelProps {
  summary: string
  onEdit: () => void
  onRegenerate?: () => void
  isRegenerating?: boolean
}

export function SummaryPanel({ summary, onEdit, onRegenerate, isRegenerating }: SummaryPanelProps) {
  return (
    <div className="flex flex-col h-full">
      {/* 标题栏 */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-green-600">📋</span>
          <span className="text-sm font-medium">会议总结</span>
        </div>
        <div className="flex items-center gap-1">
          {onRegenerate && (
            <button
              onClick={onRegenerate}
              disabled={isRegenerating}
              className="p-1 text-gray-500 hover:text-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
              title="重新生成总结"
            >
              <RefreshCw className={`w-4 h-4 ${isRegenerating ? 'animate-spin' : ''}`} />
            </button>
          )}
          <button
            onClick={onEdit}
            className="p-1 text-gray-500 hover:text-gray-700"
            title="编辑总结"
          >
            <Edit2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* 内容区 */}
      <div className="flex-1 border border-gray-300 rounded-lg p-4 bg-white overflow-auto">
        {summary ? (
          <div className="prose prose-sm max-w-none prose-headings:text-gray-800 prose-p:text-gray-600 prose-li:text-gray-600">
            <ReactMarkdown>{summary}</ReactMarkdown>
          </div>
        ) : (
          <p className="text-gray-400 text-sm italic">
            处理音频后将在此显示会议总结...
          </p>
        )}
      </div>
    </div>
  )
}
