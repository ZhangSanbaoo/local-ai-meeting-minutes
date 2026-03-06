import { useEffect, useRef, useState } from 'react'
import { Edit2, RefreshCw } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { ChatPanel } from './ChatPanel'

type TabType = 'summary' | 'chat'

interface SummaryPanelProps {
  summary: string
  onEdit: () => void
  onRegenerate?: () => void
  isRegenerating?: boolean
  historyId?: string | null
}

export function SummaryPanel({
  summary,
  onEdit,
  onRegenerate,
  isRegenerating,
  historyId,
}: SummaryPanelProps) {
  const [activeTab, setActiveTab] = useState<TabType>('summary')
  const [elapsed, setElapsed] = useState(0)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    if (isRegenerating) {
      setElapsed(0)
      timerRef.current = setInterval(() => setElapsed((t) => t + 1), 1000)
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [isRegenerating])

  return (
    <div className="flex flex-col h-full">
      {/* 标题栏 + Tab */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-1">
          {/* Tab 切换 */}
          <button
            onClick={() => setActiveTab('summary')}
            className={`px-3 py-1 text-sm rounded-t transition-colors ${
              activeTab === 'summary'
                ? 'bg-white dark:bg-gray-800 border border-b-0 border-gray-300 dark:border-gray-600 font-medium text-gray-800 dark:text-gray-100'
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700'
            }`}
          >
            总结
          </button>
          <button
            onClick={() => setActiveTab('chat')}
            className={`px-3 py-1 text-sm rounded-t transition-colors ${
              activeTab === 'chat'
                ? 'bg-white dark:bg-gray-800 border border-b-0 border-gray-300 dark:border-gray-600 font-medium text-gray-800 dark:text-gray-100'
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700'
            }`}
          >
            AI 对话
          </button>
        </div>

        {/* 操作按钮（仅在总结 Tab 显示） */}
        {activeTab === 'summary' && (
          <div className="flex items-center gap-1">
            {onRegenerate && (
              <button
                onClick={onRegenerate}
                disabled={isRegenerating}
                className="p-1 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 disabled:opacity-50 disabled:cursor-not-allowed"
                title="重新生成总结"
              >
                <RefreshCw className={`w-4 h-4 ${isRegenerating ? 'animate-spin' : ''}`} />
              </button>
            )}
            <button
              onClick={onEdit}
              className="p-1 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"
              title="编辑总结"
            >
              <Edit2 className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>

      {/* 重新生成进度提示 */}
      {isRegenerating && activeTab === 'summary' && (
        <div className="flex items-center gap-2 mb-2 px-3 py-2 bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-700 rounded-lg">
          <RefreshCw className="w-4 h-4 text-blue-500 animate-spin flex-shrink-0" />
          <span className="text-sm text-blue-700 dark:text-blue-400">
            正在重新生成总结... ({elapsed}s)
          </span>
        </div>
      )}

      {/* 内容区 - CSS hidden 切换保持状态 */}
      <div className={`flex-1 min-h-0 ${activeTab === 'summary' ? '' : 'hidden'}`}>
        <div className="h-full border border-gray-300 dark:border-gray-600 rounded-lg p-4 bg-white dark:bg-gray-800 overflow-auto">
          {summary ? (
            <div className="prose prose-sm dark:prose-invert max-w-none prose-headings:text-gray-800 dark:prose-headings:text-gray-100 prose-p:text-gray-600 dark:prose-p:text-gray-300 prose-li:text-gray-600 dark:prose-li:text-gray-300">
              <ReactMarkdown>{summary}</ReactMarkdown>
            </div>
          ) : (
            <p className="text-gray-400 dark:text-gray-500 text-sm italic">
              处理音频后将在此显示会议总结...
            </p>
          )}
        </div>
      </div>

      <div className={`flex-1 min-h-0 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 overflow-hidden ${activeTab === 'chat' ? '' : 'hidden'}`}>
        <ChatPanel
          historyId={historyId ?? null}
          disabled={!historyId}
        />
      </div>
    </div>
  )
}
