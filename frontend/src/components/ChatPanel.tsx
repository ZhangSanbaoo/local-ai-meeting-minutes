import { useCallback, useEffect, useRef, useState } from 'react'
import { Send, Trash2, Square, Loader2 } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { useAppStore } from '../stores/appStore'
import * as api from '../api/client'
import type { ChatMessage } from '../types'

const API_BASE = '/api'

interface ChatPanelProps {
  historyId: string | null
  disabled?: boolean
}

export function ChatPanel({ historyId, disabled }: ChatPanelProps) {
  const llmModels = useAppStore((s) => s.llmModels)

  // LLM 状态
  const [selectedModel, setSelectedModel] = useState('')
  const [llmState, setLlmState] = useState<'unloaded' | 'loading' | 'loaded'>('unloaded')
  const [loadedModelName, setLoadedModelName] = useState<string | null>(null)

  // 聊天状态
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const prevHistoryIdRef = useRef<string | null>(null)

  // 初始化：查询 LLM 状态
  useEffect(() => {
    api.getLlmStatus().then((status) => {
      if (status.state === 'loaded') {
        setLlmState('loaded')
        setLoadedModelName(status.model_name)
        if (status.model_name) setSelectedModel(status.model_name)
      }
    }).catch(() => {})
  }, [])

  // 默认选中第一个非 disabled 的 LLM 模型
  useEffect(() => {
    if (!selectedModel && llmModels.length > 0) {
      const first = llmModels.find((m) => m.name !== 'disabled')
      if (first) setSelectedModel(first.name)
    }
  }, [llmModels, selectedModel])

  // 切换 historyId 时清空消息
  useEffect(() => {
    if (prevHistoryIdRef.current !== historyId) {
      setMessages([])
      setInput('')
      prevHistoryIdRef.current = historyId
    }
  }, [historyId])

  // 自动滚动到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // 加载 LLM
  const handleLoad = useCallback(async () => {
    if (!selectedModel || llmState === 'loading') return
    setLlmState('loading')
    try {
      const result = await api.loadLlm(selectedModel)
      setLlmState('loaded')
      setLoadedModelName(result.model_name)
    } catch (err: unknown) {
      setLlmState('unloaded')
      const msg = err instanceof Error ? err.message : '加载失败'
      alert(`LLM 加载失败: ${msg}`)
    }
  }, [selectedModel, llmState])

  // 卸载 LLM
  const handleUnload = useCallback(async () => {
    if (isGenerating) return
    try {
      await api.unloadLlm()
      setLlmState('unloaded')
      setLoadedModelName(null)
    } catch (err) {
      console.error('卸载失败:', err)
    }
  }, [isGenerating])

  // 发送消息
  const handleSend = useCallback(async () => {
    const text = input.trim()
    if (!text || !historyId || isGenerating || llmState !== 'loaded') return

    const userMsg: ChatMessage = { role: 'user', content: text }
    const newMessages = [...messages, userMsg]
    setMessages(newMessages)
    setInput('')
    setIsGenerating(true)

    const history = messages.map(({ role, content }) => ({ role, content }))
    const controller = new AbortController()
    abortRef.current = controller

    const aiMsgIndex = newMessages.length
    setMessages([...newMessages, { role: 'assistant', content: '' }])

    try {
      const res = await fetch(`${API_BASE}/history/${historyId}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, history }),
        signal: controller.signal,
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail || '请求失败')
      }

      const reader = res.body?.getReader()
      if (!reader) throw new Error('无法读取响应流')

      const decoder = new TextDecoder()
      let accumulated = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const jsonStr = line.slice(6).trim()
          if (!jsonStr) continue

          try {
            const data = JSON.parse(jsonStr)
            if (data.done) break
            if (data.error) {
              accumulated += `\n[错误] ${data.error}`
              break
            }
            if (data.token) {
              accumulated += data.token
              setMessages((prev) => {
                const updated = [...prev]
                updated[aiMsgIndex] = { role: 'assistant', content: accumulated }
                return updated
              })
            }
          } catch {
            // 忽略解析错误
          }
        }
      }

      setMessages((prev) => {
        const updated = [...prev]
        updated[aiMsgIndex] = { role: 'assistant', content: accumulated }
        return updated
      })
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        setMessages((prev) => {
          const updated = [...prev]
          if (updated[aiMsgIndex] && !updated[aiMsgIndex].content) {
            updated[aiMsgIndex] = { role: 'assistant', content: '[已取消]' }
          }
          return updated
        })
      } else {
        const message = err instanceof Error ? err.message : '未知错误'
        setMessages((prev) => {
          const updated = [...prev]
          updated[aiMsgIndex] = { role: 'assistant', content: `[错误] ${message}` }
          return updated
        })
      }
    } finally {
      setIsGenerating(false)
      abortRef.current = null
    }
  }, [input, historyId, isGenerating, llmState, messages])

  const handleStop = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  const handleClear = useCallback(() => {
    if (isGenerating) return
    setMessages([])
  }, [isGenerating])

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        handleSend()
      }
    },
    [handleSend],
  )

  const noHistory = disabled || !historyId
  const canChat = !noHistory && llmState === 'loaded'
  const availableModels = llmModels.filter((m) => m.name !== 'disabled')

  return (
    <div className="flex flex-col h-full">
      {/* LLM 模型控制栏 */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-gray-200 dark:border-gray-600 bg-gray-50 dark:bg-gray-900">
        <span className="text-xs text-gray-500 dark:text-gray-400 whitespace-nowrap">LLM</span>
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          disabled={llmState === 'loaded' || llmState === 'loading'}
          className="flex-1 min-w-0 px-2 py-1 border border-gray-300 dark:border-gray-600 rounded text-xs disabled:bg-gray-100 dark:disabled:bg-gray-700 disabled:cursor-not-allowed dark:bg-gray-700 dark:text-gray-200"
        >
          {availableModels.length === 0 ? (
            <option value="">无可用模型</option>
          ) : (
            availableModels.map((m) => (
              <option key={m.name} value={m.name}>{m.display_name}</option>
            ))
          )}
        </select>

        {llmState === 'loaded' ? (
          <button
            onClick={handleUnload}
            disabled={isGenerating}
            className="px-2 py-1 text-xs border border-red-300 dark:border-red-700 text-red-600 dark:text-red-400 rounded hover:bg-red-50 dark:hover:bg-red-900/30 disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap"
          >
            卸载
          </button>
        ) : (
          <button
            onClick={handleLoad}
            disabled={llmState === 'loading' || !selectedModel}
            className="px-2 py-1 text-xs border border-blue-300 dark:border-blue-700 text-blue-600 dark:text-blue-400 rounded hover:bg-blue-50 dark:hover:bg-blue-900/30 disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap flex items-center gap-1"
          >
            {llmState === 'loading' && <Loader2 className="w-3 h-3 animate-spin" />}
            {llmState === 'loading' ? '加载中' : '加载'}
          </button>
        )}

        {/* 状态指示 */}
        <div className={`w-2 h-2 rounded-full flex-shrink-0 ${
          llmState === 'loaded' ? 'bg-green-500' :
          llmState === 'loading' ? 'bg-yellow-500 animate-pulse' :
          'bg-gray-300 dark:bg-gray-600'
        }`} title={
          llmState === 'loaded' ? `已加载: ${loadedModelName}` :
          llmState === 'loading' ? '加载中...' : '未加载'
        } />
      </div>

      {/* 消息列表 */}
      <div className="flex-1 overflow-auto p-3 space-y-3">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-400 dark:text-gray-500 text-sm text-center px-4">
            {noHistory
              ? '请先选择一条历史记录'
              : llmState !== 'loaded'
              ? '请先选择并加载 LLM 模型'
              : '基于会议内容提问，例如："会议讨论了哪些要点？"'}
          </div>
        ) : (
          messages.map((msg, i) => (
            <div
              key={i}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[85%] rounded-lg px-3 py-2 text-sm ${
                  msg.role === 'user'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200'
                }`}
              >
                {msg.role === 'assistant' ? (
                  <div className="prose prose-sm dark:prose-invert max-w-none prose-p:my-1 prose-li:my-0.5">
                    <ReactMarkdown>{msg.content || '...'}</ReactMarkdown>
                  </div>
                ) : (
                  <span className="whitespace-pre-wrap">{msg.content}</span>
                )}
              </div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* 底部输入区 */}
      <div className="border-t border-gray-200 dark:border-gray-600 p-2">
        <div className="flex items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              noHistory ? '请先选择历史记录...' :
              llmState !== 'loaded' ? '请先加载 LLM 模型...' :
              '输入问题...'
            }
            disabled={!canChat}
            rows={1}
            className="flex-1 resize-none border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-400 disabled:bg-gray-100 dark:disabled:bg-gray-700 disabled:cursor-not-allowed dark:bg-gray-700 dark:text-gray-200 dark:placeholder-gray-500"
            style={{ maxHeight: '80px' }}
          />
          <div className="flex gap-1">
            {isGenerating ? (
              <button
                onClick={handleStop}
                className="p-2 text-red-500 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/30 rounded-lg"
                title="停止生成"
              >
                <Square className="w-4 h-4" />
              </button>
            ) : (
              <button
                onClick={handleSend}
                disabled={!canChat || !input.trim()}
                className="p-2 text-blue-500 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/30 rounded-lg disabled:text-gray-300 dark:disabled:text-gray-600 disabled:cursor-not-allowed"
                title="发送 (Enter)"
              >
                <Send className="w-4 h-4" />
              </button>
            )}
            <button
              onClick={handleClear}
              disabled={!canChat || isGenerating || messages.length === 0}
              className="p-2 text-gray-400 dark:text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg disabled:text-gray-200 dark:disabled:text-gray-600 disabled:cursor-not-allowed"
              title="清空对话"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
