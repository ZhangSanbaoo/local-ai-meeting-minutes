import { X, Scissors } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

interface DialogProps {
  open: boolean
  onClose: () => void
  title: string
  children: React.ReactNode
  actions?: React.ReactNode
}

export function Dialog({ open, onClose, title, children, actions }: DialogProps) {
  const dialogRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (open) {
      // 聚焦到对话框
      dialogRef.current?.focus()
      // 禁止背景滚动
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = ''
    }
    return () => {
      document.body.style.overflow = ''
    }
  }, [open])

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* 背景遮罩 */}
      <div
        className="absolute inset-0 bg-black/50"
        onClick={onClose}
      />

      {/* 对话框 */}
      <div
        ref={dialogRef}
        tabIndex={-1}
        className="relative bg-white rounded-lg shadow-xl max-w-md w-full mx-4 max-h-[90vh] flex flex-col"
      >
        {/* 标题栏 */}
        <div className="flex items-center justify-between px-4 py-3 border-b">
          <h3 className="text-lg font-medium">{title}</h3>
          <button
            onClick={onClose}
            className="p-1 text-gray-400 hover:text-gray-600"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* 内容 */}
        <div className="flex-1 px-4 py-4 overflow-auto">
          {children}
        </div>

        {/* 操作按钮 */}
        {actions && (
          <div className="flex items-center justify-end gap-2 px-4 py-3 border-t bg-gray-50">
            {actions}
          </div>
        )}
      </div>
    </div>
  )
}

// 编辑片段对话框
interface EditSegmentDialogProps {
  open: boolean
  onClose: () => void
  segment: { speakerId: string; text: string } | null
  speakers: Array<{ id: string; name: string }>
  onSave: (speakerId: string, text: string, newSpeakerName?: string) => void
}

export function EditSegmentDialog({
  open,
  onClose,
  segment,
  speakers,
  onSave,
}: EditSegmentDialogProps) {
  const [selectedSpeaker, setSelectedSpeaker] = useState('')
  const [isNewSpeaker, setIsNewSpeaker] = useState(false)
  const [newSpeakerName, setNewSpeakerName] = useState('')
  const textRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (open && segment) {
      setSelectedSpeaker(segment.speakerId)
      setIsNewSpeaker(false)
      setNewSpeakerName('')
      if (textRef.current) textRef.current.value = segment.text
    }
  }, [open, segment])

  // 生成新的说话人 ID
  const generateNewSpeakerId = () => {
    const existingNums = speakers
      .map((s) => {
        const match = s.id.match(/^SPEAKER_(\d+)$/)
        return match ? parseInt(match[1], 10) : -1
      })
      .filter((n) => n >= 0)
    const nextNum = existingNums.length > 0 ? Math.max(...existingNums) + 1 : 0
    return `SPEAKER_${String(nextNum).padStart(2, '0')}`
  }

  const handleSpeakerChange = (value: string) => {
    if (value === '__new__') {
      setIsNewSpeaker(true)
      setSelectedSpeaker(generateNewSpeakerId())
    } else {
      setIsNewSpeaker(false)
      setSelectedSpeaker(value)
    }
  }

  const handleSave = () => {
    if (isNewSpeaker && !newSpeakerName.trim()) return
    const text = textRef.current?.value || ''
    onSave(selectedSpeaker, text, isNewSpeaker ? newSpeakerName.trim() : undefined)
    onClose()
  }

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="编辑对话"
      actions={
        <>
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800"
          >
            取消
          </button>
          <button
            onClick={handleSave}
            disabled={isNewSpeaker && !newSpeakerName.trim()}
            className="px-4 py-2 text-sm bg-primary-600 text-white rounded hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            保存
          </button>
        </>
      }
    >
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            说话人
          </label>
          <select
            value={isNewSpeaker ? '__new__' : selectedSpeaker}
            onChange={(e) => handleSpeakerChange(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-primary-500"
          >
            {speakers.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name} ({s.id})
              </option>
            ))}
            <option value="__new__">+ 新建说话人</option>
          </select>
          {isNewSpeaker && (
            <input
              type="text"
              value={newSpeakerName}
              onChange={(e) => setNewSpeakerName(e.target.value)}
              placeholder="输入新说话人的名字"
              className="w-full mt-2 px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-primary-500"
              autoFocus
            />
          )}
          {!isNewSpeaker && selectedSpeaker !== segment?.speakerId && (
            <p className="text-xs text-orange-600 mt-1">
              将把此片段重新分配给 {speakers.find(s => s.id === selectedSpeaker)?.name || selectedSpeaker}
            </p>
          )}
          {isNewSpeaker && newSpeakerName.trim() && (
            <p className="text-xs text-green-600 mt-1">
              将创建新说话人 "{newSpeakerName.trim()}" ({selectedSpeaker})
            </p>
          )}
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            内容
          </label>
          <textarea
            ref={textRef}
            rows={4}
            className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-primary-500 resize-none"
          />
        </div>
      </div>
    </Dialog>
  )
}

// 重命名说话人对话框
interface RenameSpeakerDialogProps {
  open: boolean
  onClose: () => void
  speaker: { id: string; name: string; count: number } | null
  onSave: (newName: string) => void
}

export function RenameSpeakerDialog({
  open,
  onClose,
  speaker,
  onSave,
}: RenameSpeakerDialogProps) {
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open && speaker && inputRef.current) {
      inputRef.current.value = speaker.name
      inputRef.current.select()
    }
  }, [open, speaker])

  const handleSave = () => {
    const newName = inputRef.current?.value?.trim()
    if (newName) {
      onSave(newName)
      onClose()
    }
  }

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="重命名说话人"
      actions={
        <>
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800"
          >
            取消
          </button>
          <button
            onClick={handleSave}
            className="px-4 py-2 text-sm bg-primary-600 text-white rounded hover:bg-primary-700"
          >
            确认修改
          </button>
        </>
      }
    >
      <div className="space-y-4">
        <div className="flex items-center gap-2 text-sm text-gray-600">
          <span>当前名字:</span>
          <span className="font-medium">{speaker?.name}</span>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            新名字
          </label>
          <input
            ref={inputRef}
            type="text"
            className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-primary-500"
            onKeyDown={(e) => e.key === 'Enter' && handleSave()}
          />
        </div>
        <p className="text-xs text-gray-500">
          将影响 {speaker?.count || 0} 条对话记录
        </p>
        <p className="text-xs text-orange-600">
          修改后将替换所有该说话人的显示名
        </p>
      </div>
    </Dialog>
  )
}

// 编辑总结对话框
interface EditSummaryDialogProps {
  open: boolean
  onClose: () => void
  summary: string
  onSave: (summary: string) => void
}

export function EditSummaryDialog({
  open,
  onClose,
  summary,
  onSave,
}: EditSummaryDialogProps) {
  const textRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (open && textRef.current) {
      textRef.current.value = summary
    }
  }, [open, summary])

  const handleSave = () => {
    const newSummary = textRef.current?.value || ''
    onSave(newSummary)
    onClose()
  }

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="编辑会议总结"
      actions={
        <>
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800"
          >
            取消
          </button>
          <button
            onClick={handleSave}
            className="px-4 py-2 text-sm bg-primary-600 text-white rounded hover:bg-primary-700"
          >
            保存
          </button>
        </>
      }
    >
      <textarea
        ref={textRef}
        rows={15}
        className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-primary-500 resize-none font-mono text-sm"
        placeholder="支持 Markdown 格式..."
      />
    </Dialog>
  )
}

// 重命名历史记录对话框
interface RenameHistoryDialogProps {
  open: boolean
  onClose: () => void
  historyId: string
  currentName: string
  onSave: (newName: string) => void
}

export function RenameHistoryDialog({
  open,
  onClose,
  historyId,
  currentName,
  onSave,
}: RenameHistoryDialogProps) {
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open && inputRef.current) {
      // 提取名称部分（去掉时间戳）
      const parts = currentName.split('_')
      let nameOnly = currentName
      if (parts.length >= 3) {
        // 假设格式: name_YYYYMMDD_HHMMSS
        const last = parts[parts.length - 1]
        const secondLast = parts[parts.length - 2]
        if (last.length === 6 && secondLast.length === 8) {
          nameOnly = parts.slice(0, -2).join('_')
        }
      }
      inputRef.current.value = nameOnly
      inputRef.current.select()
    }
  }, [open, currentName])

  const handleSave = () => {
    const newName = inputRef.current?.value?.trim()
    if (newName) {
      onSave(newName)
    }
  }

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="重命名会议记录"
      actions={
        <>
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800"
          >
            取消
          </button>
          <button
            onClick={handleSave}
            className="px-4 py-2 text-sm bg-primary-600 text-white rounded hover:bg-primary-700"
          >
            确认重命名
          </button>
        </>
      }
    >
      <div className="space-y-4">
        <div className="flex items-center gap-2 text-sm text-gray-600">
          <span>当前名称:</span>
          <span className="font-medium truncate">{currentName}</span>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            新名称
          </label>
          <input
            ref={inputRef}
            type="text"
            className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-primary-500"
            onKeyDown={(e) => e.key === 'Enter' && handleSave()}
            placeholder="输入新的会议名称"
          />
        </div>
        <p className="text-xs text-gray-500">
          重命名后会保留原始时间戳
        </p>
      </div>
    </Dialog>
  )
}

// 分割片段对话框
interface SplitSegmentDialogProps {
  open: boolean
  onClose: () => void
  segment: { id: number; text: string; speaker: string } | null
  speakers: Array<{ id: string; name: string }>
  onSplit: (splitPosition: number, newSpeaker?: string) => void
}

export function SplitSegmentDialog({
  open,
  onClose,
  segment,
  speakers,
  onSplit,
}: SplitSegmentDialogProps) {
  const [splitPosition, setSplitPosition] = useState<number | null>(null)
  const [newSpeaker, setNewSpeaker] = useState<string>('')
  const textRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (open && segment) {
      setSplitPosition(null)
      setNewSpeaker('')
    }
  }, [open, segment])

  const handleTextClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!textRef.current || !segment) return

    const selection = window.getSelection()
    if (selection && selection.rangeCount > 0) {
      const range = selection.getRangeAt(0)
      // 获取选区的起始位置作为分割点
      if (range.startContainer.parentElement === textRef.current ||
          range.startContainer === textRef.current) {
        const pos = range.startOffset
        if (pos > 0 && pos < segment.text.length) {
          setSplitPosition(pos)
        }
      }
    }
  }

  const handleSplit = () => {
    if (splitPosition !== null && splitPosition > 0) {
      onSplit(splitPosition, newSpeaker || undefined)
      onClose()
    }
  }

  const text1 = segment && splitPosition ? segment.text.slice(0, splitPosition) : ''
  const text2 = segment && splitPosition ? segment.text.slice(splitPosition) : ''

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="分割对话片段"
      actions={
        <>
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800"
          >
            取消
          </button>
          <button
            onClick={handleSplit}
            disabled={splitPosition === null}
            className="px-4 py-2 text-sm bg-primary-600 text-white rounded hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
          >
            <Scissors className="w-4 h-4" />
            分割
          </button>
        </>
      }
    >
      <div className="space-y-4">
        <p className="text-sm text-gray-600">
          点击文本中要分割的位置，或选中文字确定分割点
        </p>

        {/* 原始文本，点击选择分割位置 */}
        <div
          ref={textRef}
          onClick={handleTextClick}
          className="p-3 bg-gray-50 rounded border cursor-text text-sm leading-relaxed select-text"
          style={{ minHeight: '80px' }}
        >
          {segment?.text}
        </div>

        {/* 分割预览 */}
        {splitPosition !== null && (
          <div className="space-y-2">
            <p className="text-sm font-medium text-gray-700">分割预览：</p>
            <div className="p-2 bg-blue-50 rounded border border-blue-200 text-sm">
              <span className="text-xs text-blue-600 font-medium">[片段 1 - {segment?.speaker}]</span>
              <p className="mt-1">{text1}</p>
            </div>
            <div className="p-2 bg-green-50 rounded border border-green-200 text-sm">
              <span className="text-xs text-green-600 font-medium">[片段 2]</span>
              <p className="mt-1">{text2}</p>
            </div>
          </div>
        )}

        {/* 选择第二段的说话人 */}
        {splitPosition !== null && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              片段 2 的说话人
            </label>
            <select
              value={newSpeaker}
              onChange={(e) => setNewSpeaker(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              <option value="">保持原说话人 ({segment?.speaker})</option>
              {speakers.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
          </div>
        )}

        <p className="text-xs text-gray-500">
          提示：分割后的时间会按文本比例自动计算
        </p>
      </div>
    </Dialog>
  )
}
