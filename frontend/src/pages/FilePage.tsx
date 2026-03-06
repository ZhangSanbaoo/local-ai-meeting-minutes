import { useCallback, useEffect, useRef, useState } from 'react'
import { Upload, Play, FileText, Code, FileDown, RefreshCw, Pencil, Trash2, Merge } from 'lucide-react'
import {
  AudioPlayer,
  SegmentCard,
  SummaryPanel,
  ProgressBar,
  EditSegmentDialog,
  RenameSpeakerDialog,
  EditSummaryDialog,
  RenameHistoryDialog,
  SplitSegmentDialog,
} from '../components'
import { useAppStore } from '../stores/appStore'
import * as api from '../api/client'
import type { Segment } from '../types'

export function FilePage() {
  const {
    // 模型
    asrModels,
    llmModels,
    genderModels,
    selectedAsrModel,
    selectedLlmModel,
    selectedGenderModel,
    setSelectedAsrModel,
    setSelectedLlmModel,
    setSelectedGenderModel,

    // 选项
    enableNaming,
    enableCorrection,
    enableSummary,
    enhanceMode,
    setProcessOptions,

    // 任务状态
    currentJobId,
    isProcessing,
    progress,
    progressMessage,
    setCurrentJob,
    setProcessing,
    setProgress,

    // 结果
    segments,
    speakers,
    summary,
    audioUrl,
    audioOriginalUrl,
    useOriginalAudio,
    toggleAudioSource,
    setResult,
    deleteSegment,
    updateSegmentText,
    updateSpeakerName,
    updateSummary,

    // 播放
    isPlaying,
    currentSegmentId,
    setPlayback,

    // 历史
    historyItems,
    setHistoryItems,
    pendingHistoryId,
    setPendingHistoryId,
  } = useAppStore()

  const [sourceType, setSourceType] = useState<'new' | 'history'>('new')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [selectedHistoryId, setSelectedHistoryId] = useState<string>('')
  const [meetingName, setMeetingName] = useState('')
  // 使用对象来包含时间和触发 ID，确保每次点击都触发跳转
  const [seekRequest, setSeekRequest] = useState<{ time: number; id: number; autoPlay?: boolean } | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // 对话框状态
  const [editSegment, setEditSegment] = useState<{
    id: number
    speakerId: string
    text: string
  } | null>(null)
  const [renameSpeaker, setRenameSpeaker] = useState<{
    id: string
    name: string
    count: number
  } | null>(null)
  const [editingSummary, setEditingSummary] = useState(false)
  const [renameHistory, setRenameHistory] = useState<{
    id: string
    name: string
  } | null>(null)
  const [splitSegment, setSplitSegment] = useState<{
    id: number
    text: string
    speaker: string
  } | null>(null)

  // 合并模式状态
  const [isMergeMode, setIsMergeMode] = useState(false)
  const [selectedForMerge, setSelectedForMerge] = useState<number[]>([])
  const [isMerging, setIsMerging] = useState(false)  // 防止重复提交

  // 重新生成总结状态
  const [isRegeneratingSummary, setIsRegeneratingSummary] = useState(false)

  // 重新生成对话状态
  const [isRegeneratingTranscript, setIsRegeneratingTranscript] = useState(false)

  // 轮询任务状态
  const pollingRef = useRef<number | null>(null)

  // 加载模型列表
  const refreshModels = useCallback(async () => {
    try {
      const data = await api.getModels()
      const asr = data.asr_models?.length ? data.asr_models : data.whisper_models
      useAppStore.getState().setModels(asr, data.llm_models, data.gender_models)
      // 不自动选择第一个模型，强制用户手动选择
      // if (asr.length > 0 && !selectedAsrModel) {
      //   setSelectedAsrModel(asr[0].name)
      // }
    } catch (err) {
      console.error('加载模型列表失败:', err)
    }
  }, [selectedAsrModel, setSelectedAsrModel])

  useEffect(() => {
    refreshModels()
  }, [])

  // 后端连接状态
  const [backendError, setBackendError] = useState<string | null>(null)

  // 加载历史记录
  useEffect(() => {
    api.getHistory()
      .then((data) => {
        setHistoryItems(data.items)
        setBackendError(null)
      })
      .catch((err) => {
        console.error('加载历史记录失败:', err)
        setBackendError('后端未连接')
      })
  }, [setHistoryItems])

  // 清理轮询
  useEffect(() => {
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current)
      }
    }
  }, [])

  // 处理从实时录音跳转过来的情况
  useEffect(() => {
    if (!pendingHistoryId) return

    // 新录音刚完成，historyItems 可能还是旧的，需要先刷新再选中
    const loadAndSelect = async () => {
      try {
        const data = await api.getHistory()
        setHistoryItems(data.items)

        const exists = data.items.some((item) => item.id === pendingHistoryId)
        if (exists) {
          setSourceType('history')
          setSelectedHistoryId(pendingHistoryId)
          const result = await api.getHistoryItem(pendingHistoryId)
          setResult(result)
          setCurrentJob(pendingHistoryId)
        }
      } catch (err) {
        console.error('刷新历史记录失败:', err)
      }
      setPendingHistoryId(null)
    }

    loadAndSelect()
  }, [pendingHistoryId, setPendingHistoryId, setHistoryItems, setResult, setCurrentJob])

  // 文件选择
  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedFile(file)
    }
  }, [])

  // 加载历史记录
  const handleHistorySelect = useCallback(async (historyId: string) => {
    if (!historyId) return
    setSelectedHistoryId(historyId)
    // 切换历史记录时退出合并模式
    setIsMergeMode(false)
    setSelectedForMerge([])
    try {
      const result = await api.getHistoryItem(historyId)
      setResult(result)
      setCurrentJob(historyId)
    } catch (err) {
      console.error('加载历史记录失败:', err)
    }
  }, [setResult, setCurrentJob])

  // 开始处理
  const handleProcess = useCallback(async () => {
    if (sourceType === 'new' && !selectedFile) {
      alert('请先选择音频文件')
      return
    }
    if (sourceType === 'history' && selectedHistoryId) {
      // 已加载历史记录
      return
    }

    if (!selectedFile) return

    // 验证必选项
    if (!selectedAsrModel) {
      alert('请选择语音识别模型')
      return
    }

    // 勾选了需要 LLM 的功能但没选 LLM
    const needsLlm = enableNaming || enableCorrection || enableSummary
    if (needsLlm && (!selectedLlmModel || selectedLlmModel === 'disabled')) {
      alert('已勾选智能命名/错别字校正/会议总结，请先选择一个 LLM 模型')
      return
    }

    setProcessing(true)
    setProgress(0, '释放显存...')

    // 卸载聊天加载的 LLM，释放显存给 ASR/diarization
    try { await api.unloadLlm() } catch { /* 忽略 */ }

    setProgress(0, '上传文件...')

    try {
      const job = await api.uploadAndProcess(selectedFile, {
        name: meetingName.trim() || undefined,  // 自定义会议名称
        asr_model: selectedAsrModel,
        llm_model: selectedLlmModel !== 'disabled' ? selectedLlmModel : undefined,
        gender_model: selectedGenderModel || undefined,
        enable_naming: enableNaming,
        enable_correction: enableCorrection,
        enable_summary: enableSummary,
        enhance_mode: enhanceMode,
      })

      setCurrentJob(job.job_id)

      // 开始轮询
      pollingRef.current = window.setInterval(async () => {
        try {
          const status = await api.getJobStatus(job.job_id)
          setProgress(status.progress, status.message)

          if (status.status === 'completed') {
            clearInterval(pollingRef.current!)
            pollingRef.current = null
            const result = await api.getJobResult(job.job_id)
            setResult(result)
            setProcessing(false)
            // 刷新历史记录
            const history = await api.getHistory()
            setHistoryItems(history.items)
            // 自动切换到历史记录视图并选中刚处理完的项目
            const dirName = result.output_dir.split(/[/\\]/).pop() || ''
            const matchItem = history.items.find((item) => item.id === dirName)
            if (matchItem) {
              setSourceType('history')
              setSelectedHistoryId(matchItem.id)
              setCurrentJob(matchItem.id)
            }
          } else if (status.status === 'failed') {
            clearInterval(pollingRef.current!)
            pollingRef.current = null
            setProcessing(false)
            alert(`处理失败: ${status.message}`)
          }
        } catch (err) {
          console.error('获取状态失败:', err)
        }
      }, 1000)
    } catch (err) {
      console.error('上传失败:', err)
      setProcessing(false)
      alert('上传失败，请重试')
    }
  }, [
    sourceType,
    selectedFile,
    selectedHistoryId,
    meetingName,
    selectedAsrModel,
    selectedLlmModel,
    enableNaming,
    enableCorrection,
    enableSummary,
    enhanceMode,
    setProcessing,
    setProgress,
    setCurrentJob,
    setResult,
    setHistoryItems,
  ])

  // 点击片段跳转播放
  const handleSegmentClick = useCallback((segment: Segment) => {
    setPlayback({ currentSegmentId: segment.id })
    setSeekRequest({ time: segment.start, id: Date.now() })
  }, [setPlayback])

  // 音频时间更新，更新当前片段高亮
  const handleTimeUpdate = useCallback((currentTime: number) => {
    const current = segments.find(
      (seg) => seg.start <= currentTime && currentTime < seg.end
    )
    if (current && current.id !== currentSegmentId) {
      setPlayback({ currentSegmentId: current.id, currentTime })
    } else {
      setPlayback({ currentTime })
    }
  }, [segments, currentSegmentId, setPlayback])

  // 保存编辑
  const handleSaveSegment = useCallback(async (speakerId: string, text: string, newSpeakerName?: string) => {
    if (!editSegment || !currentJobId) return

    const oldSpeakerId = editSegment.speakerId
    const speakerChanged = speakerId !== oldSpeakerId
    const textChanged = text !== editSegment.text

    // 如果是新建说话人，先在本地添加
    if (newSpeakerName) {
      useAppStore.getState().addSpeaker(speakerId, newSpeakerName)
    }

    // 更新本地状态
    if (textChanged) updateSegmentText(editSegment.id, text)
    if (speakerChanged) useAppStore.getState().updateSegmentSpeaker(editSegment.id, speakerId)

    // 调用后端 API
    try {
      const updates: { text?: string; speaker?: string; speaker_name?: string } = {}
      if (textChanged) updates.text = text
      if (speakerChanged) updates.speaker = speakerId
      if (newSpeakerName) updates.speaker_name = newSpeakerName

      if (Object.keys(updates).length > 0) {
        if (sourceType === 'history' && selectedHistoryId) {
          await api.updateHistorySegment(selectedHistoryId, editSegment.id, updates)
        } else {
          await api.updateSegment(currentJobId, editSegment.id, updates)
        }
      }
    } catch (err) {
      console.error('保存失败:', err)
    }
    setEditSegment(null)
  }, [editSegment, currentJobId, sourceType, selectedHistoryId, updateSegmentText])

  // 删除片段
  const handleDeleteSegment = useCallback(async (segmentId: number) => {
    if (!currentJobId) return
    if (!confirm('确定要删除这条对话记录吗？')) return
    try {
      if (sourceType === 'history' && selectedHistoryId) {
        await api.deleteHistorySegment(selectedHistoryId, segmentId)
      } else {
        await api.deleteSegment(currentJobId, segmentId)
      }
      deleteSegment(segmentId)
    } catch (err) {
      console.error('删除失败:', err)
    }
  }, [currentJobId, sourceType, selectedHistoryId, deleteSegment])

  // 保存重命名
  const handleSaveRename = useCallback(async (newName: string) => {
    if (!renameSpeaker || !currentJobId) return
    updateSpeakerName(renameSpeaker.id, newName)
    try {
      if (sourceType === 'history' && selectedHistoryId) {
        await api.renameHistorySpeaker(selectedHistoryId, renameSpeaker.id, newName)
      } else {
        await api.renameSpeaker(currentJobId, renameSpeaker.id, newName)
      }
    } catch (err) {
      console.error('重命名失败:', err)
    }
    setRenameSpeaker(null)
  }, [renameSpeaker, currentJobId, sourceType, selectedHistoryId, updateSpeakerName])

  // 保存总结
  const handleSaveSummary = useCallback(async (newSummary: string) => {
    if (!currentJobId) return
    updateSummary(newSummary)
    try {
      if (sourceType === 'history' && selectedHistoryId) {
        await api.updateHistorySummary(selectedHistoryId, newSummary)
      } else {
        await api.updateSummary(currentJobId, newSummary)
      }
    } catch (err) {
      console.error('保存总结失败:', err)
    }
    setEditingSummary(false)
  }, [currentJobId, sourceType, selectedHistoryId, updateSummary])

  // 重新生成总结
  const handleRegenerateSummary = useCallback(async () => {
    if (!selectedHistoryId || isRegeneratingSummary) return

    if (!selectedLlmModel || selectedLlmModel === 'disabled') {
      alert('请先在下方选择一个 LLM 模型')
      return
    }

    if (!confirm('确定要重新生成会议总结吗？这将覆盖当前的总结内容。')) {
      return
    }

    setIsRegeneratingSummary(true)
    try {
      const result = await api.regenerateSummary(selectedHistoryId, selectedLlmModel)
      updateSummary(result.summary)
    } catch (err: unknown) {
      console.error('重新生成总结失败:', err)
      const message = err instanceof Error ? err.message : '未知错误'
      alert(`重新生成总结失败: ${message}`)
    } finally {
      setIsRegeneratingSummary(false)
    }
  }, [selectedHistoryId, isRegeneratingSummary, updateSummary, selectedLlmModel])

  // 重新生成对话（ASR 重新转写 + 对齐 + 校正 + 命名）
  const handleRegenerateTranscript = useCallback(async () => {
    if (!selectedHistoryId || isRegeneratingTranscript) return

    // ASR 模型必选
    if (!selectedAsrModel) {
      alert('请选择语音识别模型')
      return
    }

    // 勾选了需要 LLM 的功能但没选 LLM
    const needsLlm = enableNaming || enableCorrection
    if (needsLlm && (!selectedLlmModel || selectedLlmModel === 'disabled')) {
      alert('已勾选智能命名/错别字校正，请先选择一个 LLM 模型')
      return
    }

    if (!confirm('确定要重新生成对话吗？这将使用选择的 ASR 模型重新转写音频，并运行对齐、校正和命名。')) {
      return
    }

    setIsRegeneratingTranscript(true)
    try {
      await api.regenerateTranscript(selectedHistoryId, {
        asr_model: selectedAsrModel,
        llm_model: selectedLlmModel !== 'disabled' ? selectedLlmModel : undefined,
        gender_model: selectedGenderModel || undefined,
        enable_naming: enableNaming,
        enable_correction: enableCorrection,
      })
      // 重新加载完整历史记录数据更新 store
      const historyResult = await api.getHistoryItem(selectedHistoryId)
      setResult(historyResult)
    } catch (err: unknown) {
      console.error('重新生成对话失败:', err)
      const message = err instanceof Error ? err.message : '未知错误'
      alert(`重新生成对话失败: ${message}`)
    } finally {
      setIsRegeneratingTranscript(false)
    }
  }, [selectedHistoryId, isRegeneratingTranscript, selectedAsrModel, selectedLlmModel, selectedGenderModel, enableNaming, enableCorrection, setResult])

  // 分割片段
  const handleSplitSegment = useCallback(async (splitPosition: number, newSpeaker?: string) => {
    if (!splitSegment || !currentJobId) return
    try {
      // 根据来源调用不同的 API
      if (sourceType === 'history' && selectedHistoryId) {
        await api.splitHistorySegment(selectedHistoryId, splitSegment.id, splitPosition, newSpeaker)
        // 重新加载历史记录数据
        const data = await api.getHistoryItem(selectedHistoryId)
        setResult(data)
      } else {
        await api.splitSegment(currentJobId, splitSegment.id, splitPosition, newSpeaker)
        // 重新加载任务结果
        const data = await api.getJobResult(currentJobId)
        setResult(data)
      }
    } catch (err) {
      console.error('分割失败:', err)
      alert('分割片段失败')
    }
    setSplitSegment(null)
  }, [splitSegment, currentJobId, sourceType, selectedHistoryId, setResult])

  // 切换片段的合并选中状态
  const handleToggleMergeSelect = useCallback((segmentId: number) => {
    setSelectedForMerge((prev) => {
      if (prev.includes(segmentId)) {
        return prev.filter((id) => id !== segmentId)
      }
      return [...prev, segmentId].sort((a, b) => a - b)
    })
  }, [])

  // 执行合并
  const handleMergeSegments = useCallback(async () => {
    if (isMerging) return  // 防止重复提交
    if (selectedForMerge.length < 2) {
      alert('请至少选择两个片段进行合并')
      return
    }

    // 检查是否连续
    const sorted = [...selectedForMerge].sort((a, b) => a - b)
    for (let i = 0; i < sorted.length - 1; i++) {
      if (sorted[i + 1] - sorted[i] !== 1) {
        alert('只能合并连续的片段')
        return
      }
    }

    setIsMerging(true)
    try {
      if (sourceType === 'history' && selectedHistoryId) {
        await api.mergeHistorySegments(selectedHistoryId, sorted)
        // 重新加载历史记录数据
        const data = await api.getHistoryItem(selectedHistoryId)
        setResult(data)
      }
      // 退出合并模式
      setIsMergeMode(false)
      setSelectedForMerge([])
    } catch (err) {
      console.error('合并失败:', err)
      alert('合并片段失败')
    } finally {
      setIsMerging(false)
    }
  }, [isMerging, selectedForMerge, sourceType, selectedHistoryId, setResult])

  // 退出合并模式
  const handleCancelMerge = useCallback(() => {
    setIsMergeMode(false)
    setSelectedForMerge([])
  }, [])

  // 导出
  const handleExport = useCallback((format: 'txt' | 'json' | 'md') => {
    if (!currentJobId) return
    const url = api.getExportUrl(currentJobId, format)
    window.open(url, '_blank')
  }, [currentJobId])

  // 重命名历史记录
  const handleRenameHistory = useCallback(async (newName: string) => {
    if (!renameHistory) return
    try {
      const result = await api.renameHistoryItem(renameHistory.id, newName)
      // 刷新历史记录列表
      const history = await api.getHistory()
      setHistoryItems(history.items)
      // 如果当前选中的就是被重命名的，更新选中项
      if (selectedHistoryId === renameHistory.id) {
        setSelectedHistoryId(result.new_id)
      }
      setRenameHistory(null)
    } catch (err) {
      console.error('重命名失败:', err)
      alert('重命名失败')
    }
  }, [renameHistory, selectedHistoryId, setHistoryItems])

  // 删除历史记录
  const handleDeleteHistory = useCallback(async () => {
    if (!selectedHistoryId) return
    const item = historyItems.find(h => h.id === selectedHistoryId)
    if (!item) return

    if (!confirm(`确定要删除 "${item.name}" 吗？此操作不可撤销。`)) {
      return
    }

    try {
      await api.deleteHistoryItem(selectedHistoryId)
      // 刷新历史记录列表
      const history = await api.getHistory()
      setHistoryItems(history.items)
      // 清空当前选择
      setSelectedHistoryId('')
      setResult(null)
      setCurrentJob(null)
    } catch (err) {
      console.error('删除失败:', err)
      alert('删除失败')
    }
  }, [selectedHistoryId, historyItems, setHistoryItems, setResult, setCurrentJob])

  return (
    <div className="flex flex-col h-full">
      {/* 主内容区：对话记录 + 会议总结 */}
      <div className="flex-1 flex min-h-0">
        {/* 左侧：对话记录 */}
        <div className="flex-[6] flex flex-col p-2 min-w-0">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-primary-600">💬</span>
            <span className="text-sm font-medium">对话记录</span>
            <div className="flex-1" />
            {/* 合并模式控制 */}
            {sourceType === 'history' && selectedHistoryId && segments.length > 1 && (
              isMergeMode ? (
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    已选择 {selectedForMerge.length} 个片段
                  </span>
                  <button
                    onClick={handleMergeSegments}
                    disabled={selectedForMerge.length < 2 || isMerging}
                    className="px-2 py-1 text-xs bg-primary-600 text-white rounded hover:bg-primary-700 disabled:bg-gray-300 dark:disabled:bg-gray-600 disabled:cursor-not-allowed"
                  >
                    {isMerging ? '合并中...' : '确认合并'}
                  </button>
                  <button
                    onClick={handleCancelMerge}
                    className="px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-100 dark:hover:bg-gray-700"
                  >
                    取消
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => {
                    if (isPlaying) {
                      alert('请先暂停音频播放再进行合并操作')
                      return
                    }
                    setIsMergeMode(true)
                  }}
                  disabled={isPlaying}
                  className="flex items-center gap-1 px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                  title={isPlaying ? '请先暂停音频' : '合并相邻片段'}
                >
                  <Merge className="w-3 h-3" />
                  合并
                </button>
              )
            )}
          </div>

          {/* 对话列表 */}
          <div className="flex-1 border border-gray-300 dark:border-gray-600 rounded-lg overflow-auto">
            {segments.length > 0 ? (
              <div className="p-2 space-y-1">
                {segments.map((segment) => (
                  <div
                    key={segment.id}
                    className={`flex items-start gap-2 ${isMergeMode ? 'cursor-pointer' : ''}`}
                    onClick={isMergeMode ? () => handleToggleMergeSelect(segment.id) : undefined}
                  >
                    {/* 合并模式选择框 */}
                    {isMergeMode && (
                      <input
                        type="checkbox"
                        checked={selectedForMerge.includes(segment.id)}
                        onChange={() => handleToggleMergeSelect(segment.id)}
                        onClick={(e) => e.stopPropagation()}
                        className="mt-3 w-4 h-4 cursor-pointer"
                      />
                    )}
                    <div className="flex-1">
                      <SegmentCard
                        segment={segment}
                        speaker={speakers[segment.speaker]}
                        isPlaying={segment.id === currentSegmentId}
                        isSelected={selectedForMerge.includes(segment.id)}
                        onClick={isMergeMode ? undefined : () => handleSegmentClick(segment)}
                        onEdit={isMergeMode ? undefined : () =>
                          setEditSegment({
                            id: segment.id,
                            speakerId: segment.speaker,
                            text: segment.text,
                          })
                        }
                        onSpeakerClick={isMergeMode ? undefined : () =>
                          setRenameSpeaker({
                            id: segment.speaker,
                            name: speakers[segment.speaker]?.display_name || segment.speaker,
                            count: segments.filter((s) => s.speaker === segment.speaker).length,
                          })
                        }
                        onSplit={isMergeMode ? undefined : () =>
                          setSplitSegment({
                            id: segment.id,
                            text: segment.text,
                            speaker: segment.speaker,
                          })
                        }
                        onDelete={isMergeMode ? undefined : () => handleDeleteSegment(segment.id)}
                      />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex items-center justify-center h-full text-gray-400 dark:text-gray-500 text-sm">
                处理音频后将在此显示对话记录...
              </div>
            )}
          </div>

          {/* 音频播放器 */}
          <div className="mt-2">
            {audioOriginalUrl && audioUrl && audioOriginalUrl !== audioUrl && (
              <div className="flex items-center gap-2 mb-1 px-1">
                <span className="text-xs text-gray-500 dark:text-gray-400 mr-0.5">播放:</span>
                <div className="inline-flex rounded border border-gray-300 dark:border-gray-600 overflow-hidden">
                  <button
                    onClick={() => {
                      if (useOriginalAudio) {
                        const t = useAppStore.getState().currentTime
                        toggleAudioSource()
                        setSeekRequest({ time: t, id: Date.now(), autoPlay: false })
                      }
                    }}
                    className={`text-xs px-2.5 py-1 transition-colors ${
                      !useOriginalAudio
                        ? 'bg-blue-500 text-white font-medium'
                        : 'bg-white dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
                    }`}
                  >
                    增强音频
                  </button>
                  <button
                    onClick={() => {
                      if (!useOriginalAudio) {
                        const t = useAppStore.getState().currentTime
                        toggleAudioSource()
                        setSeekRequest({ time: t, id: Date.now(), autoPlay: false })
                      }
                    }}
                    className={`text-xs px-2.5 py-1 border-l border-gray-300 dark:border-gray-600 transition-colors ${
                      useOriginalAudio
                        ? 'bg-blue-500 text-white font-medium'
                        : 'bg-white dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
                    }`}
                  >
                    原始音频
                  </button>
                </div>
              </div>
            )}
            <AudioPlayer
              src={useOriginalAudio ? audioOriginalUrl : audioUrl}
              onTimeUpdate={handleTimeUpdate}
              onPlayStateChange={(playing) => setPlayback({ isPlaying: playing })}
              seekTo={seekRequest?.time ?? null}
              seekId={seekRequest?.id}
              seekAutoPlay={seekRequest?.autoPlay ?? true}
            />
          </div>
        </div>

        {/* 分隔线 */}
        <div className="w-1.5 bg-gray-200 dark:bg-gray-700" />

        {/* 右侧：会议总结 */}
        <div className="flex-[4] p-2 min-w-0">
          <SummaryPanel
            summary={summary}
            onEdit={() => setEditingSummary(true)}
            onRegenerate={sourceType === 'history' && selectedHistoryId ? handleRegenerateSummary : undefined}
            isRegenerating={isRegeneratingSummary}
            historyId={sourceType === 'history' ? selectedHistoryId : currentJobId}
          />
        </div>
      </div>

      {/* 底部控制区 */}
      <div className="border-t bg-gray-50 dark:bg-gray-900 dark:border-gray-700 px-4 py-3 space-y-3">
        {/* 第一行：文件选择 + 模型 + 开始按钮 */}
        <div className="flex items-center gap-3 flex-wrap">
          {sourceType === 'new' ? (
            <>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="flex items-center gap-2 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-100 dark:hover:bg-gray-700"
              >
                <Upload className="w-4 h-4" />
                浏览...
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                className="hidden"
                onChange={handleFileSelect}
              />
              <span className="text-sm text-gray-600 dark:text-gray-300 truncate max-w-[150px]">
                {selectedFile ? selectedFile.name : '未选择文件'}
              </span>
              <input
                type="text"
                value={meetingName}
                onChange={(e) => setMeetingName(e.target.value)}
                placeholder="会议名称（可选）"
                disabled={isProcessing}
                className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded text-sm w-36 disabled:bg-gray-200 dark:disabled:bg-gray-700 dark:bg-gray-800 dark:text-gray-200"
              />
            </>
          ) : (
            <>
              <select
                value={selectedHistoryId}
                onChange={(e) => handleHistorySelect(e.target.value)}
                disabled={!!backendError}
                className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded text-sm min-w-[200px] disabled:bg-gray-200 dark:disabled:bg-gray-700 disabled:cursor-not-allowed dark:bg-gray-800 dark:text-gray-200"
              >
                {backendError ? (
                  <option value="">后端未连接</option>
                ) : historyItems.length === 0 ? (
                  <option value="">暂无历史记录</option>
                ) : (
                  <>
                    <option value="">选择历史记录</option>
                    {historyItems.map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.name}
                      </option>
                    ))}
                  </>
                )}
              </select>
              {selectedHistoryId && (
                <>
                  <button
                    onClick={() => {
                      const item = historyItems.find(h => h.id === selectedHistoryId)
                      if (item) {
                        setRenameHistory({ id: item.id, name: item.name })
                      }
                    }}
                    className="p-2 border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-100 dark:hover:bg-gray-700"
                    title="重命名"
                  >
                    <Pencil className="w-4 h-4" />
                  </button>
                  <button
                    onClick={handleRegenerateTranscript}
                    disabled={isRegeneratingTranscript}
                    className="flex items-center gap-1 px-2 py-2 border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                    title="重新生成对话（ASR重新转写+对齐+校正+命名）"
                  >
                    <RefreshCw className={`w-4 h-4 ${isRegeneratingTranscript ? 'animate-spin' : ''}`} />
                    {isRegeneratingTranscript ? '生成中...' : '重新生成对话'}
                  </button>
                  <button
                    onClick={handleDeleteHistory}
                    className="p-2 border border-red-300 dark:border-red-700 rounded hover:bg-red-50 dark:hover:bg-red-900/30 text-red-600 dark:text-red-400"
                    title="删除"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </>
              )}
            </>
          )}

          <div className="flex items-center gap-1">
            <span className="text-xs text-gray-500 dark:text-gray-400 whitespace-nowrap">识别</span>
            <select
              value={selectedAsrModel}
              onChange={(e) => setSelectedAsrModel(e.target.value)}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded text-sm max-w-[150px] dark:bg-gray-800 dark:text-gray-200"
              title="语音识别模型"
            >
              <option value="">请选择</option>
              {(() => {
                const grouped: Record<string, typeof asrModels> = {}
                for (const m of asrModels) {
                  const eng = m.engine || 'faster-whisper'
                  if (!grouped[eng]) grouped[eng] = []
                  grouped[eng].push(m)
                }
                const engineLabels: Record<string, string> = {
                  'faster-whisper': 'Whisper',
                  'funasr': 'FunASR',
                  'fireredasr': 'FireRedASR',
                }
                const entries = Object.entries(grouped)
                if (entries.length <= 1) {
                  return asrModels.map((m) => (
                    <option key={m.name} value={m.name}>{m.display_name}</option>
                  ))
                }
                return entries.map(([eng, models]) => (
                  <optgroup key={eng} label={engineLabels[eng] || eng}>
                    {models.map((m) => (
                      <option key={m.name} value={m.name}>{m.display_name}</option>
                    ))}
                  </optgroup>
                ))
              })()}
            </select>
          </div>

          <div className="flex items-center gap-1">
            <span className="text-xs text-gray-500 dark:text-gray-400 whitespace-nowrap">LLM</span>
            <select
              value={selectedLlmModel}
              onChange={(e) => setSelectedLlmModel(e.target.value)}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded text-sm max-w-[150px] dark:bg-gray-800 dark:text-gray-200"
              title="LLM 模型（智能命名/总结）"
            >
              {llmModels.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.display_name}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-1">
            <span className="text-xs text-gray-500 dark:text-gray-400 whitespace-nowrap">性别</span>
            <select
              value={selectedGenderModel}
              onChange={(e) => setSelectedGenderModel(e.target.value)}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded text-sm max-w-[130px] dark:bg-gray-800 dark:text-gray-200"
              title="性别检测模型"
            >
              {genderModels.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.display_name}
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={refreshModels}
            className="p-2 border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-100 dark:hover:bg-gray-700"
            title="刷新模型列表"
          >
            <RefreshCw className="w-4 h-4" />
          </button>

          <div className="flex-1" />

          {/* 只在"选择新文件"模式下显示开始处理按钮 */}
          {sourceType === 'new' && (
            <>
              <button
                onClick={handleProcess}
                disabled={isProcessing || !selectedFile}
                className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded hover:bg-primary-700 disabled:bg-gray-300 dark:disabled:bg-gray-600 disabled:cursor-not-allowed"
              >
                <Play className="w-4 h-4" />
                开始处理
              </button>

              {isProcessing && (
                <ProgressBar
                  progress={progress}
                  message={progressMessage}
                  className="flex-1 min-w-[180px]"
                />
              )}
            </>
          )}
        </div>

        {/* 第二行：选项 + 导出 */}
        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-1.5 text-sm dark:text-gray-200">
              <input
                type="radio"
                name="source"
                checked={sourceType === 'new'}
                onChange={() => {
                  setSourceType('new')
                  setResult(null)
                  setCurrentJob(null)
                  setSelectedHistoryId('')
                  // 清空文件选择，避免用户混淆
                  setSelectedFile(null)
                  setMeetingName('')
                }}
              />
              选择新文件
            </label>
            <label className="flex items-center gap-1.5 text-sm dark:text-gray-200">
              <input
                type="radio"
                name="source"
                checked={sourceType === 'history'}
                onChange={() => setSourceType('history')}
              />
              历史记录
            </label>
          </div>

          <div className="w-px h-4 bg-gray-300 dark:bg-gray-600" />

          <label className="flex items-center gap-1.5 text-sm dark:text-gray-200">
            <input
              type="checkbox"
              checked={enableNaming}
              onChange={(e) => setProcessOptions({ enableNaming: e.target.checked })}
            />
            智能命名
          </label>
          <label className="flex items-center gap-1.5 text-sm dark:text-gray-200">
            <input
              type="checkbox"
              checked={enableCorrection}
              onChange={(e) => setProcessOptions({ enableCorrection: e.target.checked })}
            />
            错别字校正
          </label>
          <label className="flex items-center gap-1.5 text-sm dark:text-gray-200">
            <input
              type="checkbox"
              checked={enableSummary}
              onChange={(e) => setProcessOptions({ enableSummary: e.target.checked })}
            />
            会议总结
          </label>

          <select
            value={enhanceMode}
            onChange={(e) => setProcessOptions({ enhanceMode: e.target.value as typeof enhanceMode })}
            className="px-2 py-1 border border-gray-300 dark:border-gray-600 rounded text-sm dark:bg-gray-800 dark:text-gray-200"
          >
            <option value="none">不增强</option>
            <option value="denoise">专业降噪 (DeepFilterNet3)</option>
            <option value="enhance">降噪+清晰化 (DeepFilterNet3 + Resemble)</option>
            <option value="vocal">人声分离 (Demucs)</option>
            <option value="full">完整增强 (全部模型)</option>
          </select>

          <div className="flex-1" />

          <div className="flex items-center gap-2">
            <button
              onClick={() => handleExport('txt')}
              disabled={!currentJobId}
              className="flex items-center gap-1 px-3 py-1.5 border border-gray-300 dark:border-gray-600 rounded text-sm hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <FileText className="w-4 h-4" />
              TXT
            </button>
            <button
              onClick={() => handleExport('json')}
              disabled={!currentJobId}
              className="flex items-center gap-1 px-3 py-1.5 border border-gray-300 dark:border-gray-600 rounded text-sm hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Code className="w-4 h-4" />
              JSON
            </button>
            <button
              onClick={() => handleExport('md')}
              disabled={!currentJobId}
              className="flex items-center gap-1 px-3 py-1.5 border border-gray-300 dark:border-gray-600 rounded text-sm hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <FileDown className="w-4 h-4" />
              Markdown
            </button>
          </div>
        </div>
      </div>

      {/* 对话框 */}
      <EditSegmentDialog
        open={!!editSegment}
        onClose={() => setEditSegment(null)}
        segment={editSegment}
        speakers={Object.entries(speakers).map(([id, s]) => ({
          id,
          name: s.display_name || id,
        }))}
        onSave={handleSaveSegment}
      />
      <RenameSpeakerDialog
        open={!!renameSpeaker}
        onClose={() => setRenameSpeaker(null)}
        speaker={renameSpeaker}
        onSave={handleSaveRename}
      />
      <EditSummaryDialog
        open={editingSummary}
        onClose={() => setEditingSummary(false)}
        summary={summary}
        onSave={handleSaveSummary}
      />
      <RenameHistoryDialog
        open={!!renameHistory}
        onClose={() => setRenameHistory(null)}
        historyId={renameHistory?.id || ''}
        currentName={renameHistory?.name || ''}
        onSave={handleRenameHistory}
      />
      <SplitSegmentDialog
        open={!!splitSegment}
        onClose={() => setSplitSegment(null)}
        segment={splitSegment}
        speakers={Object.entries(speakers).map(([id, s]) => ({
          id,
          name: s.display_name || id,
        }))}
        onSplit={handleSplitSegment}
      />
    </div>
  )
}
