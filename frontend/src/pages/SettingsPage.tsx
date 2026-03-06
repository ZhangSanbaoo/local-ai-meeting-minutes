import { useCallback, useEffect, useRef, useState } from 'react'
import { Upload, Trash2, RefreshCw, HardDrive, Cpu, Info } from 'lucide-react'
import { clsx } from 'clsx'
import { useAppStore } from '../stores/appStore'
import * as api from '../api/client'
import type { LLMSettings, SystemInfo } from '../types'

export function SettingsPage() {
  const { asrModels, llmModels, genderModels } = useAppStore()
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState('')
  const whisperInputRef = useRef<HTMLInputElement>(null)
  const asrInputRef = useRef<HTMLInputElement>(null)
  const llmInputRef = useRef<HTMLInputElement>(null)
  const genderInputRef = useRef<HTMLInputElement>(null)

  // LLM 参数设置
  const [llmSettings, setLlmSettings] = useState<LLMSettings | null>(null)
  const [nCtxValue, setNCtxValue] = useState(4096)
  const [nCtxInput, setNCtxInput] = useState('4096')
  const [llmSaving, setLlmSaving] = useState(false)
  const [llmSaveMsg, setLlmSaveMsg] = useState<string | null>(null)

  // 加载系统信息
  useEffect(() => {
    api.getSystemInfo().then(setSystemInfo).catch(console.error)
  }, [])

  // 加载 LLM 参数
  useEffect(() => {
    api.getLLMSettings().then((s) => {
      setLlmSettings(s)
      setNCtxValue(s.n_ctx)
      setNCtxInput(String(s.n_ctx))
    }).catch(console.error)
  }, [])

  const handleNCtxSlider = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const v = Number(e.target.value)
    setNCtxValue(v)
    setNCtxInput(String(v))
  }, [])

  const handleNCtxInputBlur = useCallback(() => {
    let v = parseInt(nCtxInput, 10)
    if (isNaN(v)) v = 4096
    v = Math.max(1024, Math.min(32768, Math.round(v / 1024) * 1024))
    setNCtxValue(v)
    setNCtxInput(String(v))
  }, [nCtxInput])

  const handleSaveLlmSettings = useCallback(async () => {
    setLlmSaving(true)
    setLlmSaveMsg(null)
    try {
      await api.updateLLMSettings(nCtxValue)
      setLlmSaveMsg('已保存')
      if (llmSettings) {
        setLlmSettings({ ...llmSettings, n_ctx: nCtxValue })
      }
      setTimeout(() => setLlmSaveMsg(null), 2000)
    } catch (err) {
      console.error('保存失败:', err)
      setLlmSaveMsg('保存失败')
      setTimeout(() => setLlmSaveMsg(null), 3000)
    } finally {
      setLlmSaving(false)
    }
  }, [nCtxValue, llmSettings])

  const handleApplyRecommended = useCallback(() => {
    if (!llmSettings) return
    setNCtxValue(llmSettings.recommended_n_ctx)
    setNCtxInput(String(llmSettings.recommended_n_ctx))
  }, [llmSettings])

  // 刷新模型列表
  const refreshModels = useCallback(async () => {
    try {
      const data = await api.getModels()
      const asr = data.asr_models?.length ? data.asr_models : data.whisper_models
      useAppStore.getState().setModels(asr, data.llm_models, data.gender_models)
    } catch (err) {
      console.error('加载模型列表失败:', err)
    }
  }, [])

  useEffect(() => {
    refreshModels()
  }, [refreshModels])

  // 上传 Whisper 模型
  const handleWhisperUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    if (!file.name.endsWith('.zip') && !file.name.endsWith('.tar.gz')) {
      alert('只支持 .zip 或 .tar.gz 格式')
      return
    }

    setIsUploading(true)
    setUploadProgress(`上传 Whisper 模型: ${file.name}...`)

    try {
      await api.uploadWhisperModel(file)
      await refreshModels()
      alert('上传成功')
    } catch (err) {
      console.error('上传失败:', err)
      alert('上传失败，请重试')
    } finally {
      setIsUploading(false)
      setUploadProgress('')
      if (whisperInputRef.current) {
        whisperInputRef.current.value = ''
      }
    }
  }, [refreshModels])

  // 上传 ASR 模型（非 Whisper）
  const handleAsrUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    if (!file.name.endsWith('.zip') && !file.name.endsWith('.tar.gz')) {
      alert('只支持 .zip 或 .tar.gz 格式')
      return
    }

    setIsUploading(true)
    setUploadProgress(`上传 ASR 模型: ${file.name}...`)

    try {
      await api.uploadAsrModel(file)
      await refreshModels()
      alert('上传成功')
    } catch (err) {
      console.error('上传失败:', err)
      alert('上传失败，请重试')
    } finally {
      setIsUploading(false)
      setUploadProgress('')
      if (asrInputRef.current) {
        asrInputRef.current.value = ''
      }
    }
  }, [refreshModels])

  // 上传 LLM 模型
  const handleLlmUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    if (!file.name.endsWith('.gguf')) {
      alert('只支持 .gguf 格式')
      return
    }

    setIsUploading(true)
    setUploadProgress(`上传 LLM 模型: ${file.name}...`)

    try {
      await api.uploadLlmModel(file)
      await refreshModels()
      alert('上传成功')
    } catch (err) {
      console.error('上传失败:', err)
      alert('上传失败，请重试')
    } finally {
      setIsUploading(false)
      setUploadProgress('')
      if (llmInputRef.current) {
        llmInputRef.current.value = ''
      }
    }
  }, [refreshModels])

  // 上传性别检测模型
  const handleGenderUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    if (!file.name.endsWith('.zip') && !file.name.endsWith('.tar.gz')) {
      alert('只支持 .zip 或 .tar.gz 格式')
      return
    }

    setIsUploading(true)
    setUploadProgress(`上传性别检测模型: ${file.name}...`)

    try {
      await api.uploadGenderModel(file)
      await refreshModels()
      alert('上传成功')
    } catch (err) {
      console.error('上传失败:', err)
      alert('上传失败，请重试')
    } finally {
      setIsUploading(false)
      setUploadProgress('')
      if (genderInputRef.current) {
        genderInputRef.current.value = ''
      }
    }
  }, [refreshModels])

  // 删除 Whisper 模型
  const handleDeleteWhisper = useCallback(async (modelName: string) => {
    if (!confirm(`确定要删除 Whisper 模型 "${modelName}" 吗？`)) return

    try {
      await api.deleteWhisperModel(modelName)
      await refreshModels()
    } catch (err) {
      console.error('删除失败:', err)
      alert('删除失败')
    }
  }, [refreshModels])

  // 删除 ASR 模型（非 Whisper）
  const handleDeleteAsr = useCallback(async (modelName: string) => {
    if (!confirm(`确定要删除 ASR 模型 "${modelName}" 吗？`)) return

    try {
      await api.deleteAsrModel(modelName)
      await refreshModels()
    } catch (err) {
      console.error('删除失败:', err)
      alert('删除失败')
    }
  }, [refreshModels])

  // 删除 LLM 模型
  const handleDeleteLlm = useCallback(async (modelName: string) => {
    if (!confirm(`确定要删除 LLM 模型 "${modelName}" 吗？`)) return

    try {
      await api.deleteLlmModel(modelName)
      await refreshModels()
    } catch (err) {
      console.error('删除失败:', err)
      alert('删除失败')
    }
  }, [refreshModels])

  // 删除性别检测模型
  const handleDeleteGender = useCallback(async (modelName: string) => {
    if (!confirm(`确定要删除性别检测模型 "${modelName}" 吗？`)) return

    try {
      await api.deleteGenderModel(modelName)
      await refreshModels()
    } catch (err) {
      console.error('删除失败:', err)
      alert('删除失败')
    }
  }, [refreshModels])

  return (
    <div className="h-full overflow-auto p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* 系统信息 */}
        <section className="bg-white dark:bg-gray-800 rounded-lg shadow dark:shadow-gray-900/50 p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Cpu className="w-5 h-5" />
            系统信息
          </h2>
          {systemInfo ? (
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-500 dark:text-gray-400">版本:</span>{' '}
                <span className="font-medium">{systemInfo.version}</span>
              </div>
              <div>
                <span className="text-gray-500 dark:text-gray-400">CUDA:</span>{' '}
                <span className={clsx('font-medium', systemInfo.cuda_available ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400')}>
                  {systemInfo.cuda_available ? `可用 (${systemInfo.cuda_version})` : '不可用'}
                </span>
              </div>
              {systemInfo.gpu_name && (
                <div className="col-span-2">
                  <span className="text-gray-500 dark:text-gray-400">GPU:</span>{' '}
                  <span className="font-medium">
                    {systemInfo.gpu_name}
                    {systemInfo.gpu_vram_gb != null && ` (${systemInfo.gpu_vram_gb}GB)`}
                  </span>
                </div>
              )}
              <div className="col-span-2">
                <span className="text-gray-500 dark:text-gray-400">模型目录:</span>{' '}
                <span className="font-mono text-xs">{systemInfo.models_dir}</span>
              </div>
              <div className="col-span-2">
                <span className="text-gray-500 dark:text-gray-400">输出目录:</span>{' '}
                <span className="font-mono text-xs">{systemInfo.output_dir}</span>
              </div>
            </div>
          ) : (
            <div className="text-gray-400 dark:text-gray-500">加载中...</div>
          )}
        </section>

        {/* ASR 语音识别模型管理 */}
        <section className="bg-white dark:bg-gray-800 rounded-lg shadow dark:shadow-gray-900/50 p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <HardDrive className="w-5 h-5" />
              ASR 语音识别模型
            </h2>
            <div className="flex items-center gap-2">
              <button
                onClick={refreshModels}
                className="p-2 border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-100 dark:hover:bg-gray-700"
                title="刷新"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
              <button
                onClick={() => whisperInputRef.current?.click()}
                disabled={isUploading}
                className="flex items-center gap-2 px-3 py-2 bg-primary-600 text-white rounded hover:bg-primary-700 disabled:bg-gray-300 dark:disabled:bg-gray-600"
                title="上传 faster-whisper 模型到 models/whisper/"
              >
                <Upload className="w-4 h-4" />
                上传 Whisper
              </button>
              <input
                ref={whisperInputRef}
                type="file"
                accept=".zip,.tar.gz"
                className="hidden"
                onChange={handleWhisperUpload}
              />
              <button
                onClick={() => asrInputRef.current?.click()}
                disabled={isUploading}
                className="flex items-center gap-2 px-3 py-2 bg-primary-600 text-white rounded hover:bg-primary-700 disabled:bg-gray-300 dark:disabled:bg-gray-600"
                title="上传 FunASR/FireRedASR 等模型到 models/asr/"
              >
                <Upload className="w-4 h-4" />
                上传 ASR
              </button>
              <input
                ref={asrInputRef}
                type="file"
                accept=".zip,.tar.gz"
                className="hidden"
                onChange={handleAsrUpload}
              />
            </div>
          </div>

          <div className="text-sm text-gray-500 dark:text-gray-400 mb-4 space-y-1">
            <p>上传 .zip 或 .tar.gz 压缩包，系统根据文件特征自动识别引擎类型：</p>
            <ul className="list-disc list-inside ml-2 text-xs space-y-0.5">
              <li><span className="font-medium text-blue-600 dark:text-blue-400">Whisper</span> — 含 vocabulary.json + model.bin（faster-whisper 格式）</li>
              <li><span className="font-medium text-green-600 dark:text-green-400">FunASR</span> — 含 configuration.json 或 model.py（SenseVoice / Paraformer 等）</li>
              <li><span className="font-medium text-orange-600 dark:text-orange-400">FireRedASR</span> — 含 spm.model + model.pt</li>
              <li><span className="font-medium text-purple-600 dark:text-purple-400">Qwen3-ASR</span> — 含 config.json + preprocessor_config.json + vocab.json（HuggingFace 格式）</li>
            </ul>
            <p className="text-xs">不符合以上格式的模型无法使用。同框架内的模型可自由替换。</p>
          </div>

          {asrModels.length > 0 ? (
            <div className="space-y-2">
              {asrModels.map((model) => {
                const engineLabel: Record<string, string> = {
                  'faster-whisper': 'Whisper',
                  'funasr': 'FunASR',
                  'fireredasr': 'FireRedASR',
                  'qwen3-asr': 'Qwen3-ASR',
                }
                const engineColor: Record<string, string> = {
                  'faster-whisper': 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400',
                  'funasr': 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
                  'fireredasr': 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400',
                  'qwen3-asr': 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400',
                }
                const eng = model.engine || 'faster-whisper'
                return (
                  <div
                    key={model.name}
                    className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded-lg"
                  >
                    <div className="flex items-center gap-2">
                      <span className={clsx('text-xs px-1.5 py-0.5 rounded font-medium', engineColor[eng] || 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300')}>
                        {engineLabel[eng] || eng}
                      </span>
                      <span className="font-medium">{model.display_name}</span>
                      {model.size_mb != null && (
                        <span className="text-gray-500 dark:text-gray-400 text-sm">
                          ({model.size_mb > 1024 ? `${(model.size_mb / 1024).toFixed(1)} GB` : `${model.size_mb} MB`})
                        </span>
                      )}
                    </div>
                    <button
                      onClick={() => eng === 'faster-whisper' ? handleDeleteWhisper(model.name) : handleDeleteAsr(model.name)}
                      className="p-2 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/30 rounded"
                      title="删除"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                )
              })}
            </div>
          ) : (
            <div className="text-center text-gray-400 dark:text-gray-500 py-8">
              暂无 ASR 模型，请上传或手动放入 models/whisper/ 或 models/asr/ 目录
            </div>
          )}
        </section>

        {/* LLM 模型管理 */}
        <section className="bg-white dark:bg-gray-800 rounded-lg shadow dark:shadow-gray-900/50 p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <HardDrive className="w-5 h-5" />
              LLM 模型 (智能命名/总结)
            </h2>
            <div className="flex items-center gap-2">
              <button
                onClick={() => llmInputRef.current?.click()}
                disabled={isUploading}
                className="flex items-center gap-2 px-3 py-2 bg-primary-600 text-white rounded hover:bg-primary-700 disabled:bg-gray-300 dark:disabled:bg-gray-600"
              >
                <Upload className="w-4 h-4" />
                上传模型
              </button>
              <input
                ref={llmInputRef}
                type="file"
                accept=".gguf"
                className="hidden"
                onChange={handleLlmUpload}
              />
            </div>
          </div>

          <div className="text-sm text-gray-500 dark:text-gray-400 mb-4 space-y-1">
            <p>框架: llama-cpp-python。格式: .gguf（GGML 量化模型）</p>
            <p className="text-xs">推荐 Qwen2.5-7B-Instruct Q4_K_M。同框架的 GGUF 模型可自由替换。</p>
          </div>

          {llmModels.filter(m => m.name !== 'disabled').length > 0 ? (
            <div className="space-y-2">
              {llmModels
                .filter((model) => model.name !== 'disabled')
                .map((model) => (
                  <div
                    key={model.name}
                    className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded-lg"
                  >
                    <div>
                      <span className="font-medium">{model.display_name}</span>
                      {model.size_mb && (
                        <span className="text-gray-500 dark:text-gray-400 text-sm ml-2">
                          ({(model.size_mb / 1024).toFixed(1)} GB)
                        </span>
                      )}
                    </div>
                    <button
                      onClick={() => handleDeleteLlm(model.name)}
                      className="p-2 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/30 rounded"
                      title="删除"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                ))}
            </div>
          ) : (
            <div className="text-center text-gray-400 dark:text-gray-500 py-8">
              暂无 LLM 模型，请上传或手动放入 models/llm/ 目录
            </div>
          )}
        </section>


        {/* LLM 参数设置 */}
        <section className="bg-white dark:bg-gray-800 rounded-lg shadow dark:shadow-gray-900/50 p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Cpu className="w-5 h-5" />
            LLM 参数
          </h2>

          {llmSettings ? (
            <div className="space-y-4">
              {/* 滑块 */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-200">上下文长度 (n_ctx)</label>
                  <input
                    type="text"
                    value={nCtxInput}
                    onChange={(e) => setNCtxInput(e.target.value)}
                    onBlur={handleNCtxInputBlur}
                    onKeyDown={(e) => { if (e.key === 'Enter') handleNCtxInputBlur() }}
                    className="w-20 text-right text-sm font-mono border border-gray-300 dark:border-gray-600 rounded px-2 py-1 dark:bg-gray-700 dark:text-gray-200"
                  />
                </div>
                <input
                  type="range"
                  min={1024}
                  max={32768}
                  step={1024}
                  value={nCtxValue}
                  onChange={handleNCtxSlider}
                  className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-primary-600"
                />
                <div className="flex justify-between text-xs text-gray-400 dark:text-gray-500 mt-1">
                  <span>1024</span>
                  <span>32768</span>
                </div>
              </div>

              {/* 推荐值 */}
              <div className="flex items-center gap-3 text-sm">
                <span className="text-gray-500 dark:text-gray-400">
                  推荐:{' '}
                  <span className="font-medium text-green-600 dark:text-green-400">{llmSettings.recommended_n_ctx}</span>
                  {llmSettings.gpu_vram_gb != null && (
                    <span className="text-gray-400 dark:text-gray-500 ml-1">
                      (基于 {llmSettings.gpu_vram_gb}GB 显存)
                    </span>
                  )}
                </span>
                {nCtxValue !== llmSettings.recommended_n_ctx && (
                  <button
                    onClick={handleApplyRecommended}
                    className="text-xs px-2 py-0.5 bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-400 border border-green-200 dark:border-green-700 rounded hover:bg-green-100 dark:hover:bg-green-900/50"
                  >
                    应用推荐
                  </button>
                )}
              </div>

              {/* 提示 */}
              <div className="flex items-start gap-2 text-xs text-gray-500 dark:text-gray-400 bg-gray-50 dark:bg-gray-900 rounded p-3">
                <Info className="w-4 h-4 mt-0.5 flex-shrink-0 text-gray-400 dark:text-gray-500" />
                <div>
                  <p>较大值可处理更长会议，但占用更多显存。</p>
                  <p>修改后在下次使用 LLM 时生效。重启后端将恢复 .env 默认值。</p>
                </div>
              </div>

              {/* 保存按钮 */}
              <div className="flex items-center justify-end gap-3">
                {llmSaveMsg && (
                  <span className={clsx(
                    'text-sm',
                    llmSaveMsg === '已保存' ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                  )}>
                    {llmSaveMsg}
                  </span>
                )}
                <button
                  onClick={handleSaveLlmSettings}
                  disabled={llmSaving || nCtxValue === llmSettings.n_ctx}
                  className="px-4 py-2 bg-primary-600 text-white rounded hover:bg-primary-700 disabled:bg-gray-300 dark:disabled:bg-gray-600 disabled:cursor-not-allowed text-sm"
                >
                  {llmSaving ? '保存中...' : '保存'}
                </button>
              </div>
            </div>
          ) : (
            <div className="text-gray-400 dark:text-gray-500">加载中...</div>
          )}
        </section>

        {/* 性别检测模型管理 */}
        <section className="bg-white dark:bg-gray-800 rounded-lg shadow dark:shadow-gray-900/50 p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <HardDrive className="w-5 h-5" />
              性别检测模型
            </h2>
            <div className="flex items-center gap-2">
              <button
                onClick={() => genderInputRef.current?.click()}
                disabled={isUploading}
                className="flex items-center gap-2 px-3 py-2 bg-primary-600 text-white rounded hover:bg-primary-700 disabled:bg-gray-300 dark:disabled:bg-gray-600"
              >
                <Upload className="w-4 h-4" />
                上传模型
              </button>
              <input
                ref={genderInputRef}
                type="file"
                accept=".zip,.tar.gz"
                className="hidden"
                onChange={handleGenderUpload}
              />
            </div>
          </div>

          <div className="text-sm text-gray-500 dark:text-gray-400 mb-4 space-y-1">
            <p>上传 .zip 或 .tar.gz 压缩包。支持的引擎：</p>
            <ul className="list-disc list-inside ml-2 text-xs space-y-0.5">
              <li><span className="font-medium text-gray-700 dark:text-gray-200">基频分析 (f0)</span> — 内置，无需模型文件，不可删除</li>
              <li><span className="font-medium text-blue-600 dark:text-blue-400">transformers</span> — 含 config.json + model.safetensors（ECAPA-TDNN / Wav2Vec2 等音频分类模型）</li>
            </ul>
            <p className="text-xs">transformers 框架内的 AutoModelForAudioClassification 模型可自由替换。</p>
          </div>

          {genderModels.length > 0 ? (
            <div className="space-y-2">
              {genderModels.map((model) => (
                <div
                  key={model.name}
                  className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded-lg"
                >
                  <div>
                    <span className="font-medium">{model.display_name}</span>
                    {model.size_mb != null && (
                      <span className="text-gray-500 dark:text-gray-400 text-sm ml-2">
                        ({model.size_mb > 1024 ? `${(model.size_mb / 1024).toFixed(1)} GB` : `${model.size_mb} MB`})
                      </span>
                    )}
                  </div>
                  {model.name !== 'f0' && (
                    <button
                      onClick={() => handleDeleteGender(model.name)}
                      className="p-2 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/30 rounded"
                      title="删除"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-gray-400 dark:text-gray-500 py-8">
              暂无性别检测模型
            </div>
          )}
        </section>

        {/* 上传进度 */}
        {isUploading && (
          <div className="fixed bottom-4 right-4 bg-white dark:bg-gray-800 shadow-lg dark:shadow-gray-900/50 rounded-lg p-4 flex items-center gap-3">
            <div className="animate-spin rounded-full h-5 w-5 border-2 border-primary-600 border-t-transparent" />
            <span className="text-sm">{uploadProgress}</span>
          </div>
        )}
      </div>
    </div>
  )
}
