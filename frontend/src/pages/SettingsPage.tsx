import { useCallback, useEffect, useRef, useState } from 'react'
import { Upload, Trash2, RefreshCw, HardDrive, Cpu } from 'lucide-react'
import { clsx } from 'clsx'
import { useAppStore } from '../stores/appStore'
import * as api from '../api/client'
import type { SystemInfo } from '../types'

export function SettingsPage() {
  const { whisperModels, llmModels, diarizationModels, genderModels } = useAppStore()
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState('')
  const whisperInputRef = useRef<HTMLInputElement>(null)
  const llmInputRef = useRef<HTMLInputElement>(null)
  const diarInputRef = useRef<HTMLInputElement>(null)
  const genderInputRef = useRef<HTMLInputElement>(null)

  // 加载系统信息
  useEffect(() => {
    api.getSystemInfo().then(setSystemInfo).catch(console.error)
  }, [])

  // 刷新模型列表
  const refreshModels = useCallback(async () => {
    try {
      const data = await api.getModels()
      useAppStore.getState().setModels(
        data.whisper_models,
        data.llm_models,
        data.diarization_models,
        data.gender_models,
      )
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

  // 上传说话人分离模型
  const handleDiarUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    if (!file.name.endsWith('.zip') && !file.name.endsWith('.tar.gz')) {
      alert('只支持 .zip 或 .tar.gz 格式')
      return
    }

    setIsUploading(true)
    setUploadProgress(`上传说话人分离模型: ${file.name}...`)

    try {
      await api.uploadDiarizationModel(file)
      await refreshModels()
      alert('上传成功')
    } catch (err) {
      console.error('上传失败:', err)
      alert('上传失败，请重试')
    } finally {
      setIsUploading(false)
      setUploadProgress('')
      if (diarInputRef.current) {
        diarInputRef.current.value = ''
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

  // 删除说话人分离模型
  const handleDeleteDiar = useCallback(async (modelName: string) => {
    if (!confirm(`确定要删除说话人分离模型 "${modelName}" 吗？`)) return

    try {
      await api.deleteDiarizationModel(modelName)
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
        <section className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Cpu className="w-5 h-5" />
            系统信息
          </h2>
          {systemInfo ? (
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-500">版本:</span>{' '}
                <span className="font-medium">{systemInfo.version}</span>
              </div>
              <div>
                <span className="text-gray-500">CUDA:</span>{' '}
                <span className={clsx('font-medium', systemInfo.cuda_available ? 'text-green-600' : 'text-red-600')}>
                  {systemInfo.cuda_available ? `可用 (${systemInfo.cuda_version})` : '不可用'}
                </span>
              </div>
              {systemInfo.gpu_name && (
                <div className="col-span-2">
                  <span className="text-gray-500">GPU:</span>{' '}
                  <span className="font-medium">{systemInfo.gpu_name}</span>
                </div>
              )}
              <div className="col-span-2">
                <span className="text-gray-500">模型目录:</span>{' '}
                <span className="font-mono text-xs">{systemInfo.models_dir}</span>
              </div>
              <div className="col-span-2">
                <span className="text-gray-500">输出目录:</span>{' '}
                <span className="font-mono text-xs">{systemInfo.output_dir}</span>
              </div>
            </div>
          ) : (
            <div className="text-gray-400">加载中...</div>
          )}
        </section>

        {/* Whisper 模型管理 */}
        <section className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <HardDrive className="w-5 h-5" />
              Whisper 模型 (语音转写)
            </h2>
            <div className="flex items-center gap-2">
              <button
                onClick={refreshModels}
                className="p-2 border border-gray-300 rounded hover:bg-gray-100"
                title="刷新"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
              <button
                onClick={() => whisperInputRef.current?.click()}
                disabled={isUploading}
                className="flex items-center gap-2 px-3 py-2 bg-primary-600 text-white rounded hover:bg-primary-700 disabled:bg-gray-300"
              >
                <Upload className="w-4 h-4" />
                上传模型
              </button>
              <input
                ref={whisperInputRef}
                type="file"
                accept=".zip,.tar.gz"
                className="hidden"
                onChange={handleWhisperUpload}
              />
            </div>
          </div>

          <p className="text-sm text-gray-500 mb-4">
            支持格式: .zip, .tar.gz (faster-whisper 模型)
          </p>

          {whisperModels.length > 0 ? (
            <div className="space-y-2">
              {whisperModels.map((model) => (
                <div
                  key={model.name}
                  className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                >
                  <div>
                    <span className="font-medium">{model.display_name}</span>
                    {model.size_mb && (
                      <span className="text-gray-500 text-sm ml-2">
                        ({model.size_mb} MB)
                      </span>
                    )}
                  </div>
                  <button
                    onClick={() => handleDeleteWhisper(model.name)}
                    className="p-2 text-red-600 hover:bg-red-50 rounded"
                    title="删除"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-gray-400 py-8">
              暂无 Whisper 模型，请上传或手动放入 models/whisper/ 目录
            </div>
          )}
        </section>

        {/* LLM 模型管理 */}
        <section className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <HardDrive className="w-5 h-5" />
              LLM 模型 (智能命名/总结)
            </h2>
            <div className="flex items-center gap-2">
              <button
                onClick={() => llmInputRef.current?.click()}
                disabled={isUploading}
                className="flex items-center gap-2 px-3 py-2 bg-primary-600 text-white rounded hover:bg-primary-700 disabled:bg-gray-300"
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

          <p className="text-sm text-gray-500 mb-4">
            支持格式: .gguf (GGML 量化模型，推荐 Qwen2.5-7B-Instruct)
          </p>

          {llmModels.filter(m => m.name !== 'disabled').length > 0 ? (
            <div className="space-y-2">
              {llmModels
                .filter((model) => model.name !== 'disabled')
                .map((model) => (
                  <div
                    key={model.name}
                    className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                  >
                    <div>
                      <span className="font-medium">{model.display_name}</span>
                      {model.size_mb && (
                        <span className="text-gray-500 text-sm ml-2">
                          ({(model.size_mb / 1024).toFixed(1)} GB)
                        </span>
                      )}
                    </div>
                    <button
                      onClick={() => handleDeleteLlm(model.name)}
                      className="p-2 text-red-600 hover:bg-red-50 rounded"
                      title="删除"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                ))}
            </div>
          ) : (
            <div className="text-center text-gray-400 py-8">
              暂无 LLM 模型，请上传或手动放入 models/llm/ 目录
            </div>
          )}
        </section>

        {/* 说话人分离模型管理 */}
        <section className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <HardDrive className="w-5 h-5" />
              说话人分离模型
            </h2>
            <div className="flex items-center gap-2">
              <button
                onClick={() => diarInputRef.current?.click()}
                disabled={isUploading}
                className="flex items-center gap-2 px-3 py-2 bg-primary-600 text-white rounded hover:bg-primary-700 disabled:bg-gray-300"
              >
                <Upload className="w-4 h-4" />
                上传模型
              </button>
              <input
                ref={diarInputRef}
                type="file"
                accept=".zip,.tar.gz"
                className="hidden"
                onChange={handleDiarUpload}
              />
            </div>
          </div>

          <p className="text-sm text-gray-500 mb-4">
            支持: pyannote (config.yaml), 3D-Speaker/ModelScope (configuration.json)。
            压缩包解压后放入 models/diarization/ 目录。
          </p>

          {diarizationModels.length > 0 ? (
            <div className="space-y-2">
              {diarizationModels.map((model) => (
                <div
                  key={model.name}
                  className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                >
                  <div>
                    <span className="font-medium">{model.display_name}</span>
                    {model.size_mb != null && (
                      <span className="text-gray-500 text-sm ml-2">
                        ({model.size_mb > 1024 ? `${(model.size_mb / 1024).toFixed(1)} GB` : `${model.size_mb} MB`})
                      </span>
                    )}
                  </div>
                  <button
                    onClick={() => handleDeleteDiar(model.name)}
                    className="p-2 text-red-600 hover:bg-red-50 rounded"
                    title="删除"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-gray-400 py-8">
              暂无说话人分离模型，请上传或手动放入 models/diarization/ 目录
            </div>
          )}
        </section>

        {/* 性别检测模型管理 */}
        <section className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <HardDrive className="w-5 h-5" />
              性别检测模型
            </h2>
            <div className="flex items-center gap-2">
              <button
                onClick={() => genderInputRef.current?.click()}
                disabled={isUploading}
                className="flex items-center gap-2 px-3 py-2 bg-primary-600 text-white rounded hover:bg-primary-700 disabled:bg-gray-300"
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

          <p className="text-sm text-gray-500 mb-4">
            内置: 基频分析 (f0)。可上传: ECAPA-TDNN, Wav2Vec2 等 transformers 模型。
            压缩包解压后放入 models/gender/ 目录。
          </p>

          {genderModels.length > 0 ? (
            <div className="space-y-2">
              {genderModels.map((model) => (
                <div
                  key={model.name}
                  className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                >
                  <div>
                    <span className="font-medium">{model.display_name}</span>
                    {model.size_mb != null && (
                      <span className="text-gray-500 text-sm ml-2">
                        ({model.size_mb > 1024 ? `${(model.size_mb / 1024).toFixed(1)} GB` : `${model.size_mb} MB`})
                      </span>
                    )}
                  </div>
                  {model.name !== 'f0' && (
                    <button
                      onClick={() => handleDeleteGender(model.name)}
                      className="p-2 text-red-600 hover:bg-red-50 rounded"
                      title="删除"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-gray-400 py-8">
              暂无性别检测模型
            </div>
          )}
        </section>

        {/* 上传进度 */}
        {isUploading && (
          <div className="fixed bottom-4 right-4 bg-white shadow-lg rounded-lg p-4 flex items-center gap-3">
            <div className="animate-spin rounded-full h-5 w-5 border-2 border-primary-600 border-t-transparent" />
            <span className="text-sm">{uploadProgress}</span>
          </div>
        )}
      </div>
    </div>
  )
}
