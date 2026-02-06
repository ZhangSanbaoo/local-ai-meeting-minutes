import { Mic, FolderOpen, Settings } from 'lucide-react'
import { clsx } from 'clsx'
import { FilePage } from './pages/FilePage'
import { RealtimePage } from './pages/RealtimePage'
import { SettingsPage } from './pages/SettingsPage'
import { useAppStore } from './stores/appStore'

export default function App() {
  const activeTab = useAppStore((s) => s.activeTab)
  const setActiveTab = useAppStore((s) => s.setActiveTab)

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      {/* 标题栏 */}
      <header className="bg-white px-5 py-4 shadow-sm">
        <div className="flex items-center gap-3">
          <span className="text-2xl">🎤</span>
          <h1 className="text-xl font-bold text-gray-800">会议纪要 AI</h1>
        </div>
      </header>

      {/* Tab 切换栏 */}
      <nav className="bg-gray-100 px-4 py-2 flex gap-2">
        <button
          onClick={() => setActiveTab('realtime')}
          className={clsx(
            'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors',
            activeTab === 'realtime'
              ? 'bg-blue-50 text-primary-600'
              : 'text-gray-600 hover:bg-gray-200'
          )}
        >
          <Mic className="w-4 h-4" />
          实时录音
        </button>
        <button
          onClick={() => setActiveTab('file')}
          className={clsx(
            'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors',
            activeTab === 'file'
              ? 'bg-blue-50 text-primary-600'
              : 'text-gray-600 hover:bg-gray-200'
          )}
        >
          <FolderOpen className="w-4 h-4" />
          音频文件
        </button>
        <div className="flex-1" />
        <button
          onClick={() => setActiveTab('settings')}
          className={clsx(
            'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors',
            activeTab === 'settings'
              ? 'bg-blue-50 text-primary-600'
              : 'text-gray-600 hover:bg-gray-200'
          )}
        >
          <Settings className="w-4 h-4" />
          设置
        </button>
      </nav>

      {/* 内容区 — FilePage/RealtimePage 用 CSS 隐藏保持状态不丢失 */}
      <main className="flex-1 min-h-0">
        <div className={activeTab === 'realtime' ? 'h-full' : 'hidden'}>
          <RealtimePage />
        </div>
        <div className={activeTab === 'file' ? 'h-full' : 'hidden'}>
          <FilePage />
        </div>
        {activeTab === 'settings' && <SettingsPage />}
      </main>
    </div>
  )
}
