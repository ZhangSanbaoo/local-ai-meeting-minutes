import { useEffect, useRef } from 'react'
import { Mic, FolderOpen, Settings, Sun, Moon } from 'lucide-react'
import { clsx } from 'clsx'
import { FilePage } from './pages/FilePage'
import { RealtimePage } from './pages/RealtimePage'
import { SettingsPage } from './pages/SettingsPage'
import { useAppStore } from './stores/appStore'

export default function App() {
  const activeTab = useAppStore((s) => s.activeTab)
  const setActiveTab = useAppStore((s) => s.setActiveTab)
  const isDarkMode = useAppStore((s) => s.isDarkMode)
  const toggleDarkMode = useAppStore((s) => s.toggleDarkMode)
  const toggleBtnRef = useRef<HTMLButtonElement>(null)

  // 页面加载时同步 dark 类到 <html>
  useEffect(() => {
    document.documentElement.classList.toggle('dark', isDarkMode)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // 从按钮中心向外扩散的圆形主题切换动效
  const handleToggleDarkMode = () => {
    const btn = toggleBtnRef.current
    if (btn) {
      const rect = btn.getBoundingClientRect()
      const x = Math.round(rect.left + rect.width / 2)
      const y = Math.round(rect.top + rect.height / 2)
      document.documentElement.style.setProperty('--vt-x', `${x}px`)
      document.documentElement.style.setProperty('--vt-y', `${y}px`)
    }
    // View Transitions API（Chrome 111+），降级直接切换
    const startVT = (document as Document & { startViewTransition?: (cb: () => void) => void }).startViewTransition
    if (startVT) {
      startVT.call(document, toggleDarkMode)
    } else {
      toggleDarkMode()
    }
  }

  return (
    <div className="h-screen flex flex-col bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-200">
      {/* 标题栏 */}
      <header className="bg-white dark:bg-gray-800 px-5 py-4 shadow-sm dark:shadow-gray-900/50">
        <div className="flex items-center gap-3">
          <span className="text-2xl">🎤</span>
          <h1 className="text-xl font-bold text-gray-800 dark:text-gray-100">会议纪要 AI</h1>
          <div className="flex-1" />
          {/* 暗黑模式切换滑块 */}
          <button
            ref={toggleBtnRef}
            onClick={handleToggleDarkMode}
            className="relative w-14 h-7 rounded-full transition-colors duration-300 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary-500
              bg-gray-300 dark:bg-gray-600"
            title={isDarkMode ? '切换到亮色模式' : '切换到暗黑模式'}
            role="switch"
            aria-checked={isDarkMode}
          >
            {/* 滑块圆球 */}
            <span
              className={clsx(
                'absolute top-0.5 left-0.5 w-6 h-6 rounded-full bg-white shadow-md flex items-center justify-center transition-transform duration-500 ease-[cubic-bezier(0.34,1.56,0.64,1)]',
                isDarkMode && 'translate-x-7'
              )}
            >
              {isDarkMode
                ? <Moon className="w-3.5 h-3.5 text-indigo-500" />
                : <Sun className="w-3.5 h-3.5 text-amber-500" />
              }
            </span>
          </button>
        </div>
      </header>

      {/* Tab 切换栏 */}
      <nav className="bg-gray-100 dark:bg-gray-800/50 px-4 py-2 flex gap-2">
        <button
          onClick={() => setActiveTab('realtime')}
          className={clsx(
            'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors',
            activeTab === 'realtime'
              ? 'bg-blue-50 dark:bg-blue-900/30 text-primary-600 dark:text-blue-400'
              : 'text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700'
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
              ? 'bg-blue-50 dark:bg-blue-900/30 text-primary-600 dark:text-blue-400'
              : 'text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700'
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
              ? 'bg-blue-50 dark:bg-blue-900/30 text-primary-600 dark:text-blue-400'
              : 'text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700'
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
