import { useCallback, useEffect, useRef, useState } from 'react'

interface UseRecordingTimerReturn {
  /** Current elapsed time in seconds */
  elapsedTime: number
  /** Is the timer currently running */
  isRunning: boolean
  /** Start or resume the timer */
  start: () => void
  /** Pause the timer */
  pause: () => void
  /** Stop and reset the timer */
  reset: () => void
}

/**
 * High-precision recording timer using requestAnimationFrame
 * Provides smooth 60fps updates for time display
 */
export function useRecordingTimer(): UseRecordingTimerReturn {
  const [elapsedTime, setElapsedTime] = useState(0)
  const [isRunning, setIsRunning] = useState(false)

  // Refs for timing
  const startTimeRef = useRef<number>(0)
  const pausedTimeRef = useRef<number>(0)
  const rafIdRef = useRef<number | null>(null)

  // Animation loop for smooth time updates
  const updateTime = useCallback(() => {
    const now = performance.now()
    const elapsed = (now - startTimeRef.current + pausedTimeRef.current) / 1000
    setElapsedTime(elapsed)
    rafIdRef.current = requestAnimationFrame(updateTime)
  }, [])

  const start = useCallback(() => {
    if (isRunning) return

    startTimeRef.current = performance.now()
    setIsRunning(true)
    rafIdRef.current = requestAnimationFrame(updateTime)
  }, [isRunning, updateTime])

  const pause = useCallback(() => {
    if (!isRunning) return

    if (rafIdRef.current !== null) {
      cancelAnimationFrame(rafIdRef.current)
      rafIdRef.current = null
    }

    // Save elapsed time for resume
    const now = performance.now()
    pausedTimeRef.current += now - startTimeRef.current

    setIsRunning(false)
  }, [isRunning])

  const reset = useCallback(() => {
    if (rafIdRef.current !== null) {
      cancelAnimationFrame(rafIdRef.current)
      rafIdRef.current = null
    }

    startTimeRef.current = 0
    pausedTimeRef.current = 0
    setElapsedTime(0)
    setIsRunning(false)
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (rafIdRef.current !== null) {
        cancelAnimationFrame(rafIdRef.current)
      }
    }
  }, [])

  return {
    elapsedTime,
    isRunning,
    start,
    pause,
    reset,
  }
}
