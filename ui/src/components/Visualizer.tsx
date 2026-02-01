import { useEffect, useRef } from 'react'

interface VisualizerProps {
  getFrequencyData: () => Uint8Array
  isActive: boolean
  barCount?: number
}

export function Visualizer({
  getFrequencyData,
  isActive,
  barCount = 32,
}: VisualizerProps) {
  const barsRef = useRef<(HTMLDivElement | null)[]>([])
  const animationRef = useRef<number | null>(null)

  useEffect(() => {
    const animate = () => {
      if (!isActive) {
        barsRef.current.forEach((bar) => {
          if (bar) bar.style.height = '4px'
        })
        return
      }

      const data = getFrequencyData()
      barsRef.current.forEach((bar, i) => {
        if (bar) {
          const value = data[i] || 0
          bar.style.height = `${Math.max(4, value / 4)}px`
        }
      })

      animationRef.current = requestAnimationFrame(animate)
    }

    if (isActive) {
      animate()
    } else {
      barsRef.current.forEach((bar) => {
        if (bar) bar.style.height = '4px'
      })
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isActive, getFrequencyData])

  return (
    <div className="flex h-16 items-center justify-center gap-1 rounded-lg bg-card p-4">
      {Array.from({ length: barCount }).map((_, i) => (
        <div
          key={i}
          ref={(el) => { barsRef.current[i] = el }}
          className="w-1 rounded-sm bg-primary transition-[height] duration-100"
          style={{ height: '4px' }}
        />
      ))}
    </div>
  )
}
