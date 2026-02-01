import { useCallback, useEffect, useRef, useState } from 'react'

interface TranscriptMessage {
  type: 'partial' | 'final' | 'chunk' | 'pong'
  text?: string
}

interface UseWebSocketOptions {
  url: string
  onChunk?: (text: string) => void
  onPartial?: (text: string) => void
  onFinal?: (text: string) => void
  pingInterval?: number
}

export function useWebSocket({
  url,
  onChunk,
  onPartial,
  onFinal,
  pingInterval = 15000,
}: UseWebSocketOptions) {
  const wsRef = useRef<WebSocket | null>(null)
  const pingIntervalRef = useRef<number | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      setIsConnected(true)
      setError(null)

      // Start ping interval to keep connection alive
      pingIntervalRef.current = window.setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'ping' }))
        }
      }, pingInterval)
    }

    ws.onmessage = (event) => {
      try {
        const data: TranscriptMessage = JSON.parse(event.data)
        if (data.type === 'chunk' && data.text) {
          onChunk?.(data.text)
        } else if (data.type === 'partial' && data.text) {
          onPartial?.(data.text)
        } else if (data.type === 'final' && data.text) {
          onFinal?.(data.text)
        }
      } catch {
        // Ignore parse errors
      }
    }

    ws.onerror = () => {
      setError('Connection error')
    }

    ws.onclose = () => {
      setIsConnected(false)
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current)
        pingIntervalRef.current = null
      }
    }
  }, [url, onChunk, onPartial, onFinal, pingInterval])

  const disconnect = useCallback(() => {
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current)
      pingIntervalRef.current = null
    }
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setIsConnected(false)
  }, [])

  const sendAudio = useCallback((data: ArrayBuffer) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(data)
    }
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [disconnect])

  return {
    connect,
    disconnect,
    sendAudio,
    isConnected,
    error,
  }
}
