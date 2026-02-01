import { useCallback, useState } from 'react'
import { TranscriptBox } from '@/components/TranscriptBox'
import { useWebSocket } from '@/hooks/useWebSocket'
import { useAudioRecorder } from '@/hooks/useAudioRecorder'

const WS_URL = 'ws://localhost:8000/ws/transcribe'

// Simple audio bars animation - CSS-based for reliability
const AudioBars = () => (
  <div className="flex items-center justify-center gap-1">
    {[0, 1, 2, 3, 4].map((i) => (
      <div
        key={i}
        className="w-2 rounded-full bg-red-500"
        style={{
          animation: `audioBar 0.8s ease-in-out infinite`,
          animationDelay: `${i * 0.1}s`,
          height: '40px',
        }}
      />
    ))}
    <style>{`
      @keyframes audioBar {
        0%, 100% { transform: scaleY(0.3); }
        50% { transform: scaleY(1); }
      }
    `}</style>
  </div>
)

// Microphone icon for idle state
const MicIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
    <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
    <line x1="12" x2="12" y1="19" y2="22"/>
  </svg>
)

function App() {
  const [transcript, setTranscript] = useState('')

  const handleChunk = useCallback((text: string) => {
    setTranscript(prev => prev ? prev + ' ' + text : text)
  }, [])

  const handlePartial = useCallback((_text: string) => {
    // Partial updates are handled by chunks
  }, [])

  const handleFinal = useCallback((text: string) => {
    if (text.length > transcript.length) {
      setTranscript(text)
    }
  }, [transcript])

  const { connect, disconnect, sendAudio, error: wsError } = useWebSocket({
    url: WS_URL,
    onChunk: handleChunk,
    onPartial: handlePartial,
    onFinal: handleFinal,
  })

  const { startRecording, stopRecording, isRecording, error: audioError } = useAudioRecorder({
    onAudioData: sendAudio,
  })

  const handleToggle = useCallback(async () => {
    if (isRecording) {
      stopRecording()
      disconnect()
    } else {
      setTranscript('')
      connect()
      await startRecording()
    }
  }, [isRecording, connect, disconnect, startRecording, stopRecording])

  const error = wsError || audioError

  return (
    <div className="dark mx-auto flex min-h-screen max-w-2xl flex-col items-center justify-center p-6">
      {/* Recording Button */}
      <button
        onClick={handleToggle}
        className={`relative mb-4 flex h-32 w-32 items-center justify-center rounded-full transition-all duration-300 ${
          isRecording
            ? 'bg-red-500/20 shadow-lg shadow-red-500/30'
            : 'bg-zinc-800 hover:bg-zinc-700 hover:shadow-lg'
        }`}
      >
        {isRecording ? (
          <AudioBars />
        ) : (
          <div className="text-zinc-300">
            <MicIcon />
          </div>
        )}
      </button>

      {/* Status Text */}
      <p className="mb-8 text-sm text-zinc-500">
        {error ? (
          <span className="text-red-400">{error}</span>
        ) : isRecording ? (
          'Tap to stop recording'
        ) : (
          'Tap to start recording'
        )}
      </p>

      {/* Transcript */}
      <div className="w-full">
        <TranscriptBox transcript={transcript} isRecording={isRecording} />
      </div>
    </div>
  )
}

export default App
