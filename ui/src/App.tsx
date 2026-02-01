import { useCallback, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Visualizer } from '@/components/Visualizer'
import { TranscriptBox } from '@/components/TranscriptBox'
import { useWebSocket } from '@/hooks/useWebSocket'
import { useAudioRecorder } from '@/hooks/useAudioRecorder'

const WS_URL = 'ws://localhost:8000/ws/transcribe'

function App() {
  const [transcript, setTranscript] = useState('')
  const [currentChunk, setCurrentChunk] = useState('')

  // Handle new chunk from server - accumulate into transcript
  const handleChunk = useCallback((text: string) => {
    setCurrentChunk(text)
    setTranscript(prev => prev ? prev + ' ' + text : text)
  }, [])

  const handlePartial = useCallback((text: string) => {
    setCurrentChunk(text)
  }, [])

  const handleFinal = useCallback((text: string) => {
    if (text.length > transcript.length) {
      setTranscript(text)
    }
    setCurrentChunk('')
  }, [transcript])

  const { connect, disconnect, sendAudio, isConnected, error: wsError } = useWebSocket({
    url: WS_URL,
    onChunk: handleChunk,
    onPartial: handlePartial,
    onFinal: handleFinal,
  })

  const { startRecording, stopRecording, getFrequencyData, isRecording, error: audioError } = useAudioRecorder({
    onAudioData: sendAudio,
  })

  const handleStart = useCallback(async () => {
    setTranscript('')
    setCurrentChunk('')
    connect()
    await startRecording()
  }, [connect, startRecording])

  const handleStop = useCallback(() => {
    stopRecording()
    disconnect()
  }, [stopRecording, disconnect])

  const error = wsError || audioError

  return (
    <div className="dark mx-auto min-h-screen max-w-2xl p-6">
      <div className="mb-8 text-center">
        <h1 className="mb-2 text-3xl font-bold text-primary">Lightning Whisper</h1>
        <p className="text-muted-foreground">
          Real-time Speech-to-Text on Apple Silicon with SimulWhisper
        </p>
      </div>

      <div className="mb-6 flex justify-center">
        {error ? (
          <Badge variant="destructive">{error}</Badge>
        ) : isRecording ? (
          <Badge variant="destructive" className="animate-pulse">
            Recording...
          </Badge>
        ) : isConnected ? (
          <Badge variant="secondary">Connected</Badge>
        ) : (
          <Badge variant="outline">Click Start to begin</Badge>
        )}
      </div>

      <div className="mb-6">
        <Visualizer
          getFrequencyData={getFrequencyData}
          isActive={isRecording}
        />
      </div>

      <div className="mb-6 flex justify-center gap-4">
        <Button
          size="lg"
          onClick={handleStart}
          disabled={isRecording}
          className={isRecording ? 'animate-pulse bg-destructive' : ''}
        >
          {isRecording ? 'Recording...' : 'Start Recording'}
        </Button>
        <Button
          size="lg"
          variant="destructive"
          onClick={handleStop}
          disabled={!isRecording}
        >
          Stop
        </Button>
      </div>

      <TranscriptBox transcript={transcript} partialText={currentChunk} isRecording={isRecording} />
    </div>
  )
}

export default App
