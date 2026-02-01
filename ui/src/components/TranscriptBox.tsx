interface TranscriptBoxProps {
  transcript: string
  isRecording?: boolean
}

// Pulsing dots indicator
function WaitingIndicator() {
  return (
    <span className="ml-1 inline-flex items-center gap-0.5">
      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-zinc-500 [animation-delay:-0.3s]" />
      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-zinc-500 [animation-delay:-0.15s]" />
      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-zinc-500" />
    </span>
  )
}

export function TranscriptBox({ transcript, isRecording }: TranscriptBoxProps) {
  const displayText = transcript || ''
  const isEmpty = !displayText && !isRecording

  return (
    <div className="min-h-[200px] rounded-2xl bg-zinc-900 p-6">
      {isEmpty ? (
        <p className="text-center text-zinc-600">Your transcription will appear here...</p>
      ) : (
        <div className="whitespace-pre-wrap text-lg leading-relaxed">
          <span className="text-zinc-100">{displayText}</span>
          {isRecording && <WaitingIndicator />}
        </div>
      )}
    </div>
  )
}
