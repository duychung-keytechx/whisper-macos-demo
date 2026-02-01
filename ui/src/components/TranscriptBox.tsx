import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

interface TranscriptBoxProps {
  transcript: string
  partialText: string
  isRecording?: boolean
}

// Pulsing dots indicator
function WaitingIndicator() {
  return (
    <span className="inline-flex items-center gap-0.5 ml-1">
      <span className="h-2 w-2 rounded-full bg-gray-400 animate-bounce [animation-delay:-0.3s]" />
      <span className="h-2 w-2 rounded-full bg-gray-400 animate-bounce [animation-delay:-0.15s]" />
      <span className="h-2 w-2 rounded-full bg-gray-400 animate-bounce" />
    </span>
  )
}

export function TranscriptBox({ transcript, partialText, isRecording }: TranscriptBoxProps) {
  // Show accumulated transcript
  const displayText = transcript || ''

  return (
    <Card className="min-h-[300px] bg-white">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-normal text-gray-600">
          Transcription
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="whitespace-pre-wrap text-lg leading-relaxed">
          <span className="text-black">{displayText}</span>
          {isRecording && <WaitingIndicator />}
        </div>
      </CardContent>
    </Card>
  )
}
