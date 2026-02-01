import { useCallback, useRef, useState } from 'react'

interface UseAudioRecorderOptions {
  onAudioData: (data: ArrayBuffer) => void
  sampleRate?: number
}

export function useAudioRecorder({
  onAudioData,
  sampleRate = 16000,
}: UseAudioRecorderOptions) {
  const audioContextRef = useRef<AudioContext | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const [isRecording, setIsRecording] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate,
          echoCancellation: true,
          noiseSuppression: true,
        },
      })
      streamRef.current = stream

      const audioContext = new AudioContext({ sampleRate })
      audioContextRef.current = audioContext

      const source = audioContext.createMediaStreamSource(stream)

      // Create analyser for visualization
      const analyser = audioContext.createAnalyser()
      analyser.fftSize = 64
      analyserRef.current = analyser
      source.connect(analyser)

      // Create processor for PCM data
      const processor = audioContext.createScriptProcessor(4096, 1, 1)
      processorRef.current = processor

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0)
        // Convert to 16-bit PCM
        const pcmData = new Int16Array(inputData.length)
        for (let i = 0; i < inputData.length; i++) {
          pcmData[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768))
        }
        onAudioData(pcmData.buffer)
      }

      source.connect(processor)
      processor.connect(audioContext.destination)

      setIsRecording(true)
      setError(null)
    } catch (err) {
      setError('Microphone access denied')
      console.error('Error starting recording:', err)
    }
  }, [onAudioData, sampleRate])

  const stopRecording = useCallback(() => {
    if (processorRef.current) {
      processorRef.current.disconnect()
      processorRef.current = null
    }

    if (audioContextRef.current) {
      audioContextRef.current.close()
      audioContextRef.current = null
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }

    analyserRef.current = null
    setIsRecording(false)
  }, [])

  const getFrequencyData = useCallback(() => {
    if (!analyserRef.current) {
      return new Uint8Array(32)
    }
    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount)
    analyserRef.current.getByteFrequencyData(dataArray)
    return dataArray
  }, [])

  return {
    startRecording,
    stopRecording,
    getFrequencyData,
    isRecording,
    error,
  }
}
