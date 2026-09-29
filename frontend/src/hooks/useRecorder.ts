import { useRef, useState } from 'react'
export function useRecorder() {
  const recorder = useRef<MediaRecorder | null>(null); const chunks = useRef<Blob[]>([])
  const [recording, setRecording] = useState(false); const [supported] = useState(() => typeof MediaRecorder !== 'undefined')
  async function start() { const stream = await navigator.mediaDevices.getUserMedia({ audio: true }); const media = new MediaRecorder(stream); chunks.current = []; media.ondataavailable = e => chunks.current.push(e.data); media.start(); recorder.current = media; setRecording(true) }
  function stop(): Promise<Blob> { return new Promise((resolve, reject) => { if (!recorder.current) return reject(new Error('No recording in progress')); const media = recorder.current; media.onstop = () => { media.stream.getTracks().forEach(track => track.stop()); setRecording(false); resolve(new Blob(chunks.current, { type: media.mimeType || 'audio/webm' })) }; media.stop() }) }
  return { recording, supported, start, stop }
}
