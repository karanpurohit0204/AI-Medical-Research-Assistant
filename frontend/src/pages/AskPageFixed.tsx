import { FormEvent, useState } from 'react'
import { Mic, Send, Square, Upload } from 'lucide-react'
import { api } from '../lib/api'
import { useRecorder } from '../hooks/useRecorder'
import { Answer } from '../components/Answer'
import { Status } from '../components/Status'
import type { AskResponse } from '../types'

const suggestions = ['What is the latest evidence on GLP-1 medications for obesity?', 'How effective are SGLT2 inhibitors in heart failure?', 'What are evidence-based treatments for migraine prevention?']

export function AskPage() {
  const [question, setQuestion] = useState(''); const [result, setResult] = useState<AskResponse>(); const [loading, setLoading] = useState(false); const [error, setError] = useState<string | null>(null); const { recording, supported, start, stop } = useRecorder()
  async function execute(action: () => Promise<AskResponse>) { setError(null); setLoading(true); try { setResult(await action()) } catch (error) { setError(error instanceof Error ? error.message : 'Something went wrong.') } finally { setLoading(false) } }
  function submit(event: FormEvent) { event.preventDefault(); if (question.trim().length < 5) return setError('Please enter a medical question of at least 5 characters.'); void execute(() => api.ask(question.trim())) }
  async function toggleRecording() { try { if (recording) { const audio = await stop(); void execute(() => api.voiceAsk(audio)) } else await start() } catch (error) { setError(error instanceof Error ? error.message : 'Microphone access was not available.') } }
  function upload(file?: File) { if (file) void execute(() => api.voiceAsk(file)) }
  return <div className="ask-page"><section className="hero"><p className="eyebrow">Evidence-based research assistant</p><h1>Ask a better clinical question.</h1><p>Search current PubMed evidence and receive a cited, concise answer.</p></section><section className="query-card"><form onSubmit={submit}><label htmlFor="question">Your research question</label><textarea id="question" value={question} onChange={event => setQuestion(event.target.value)} placeholder="e.g. What does recent evidence say about…" rows={4} disabled={loading}/><div className="actions"><button type="button" className={`mic ${recording ? 'recording' : ''}`} onClick={() => void toggleRecording()} disabled={loading || !supported} aria-label={recording ? 'Stop recording and ask' : 'Ask with voice'}>{recording ? <Square size={18}/> : <Mic size={20}/>}<span>{recording ? 'Stop & ask' : 'Voice question'}</span></button><label className="upload"><Upload size={17}/> Upload audio<input type="file" accept="audio/*,.webm" onChange={event => upload(event.target.files?.[0])} disabled={loading}/></label><button className="submit" type="submit" disabled={loading}><Send size={17}/> Ask Evidence</button></div></form></section><Status loading={loading} error={error}/>{!result && !loading && <section className="suggestions"><p>Try a question</p>{suggestions.map(suggestion => <button key={suggestion} onClick={() => setQuestion(suggestion)}>{suggestion}</button>)}</section>}{result && <Answer result={result}/>}</div>
}
