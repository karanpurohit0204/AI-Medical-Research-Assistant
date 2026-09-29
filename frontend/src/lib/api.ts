import type { AskResponse, Health, RAGResponse } from '../types'

export const API_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '')
class ApiError extends Error { constructor(message: string, public status?: number) { super(message) } }
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try { response = await fetch(`${API_URL}${path}`, init) } catch { throw new ApiError('Unable to reach the API. Check that the backend is running.') }
  if (!response.ok) { const body = await response.json().catch(() => null); throw new ApiError(body?.detail || `Request failed (${response.status})`, response.status) }
  return response.json() as Promise<T>
}
export const api = {
  health: () => request<Health>('/health'),
  ask: (query: string) => request<AskResponse>('/api/query/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query, max_papers: 5, top_k: 3 }) }),
  search: (query: string) => request<RAGResponse>('/api/rag/search', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query, max_pubmed_results: 8, top_k: 5 }) }),
  transcribe: (file: Blob) => { const data = new FormData(); data.append('file', file, 'recording.webm'); return request<{ text: string }>('/api/stt/transcribe', { method: 'POST', body: data }) },
  voiceAsk: (file: Blob) => { const data = new FormData(); data.append('file', file, 'recording.webm'); data.append('max_papers', '5'); data.append('top_k', '3'); return request<AskResponse>('/api/voice/ask', { method: 'POST', body: data }) },
  audioUrl: (filename: string) => `${API_URL}/api/voice/audio/${encodeURIComponent(filename)}`
}
