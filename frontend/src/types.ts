export interface Citation { pmid: string; title: string; authors: string; journal: string; year: string; url: string; relevance_score: number }
export interface AskResponse { query?: string; transcribed_query?: string; answer: string; citations: Citation[]; citations_text: string; confidence_score: number; confidence_reasoning: string; llm_model: string; processing_time_ms: number; audio_filename?: string; stages_ms?: Record<string, number> }
export interface RAGResponse { query: string; papers: Citation[]; total_papers_fetched: number; total_papers_stored: number; processing_time_ms: number }
export interface Health { status: string; app: string; whisper_model_loaded: boolean }
