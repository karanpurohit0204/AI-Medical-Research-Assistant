# Evidence frontend

Responsive React client for the FastAPI medical research assistant. It uses the existing backend APIs directly:

| UI flow | API call |
| --- | --- |
| Typed question | `POST /api/query/ask` with JSON `{ query, max_papers, top_k }` |
| Recorded/uploaded question | `POST /api/voice/ask` as `multipart/form-data` |
| Evidence explorer | `POST /api/rag/search` with JSON |
| Generated speech | `GET /api/voice/audio/{filename}` in an HTML audio player |

## Run locally

1. Start the FastAPI server from the project root: `uvicorn backend.main:app --reload --port 8000`
2. In this directory, copy `.env.example` to `.env` and change `VITE_API_BASE_URL` only if the API is hosted elsewhere.
3. Install and run: `npm install` then `npm run dev`

The API client is in `src/lib/api.ts`. All responses are checked before JSON parsing; API error details, loading states, microphone failures, and invalid inputs are surfaced in the UI. This backend currently has no authentication endpoints, so the client intentionally does not fabricate an auth flow.

## Production

Run `npm run build` to create `dist/`. Configure the backend CORS `allow_origins` with the deployed frontend origin before deployment.
