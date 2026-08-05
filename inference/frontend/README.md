# Demand Prediction Frontend

The frontend is a Preact, Vite, TypeScript, Tailwind CSS, and Leaflet application. Commands are run from the repository root:

```powershell
npm install
npm run backend
```

Start the frontend in a second terminal:

```powershell
npm run frontend
```

The development server is available at `http://localhost:5173` and proxies `/api` requests to FastAPI at `http://localhost:8000`.

The `/chat` page uses Vercel AI Gateway by default. Configure its browser-safe
settings—provider, model, LLM base URL, and MCP endpoint—in
`src/pages/Chat/config.ts`. This file is bundled into the frontend, so it must
never contain an API key.

Set `AI_GATEWAY_API_KEY` only for the FastAPI backend, for example in the
repository-root `.env` used by `npm run backend`. FastAPI relays
`/api/gateway/language-model` to Vercel AI Gateway and injects the key
server-side; the browser never receives it.

To use an OpenAI-compatible endpoint, change `provider`, `model`, and
`openAICompatibleBaseUrl` in `src/pages/Chat/config.ts`, then set
`OPENAI_COMPATIBLE_API_KEY` in the repository-root `.env`. The AI SDK still
runs in the frontend, but its request is sent to
`/api/openai-compatible/{path}`; FastAPI adds the key before forwarding it.
The key is never included in the frontend bundle or browser request.

Build or preview the production bundle with:

```powershell
npm run frontend:build
npm run frontend:preview
```

The compiled files are written to `inference/frontend/dist/`.
