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

The `/chat` page uses Vercel AI Gateway by default. Configure the frontend before starting Vite:

```powershell
$env:VITE_LLM_PROVIDER="gateway"
$env:VITE_LLM_MODEL="openai/gpt-5-mini"
$env:AI_GATEWAY_API_KEY="your-gateway-key"
```

Start the FastAPI server in the same environment. It relays `/api/gateway/language-model` to Vercel AI Gateway and injects `AI_GATEWAY_API_KEY` server-side; the key is never included in the frontend bundle.

For a future OpenAI-compatible endpoint, use `VITE_LLM_PROVIDER="openai-compatible"`, then set `VITE_LLM_BASE_URL`, `VITE_LLM_MODEL`, and optionally `VITE_LLM_API_KEY`.

Build or preview the production bundle with:

```powershell
npm run frontend:build
npm run frontend:preview
```

The compiled files are written to `inference/frontend/dist/`.
