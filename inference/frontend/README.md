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

Build or preview the production bundle with:

```powershell
npm run frontend:build
npm run frontend:preview
```

The compiled files are written to `inference/frontend/dist/`.
