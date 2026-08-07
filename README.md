# Bus Demand Prediction

[Türkçe dokümantasyon](README.tr.md)

This repository predicts the expected passenger demand for a proposed intercity bus trip. It is a **demand model**, not a profitability model: fares, capacity, operating costs, commissions, and taxes are outside its scope.

The prediction input is a company, its route code, a departure date, and a departure time. The serving pipeline returns a numeric demand estimate, threshold probabilities, a demand label, and an evidence-based reliability assessment.

> **Target:** `SEFER_SAYISI` (used in this project as passenger demand)<br>
> **Data:** 9,103,971 cleaned trip rows, from 2023-01-01 through 2026-04-14<br>
> **Current serving candidate:** v4.2 hybrid numeric prediction + v4.4 threshold classifiers

## Repository map

```text
training/
├── analysis.duckdb              # DuckDB database used by training and inference
├── models/                      # Frozen CatBoost models and classifier metadata
├── results/                     # Frozen v4.2 hybrid rule and experiment outputs
├── inference/                   # Self-contained, read-only production inference
│   ├── engine.py                # Model loading, prediction flow, reliability
│   ├── features.py              # Feature contract and DuckDB configuration
│   ├── batch_features.py        # Vectorized feature builder for proposals
│   ├── hybrid.py                # Frozen v4.2 blending rule
│   ├── classifiers.py           # v4.4 model/metadata validation
│   ├── predict_trip.py          # One-trip CLI
│   ├── predict_trips_batch.py   # Batch CSV CLI
│   ├── check_trips.py           # Audit CLI with review flags
│   ├── stop_addition/           # Proposed-stop inference, uplift, and rules
│   ├── api/                     # Shared FastAPI service for both projects
│   └── frontend/                # Preact/Vite interface
├── scripts/                     # Versioned training and evaluation workflows
│   ├── shared/                  # Shared constants, DuckDB pipeline, metrics, and model loading
│   ├── baseline/                # Historical-average baselines using the shared runner
│   ├── stop_addition/           # Stop-addition data, scenarios, and model training
│   │   ├── build_training_data.py
│   │   ├── build_training_scenarios.py
│   │   └── train_demand_model.py
│   └── catboost/v1 … v4_5/      # Versioned model experiments and evaluations
├── archive/                     # Local historical notes and test artifacts
├── pyproject.toml               # Python dependencies and version requirement
├── package.json                 # Frontend/backend development commands
└── README.tr.md                 # Turkish documentation
```

`scripts/` contains the training and evaluation workflows. The newer workflows use `scripts/shared/`; older versioned experiments remain as standalone historical implementations. Inference is intentionally self-contained and does not import from `scripts/`. The shared modules provide:

| Module | Purpose |
|---|---|
| `scripts/shared/paths.py` | Project directory, database, and model paths |
| `scripts/shared/constants.py` | Feature-column definitions, grouping keys, and source columns |
| `scripts/shared/period_config.py` | `PeriodConfig` dataclass and standard train/validation/test/final periods |
| `scripts/shared/feature_pipeline.py` | DuckDB feature construction: source tables, long-term statistics, recent windows, feature assembly, and leakage validation |
| `scripts/shared/metrics.py` | Regression, classification, and probability metrics |
| `scripts/shared/model_utils.py` | CatBoost model loading with feature-contract validation |
| `scripts/shared/baseline_runner.py` | Parametric hierarchical baseline runner |

`GUZERGAH_KODU` is company-specific. The safe route identity is therefore `FIRMA_ID + GUZERGAH_KODU`. `guzergah_canonical` also maps each company route to a canonical physical route, so the model can use both company-specific history and shared physical-route history.

## Stack

Python 3.14+ is managed with [uv](https://docs.astral.sh/uv/). The core Python libraries are CatBoost, DuckDB, pandas, scikit-learn, FastAPI, Uvicorn, and Matplotlib. Training uses CatBoost GPU mode (`task_type="GPU"`, device `0`); inference only needs a compatible CatBoost runtime and the frozen model files.

The web client uses Preact, Vite, TypeScript, Tailwind CSS, Leaflet, and OpenStreetMap tiles. Its dependencies are managed with npm or Bun, according to `package.json` and `bun.lock`.

DuckDB performs the heavy work—filtering, joins, grouped aggregates, quantiles, and feature assembly. Pandas receives only completed feature matrices and handles small outputs, metrics, and CSV files. This avoids loading millions of raw history rows into a pandas `groupby` workflow.

## Install and validate

Run commands from the repository root.

```powershell
uv sync
```

Before serving predictions, confirm that `analysis.duckdb`, the required frozen model files in `models/`, and `results/catboost_v4_2_hybrid_rule.json` are present. The database is read during inference; the inference pipeline does not train models, change model files, or write permanent database objects.

The frozen serving models and the small runtime result files (the v4.2 hybrid rule and the stop-addition route options) are committed to Git with Git LFS, so a fresh clone already contains them. `analysis.duckdb` is large and remains out of version control; provide it separately before inference or the API demo can run. Experiment outputs under `results/` that are not part of the frozen serving set, and the local evaluation archive, also remain excluded from Git.

## Data model and cleaning

`analysis.duckdb` is the main database. `model_data_base` is the model-facing trip relation and contains the trip ID, departure date and time, company and route IDs, canonical route ID, target, calendar fields, departure minute, and 30-minute departure bucket. The target is copied from `SEFER_SAYISI` and is limited to the cleaned range 1–300.

The model dataset was produced with three important safeguards:

- records above 300 were excluded as suspicious source errors rather than clipped;
- trips without a valid company-route-to-canonical-route mapping were excluded;
- duplicate route-stop records were removed before ordered stop lists were built.

Raw route-stop rows must never be joined directly to trip rows. Doing so would duplicate each trip once per stop and corrupt counts, averages, and targets.

The cleaned dataset contains 9,103,971 rows:

| Year | Rows | Mean target | Median target |
|---|---:|---:|---:|
| 2023 | 2,587,983 | 32.13 | 31 |
| 2024 | 2,797,931 | 31.17 | 29 |
| 2025 | 2,937,192 | 31.79 | 29 |
| 2026 through April 14 | 780,865 | 30.81 | 28 |

## Run inference

### One proposed trip

```powershell
uv run python inference/predict_trip.py `
  --firma-id 123 --guzergah-kodu 456 `
  --date 2026-08-01 --time 14:30
```

Use `--debug` to print normalized lookup keys, matching-history counts, and the selected baseline source.

### CSV batch

```powershell
uv run python inference/predict_trips_batch.py --input input.csv --output output.csv
```

The input must contain these columns:

```csv
FIRMA_ID,GUZERGAH_KODU,SEFER_TARIHI,SEFER_SAATI
123,456,2026-08-01,14:30
```

### Audit selected trips

```powershell
uv run python inference/check_trips.py --input input.csv --output results/check.csv
```

In addition to the prediction, the audit output includes strict-earlier route, exact-time, and weekday-time history statistics, nine review flags, and `any_review_flag`.

### Stop-addition evaluation

The stop-addition runtime is part of the same `inference/` backend and does not
import anything from `scripts/`. The optional frozen 65-row regression fixture
is kept in the local archive and can be run when that archive is available:

```powershell
uv run python -m inference.stop_addition.evaluator `
  --input archive/stop_addition_tests/fixtures/requests_65.csv `
  --output stop_addition_evaluation.csv
```

The web demo offers two stop-addition modes:

- **Scheduled trip** uses the requested departure date and time and keeps the
  existing stop-addition demand model unchanged.
- **General route** needs only the company, current route, and candidate stop.
  It estimates typical demand from historical route averages without using a
  departure date, departure time, or demand classifier.

Both modes validate the request, insert the proposed stop at the
minimum-Haversine detour position, compare proposed and current demand, and
apply the same `APPROVE`/`REVIEW`/`REJECT` rules. Distance increase, insertion
position, route similarity, company-origin approval, current-route viability,
and evidence-quality checks remain active in both modes.

For general-route demand, historical evidence is used in this order:

1. same company on the exact proposed route;
2. all companies on the exact proposed route;
3. similar routes;
4. the current route;
5. the company's overall history.

The time-independent CatBoost candidate was not promoted: validation grouped
MAE was `6.1737` versus `5.3551` for the historical estimator, and approval
accuracy was also lower. The general-route mode therefore uses the historical
estimate directly and assigns demand bands with the fixed 10/20/30/43 limits.

### How proposed-route prediction works

Stop addition answers a counterfactual question: **what demand should we expect
if a candidate stop is inserted into an existing route?** It does not simply add
a fixed number of passengers to the current-route prediction.

1. The request identifies the company route and candidate stop. Scheduled mode
   also includes departure date and time; general mode does not.
2. The route geometry places that stop between the adjacent stops that produce
   the smallest additional Haversine distance. This creates the proposed ordered
   stop sequence and measures its detour distance and ratio.
3. The feature builder searches only history strictly earlier than the proposal
   date. Evidence falls back from the same company on the exact proposed route,
   to other companies on that route, to geometrically similar routes, and then
   to broader company/route history. The response exposes the resulting evidence
   scenario, so an exact-history estimate can be distinguished from a cold start.
4. The frozen stop-addition demand model predicts demand for the constructed
   route. The ordinary production demand pipeline independently predicts the
   unchanged current route for the same departure.
5. `predicted_uplift` is the proposed-route prediction minus the current-route
   prediction. Business rules combine this uplift with detour size, predicted
   demand, and evidence strength to return `APPROVE`, `REVIEW`, or `REJECT`,
   together with warnings and supporting evidence.

The decision is decision support rather than a profitability forecast. It does
not model ticket price, vehicle capacity, operating cost, or whether a proposed
stop is operationally feasible beyond the route-distance checks.

## What the production pipeline does

For every proposal, it uses only trips dated **strictly before** the proposed departure date. This is the key protection against target leakage.

1. Validates and normalizes the input.
2. Maps the company route to its canonical route.
3. Builds historical features with vectorized DuckDB queries.
4. Predicts a numeric value with the frozen v4.1 CatBoost regressor.
5. Builds a weekday historical baseline and applies the frozen v4.2 rule when its high-demand evidence justifies blending upward.
6. Runs the four v4.4 classifiers for `>=10`, `>=20`, `>=30`, and `>=43` demand.
7. Corrects inconsistent threshold decisions, assigns a demand label, and reports reliability.

The v4.1 regressor expects **64 features**: 6 categorical, 2 calendar/time, 40 long-term historical, and 16 recent-history features. The v4.4 classifiers use those 64 plus 8 distribution features, for **72 features**. Feature names, ordering, types, grouping keys, and fallback behavior are part of the model contract and must not be changed while reusing a frozen `.cbm` file.

The required frozen serving files are:

| File | Role |
|---|---|
| `models/catboost_demand_model_v4_1_recent_mae_6000.cbm` | 64-feature v4.1 demand regressor |
| `results/catboost_v4_2_hybrid_rule.json` | selected v4.2 blending rule |
| `models/catboost_demand_model_v4_4_classifier_ge_10_class_weights_none.cbm` | probability of demand >=10 |
| `models/catboost_demand_model_v4_4_classifier_ge_20_class_weights_none.cbm` | probability of demand >=20 |
| `models/catboost_demand_model_v4_4_classifier_ge_30_class_weights_none.cbm` | probability of demand >=30 |
| `models/catboost_demand_model_v4_4_classifier_ge_43.cbm` | probability of demand >=43 |
| matching `*_metadata.json` files | frozen decision cutoffs and feature contracts |

The `>=10`, `>=20`, and `>=30` classifiers are unweighted variants; `>=43` uses the balanced variant. On startup, inference checks that all files exist and that classifier feature names match the stored metadata and model schema.

### Feature groups

The long-term history is summarized at five levels:

1. company + canonical route + 30-minute bucket + weekday;
2. company + canonical route + 30-minute bucket;
3. canonical route + 30-minute bucket + weekday;
4. canonical route;
5. company.

Each level produces average, median, sample standard deviation, maximum, p90, rates above 60 and 100, and observation count. Recent features use 30, 60, 90, and 180-day windows at the company-route-time-weekday and canonical-route-time-weekday levels. The eight classifier-only distribution features are p10, p25, p75, below-10 rate, and rates above 10, 20, 30, and 40.

The weekday baseline uses the following fallback order: company-route-time-weekday, canonical-route-time-weekday, canonical route, and finally the global historical average.

## How the model was developed

The project progressed from an interpretable historical baseline to a mixed regression and classification system:

| Version | Main change | Outcome |
|---|---|---|
| Baseline 1 | company + route + departure time | initial reference |
| Baseline 2 | added weekday | selected baseline |
| Baseline 3 | added month | rejected because groups became too sparse |
| CatBoost v1 | IDs and calendar features | did not beat the baseline |
| CatBoost v2 | historical means and counts | first clear ML improvement |
| CatBoost v3 | broader historical distribution statistics | improved MAE; moved heavy work to DuckDB |
| CatBoost v4.1 | 30/60/90/180-day recent history | selected production regression |
| CatBoost v4.2 | blends v4.1 with the weekday baseline under high-demand evidence | selected numeric prediction |
| CatBoost v4.3 | eight extra distribution features | rejected after high-demand guardrails worsened |
| CatBoost v4.4 | four independent demand-threshold classifiers | selected classification layer |
| CatBoost v4.5 | religious-holiday features | isolated experiment; not promoted |

MAE is the primary metric because it directly describes average absolute passenger-count error. RMSE is tracked to expose larger misses, while bias is examined by demand band to avoid hiding systematic over- or underprediction.

## Results and limitations

The selected v4.2 hybrid achieved the following MAE / RMSE results:

| Evaluation period | MAE | RMSE |
|---|---:|---:|
| 2025 H1 validation | 9.8537 | 14.5903 |
| 2025 H2 test | 9.8704 | 16.9336 |
| 2026 final evaluation | 9.7391 | 13.9263 |

The 2026 period has already been used for the official final evaluation. It may be rerun to reproduce the frozen result, but must not be used to select a changed model.

The model is strongest in the common demand range and underpredicts rare high-demand trips. It also has no explicit holiday/event, competition, capacity, fare, cancellation, or operating-cost features. Treat `prediction_reliability`, history counts, p90-style context, and threshold probabilities as important companions to the point estimate.

## Reproduce research and train a new experiment

The `scripts/` tree contains the training and evaluation workflows. Each versioned experiment gets its constants and pipeline helpers from `scripts/shared/` and is independent of the other versions. Run each script from the project root rather than moving files or changing its relative paths.

```powershell
# Selected historical-average baseline
uv run python scripts/baseline/time_dayofweek.py

# Frozen v4.1 regression training workflow
uv run python scripts/catboost/v4_1/train.py

# v4.2 hybrid-rule selection on 2025 H1
uv run python scripts/catboost/v4_2/validation.py

# v4.4 threshold-classifier training
uv run python scripts/catboost/v4_4/train.py

# Stop-addition proposed-route training pipeline
uv run python scripts/stop_addition/build_training_data.py
uv run python scripts/stop_addition/build_training_scenarios.py
uv run python scripts/stop_addition/train_demand_model.py

# Chronological evaluation examples
uv run python scripts/catboost/v4_1/test.py
uv run python scripts/catboost/v4_2/test.py
uv run python scripts/catboost/v4_4/test.py
```

These commands can be computationally expensive, need adequate temporary disk space for DuckDB, and GPU training requires CUDA support. The scripts write model/result paths associated with their experiment; do **not** overwrite the frozen production files. Use a new filename and feature-version name for new work.

All evaluations must remain chronological. The established split is 2024 supervised training rows with 2023 history; 2025 H1 validation with 2023–2024 history; 2025 H2 testing with history through 2025-06-30; and the official 2026 final period with history through 2025-12-31.

| Role | Period | Rows |
|---|---|---:|
| Broad historical training source | 2023-01-01 through 2024-12-31 | 5,385,914 |
| Validation | 2025-01-01 through 2025-06-30 | 1,420,182 |
| Test | 2025-07-01 through 2025-12-31 | 1,517,010 |
| Official final evaluation | 2026-01-01 through 2026-04-14 | 780,865 |

The actual v4.1 supervised matrix uses 2024 rows with features derived from 2023. This is why 2023 is history rather than direct supervised training data: equivalent earlier history is not available for 2023 rows.

## API and web interface

Start the backend:

```powershell
npm run backend
```

Open `http://localhost:8000/docs` for the OpenAPI interface. The shared API
exposes scheduled demand inference at `POST /predict`, general route demand at
`POST /predict-general`, scheduled stop-addition evaluation at
`POST /predict-stop-addition`, and time-independent stop-addition evaluation at
`POST /predict-stop-addition-general`. It also provides `GET /health`, stop
lookup under `GET /durak`, company-route lookup under `GET /route`, and
historically observed stop-addition options at
`GET /stop-addition/available-routes`.

In a second terminal, install packages and start the frontend:

```powershell
npm install
npm run frontend
```

Vite proxies `/api` requests to the local FastAPI server. Use `npm run frontend:build` to write the production frontend to `inference/frontend/dist/`.

The `/chat` page keeps its browser-safe provider, model, OpenAI-compatible
base URL, and MCP URL in `inference/frontend/src/pages/Chat/config.ts`. The
current default is `provider: 'openai-compatible'`: the AI SDK sends its
request through `/api/openai-compatible/{path}`, and FastAPI reads
`OPENAI_COMPATIBLE_API_KEY` from the repository-root `.env`, adds it as a
Bearer credential, and relays the request and response stream to the
configured OpenAI-compatible API. The credential is never bundled into the
frontend.

The chat page can alternatively use the Vercel AI Gateway by changing
`provider` to `'gateway'`. That mode sends requests through
`/api/gateway/language-model`; FastAPI reads `AI_GATEWAY_API_KEY` from the
repository-root `.env` and injects it server-side. Neither credential is
bundled into the frontend.

## Rules for future changes

- Use DuckDB for large scans, joins, grouped statistics, and quantiles; fetch only completed model matrices into pandas.
- Use end-exclusive date ranges and only history available before the predicted period.
- Never use a random train/test split for this time-dependent problem.
- Do not tune on the official 2026 final period.
- Do not overwrite the frozen production models, classifier metadata, or hybrid rule.
- Treat any feature-name, order, type, grouping, statistic, or fallback change as a new feature version.
- Verify `model.feature_names_` against the exact ordered feature list before prediction.
- Confirm the upstream business meaning of `SEFER_SAYISI` before presenting the output as passenger demand.
