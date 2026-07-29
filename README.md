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
│   ├── artifacts/stop_addition/ # Selected stop-addition model and contract
│   ├── tests/stop_addition/     # Frozen 65-row fixture and baseline
│   ├── api/                     # Shared FastAPI service for both projects
│   └── frontend/                # Preact/Vite interface
├── scripts/                     # Versioned training and evaluation workflows
│   ├── shared/                  # Shared constants, DuckDB pipeline, metrics, and model loading
│   ├── baseline/                # Historical-average baselines using the shared runner
│   ├── stop_addition/           # Stop-addition training/reproducibility only
│   └── catboost/v1 … v4_5/      # Reproducible model experiments using shared modules only
├── archive/stop_addition/       # Preserved stop-addition analyses and frozen staging
├── pyproject.toml               # Python dependencies and version requirement
├── package.json                 # Frontend/backend development commands
└── README.tr.md                 # Turkish documentation
```

`scripts/` contains the training and evaluation workflows. Each versioned experiment (`v4_1` through `v4_5`) uses `scripts/shared/` and does not import code from another version. The shared modules provide:

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
import anything from `scripts/`. The frozen 65-row regression can be run with:

```powershell
uv run python -m inference.stop_addition.evaluator `
  --input inference/tests/stop_addition/fixtures/requests_65.csv `
  --output stop_addition_evaluation.csv
```

It validates the request, inserts the proposed stop at the minimum-Haversine
detour position, predicts proposed and current demand, calculates uplift, and
applies the `APPROVE`/`REVIEW`/`REJECT` rules.

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
exposes demand inference at `POST /predict`, complete stop-addition evaluation
at `POST /predict-stop-addition`, `GET /health`, stop lookup under
`GET /durak`, and company-route lookup under `GET /route`.

In a second terminal, install packages and start the frontend:

```powershell
npm install
npm run frontend
```

Vite proxies `/api` requests to the local FastAPI server. Use `npm run frontend:build` to write the production frontend to `inference/frontend/dist/`.

## Rules for future changes

- Use DuckDB for large scans, joins, grouped statistics, and quantiles; fetch only completed model matrices into pandas.
- Use end-exclusive date ranges and only history available before the predicted period.
- Never use a random train/test split for this time-dependent problem.
- Do not tune on the official 2026 final period.
- Do not overwrite the frozen production models, classifier metadata, or hybrid rule.
- Treat any feature-name, order, type, grouping, statistic, or fallback change as a new feature version.
- Verify `model.feature_names_` against the exact ordered feature list before prediction.
- Confirm the upstream business meaning of `SEFER_SAYISI` before presenting the output as passenger demand.
