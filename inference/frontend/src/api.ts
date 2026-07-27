// API client for the passenger-demand demo backend.
//
// All requests go through the Vite dev-server proxy: the `/api` prefix is
// stripped by the proxy (see vite.config.ts) and forwarded to the FastAPI
// backend running at http://localhost:8000.

const API_BASE = '/api';

export type DemandLabel =
	| 'CLEAR_FAILURE'
	| 'WEAK_DEMAND'
	| 'MODERATE_DEMAND'
	| 'STRONG_DEMAND'
	| 'CAPACITY_PRESSURE';

export type Reliability =
	| 'NO_HISTORY'
	| 'UNSAFE'
	| 'LOW'
	| 'MEDIUM'
	| 'HIGH';

export interface ThresholdProbabilities {
	ge_10: number;
	ge_20: number;
	ge_30: number;
	ge_43: number;
}

export interface ReliabilityEvidence {
	exact_time_weekday_count: number;
	exact_time_count: number;
	company_route_count: number;
	canonical_time_weekday_count: number;
	canonical_route_count: number;
	baseline_source: string;
	frequent_departure_times: FrequentDepartureTime[];
}

export interface FrequentDepartureTime {
	departure_time: string;
	trip_count: number;
	route_share: number;
}

export interface HealthResponse {
	status: 'ok';
	database: 'ok';
	artifacts: 'loaded';
}

export interface SimplifiedPrediction {
	expected_demand: number;
	baseline_demand: number;
	demand_label: DemandLabel;
	reliability: Reliability;
	reliability_reason: string;
	probabilities: ThresholdProbabilities;
	reliability_evidence: ReliabilityEvidence;
}

export interface PredictionRequest {
	firma_id: number;
	guzergah_kodu: number;
	sefer_tarihi: string; // YYYY-MM-DD
	sefer_saati: string; // HH:MM
}

export interface RouteDurak {
	sira: number;
	durak_id: number;
	durak_adi: string | null;
	kisa_adi: string | null;
	il_id: number | null;
	ilce_id: number | null;
	enlem: number | null;
	boylam: number | null;
}

export interface RouteDetailResponse {
	firma_id: number;
	guzergah_kodu: number;
	canonical_guzergah_id: number;
	duraklar: RouteDurak[];
}

export interface Durak {
	id: number;
	uetds_kodu: string | null;
	turu: string | null;
	uetds_adi: string | null;
	il_id: number | null;
	ilce_id: number | null;
	kisa_adi: string | null;
	ulke_id: number | null;
	ulke_adi: string | null;
	enlem: number | null;
	boylam: number | null;
}

async function jsonFetch<T>(
	url: string,
	init?: RequestInit,
): Promise<T> {
	const response = await fetch(url, {
		headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
		...init,
	});
	if (!response.ok) {
		let detail = `${response.status} ${response.statusText}`;
		try {
			const body = await response.json();
			if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
		} catch {
			// ignore JSON parse errors
		}
		throw new ApiError(response.status, detail);
	}
	return (await response.json()) as T;
}

export class ApiError extends Error {
	constructor(readonly status: number, message: string) {
		super(message);
		this.name = 'ApiError';
	}
}

// GET /health
export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
	return jsonFetch<HealthResponse>(`${API_BASE}/health`, { signal });
}

// POST /predict
export function predict(
	payload: PredictionRequest,
	options: { detail?: boolean } = {},
): Promise<SimplifiedPrediction> {
	const url = `${API_BASE}/predict${options.detail ? '?detail=true' : ''}`;
	return jsonFetch<SimplifiedPrediction>(url, {
		method: 'POST',
		body: JSON.stringify(payload),
	});
}

// GET /route/{firmaId}/{guzergahKodu}
export function getRoute(
	firmaId: number,
	guzergahKodu: number,
): Promise<RouteDetailResponse> {
	return jsonFetch<RouteDetailResponse>(
		`${API_BASE}/route/${firmaId}/${guzergahKodu}`,
	);
}

// GET /durak/{durakId}
export function getDurak(durakId: number): Promise<Durak> {
	return jsonFetch<Durak>(`${API_BASE}/durak/${durakId}`);
}
