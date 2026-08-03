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

export interface GeneralPredictionRequest {
	firma_id: number;
	guzergah_kodu: number;
}

export type GeneralBaselineSource = 'company_route' | 'canonical_route' | 'global';

export interface GeneralPrediction {
	FIRMA_ID: number;
	GUZERGAH_KODU: number;
	canonical_route_id: number;
	expected_demand: number;
	baseline_source: GeneralBaselineSource;
	baseline_trip_count: number;
	company_route_count: number;
	company_route_mean: number | null;
	canonical_route_count: number;
	canonical_route_mean: number | null;
	demand_label: DemandLabel;
	prediction_reliability: Reliability;
	reliability_reason: string;
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
	firma_unvan: string | null;
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

export interface StopAdditionRequest {
	firma_id: number;
	current_guzergah_kodu: number;
	candidate_stop_uetds_yer_id: number;
	requested_date: string;
	requested_time: string;
}

export type StopAdditionDecision = 'APPROVE' | 'REVIEW' | 'REJECT';

export interface StopAdditionPrediction {
	FIRMA_ID: number;
	CURRENT_GUZERGAH_KODU: number;
	CANDIDATE_STOP_UETDS_YER_ID: number;
	REQUESTED_DATE: string;
	REQUESTED_TIME: string;
	added_stop_uetds_yer_id: number;
	base_stop_list: number[] | string;
	proposed_stop_list: number[] | string;
	selected_insertion_index: number;
	base_route_haversine_km: number;
	variant_route_haversine_km: number;
	added_haversine_km: number;
	detour_ratio: number;
	training_scenario: string;
	has_same_company_exact_proposed_history: boolean;
	has_any_exact_proposed_history: boolean;
	has_similar_route_history: boolean;
	current_route_expected_demand_proxy: number | null;
	proposed_route_hierarchical_baseline: number | null;
	proposed_route_baseline_source: string;
	proposed_route_history_same_company_time_count: number;
	proposed_route_history_same_company_route_count: number;
	proposed_route_history_all_company_time_count: number;
	proposed_route_history_all_company_route_count: number;
	proposed_route_history_similar_route_count: number;
	proposed_route_history_similar_trip_count: number;
	proposed_route_prediction: number;
	current_route_prediction: number | null;
	predicted_uplift: number | null;
	current_route_prediction_status: string;
	current_route_prediction_error: string | null;
	current_route_reliability: Reliability | null;
	current_route_reliability_reason: string | null;
	current_route_baseline_source: string | null;
	current_route_history_exact_time_weekday_count: number | null;
	current_route_history_exact_time_count: number | null;
	current_route_history_company_route_count: number | null;
	current_route_history_canonical_time_weekday_count: number | null;
	current_route_history_canonical_route_count: number | null;
	prediction_status: string;
	prediction_error: string | null;
	business_decision: StopAdditionDecision;
	decision_reason: string;
	decision_score: number;
	model_evidence_score: number;
	decision_override: string | null;
	decision_warnings: string | null;
	is_company_origin_city: boolean;
	company_origin_il_id: number | null;
	added_stop_il_id: number | null;
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

// POST /predict-general
export function predictGeneral(
	payload: GeneralPredictionRequest,
): Promise<GeneralPrediction> {
	return jsonFetch<GeneralPrediction>(`${API_BASE}/predict-general`, {
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

// POST /predict-stop-addition
export function predictStopAddition(
	payload: StopAdditionRequest,
): Promise<StopAdditionPrediction> {
	return jsonFetch<StopAdditionPrediction>(`${API_BASE}/predict-stop-addition`, {
		method: 'POST',
		body: JSON.stringify(payload),
	});
}
