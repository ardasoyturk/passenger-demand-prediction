import { useState } from 'preact/hooks';
import { ChartNoAxesCombined, LoaderCircle, TriangleAlert } from 'lucide-preact';
import { getRoute, predict, predictGeneral } from '../../api';
import type { GeneralPrediction, GeneralPredictionRequest, PredictionRequest, RouteDetailResponse, SimplifiedPrediction } from '../../api';
import { PredictionForm } from '../../components/PredictionForm';
import type { PredictionFormRequest } from '../../components/PredictionForm';
import { PredictionContext } from '../../components/PredictionContext';
import { PredictionResult } from '../../components/PredictionResult';
import { GeneralPredictionResult } from '../../components/GeneralPredictionResult';
import { RouteMap } from '../../components/RouteMap';
import { RouteTimeline } from '../../components/RouteTimeline';
import { errorMessage } from '../../lib/errors';

type Result =
	| { status: 'idle' }
	| { status: 'loading' }
	| { status: 'error'; message: string }
	| { status: 'success'; mode: 'trip'; prediction: SimplifiedPrediction; route: RouteDetailResponse | null; request: PredictionRequest }
	| { status: 'success'; mode: 'general'; prediction: GeneralPrediction; route: RouteDetailResponse | null; request: GeneralPredictionRequest };

export function Home() {
	const [result, setResult] = useState<Result>({ status: 'idle' });

	async function handleSubmit(submission: PredictionFormRequest) {
		setResult({ status: 'loading' });
		try {
			const predictionRequest = submission.mode === 'general'
				? predictGeneral(submission.request)
				: predict(submission.request);
			const [prediction, route] = await Promise.allSettled([
				predictionRequest,
				getRoute(submission.request.firma_id, submission.request.guzergah_kodu),
			]);
			if (prediction.status === 'rejected') throw prediction.reason;
			const routeValue = route.status === 'fulfilled' ? route.value : null;
			if (submission.mode === 'general') {
				setResult({ status: 'success', mode: 'general', prediction: prediction.value as GeneralPrediction, route: routeValue, request: submission.request });
			} else {
				setResult({ status: 'success', mode: 'trip', prediction: prediction.value as SimplifiedPrediction, route: routeValue, request: submission.request });
			}
		} catch (error) {
			setResult({ status: 'error', message: errorMessage(error) });
		}
	}

	return (
		<div class="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
			<div class="mb-6">
				<h1 class="text-2xl font-semibold tracking-tight">Yolcu Talep Tahmini</h1>
				<p class="mt-1 text-sm text-muted-foreground">Önerilen bir sefer için beklenen talebi ve eşik olasılıklarını hesaplayın.</p>
			</div>

			<PredictionForm onSubmit={handleSubmit} loading={result.status === 'loading'} />

			{result.status === 'error' && <ErrorBanner message={result.message} />}
			{result.status === 'loading' && <LoadingState />}

			{result.status === 'idle' && (
				<div class="mt-6 rounded-lg border border-dashed border-slate-300 bg-white/60 px-6 py-10 text-center">
					<div class="mx-auto grid size-10 place-items-center rounded-full bg-muted text-muted-foreground">
						<ChartNoAxesCombined class="size-5" aria-hidden="true" />
					</div>
					<h2 class="mt-3 text-sm font-medium">Henüz tahmin oluşturulmadı</h2>
					<p class="mt-1 text-sm text-muted-foreground">Sonuçları görmek için sefer bilgilerini girip “Tahmin et” butonuna basın.</p>
				</div>
			)}

			{result.status === 'success' && <PredictionSuccess result={result} />}
		</div>
	);
}

function LoadingState() {
	return <div class="mt-6 flex items-center gap-3 rounded-lg border border-border bg-white p-5 text-sm shadow-sm" role="status"><LoaderCircle class="size-5 animate-spin text-muted-foreground" aria-hidden="true" /><div><p class="font-medium">Tahmin hesaplanıyor</p><p class="mt-0.5 text-muted-foreground">Model ve güzergâh bilgileri yükleniyor.</p></div></div>;
}

function ErrorBanner({ message }: { message: string }) {
	return <div class="animate-enter mt-6 flex items-start gap-3 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700" role="alert"><TriangleAlert class="mt-0.5 size-4 shrink-0" aria-hidden="true" /><div><strong class="font-medium">Tahmin oluşturulamadı.</strong> <span>{message}</span></div></div>;
}

function PredictionSuccess({ result }: { result: Extract<Result, { status: 'success' }> }) {
	const trip = result.mode === 'trip';
	return (
		<div class="mt-6 space-y-4">
			<PredictionContext
				heading={trip ? 'Tahmin sonucu' : 'Genel talep tahmini'}
				companyId={result.request.firma_id}
				companyTitle={result.route?.firma_unvan}
				routeCode={result.request.guzergah_kodu}
				date={trip ? result.request.sefer_tarihi : undefined}
				time={trip ? result.request.sefer_saati : undefined}
			/>
			{trip ? <PredictionResult prediction={result.prediction} /> : <GeneralPredictionResult prediction={result.prediction} />}
			{result.route ? (
				<>
					<RouteTimeline duraklar={result.route.duraklar} />
					<RouteMap duraklar={result.route.duraklar} />
				</>
			) : <div class="rounded-lg border border-dashed border-border p-4 text-sm text-muted-foreground">Tahmin hazır; güzergâh durakları alınamadı.</div>}
		</div>
	);
}
