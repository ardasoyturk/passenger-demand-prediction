import { useState } from 'preact/hooks';
import { ChartNoAxesCombined, LoaderCircle, TriangleAlert } from 'lucide-preact';
import { ApiError, getRoute, predict } from '../../api';
import type { PredictionRequest, RouteDetailResponse, SimplifiedPrediction } from '../../api';
import { PredictionForm } from '../../components/PredictionForm';
import { PredictionResult } from '../../components/PredictionResult';
import { RouteMap } from '../../components/RouteMap';
import { RouteTimeline } from '../../components/RouteTimeline';

type Result =
	| { status: 'idle' }
	| { status: 'loading' }
	| { status: 'success'; prediction: SimplifiedPrediction; route: RouteDetailResponse | null; request: PredictionRequest }
	| { status: 'error'; message: string };

export function Home() {
	const [result, setResult] = useState<Result>({ status: 'idle' });

	async function handleSubmit(request: PredictionRequest) {
		setResult({ status: 'loading' });
		try {
			const [prediction, route] = await Promise.allSettled([predict(request), getRoute(request.firma_id, request.guzergah_kodu)]);
			if (prediction.status === 'rejected') throw prediction.reason;
			setResult({ status: 'success', prediction: prediction.value, route: route.status === 'fulfilled' ? route.value : null, request });
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

			{result.status === 'success' && (
				<div class="mt-6 space-y-4">
					<div class="flex flex-wrap items-center justify-between gap-2">
						<h2 class="text-base font-semibold">Tahmin sonucu</h2>
						<p class="text-xs text-muted-foreground">Firma {result.request.firma_id} · Güzergâh {result.request.guzergah_kodu} · {formatDate(result.request.sefer_tarihi)} {result.request.sefer_saati}</p>
					</div>
					<PredictionResult prediction={result.prediction} />
					{result.route ? (
						<>
							<RouteTimeline duraklar={result.route.duraklar} />
							<RouteMap duraklar={result.route.duraklar} />
						</>
					) : <div class="rounded-lg border border-dashed border-border p-4 text-sm text-muted-foreground">Tahmin hazır; güzergâh durakları alınamadı.</div>}
				</div>
			)}
		</div>
	);
}

function LoadingState() {
	return <div class="mt-6 flex items-center gap-3 rounded-lg border border-border bg-white p-5 text-sm shadow-sm" role="status"><LoaderCircle class="size-5 animate-spin text-muted-foreground" aria-hidden="true" /><div><p class="font-medium">Tahmin hesaplanıyor</p><p class="mt-0.5 text-muted-foreground">Model ve güzergâh bilgileri yükleniyor.</p></div></div>;
}

function ErrorBanner({ message }: { message: string }) {
	return <div class="animate-enter mt-6 flex items-start gap-3 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700" role="alert"><TriangleAlert class="mt-0.5 size-4 shrink-0" aria-hidden="true" /><div><strong class="font-medium">Tahmin oluşturulamadı.</strong> <span>{message}</span></div></div>;
}

function errorMessage(error: unknown): string {
	if (error instanceof Error) return error instanceof ApiError ? `Sunucu yanıtı: ${error.message}` : error.message;
	return 'Bilinmeyen bir hata oluştu.';
}

function formatDate(value: string) {
	const [year, month, day] = value.split('-');
	return `${day}.${month}.${year}`;
}
