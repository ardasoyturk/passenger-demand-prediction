import { Building2, Database, Route } from 'lucide-preact';
import type { GeneralPrediction } from '../api';
import { PredictionAssessment } from './DemandLabel';

export function GeneralPredictionResult({ prediction }: { prediction: GeneralPrediction }) {
	return (
		<section class="animate-enter grid gap-4 lg:grid-cols-4" aria-label="Genel talep tahmini sonucu">
			<div class="h-full rounded-lg border border-border bg-white p-5 shadow-sm">
				<p class="text-sm font-medium text-muted-foreground">Beklenen talep</p>
				<div class="mt-2 flex items-end gap-2">
					<strong class="text-4xl font-semibold tracking-tight tabular-nums">{prediction.expected_demand.toFixed(1)}</strong>
					<span class="mb-1 text-sm text-muted-foreground">yolcu</span>
				</div>
				<div class="mt-4 flex items-center justify-between border-t border-border pt-4 text-sm">
					<span class="text-muted-foreground">Tarihsel ortalama</span>
					<span class="font-medium text-foreground tabular-nums">{prediction.baseline_trip_count.toLocaleString('tr-TR')} sefer</span>
				</div>
			</div>

			<PredictionAssessment
				demandLabel={prediction.demand_label}
				reliability={prediction.prediction_reliability}
				reason={prediction.reliability_reason}
				source="baseline"
			/>

			<GeneralEvidenceCard prediction={prediction} />
		</section>
	);
}

function GeneralEvidenceCard({ prediction }: { prediction: GeneralPrediction }) {
	const items = [
		{ label: 'Aynı firma ve güzergâh', detail: 'Tüm saat ve hafta günleri', value: prediction.company_route_count, Icon: Building2 },
		{ label: 'Aynı fiziksel rota', detail: 'Firma, saat ve gün fark etmeksizin', value: prediction.canonical_route_count, Icon: Route },
	];

	return (
		<div class="rounded-lg border border-border bg-white shadow-sm lg:col-span-4">
			<div class="flex flex-wrap items-start justify-between gap-3 border-b border-border px-5 py-4">
				<div>
					<div class="flex items-center gap-2">
						<Database class="size-4 text-muted-foreground" aria-hidden="true" />
						<h2 class="text-sm font-semibold">Geçmiş veri kapsamı</h2>
					</div>
					<p class="mt-1 text-sm leading-relaxed text-muted-foreground">Bu sayılar, tahmini destekleyen geçmiş sefer eşleşmelerini gösterir. Genel tahmin, gün ve saat ayrımı yapmadan tüm geçmiş seferlerin ortalamasıdır.</p>
				</div>
				<div class="rounded-md bg-muted px-3 py-2 text-xs leading-relaxed text-muted-foreground">
					<span class="font-medium text-foreground">Baseline hesabı: </span>
					{baselineExplanation(prediction)}
				</div>
			</div>
			<div class="grid gap-px bg-border sm:grid-cols-2">
				{items.map(({ label, detail, value, Icon }) => (
					<div key={label} class="bg-white p-5">
						<div class="flex items-center gap-2 text-muted-foreground">
							<Icon class="size-4" aria-hidden="true" />
							<span class="text-xs font-semibold text-foreground/70">{label}</span>
						</div>
						<p class="mt-1 min-h-8 text-xs leading-snug text-muted-foreground">{detail}</p>
						<p class="mt-2 text-2xl font-semibold tracking-tight tabular-nums">{value.toLocaleString('tr-TR')}</p>
						<p class="mt-1 text-xs text-muted-foreground">{value === 0 ? 'Eşleşen geçmiş sefer yok' : 'eşleşen geçmiş sefer'}</p>
					</div>
				))}
			</div>
		</div>
	);
}

function baselineExplanation(prediction: GeneralPrediction) {
	const count = prediction.baseline_trip_count.toLocaleString('tr-TR');
	switch (prediction.baseline_source) {
		case 'company_route':
			return `Aynı firma ve güzergâhtaki ${count} geçmiş seferin ortalaması kullanıldı.`;
		case 'canonical_route':
			return `Aynı fiziksel rotadaki ${count} geçmiş seferin ortalaması kullanıldı.`;
		case 'global':
			return 'Daha özel bir geçmiş eşleşmesi bulunamadığı için genel tarihsel ortalama kullanıldı.';
		default:
			return 'Kullanılan tarihsel referans kaynağı belirlenemedi.';
	}
}
