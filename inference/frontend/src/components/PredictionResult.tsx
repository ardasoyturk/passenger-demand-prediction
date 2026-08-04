import { Building2, CalendarDays, Clock3, Database, Network, Route } from 'lucide-preact';
import type { ReliabilityEvidence, SimplifiedPrediction } from '../api';
import { PredictionAssessment } from './DemandLabel';
import { baselineExplanation } from '../lib/baseline';

export function PredictionResult({ prediction }: { prediction: SimplifiedPrediction }) {
	const { expected_demand, baseline_demand, demand_label, reliability, reliability_reason, probabilities, reliability_evidence } = prediction;
	const delta = expected_demand - baseline_demand;
	const bars = [
		{ label: '10+ yolcu', value: probabilities.ge_10 },
		{ label: '20+ yolcu', value: probabilities.ge_20 },
		{ label: '30+ yolcu', value: probabilities.ge_30 },
		{ label: '43+ yolcu', value: probabilities.ge_43 },
	];

	return (
		<section class="animate-enter grid gap-4 lg:grid-cols-4" aria-label="Tahmin sonucu">
			<div class="h-full rounded-lg border border-border bg-white p-5 shadow-sm">
				<p class="text-sm font-medium text-muted-foreground">Beklenen talep</p>
				<div class="mt-2 flex items-end gap-2">
					<strong class="text-4xl font-semibold tracking-tight tabular-nums">{expected_demand.toFixed(1)}</strong>
					<span class="mb-1 text-sm text-muted-foreground">yolcu</span>
				</div>
				<div class="mt-4 flex items-center justify-between border-t border-border pt-4 text-sm">
					<span class="text-muted-foreground">Referans tahmin: {baseline_demand.toFixed(1)}</span>
					<span class={delta >= 0 ? 'font-medium text-emerald-600' : 'font-medium text-orange-600'}>{delta >= 0 ? '+' : ''}{delta.toFixed(1)}</span>
				</div>
			</div>

			<PredictionAssessment
				demandLabel={demand_label}
				reliability={reliability}
				reason={reliability_reason}
			/>

			<ReliabilityEvidenceCard evidence={reliability_evidence} />

			<div class="rounded-lg border border-border bg-white shadow-sm lg:col-span-4">
				<div class="border-b border-border px-5 py-4">
					<h2 class="text-sm font-semibold">Eşik olasılıkları</h2>
					<p class="mt-1 text-sm text-muted-foreground">Her eşik için sınıflandırma modelinin ürettiği kalibre edilmemiş güven skoru. Değerler gerçek gerçekleşme yüzdesi olarak yorumlanmamalıdır.</p>
				</div>
				<div class="grid gap-5 p-5 sm:grid-cols-2">
					{bars.map((bar) => <Probability key={bar.label} {...bar} />)}
				</div>
			</div>
		</section>
	);
}

function ReliabilityEvidenceCard({ evidence }: { evidence: ReliabilityEvidence }) {
	const items = [
		{ label: 'Aynı saat ve hafta günü', detail: 'Aynı firma ve güzergâh', value: evidence.exact_time_weekday_count, Icon: CalendarDays },
		{ label: 'Aynı tam kalkış saati', detail: 'Haftanın günü fark etmeksizin', value: evidence.exact_time_count, Icon: Clock3 },
		{ label: 'Aynı firma ve güzergâh', detail: 'Tüm saat ve hafta günleri', value: evidence.company_route_count, Icon: Building2 },
		{ label: 'Aynı fiziksel rota, zaman ve gün', detail: 'Firmadan bağımsız, 30 dk zaman dilimi', value: evidence.canonical_time_weekday_count, Icon: Network },
		{ label: 'Aynı fiziksel rota', detail: 'Firma, saat ve gün fark etmeksizin', value: evidence.canonical_route_count, Icon: Route },
	];

	return (
		<div class="rounded-lg border border-border bg-white shadow-sm lg:col-span-4">
			<div class="flex flex-wrap items-start justify-between gap-3 border-b border-border px-5 py-4">
				<div>
					<div class="flex items-center gap-2">
						<Database class="size-4 text-muted-foreground" aria-hidden="true" />
						<h2 class="text-sm font-semibold">Geçmiş veri kapsamı</h2>
					</div>
					<p class="mt-1 text-sm leading-relaxed text-muted-foreground">Bu sayılar, tahmini destekleyebilecek benzer geçmiş sefer eşleşmelerini gösterir. Önerilen sefer tarihi ve sonrasındaki kayıtlar dahil edilmez.</p>
				</div>
				<div class="rounded-md bg-muted px-3 py-2 text-xs leading-relaxed text-muted-foreground">
					<span class="font-medium text-foreground">Referans hesabı: </span>
					{baselineExplanation(evidence)}
				</div>
			</div>
			<div class="grid gap-px bg-border sm:grid-cols-2 lg:grid-cols-5">
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
			{evidence.frequent_departure_times.length > 0 && (
				<div class="border-t border-border bg-slate-50/70 p-5">
					<div class="flex items-start gap-3">
						<span class="grid size-9 shrink-0 place-items-center rounded-md border border-border bg-white text-muted-foreground shadow-sm">
							<Clock3 class="size-4" aria-hidden="true" />
						</span>
						<div>
							<h3 class="text-sm font-semibold">Bu güzergâhta sık kullanılan kalkış saatleri</h3>
							<p class="mt-1 text-sm leading-relaxed text-muted-foreground">Seçilen tam saatte yeterli geçmiş eşleşmesi bulunmadığı için, aynı firma ve güzergâhtaki en sık kalkış saatleri gösteriliyor.</p>
						</div>
					</div>
					<div class="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
						{evidence.frequent_departure_times.map((item) => (
							<div key={item.departure_time} class="rounded-md border border-border bg-white p-4 shadow-sm">
								<p class="text-xl font-semibold tracking-tight tabular-nums">{item.departure_time}</p>
								<p class="mt-1 text-xs text-muted-foreground">{item.trip_count.toLocaleString('tr-TR')} geçmiş sefer</p>
								<p class="mt-2 text-xs font-medium text-foreground/70">Rota geçmişinin %{(item.route_share * 100).toLocaleString('tr-TR', { maximumFractionDigits: 1 })}'i</p>
							</div>
						))}
					</div>
				</div>
			)}
		</div>
	);
}

function Probability({ label, value }: { label: string; value: number }) {
	const percentage = Math.round(value * 100);
	return <div><div class="mb-2 flex justify-between text-sm"><span class="font-medium">{label}</span><span class="font-medium tabular-nums">%{percentage}</span></div><div class="h-2 overflow-hidden rounded-full bg-muted"><div class="h-full rounded-full bg-slate-900 transition-[width] duration-500" style={{ width: `${percentage}%` }} /></div></div>;
}
