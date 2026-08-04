import type { DemandLabel as DemandLabelType, Reliability } from '../api';
import {
	Gauge,
	Minus,
	ShieldAlert,
	ShieldCheck,
	ShieldQuestion,
	TrendingDown,
	TrendingUp,
	TriangleAlert,
} from 'lucide-preact';

const DEMAND = {
	CLEAR_FAILURE: {
		level: 'Çok düşük',
		step: 1,
		label: '10+ yolcu sinyali zayıf',
		description: 'Sınıflandırma modeli, talebin 10 yolcu eşiğini aşacağına dair yeterli sinyal üretmedi.',
		shell: 'border-red-300 bg-red-100/70 text-red-950',
		badge: 'bg-red-700 text-white',
		iconClass: 'bg-red-700 text-white',
		meter: 'bg-red-600',
		icon: TrendingDown,
	},
	WEAK_DEMAND: {
		level: 'Düşük',
		step: 2,
		label: '10+ güçlü, 20+ zayıf',
		description: 'Talebin 10 yolcuyu aşması bekleniyor; ancak 20 yolcu eşiği için sinyal yeterince güçlü değil.',
		shell: 'border-orange-300 bg-orange-100/70 text-orange-950',
		badge: 'bg-orange-700 text-white',
		iconClass: 'bg-orange-700 text-white',
		meter: 'bg-orange-600',
		icon: Minus,
	},
	MODERATE_DEMAND: {
		level: 'Orta',
		step: 3,
		label: '20+ güçlü, 30+ zayıf',
		description: 'Talebin 20 yolcuyu aşması bekleniyor; 30 yolcu eşiği ise belirsiz veya zayıf görünüyor.',
		shell: 'border-amber-300 bg-amber-100/70 text-amber-950',
		badge: 'bg-amber-700 text-white',
		iconClass: 'bg-amber-700 text-white',
		meter: 'bg-amber-500',
		icon: Gauge,
	},
	STRONG_DEMAND: {
		level: 'Yüksek',
		step: 4,
		label: '30+ yolcu sinyali',
		description: 'Sınıflandırma modeli 30 yolcu eşiğinin aşılabileceğine dair sinyal üretiyor.',
		shell: 'border-emerald-300 bg-emerald-100/70 text-emerald-950',
		badge: 'bg-emerald-700 text-white',
		iconClass: 'bg-emerald-700 text-white',
		meter: 'bg-emerald-600',
		icon: TrendingUp,
	},
	CAPACITY_PRESSURE: {
		level: 'Kapasite üstü',
		step: 5,
		label: '43+ kapasite baskısı sinyali',
		description: 'Sınıflandırma modeli talebin 43 yolcu eşiğini aşabileceğine dair sinyal üretiyor.',
		shell: 'border-emerald-400 bg-emerald-100 text-emerald-950',
		badge: 'bg-emerald-800 text-white',
		iconClass: 'bg-emerald-800 text-white',
		meter: 'bg-emerald-700',
		icon: TrendingUp,
	},
};

const RELIABILITY = {
	NO_HISTORY: {
		label: 'Sınırlı güvenilirlik',
		eyebrow: 'Geçmiş veri yok',
		guidance: 'Bu tahmini karar verirken temkinli kullanın.',
		card: 'border-amber-300 bg-amber-50/70',
		icon: TriangleAlert,
		iconClass: 'bg-amber-500 text-white',
	},
	UNSAFE: {
		label: 'Güvenilir değil',
		eyebrow: 'Riskli tahmin',
		guidance: 'Bu tahmini tek başına karar vermek için kullanmayın.',
		card: 'border-red-300 bg-red-50/70',
		icon: ShieldAlert,
		iconClass: 'bg-red-600 text-white',
	},
	LOW: {
		label: 'Düşük güvenilirlik',
		eyebrow: 'Düşük güven',
		guidance: 'Sonucu ek tarihsel bilgilerle birlikte değerlendirin.',
		card: 'border-orange-300 bg-orange-50/70',
		icon: ShieldAlert,
		iconClass: 'bg-orange-500 text-white',
	},
	MEDIUM: {
		label: 'Orta güvenilirlik',
		eyebrow: 'Orta güven',
		guidance: 'Tahmin kullanılabilir; belirsizlik hâlâ dikkate alınmalı.',
		card: 'border-slate-300 bg-slate-50',
		icon: ShieldQuestion,
		iconClass: 'bg-slate-600 text-white',
	},
	HIGH: {
		label: 'Yüksek güvenilirlik',
		eyebrow: 'Yüksek güven',
		guidance: 'Tahmin yeterli tarihsel bağlamla destekleniyor.',
		card: 'border-emerald-300 bg-emerald-50/60',
		icon: ShieldCheck,
		iconClass: 'bg-emerald-600 text-white',
	},
};

const DEMAND_BASELINE_DESCRIPTION: Record<DemandLabelType, string> = {
	CLEAR_FAILURE: 'Tarihsel ortalama talep 10 yolcu eşiğinin altında.',
	WEAK_DEMAND: 'Tarihsel ortalama talep 10 yolcu eşiğinin üzerinde; 20 yolcu eşiğinin altında.',
	MODERATE_DEMAND: 'Tarihsel ortalama talep 20 yolcu eşiğinin üzerinde; 30 yolcu eşiğinin altında.',
	STRONG_DEMAND: 'Tarihsel ortalama talep 30 yolcu eşiğinin üzerinde.',
	CAPACITY_PRESSURE: 'Tarihsel ortalama talep 43 yolcu eşiğinin üzerinde.',
};

export function DemandStatus({ label, source = 'model' }: { label: DemandLabelType; source?: 'model' | 'baseline' }) {
	const item = DEMAND[label] ?? DEMAND.CLEAR_FAILURE;
	const Icon = item.icon;
	const description = source === 'baseline' ? DEMAND_BASELINE_DESCRIPTION[label] ?? DEMAND_BASELINE_DESCRIPTION.CLEAR_FAILURE : item.description;
	return (
		<div class={`h-full rounded-lg border p-5 shadow-sm ${item.shell}`}>
			<div class="flex items-start gap-4">
				<span class={`grid size-11 shrink-0 place-items-center rounded-full ${item.iconClass}`}>
					<Icon class="size-5" aria-hidden="true" />
				</span>
				<div class="min-w-0 flex-1">
					<div class="flex flex-wrap items-center justify-between gap-2">
						<p class="text-xs font-semibold uppercase tracking-wider opacity-70">Talep seviyesi</p>
						<span class={`rounded-full px-2.5 py-1 text-xs font-semibold ${item.badge}`}>{item.level}</span>
					</div>
					<p class="mt-2 text-base font-semibold leading-snug">{item.label}</p>
					<div class="mt-3 grid grid-cols-5 gap-1.5" aria-label={`Talep seviyesi: ${item.level}`}>
						{[1, 2, 3, 4, 5].map((step) => (
							<span key={step} class={`h-1.5 rounded-full ${step <= item.step ? item.meter : 'bg-current/15'}`} />
						))}
					</div>
					<p class="mt-3 text-sm leading-relaxed opacity-80">{description}</p>
				</div>
			</div>
		</div>
	);
}

export function ReliabilityStatus({ level, reason }: { level: Reliability; reason?: string }) {
	const item = RELIABILITY[level] ?? RELIABILITY.NO_HISTORY;
	const Icon = item.icon;
	return (
		<div class={`h-full rounded-lg border p-5 shadow-sm ${item.card}`} role={level === 'UNSAFE' || level === 'NO_HISTORY' || level === 'LOW' ? 'alert' : undefined}>
			<div class="flex items-start gap-4">
				<span class={`grid size-9 shrink-0 place-items-center rounded-full ${item.iconClass}`}><Icon class="size-5" aria-hidden="true" /></span>
				<div class="min-w-0">
					<p class="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{item.eyebrow}</p>
					<h3 class="mt-1 text-lg font-semibold tracking-tight">{item.label}</h3>
					{reason && <p class="mt-2 text-sm leading-relaxed text-foreground/80">{reason}</p>}
					<p class="mt-2 text-sm font-medium text-foreground">{item.guidance}</p>
				</div>
			</div>
		</div>
	);
}

export function PredictionAssessment({
	demandLabel,
	reliability,
	reason,
	source = 'model',
}: {
	demandLabel: DemandLabelType;
	reliability: Reliability;
	reason?: string;
	source?: 'model' | 'baseline';
}) {
	const demand = DEMAND[demandLabel] ?? DEMAND.CLEAR_FAILURE;
	const confidence = RELIABILITY[reliability] ?? RELIABILITY.NO_HISTORY;
	const DemandIcon = demand.icon;
	const ConfidenceIcon = confidence.icon;
	const description = source === 'baseline'
		? DEMAND_BASELINE_DESCRIPTION[demandLabel] ?? DEMAND_BASELINE_DESCRIPTION.CLEAR_FAILURE
		: demand.description;
	const needsAttention = reliability === 'UNSAFE' || reliability === 'NO_HISTORY' || reliability === 'LOW';

	return (
		<section
			class="h-full overflow-hidden rounded-lg border border-border bg-background shadow-sm lg:col-span-3"
			aria-label="Talep ve güvenilirlik değerlendirmesi"
			role={needsAttention ? 'alert' : undefined}
		>
			<div class="grid h-full lg:grid-cols-[minmax(0,1.1fr)_minmax(20rem,.9fr)]">
				<div class={`flex items-start gap-4 p-5 sm:p-6 ${demand.shell}`}>
					<span class={`grid size-11 shrink-0 place-items-center rounded-full ${demand.iconClass}`}>
						<DemandIcon class="size-5" aria-hidden="true" />
					</span>
					<div class="min-w-0 flex-1">
						<div class="flex flex-wrap items-center gap-2">
							<p class="text-xs font-semibold uppercase tracking-wider opacity-70">Talep seviyesi</p>
							<span class={`rounded-full px-2.5 py-1 text-xs font-semibold ${demand.badge}`}>{demand.level}</span>
						</div>
						<h3 class="mt-2 text-xl font-semibold tracking-tight">{demand.label}</h3>
						<div class="mt-3 grid max-w-md grid-cols-5 gap-1.5" aria-label={`Talep seviyesi: ${demand.level}`}>
							{[1, 2, 3, 4, 5].map((step) => (
								<span key={step} class={`h-1.5 rounded-full ${step <= demand.step ? demand.meter : 'bg-current/15'}`} />
							))}
						</div>
						<p class="mt-3 max-w-xl text-sm leading-relaxed opacity-80">{description}</p>
					</div>
				</div>

				<div class={`border-t border-current/10 p-5 text-foreground sm:p-6 lg:border-l lg:border-t-0 ${confidence.card}`}>
					<p class="text-xs font-semibold uppercase tracking-wider opacity-65">Tahmin güvenilirliği</p>
					<div class="mt-3 flex items-start gap-3">
						<span class={`grid size-9 shrink-0 place-items-center rounded-full ${confidence.iconClass}`}>
							<ConfidenceIcon class="size-5" aria-hidden="true" />
						</span>
						<div class="min-w-0">
							<div class="flex flex-wrap items-center gap-2">
								<h3 class="text-lg font-semibold tracking-tight">{confidence.label}</h3>
								<span class="rounded-full border border-current/15 bg-white/60 px-2 py-0.5 text-xs font-semibold opacity-75">{confidence.eyebrow}</span>
							</div>
							{reason && <p class="mt-2 text-sm leading-relaxed opacity-80">{reason}</p>}
							<p class="mt-2 text-sm font-semibold">{confidence.guidance}</p>
						</div>
					</div>
				</div>
			</div>
		</section>
	);
}
