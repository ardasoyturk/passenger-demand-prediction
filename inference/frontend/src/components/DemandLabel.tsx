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
		label: '10+ yolcu sinyali zayıf',
		description: 'Sınıflandırma modeli, talebin 10 yolcu eşiğini aşacağına dair yeterli sinyal üretmedi.',
		className: 'border-slate-200 bg-slate-50 text-slate-700',
		icon: TrendingDown,
	},
	WEAK_DEMAND: {
		label: '10+ güçlü, 20+ zayıf',
		description: 'Talebin 10 yolcuyu aşması bekleniyor; ancak 20 yolcu eşiği için sinyal yeterince güçlü değil.',
		className: 'border-sky-200 bg-sky-50 text-sky-700',
		icon: Minus,
	},
	MODERATE_DEMAND: {
		label: '20+ güçlü, 30+ zayıf',
		description: 'Talebin 20 yolcuyu aşması bekleniyor; 30 yolcu eşiği ise belirsiz veya zayıf görünüyor.',
		className: 'border-cyan-200 bg-cyan-50 text-cyan-700',
		icon: Gauge,
	},
	STRONG_DEMAND: {
		label: '30+ yolcu sinyali',
		description: 'Sınıflandırma modeli 30 yolcu eşiğinin aşılabileceğine dair sinyal üretiyor.',
		className: 'border-emerald-200 bg-emerald-50 text-emerald-700',
		icon: TrendingUp,
	},
	CAPACITY_PRESSURE: {
		label: '43+ kapasite baskısı sinyali',
		description: 'Sınıflandırma modeli talebin 43 yolcu eşiğini aşabileceğine dair sinyal üretiyor.',
		className: 'border-blue-200 bg-blue-50 text-blue-700',
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

export function DemandStatus({ label }: { label: DemandLabelType }) {
	const item = DEMAND[label] ?? DEMAND.CLEAR_FAILURE;
	const Icon = item.icon;
	return (
		<div>
			<div class={`inline-flex items-center gap-2 rounded-md border px-3 py-2 text-base font-semibold ${item.className}`}>
				<Icon class="size-4" aria-hidden="true" />
				{item.label}
			</div>
			<p class="mt-3 text-sm leading-relaxed text-muted-foreground">{item.description}</p>
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
