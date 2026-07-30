import type { TargetedEvent } from 'preact';
import { useState } from 'preact/hooks';
import {
	AlertTriangle, BadgeCheck, Ban, Building2, BusFront, CalendarDays,
	CircleHelp, Clock3, Database, GitCompareArrows, Info, LoaderCircle,
	MapPinPlus, Network, Route, SearchCheck, TrendingUp,
} from 'lucide-preact';
import {
	ApiError, getDurak, getRoute, predictStopAddition,
} from '../../api';
import type {
	Durak, RouteDetailResponse, RouteDurak, StopAdditionPrediction,
	StopAdditionRequest,
} from '../../api';
import { RouteMap } from '../../components/RouteMap';

type Result =
	| { status: 'idle' }
	| { status: 'loading' }
	| { status: 'error'; message: string }
	| {
		status: 'success';
		prediction: StopAdditionPrediction;
		request: StopAdditionRequest;
		currentRoute: RouteDetailResponse | null;
		candidateStop: Durak | null;
	};

export function ProposedRoutes() {
	const [result, setResult] = useState<Result>({ status: 'idle' });

	async function handleSubmit(request: StopAdditionRequest) {
		setResult({ status: 'loading' });
		try {
			const prediction = await predictStopAddition(request);
			const [currentRoute, candidateStop] = await Promise.allSettled([
				getRoute(request.firma_id, request.current_guzergah_kodu),
				getDurak(request.candidate_stop_uetds_yer_id),
			]);
			setResult({
				status: 'success',
				prediction,
				request,
				currentRoute: currentRoute.status === 'fulfilled' ? currentRoute.value : null,
				candidateStop: candidateStop.status === 'fulfilled' ? candidateStop.value : null,
			});
		} catch (error) {
			setResult({ status: 'error', message: errorMessage(error) });
		}
	}

	const proposedStops = result.status === 'success'
		? buildProposedStops(result.prediction, result.currentRoute, result.candidateStop)
		: [];

	return (
		<div class="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
			<div class="mb-6 flex flex-wrap items-end justify-between gap-4">
				<div>
					<div class="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
						<Route class="size-3.5" aria-hidden="true" />
						Rota Değerlendirme
					</div>
					<h1 class="text-2xl font-semibold tracking-tight">Mevcut Rotaya Durak Ekleme</h1>
					<p class="mt-1 max-w-2xl text-sm text-muted-foreground">Mevcut güzergâha yeni bir durak eklemenin talep ve mesafe üzerindeki etkisini değerlendirin.</p>
				</div>
			</div>

			<ProposalForm onSubmit={handleSubmit} loading={result.status === 'loading'} />

			{result.status === 'error' && <ErrorBanner message={result.message} />}
			{result.status === 'loading' && <LoadingState />}
			{result.status === 'idle' && <EmptyState />}

			{result.status === 'success' && (
				<div class="mt-6 space-y-4">
					<div class="flex flex-wrap items-center justify-between gap-2">
						<h2 class="text-base font-semibold">Değerlendirme Sonucu</h2>
						<p class="text-xs text-muted-foreground">
							Firma {result.request.firma_id} · Güzergâh {result.request.current_guzergah_kodu} · {formatDate(result.request.requested_date)} {result.request.requested_time}
						</p>
					</div>
					<DecisionSummary prediction={result.prediction} candidate={result.candidateStop} />
					<Comparison prediction={result.prediction} />
					<Evidence prediction={result.prediction} />
					<ProposedTimeline stops={proposedStops} insertedStopId={result.prediction.added_stop_uetds_yer_id} />
					{proposedStops.length > 0 && (
						<RouteMap
							duraklar={proposedStops}
							highlightedStopId={result.prediction.added_stop_uetds_yer_id}
						/>
					)}
				</div>
			)}
		</div>
	);
}

function ProposalForm({ onSubmit, loading }: { onSubmit: (request: StopAdditionRequest) => void; loading: boolean }) {
	const [firma, setFirma] = useState('49');
	const [route, setRoute] = useState('5866');
	const [stop, setStop] = useState('44');
	const [date, setDate] = useState(new Date().toISOString().slice(0, 10));
	const [time, setTime] = useState('13:00');
	const [touched, setTouched] = useState(false);
	const valid = [firma, route, stop].every(validId) && Boolean(date) && /^\d{2}:\d{2}$/.test(time);

	function submit(event: TargetedEvent<HTMLFormElement>) {
		event.preventDefault();
		setTouched(true);
		if (!valid || loading) return;
		onSubmit({
			firma_id: Number(firma),
			current_guzergah_kodu: Number(route),
			candidate_stop_uetds_yer_id: Number(stop),
			requested_date: date,
			requested_time: time,
		});
	}

	return (
		<form noValidate onSubmit={submit} class="rounded-lg border border-border bg-white shadow-sm">
			<div class="border-b border-border px-5 py-4">
				<h2 class="text-sm font-semibold">Öneri Bilgileri</h2>
				<p class="mt-1 text-sm text-muted-foreground">Mevcut güzergâhı, eklenecek durağı ve planlanan sefer zamanını girin.</p>
			</div>
			<div class="grid gap-4 p-5 md:grid-cols-2 lg:grid-cols-[.85fr_1fr_1fr_1.1fr_.85fr_auto] lg:items-end">
				<Field label="Firma ID" invalid={touched && !validId(firma)}>
					<input aria-label="Firma" type="number" min={0} class={inputClass(touched && !validId(firma))} value={firma} onInput={(e) => setFirma(e.currentTarget.value)} />
				</Field>
				<Field label="Mevcut Güzergâh Kodu" invalid={touched && !validId(route)}>
					<input aria-label="Mevcut Güzergâh Kodu" type="number" min={0} class={inputClass(touched && !validId(route))} value={route} onInput={(e) => setRoute(e.currentTarget.value)} />
				</Field>
				<Field label="Eklenecek Durak" invalid={touched && !validId(stop)}>
					<input aria-label="Eklenecek Durak" type="number" min={0} class={inputClass(touched && !validId(stop))} value={stop} onInput={(e) => setStop(e.currentTarget.value)} />
				</Field>
				<Field label="Planlanan Sefer Tarihi" invalid={touched && !date}>
					<input aria-label="Planlanan Sefer Tarihi" type="date" min="2023-01-01" class={`${inputClass(touched && !date)} [color-scheme:light]`} value={date} onInput={(e) => setDate(e.currentTarget.value)} />
				</Field>
				<Field label="Planlanan Kalkış Saati" invalid={touched && !time}>
					<input aria-label="Planlanan Kalkış Saati" type="time" class={`${inputClass(touched && !time)} [color-scheme:light]`} value={time} onInput={(e) => setTime(e.currentTarget.value)} />
				</Field>
				<button type="submit" disabled={loading} class="inline-flex h-10 items-center justify-center gap-2 whitespace-nowrap rounded-md bg-primary px-4 text-sm font-medium text-white shadow-sm transition-colors hover:bg-slate-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-900 disabled:cursor-not-allowed disabled:opacity-60">
					{loading ? <LoaderCircle class="size-4 animate-spin" aria-hidden="true" /> : <GitCompareArrows class="size-4" aria-hidden="true" />}
					{loading ? 'Değerlendiriliyor...' : 'Öneriyi Değerlendir'}
				</button>
			</div>
		</form>
	);
}

function DecisionSummary({ prediction, candidate }: { prediction: StopAdditionPrediction; candidate: Durak | null }) {
	const config = decisionConfig(prediction.business_decision);
	const Icon = config.Icon;
	const factors = decisionFactors(prediction);
	return (
		<section class={`animate-enter overflow-hidden rounded-lg border shadow-sm ${config.shell}`} aria-label="Durak ekleme önerisi">
			<div class="grid lg:grid-cols-[minmax(0,1.15fr)_minmax(22rem,.85fr)]">
				<div class="flex items-start gap-4 p-5 sm:p-6">
					<span class={`grid size-11 shrink-0 place-items-center rounded-full ${config.icon}`}>
						<Icon class="size-5" aria-hidden="true" />
					</span>
					<div>
						<div class="flex flex-wrap items-center gap-2">
							<p class="text-xs font-semibold uppercase tracking-wider opacity-70">Değerlendirme sonucu</p>
							<span class={`rounded-full px-2.5 py-1 text-xs font-semibold ${config.badge}`}>{decisionLabel(prediction)}</span>
						</div>
						<h3 class="mt-2 text-xl font-semibold tracking-tight">{recommendationTitle(prediction)}</h3>
						<p class="mt-2 max-w-2xl text-sm leading-relaxed opacity-80">
							{titleCase(stopName(candidate, prediction.added_stop_uetds_yer_id))} durağı için {recommendationSummary(prediction)}
						</p>
					</div>
				</div>
				<div class="border-t border-current/10 bg-white/35 p-5 lg:border-l lg:border-t-0">
					<p class="text-xs font-semibold uppercase tracking-wider opacity-65">Bu sonucu belirleyen noktalar</p>
					<ul class="mt-3 space-y-3">
						{factors.map(({ Icon: FactorIcon, text, tone }) => (
							<li key={text} class="flex items-start gap-2.5 text-sm leading-snug">
								<FactorIcon class={`mt-0.5 size-4 shrink-0 ${tone}`} aria-hidden="true" />
								<span class="opacity-80">{text}</span>
							</li>
						))}
					</ul>
				</div>
			</div>
		</section>
	);
}

function Comparison({ prediction }: { prediction: StopAdditionPrediction }) {
	const uplift = prediction.predicted_uplift;
	const current = prediction.current_route_prediction;
	const upliftPositive = uplift !== null && uplift >= 0;
	return (
		<section class="animate-enter rounded-lg border border-border bg-white shadow-sm" aria-labelledby="comparison-heading">
			<div class="border-b border-border px-5 py-4">
				<h2 id="comparison-heading" class="text-sm font-semibold">Mevcut ve Önerilen Güzergâh Karşılaştırması</h2>
				<p class="mt-1 text-sm text-muted-foreground">Yeni durağın sefer başına beklenen yolcu sayısını ve güzergâh mesafesini nasıl değiştireceği.</p>
			</div>
			<div class="grid gap-px bg-border sm:grid-cols-2 lg:grid-cols-4">
				<Metric icon={BusFront} label="Mevcut güzergâh" value={current === null ? '—' : current.toFixed(1)} unit="yolcu" detail={currentPredictionDetail(prediction)} />
				<Metric icon={Route} label="Durak eklendiğinde" value={prediction.proposed_route_prediction.toFixed(1)} unit="yolcu" detail={proposedPredictionDetail(prediction)} />
				<Metric icon={TrendingUp} label="Beklenen yolcu değişimi" value={uplift === null ? '—' : signed(uplift, 1)} unit="yolcu" detail={upliftDescription(uplift)} accent={uplift === null ? undefined : upliftPositive ? 'positive' : 'negative'} />
				<Metric icon={MapPinPlus} label="Güzergâha eklenen mesafe" value={prediction.added_haversine_km.toFixed(1)} unit="km" detail={`${prediction.base_route_haversine_km.toFixed(1)} km'den ${prediction.variant_route_haversine_km.toFixed(1)} km'ye çıkıyor`} accent={prediction.detour_ratio >= .12 ? 'negative' : undefined} />
			</div>
		</section>
	);
}

function Metric({ icon: Icon, label, value, unit, detail, accent }: { icon: typeof Route; label: string; value: string; unit: string; detail: string; accent?: 'positive' | 'negative' }) {
	const color = accent === 'positive' ? 'text-emerald-600' : accent === 'negative' ? 'text-red-600' : 'text-foreground';
	return <div class="bg-white p-5"><div class="flex items-center gap-2 text-muted-foreground"><Icon class="size-4" aria-hidden="true" /><p class="text-xs font-semibold">{label}</p></div><div class={`mt-3 flex items-end gap-2 ${color}`}><strong class="text-3xl font-semibold tracking-tight tabular-nums">{value}</strong><span class="mb-1 text-xs">{unit}</span></div><p class="mt-2 text-xs text-muted-foreground">{detail}</p></div>;
}

function Evidence({ prediction }: { prediction: StopAdditionPrediction }) {
	const currentHistory = [
		{ label: 'Aynı firma, güzergâh, gün ve saat', value: count(prediction.current_route_history_exact_time_weekday_count), Icon: CalendarDays },
		{ label: 'Aynı firma, güzergâh ve saat', value: count(prediction.current_route_history_exact_time_count), Icon: Clock3 },
		{ label: 'Aynı firma ve güzergâh', value: count(prediction.current_route_history_company_route_count), Icon: Building2 },
		{ label: 'Aynı fiziksel güzergâh', value: count(prediction.current_route_history_canonical_route_count), Icon: Network },
	];
	const proposedHistory = [
		{ label: 'Aynı firma, önerilen güzergâh, gün ve saat', value: count(prediction.proposed_route_history_same_company_time_count), Icon: CalendarDays },
		{ label: 'Aynı firma ve önerilen güzergâh', value: count(prediction.proposed_route_history_same_company_route_count), Icon: Building2 },
		{ label: 'Tüm firmalarda aynı güzergâh', value: count(prediction.proposed_route_history_all_company_route_count), Icon: Route },
		{ label: `${count(prediction.proposed_route_history_similar_route_count)} benzer güzergâhtaki seferler`, value: count(prediction.proposed_route_history_similar_trip_count), Icon: Network },
	];
	return (
		<section class="animate-enter overflow-hidden rounded-lg border border-border bg-white shadow-sm" aria-labelledby="history-heading">
			<div class="border-b border-border px-5 py-4">
				<div class="flex items-center gap-2"><Database class="size-4 text-muted-foreground" aria-hidden="true" /><h2 id="history-heading" class="text-sm font-semibold">Tahminlerin Geçmiş Sefer Dayanağı</h2></div>
				<p class="mt-1 text-sm text-muted-foreground">İki tahminin hangi geçmiş sefer kayıtlarına dayandığını ayrı ayrı görün. Tüm sayılar planlanan tarihten önceki seferleri gösterir.</p>
			</div>
			<div class="grid gap-px bg-border lg:grid-cols-2">
				<HistoryPanel
					title="Mevcut güzergâh"
					subtitle="Durak eklenmeden önceki talep tahmini"
					status={currentHistoryStatus(prediction)}
					items={currentHistory}
					note={currentHistoryNote(prediction)}
				/>
				<HistoryPanel
					title="Durak eklenmiş güzergâh"
					subtitle="Önerilen yeni durak diziliminin talep tahmini"
					status={proposedHistoryStatus(prediction)}
					items={proposedHistory}
					note={proposedHistoryNote(prediction)}
				/>
			</div>
			{hasLimitedComparison(prediction) && (
				<div class="flex items-start gap-3 border-t border-amber-200 bg-amber-50 px-5 py-4 text-sm text-amber-900">
					<Info class="mt-0.5 size-4 shrink-0" aria-hidden="true" />
					<p><strong class="font-semibold">Karşılaştırmayı temkinli yorumlayın.</strong> Mevcut güzergâh için eşleşen geçmiş sefer bulunmadığından, yolcu değişimi doğrudan gözlenmiş iki güzergâhın karşılaştırması değildir.</p>
				</div>
			)}
		</section>
	);
}

function HistoryPanel({ title, subtitle, status, items, note }: {
	title: string;
	subtitle: string;
	status: { label: string; className: string };
	items: { label: string; value: number; Icon: typeof Route }[];
	note: string;
}) {
	return (
		<div class="bg-white p-5">
			<div class="flex flex-wrap items-start justify-between gap-3">
				<div><h3 class="text-sm font-semibold">{title}</h3><p class="mt-1 text-xs text-muted-foreground">{subtitle}</p></div>
				<span class={`rounded-full px-2.5 py-1 text-xs font-semibold ${status.className}`}>{status.label}</span>
			</div>
			<div class="mt-4 grid gap-2 sm:grid-cols-2">
				{items.map(({ label, value, Icon }) => (
					<div key={label} class="rounded-md border border-border bg-slate-50/60 p-3">
						<div class="flex items-center gap-2 text-muted-foreground"><Icon class="size-3.5" aria-hidden="true" /><span class="text-xs leading-snug">{label}</span></div>
						<p class="mt-2 text-xl font-semibold tabular-nums">{value.toLocaleString('tr-TR')} <span class="text-xs font-normal text-muted-foreground">sefer</span></p>
					</div>
				))}
			</div>
			<p class="mt-4 border-l-2 border-slate-300 pl-3 text-xs leading-relaxed text-muted-foreground">{note}</p>
		</div>
	);
}

function ProposedTimeline({ stops, insertedStopId }: { stops: RouteDurak[]; insertedStopId: number }) {
	if (!stops.length) return <div class="rounded-lg border border-dashed border-border bg-white p-5 text-sm text-muted-foreground">Önerilen durak sırası oluşturuldu; durak detayları alınamadı.</div>;
	return (
		<section class="animate-enter rounded-lg border border-border bg-white shadow-sm">
			<div class="flex flex-wrap items-center justify-between gap-3 border-b border-border px-5 py-4">
				<div><h2 class="text-sm font-semibold">Önerilen durak sırası</h2><p class="mt-1 text-sm text-muted-foreground">Yeni durak, modelin seçtiği en uygun ara konuma yerleştirildi.</p></div>
				<span class="rounded-md bg-muted px-2 py-1 text-xs font-medium text-muted-foreground">{stops.length} durak</span>
			</div>
			<div class="overflow-x-auto p-5">
				<ol class="flex min-w-max">
					{stops.map((stop, index) => {
						const inserted = stop.durak_id === insertedStopId;
						const edge = index === 0 || index === stops.length - 1;
						return <li key={`${stop.durak_id}-${index}`} class="relative w-44 shrink-0 pr-5 last:pr-0">
							{index < stops.length - 1 && <span class="absolute left-3 right-0 top-2.5 h-px bg-border" />}
							<span class={`relative z-10 block size-5 rounded-full border-4 border-white ${inserted ? 'bg-emerald-500 ring-2 ring-emerald-200' : edge ? 'bg-slate-900 ring-1 ring-slate-900' : 'bg-slate-300 ring-1 ring-slate-300'}`} />
							<p class="mt-3 max-w-36 text-sm font-medium leading-snug">{stop.durak_adi ?? stop.kisa_adi ?? `Durak ${stop.durak_id}`}</p>
							<p class={`mt-1 text-xs ${inserted ? 'font-semibold text-emerald-600' : 'text-muted-foreground'}`}>{inserted ? 'Yeni durak' : index === 0 ? 'Kalkış' : index === stops.length - 1 ? 'Varış' : `${index + 1}. durak`}</p>
						</li>;
					})}
				</ol>
			</div>
		</section>
	);
}

function buildProposedStops(prediction: StopAdditionPrediction, route: RouteDetailResponse | null, candidate: Durak | null): RouteDurak[] {
	if (!route) return [];
	const existing = new Map(route.duraklar.map((stop) => [stop.durak_id, stop]));
	return parseStopList(prediction.proposed_stop_list).map((id, index) => {
		if (id === prediction.added_stop_uetds_yer_id && candidate) {
			return { sira: index + 1, durak_id: candidate.id, durak_adi: candidate.uetds_adi, kisa_adi: candidate.kisa_adi, il_id: candidate.il_id, ilce_id: candidate.ilce_id, enlem: candidate.enlem, boylam: candidate.boylam };
		}
		const stop = existing.get(id);
		return stop ? { ...stop, sira: index + 1 } : { sira: index + 1, durak_id: id, durak_adi: null, kisa_adi: null, il_id: null, ilce_id: null, enlem: null, boylam: null };
	});
}

function parseStopList(value: number[] | string): number[] {
	if (Array.isArray(value)) return value.filter(Number.isFinite);
	try {
		const parsed: unknown = JSON.parse(value);
		return Array.isArray(parsed)
			? parsed.filter((stop): stop is number => typeof stop === 'number' && Number.isFinite(stop))
			: [];
	} catch {
		return [];
	}
}

function EmptyState() {
	return <div class="mt-6 rounded-lg border border-dashed border-slate-300 bg-white/60 px-6 py-10 text-center"><div class="mx-auto grid size-10 place-items-center rounded-full bg-muted text-muted-foreground"><MapPinPlus class="size-5" /></div><h2 class="mt-3 text-sm font-medium">Henüz bir durak ekleme önerisi değerlendirilmedi</h2><p class="mt-1 text-sm text-muted-foreground">Talep tahmini, beklenen yolcu değişimi ve ek mesafeyi görmek için öneri bilgilerini girin.</p></div>;
}

function LoadingState() {
	return <div class="mt-6 flex items-center gap-3 rounded-lg border border-border bg-white p-5 text-sm shadow-sm" role="status"><LoaderCircle class="size-5 animate-spin text-muted-foreground" /><div><p class="font-medium">Durak ekleme önerisi değerlendiriliyor</p><p class="mt-0.5 text-muted-foreground">Talep tahmini, beklenen yolcu değişimi ve ek mesafe hesaplanıyor.</p></div></div>;
}

function ErrorBanner({ message }: { message: string }) {
	return <div class="animate-enter mt-6 flex items-start gap-3 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700" role="alert"><AlertTriangle class="mt-0.5 size-4 shrink-0" /><div><strong class="font-medium">Durak ekleme önerisi değerlendirilemedi.</strong> <span>{message}</span></div></div>;
}

function Field({ label, invalid, children }: { label: string; invalid: boolean; children: preact.ComponentChildren }) {
	return <label class="grid gap-2 text-sm font-medium"><span class={invalid ? 'text-red-600' : ''}>{label}</span>{children}</label>;
}
function inputClass(error: boolean) { return `h-10 min-w-0 w-full rounded-md border bg-white px-3 text-sm shadow-sm outline-none transition-colors focus:ring-2 focus:ring-slate-400/25 ${error ? 'border-red-500' : 'border-border focus:border-slate-400'}`; }
function validId(value: string) { return value !== '' && Number.isInteger(Number(value)) && Number(value) >= 0; }
function errorMessage(error: unknown) { return error instanceof Error ? (error instanceof ApiError ? apiErrorLabel(error.message) : error.message) : 'Bilinmeyen bir hata oluştu.'; }
function apiErrorLabel(value: string) {
	const labels: Record<string, string> = {
		COMPANY_NOT_FOUND: 'Firma bulunamadı.',
		CURRENT_ROUTE_NOT_FOUND_FOR_COMPANY: 'Bu firmaya ait mevcut güzergâh bulunamadı.',
		CANDIDATE_STOP_NOT_FOUND: 'Eklenecek durak bulunamadı.',
		CURRENT_ROUTE_TOO_SHORT: 'Mevcut rota durak eklemek için çok kısa.',
		CANDIDATE_STOP_ALREADY_IN_ROUTE: 'Seçilen durak bu rotada zaten bulunuyor.',
		ROUTE_OR_CANDIDATE_COORDINATES_UNUSABLE: 'Rota veya durak koordinatları kullanılamıyor.',
		NO_USABLE_INTERMEDIATE_INSERTION: 'Durak için kullanılabilir bir ara ekleme konumu bulunamadı.',
	};
	return labels[value] ?? `Sunucu yanıtı: ${value}`;
}
function decisionConfig(decision: StopAdditionPrediction['business_decision']) {
	if (decision === 'APPROVE') return { label: 'Önerilir', shell: 'border-emerald-200 bg-emerald-50 text-emerald-950', badge: 'bg-emerald-600 text-white', icon: 'bg-emerald-600 text-white', Icon: BadgeCheck };
	if (decision === 'REJECT') return { label: 'Önerilmez', shell: 'border-red-200 bg-red-50 text-red-950', badge: 'bg-red-600 text-white', icon: 'bg-red-600 text-white', Icon: Ban };
	return { label: 'İnceleme gerekli', shell: 'border-amber-200 bg-amber-50 text-amber-950', badge: 'bg-amber-500 text-white', icon: 'bg-amber-500 text-white', Icon: CircleHelp };
}
function decisionLabel(prediction: StopAdditionPrediction) {
	return prediction.decision_override === 'ORIGIN_CITY_HARD_APPROVE'
		? 'Kesin onay'
		: decisionConfig(prediction.business_decision).label;
}
function recommendationTitle(prediction: StopAdditionPrediction) {
	if (prediction.decision_override === 'ORIGIN_CITY_HARD_APPROVE') return 'Bu durağın eklenmesi öneriliyor';
	if (prediction.business_decision === 'APPROVE') return 'Bu durağın eklenmesi öneriliyor';
	if (prediction.business_decision === 'REJECT') return 'Bu durağın eklenmesi önerilmiyor';
	return 'Karar vermeden önce manuel inceleme gerekiyor';
}
function recommendationSummary(prediction: StopAdditionPrediction) {
	const uplift = prediction.predicted_uplift;
	if (prediction.decision_override === 'ORIGIN_CITY_HARD_APPROVE') {
		return 'eklenecek durak firmanın kayıtlı çıkış iliyle aynı ilde olduğu için operasyonel kesin onay kuralı uygulandı. Talep ve mesafe göstergeleri bu kararı değiştirmedi; aşağıda ayrıca risk olarak gösteriliyor.';
	}
	if (prediction.business_decision === 'APPROVE') {
		return uplift !== null && uplift > 0
			? `durak eklendiğinde sefer başına yaklaşık ${uplift.toFixed(1)} daha fazla yolcu bekleniyor ve ek mesafe kabul edilebilir düzeyde kalıyor.`
			: 'talep ve mesafe etkisi birlikte değerlendirildiğinde ekleme uygun görünüyor.';
	}
	if (prediction.business_decision === 'REJECT') {
		if (uplift !== null && uplift < 0) {
			return `durak eklendiğinde sefer başına yaklaşık ${Math.abs(uplift).toFixed(1)} daha az yolcu bekleniyor. Güzergâh ayrıca ${prediction.added_haversine_km.toFixed(1)} km uzuyor.`;
		}
		if (prediction.detour_ratio >= .12) {
			return `güzergâh ${prediction.added_haversine_km.toFixed(1)} km uzuyor; bu artış mevcut talep beklentisine göre yüksek kalıyor.`;
		}
		return 'beklenen yolcu talebi ve ek mesafe birlikte değerlendirildiğinde ekleme uygun görünmüyor.';
	}
	return 'mevcut veriler kesin bir öneri üretmek için yeterli değil; geçmiş seferler ve operasyonel ihtiyaç birlikte incelenmeli.';
}
function decisionFactors(prediction: StopAdditionPrediction) {
	const uplift = prediction.predicted_uplift;
	const hardApprove = prediction.decision_override === 'ORIGIN_CITY_HARD_APPROVE';
	const factors = [
		...(hardApprove ? [{
			Icon: BadgeCheck,
			text: 'Eklenecek durak firmanın kayıtlı çıkış iliyle aynı ilde. Bu eşleşme operasyonel kesin onay kuralını tetikledi.',
			tone: 'text-emerald-600',
		}] : []),
		{
			Icon: TrendingUp,
			text: uplift === null
				? 'Yolcu değişimi hesaplanamadı.'
				: uplift < 0
					? `Durak eklendiğinde sefer başına ${Math.abs(uplift).toFixed(1)} daha az yolcu bekleniyor${hardApprove ? '; bu risk kesin onay kuralını geçersiz kılmadı.' : '.'}`
					: `Durak eklendiğinde sefer başına ${uplift.toFixed(1)} daha fazla yolcu bekleniyor.`,
			tone: uplift !== null && uplift < 0 ? 'text-red-600' : 'text-emerald-600',
		},
		{
			Icon: MapPinPlus,
			text: `Güzergâh ${prediction.added_haversine_km.toFixed(1)} km uzuyor; toplam mesafe %${(prediction.detour_ratio * 100).toFixed(1)} artıyor${hardApprove ? '. Bu değer kararın önüne geçmeyen önemli bir operasyonel risktir.' : '.'}`,
			tone: prediction.detour_ratio >= .12 ? 'text-red-600' : 'text-slate-500',
		},
	];
	if (hasLimitedComparison(prediction)) {
		factors.push({
			Icon: AlertTriangle,
			text: 'Mevcut güzergâh için eşleşen geçmiş sefer yok; mevcut talep tahmini daha geniş veri ortalamalarına dayanıyor.',
			tone: 'text-amber-600',
		});
	} else {
		factors.push({
			Icon: Database,
			text: `${count(prediction.current_route_history_company_route_count)} geçmiş sefer mevcut güzergâh tahminini destekliyor.`,
			tone: 'text-slate-500',
		});
	}
	return factors;
}
function signed(value: number, digits: number) { return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`; }
function titleCase(title: string) {
	return title.toLocaleLowerCase('tr').split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
}
function stopName(stop: Durak | null, id: number) { return stop?.uetds_adi ?? stop?.kisa_adi ?? `Durak ${id}`; }
function formatDate(value: string) { const [year, month, day] = value.split('-'); return `${day}.${month}.${year}`; }
function count(value: number | null | undefined) { return typeof value === 'number' && Number.isFinite(value) ? value : 0; }
function hasLimitedComparison(prediction: StopAdditionPrediction) {
	return count(prediction.current_route_history_exact_time_count) === 0
		&& count(prediction.current_route_history_company_route_count) === 0;
}
function currentPredictionDetail(prediction: StopAdditionPrediction) {
	const routeCount = count(prediction.current_route_history_company_route_count);
	if (routeCount === 0) return 'Eşleşen geçmiş sefer yok; genel veriden tahmin edildi';
	return `${routeCount.toLocaleString('tr-TR')} aynı firma-güzergâh seferiyle destekleniyor`;
}
function proposedPredictionDetail(prediction: StopAdditionPrediction) {
	const sameCompany = count(prediction.proposed_route_history_same_company_route_count);
	if (sameCompany > 0) return `${sameCompany.toLocaleString('tr-TR')} aynı firma ve durak dizilimi seferiyle destekleniyor`;
	const allCompany = count(prediction.proposed_route_history_all_company_route_count);
	if (allCompany > 0) return `${allCompany.toLocaleString('tr-TR')} aynı durak dizilimi seferiyle destekleniyor`;
	const similar = count(prediction.proposed_route_history_similar_trip_count);
	if (similar > 0) return `${similar.toLocaleString('tr-TR')} benzer güzergâh seferinden yararlanıyor`;
	return 'Bu durak dizilimi için geçmiş sefer yok';
}
function upliftDescription(uplift: number | null) {
	if (uplift === null) return 'Mevcut ve önerilen tahmin karşılaştırılamadı';
	if (uplift < 0) return `Durak eklendiğinde sefer başına ${Math.abs(uplift).toFixed(1)} daha az yolcu bekleniyor`;
	if (uplift > 0) return `Durak eklendiğinde sefer başına ${uplift.toFixed(1)} daha fazla yolcu bekleniyor`;
	return 'Durak eklenmesiyle yolcu sayısında değişim beklenmiyor';
}
function currentHistoryStatus(prediction: StopAdditionPrediction) {
	const routeCount = count(prediction.current_route_history_company_route_count);
	return routeCount > 0
		? { label: `${routeCount.toLocaleString('tr-TR')} sefer bulundu`, className: 'bg-emerald-100 text-emerald-700' }
		: { label: 'Geçmiş sefer yok', className: 'bg-amber-100 text-amber-800' };
}
function proposedHistoryStatus(prediction: StopAdditionPrediction) {
	const exact = count(prediction.proposed_route_history_all_company_route_count);
	const similar = count(prediction.proposed_route_history_similar_trip_count);
	if (exact > 0) return { label: `${exact.toLocaleString('tr-TR')} tam güzergâh seferi`, className: 'bg-emerald-100 text-emerald-700' };
	if (similar > 0) return { label: 'Benzer güzergâh verisi var', className: 'bg-sky-100 text-sky-700' };
	return { label: 'Geçmiş sefer yok', className: 'bg-amber-100 text-amber-800' };
}
function currentHistoryNote(prediction: StopAdditionPrediction) {
	if (hasLimitedComparison(prediction)) {
		return 'Bu güzergâhta firmaya ait geçmiş sefer bulunmadı. Gösterilen mevcut talep, daha geniş tarihsel veri ve model örüntülerinden hesaplandı; doğrudan güzergâh geçmişi değildir.';
	}
	if (prediction.current_route_reliability_reason) return prediction.current_route_reliability_reason;
	return 'Mevcut güzergâh tahmini, yukarıdaki geçmiş sefer eşleşmelerinden yararlanır.';
}
function proposedHistoryNote(prediction: StopAdditionPrediction) {
	const sameCompany = count(prediction.proposed_route_history_same_company_route_count);
	const allCompany = count(prediction.proposed_route_history_all_company_route_count);
	if (sameCompany > 0) return `Önerilen durak dizilimi bu firmada daha önce ${sameCompany.toLocaleString('tr-TR')} seferde kullanıldı. Talep tahmini öncelikle bu doğrudan geçmişten yararlanır.`;
	if (allCompany > 0) return `Önerilen durak dizilimi diğer firmalar dahil ${allCompany.toLocaleString('tr-TR')} geçmiş seferde görüldü. Tahmin bu güzergâh geçmişinden yararlanır.`;
	if (count(prediction.proposed_route_history_similar_trip_count) > 0) return 'Aynı durak dizilimi için geçmiş sefer yok. Tahmin, coğrafi ve durak yapısı benzer güzergâhlardaki seferlerden yararlanır.';
	return 'Önerilen durak dizilimi veya benzer güzergâhlar için geçmiş sefer bulunmadı. Tahmin daha genel firma ve ağ verilerine dayanır.';
}
