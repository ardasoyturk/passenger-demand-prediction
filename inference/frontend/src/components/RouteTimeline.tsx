import type { RouteDurak } from '../api';

export function RouteTimeline({ duraklar }: { duraklar: RouteDurak[] }) {
	if (!duraklar.length) return <p class="rounded-lg border border-border bg-white p-5 text-sm text-muted-foreground">Bu güzergâh için durak bilgisi bulunamadı.</p>;
	const departure = stopName(duraklar[0]);
	const arrival = stopName(duraklar[duraklar.length - 1]);

	return (
		<section class="animate-enter rounded-lg border border-border bg-white shadow-sm" aria-labelledby="route-heading">
			<div class="flex flex-wrap items-center justify-between gap-3 border-b border-border px-5 py-4">
				<div>
					<h2 id="route-heading" class="text-sm font-semibold">Güzergâh</h2>
					<p class="mt-1 text-sm text-muted-foreground">{departure} → {arrival}</p>
				</div>
				<span class="rounded-md bg-muted px-2 py-1 text-xs font-medium text-muted-foreground">{duraklar.length} durak</span>
			</div>
			<div class="overflow-x-auto p-5">
				<ol class="flex min-w-max">
					{duraklar.map((stop, index) => {
						const edge = index === 0 || index === duraklar.length - 1;
						return <li key={`${stop.durak_id}-${index}`} class="relative w-44 shrink-0 pr-5 last:pr-0">
							{index < duraklar.length - 1 && <span class="absolute left-3 right-0 top-2.5 h-px bg-border" aria-hidden="true" />}
							<span class={`relative z-10 block size-5 rounded-full border-4 border-white ${edge ? 'bg-slate-900 ring-1 ring-slate-900' : 'bg-slate-300 ring-1 ring-slate-300'}`} />
							<p class="mt-3 max-w-36 text-sm font-medium leading-snug">{stopName(stop)}</p>
							<p class="mt-1 text-xs text-muted-foreground">{index === 0 ? 'Kalkış' : index === duraklar.length - 1 ? 'Varış' : `${stop.sira}. durak`}</p>
						</li>;
					})}
				</ol>
			</div>
		</section>
	);
}

function stopName(stop: RouteDurak) { return stop.durak_adi ?? stop.kisa_adi ?? `Durak ${stop.durak_id}`; }
