import { useEffect, useRef } from 'preact/hooks';
import 'leaflet/dist/leaflet.css';

import type { RouteDurak } from '../api';

type MappedStop = RouteDurak & { enlem: number; boylam: number };

export function RouteMap({ duraklar }: { duraklar: RouteDurak[] }) {
	const mapElement = useRef<HTMLDivElement>(null);
	const mappedStops = duraklar.filter(hasCoordinates);
	const skippedCount = duraklar.length - mappedStops.length;

	useEffect(() => {
		if (!mapElement.current || !mappedStops.length) return;

		let disposed = false;
		let map: import('leaflet').Map | undefined;
		let resizeObserver: ResizeObserver | undefined;

		void import('leaflet').then(({ default: L }) => {
			if (disposed || !mapElement.current) return;

			map = L.map(mapElement.current, {
				zoomControl: true,
				scrollWheelZoom: true,
			});

			L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
				attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
				maxZoom: 19,
			}).addTo(map);

			const positions = mappedStops.map((stop) => L.latLng(stop.enlem, stop.boylam));
			L.polyline(positions, {
				color: '#0f172a',
				weight: 4,
				opacity: 0.78,
				lineCap: 'round',
				lineJoin: 'round',
			}).addTo(map);

			mappedStops.forEach((stop, index) => {
				const edge = index === 0 || index === mappedStops.length - 1;
				const marker = L.circleMarker([stop.enlem, stop.boylam], {
					radius: edge ? 8 : 6,
					color: '#ffffff',
					weight: 3,
					fillColor: edge ? '#0f172a' : '#64748b',
					fillOpacity: 1,
				}).addTo(map!);

				marker.bindTooltip(
					`<strong>${escapeHtml(stopName(stop))}</strong><br>${stopLabel(stop, duraklar.length)}`,
					{ direction: 'top', offset: [0, -8], opacity: 0.96 },
				);
			});

			if (positions.length === 1) {
				map.setView(positions[0], 12);
			} else {
				map.fitBounds(L.latLngBounds(positions), { padding: [36, 36], maxZoom: 12 });
			}

			resizeObserver = new ResizeObserver(() => map?.invalidateSize());
			resizeObserver.observe(mapElement.current);
		});

		return () => {
			disposed = true;
			resizeObserver?.disconnect();
			map?.remove();
		};
	}, [duraklar]);

	if (!mappedStops.length) {
		return (
			<section class="animate-enter rounded-lg border border-dashed border-border bg-white p-5">
				<h2 class="text-sm font-semibold">Güzergâh haritası</h2>
				<p class="mt-1 text-sm text-muted-foreground">Koordinat bilgisi bulunan durak olmadığı için harita gösterilemiyor.</p>
			</section>
		);
	}

	return (
		<section class="animate-enter overflow-hidden rounded-lg border border-border bg-white shadow-sm" aria-labelledby="route-map-heading">
			<div class="flex flex-wrap items-center justify-between gap-3 border-b border-border px-5 py-4">
				<div>
					<h2 id="route-map-heading" class="text-sm font-semibold">Güzergâh haritası</h2>
					<p class="mt-1 text-sm text-muted-foreground">
						{mappedStops.length} durak haritada gösteriliyor
						{skippedCount > 0 && ` · ${skippedCount} konumsuz durak atlandı`}
					</p>
				</div>
				<span class="rounded-md bg-muted px-2 py-1 text-xs font-medium text-muted-foreground">OpenStreetMap</span>
			</div>
			<div
				ref={mapElement}
				class="h-80 w-full bg-slate-100 sm:h-96"
				role="region"
				aria-label={`${mappedStops.length} duraklı güzergâh haritası`}
			/>
		</section>
	);
}

function hasCoordinates(stop: RouteDurak): stop is MappedStop {
	return (
		typeof stop.enlem === 'number'
		&& Number.isFinite(stop.enlem)
		&& stop.enlem >= -90
		&& stop.enlem <= 90
		&& typeof stop.boylam === 'number'
		&& Number.isFinite(stop.boylam)
		&& stop.boylam >= -180
		&& stop.boylam <= 180
	);
}

function stopName(stop: RouteDurak) {
	return stop.durak_adi ?? stop.kisa_adi ?? `Durak ${stop.durak_id}`;
}

function stopLabel(stop: RouteDurak, totalStops: number) {
	if (stop.sira === 1) return 'Kalkış';
	if (stop.sira === totalStops) return 'Varış';
	return `${stop.sira}. durak`;
}

function escapeHtml(value: string) {
	return value.replace(/[&<>"']/g, (character) => ({
		'&': '&amp;',
		'<': '&lt;',
		'>': '&gt;',
		'"': '&quot;',
		"'": '&#039;',
	})[character] ?? character);
}
