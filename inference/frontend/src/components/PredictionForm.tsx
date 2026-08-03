import { signal } from '@preact/signals';
import type { TargetedEvent } from 'preact';
import { useState } from 'preact/hooks';
import { ChartNoAxesCombined, LoaderCircle } from 'lucide-preact';
import type { GeneralPredictionRequest, PredictionRequest } from '../api';

export type PredictionMode = 'trip' | 'general';

export type PredictionFormRequest =
	| { mode: 'trip'; request: PredictionRequest }
	| { mode: 'general'; request: GeneralPredictionRequest };

export interface PredictionFormProps {
	onSubmit: (submission: PredictionFormRequest) => void;
	loading: boolean;
}

const mode = signal<PredictionMode>('trip');
const firmaId = signal('42185');
const guzergahKodu = signal('110926');
const seferTarihi = signal(new Date().toISOString().slice(0, 10));
const departureTime = signal('13:00');

const MODES: { value: PredictionMode; label: string }[] = [
	{ value: 'trip', label: 'Sefer bazlı' },
	{ value: 'general', label: 'Genel rota' },
];

export function PredictionForm({ onSubmit, loading }: PredictionFormProps) {
	const [touched, setTouched] = useState(false);
	const general = mode.value === 'general';
	const firmaValid = firmaId.value !== '' && Number(firmaId.value) >= 0;
	const guzergahValid = guzergahKodu.value !== '' && Number(guzergahKodu.value) >= 0;
	const dateValid = general || seferTarihi.value !== '';
	const timeValid = general || /^([01]\d|2[0-3]):[0-5]\d$/.test(departureTime.value);
	const valid = firmaValid && guzergahValid && dateValid && timeValid;

	function handleSubmit(event: TargetedEvent<HTMLFormElement>) {
		event.preventDefault();
		setTouched(true);
		if (!valid || loading) return;
		if (general) {
			onSubmit({
				mode: 'general',
				request: {
					firma_id: Number(firmaId.value),
					guzergah_kodu: Number(guzergahKodu.value),
				},
			});
			return;
		}
		onSubmit({
			mode: 'trip',
			request: {
				firma_id: Number(firmaId.value),
				guzergah_kodu: Number(guzergahKodu.value),
				sefer_tarihi: seferTarihi.value,
				sefer_saati: departureTime.value,
			},
		});
	}

	return (
		<form noValidate onSubmit={handleSubmit} class="rounded-lg border border-border bg-white shadow-sm">
			<div class="flex flex-wrap items-start justify-between gap-3 border-b border-border px-5 py-4">
				<div>
					<h2 class="text-sm font-semibold">{general ? 'Rota bilgileri' : 'Sefer bilgileri'}</h2>
					<p class="mt-1 text-sm text-muted-foreground">
						{general
							? 'Tarih ve saat olmadan, yalnızca firma ve güzergâhla genel talep tahmini oluşturun.'
							: 'Tahmin oluşturmak için sefer detaylarını girin.'}
					</p>
				</div>
				<div class="inline-flex shrink-0 rounded-md border border-border bg-muted p-0.5" role="group" aria-label="Tahmin türü">
					{MODES.map((item) => (
						<button
							key={item.value}
							type="button"
							aria-pressed={mode.value === item.value}
							onClick={() => (mode.value = item.value)}
							class={`rounded px-3 py-1.5 text-sm font-medium transition-colors ${mode.value === item.value ? 'bg-white text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}
						>
							{item.label}
						</button>
					))}
				</div>
			</div>
			<div class={`grid gap-4 p-5 md:grid-cols-2 lg:items-end ${general ? 'lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto]' : 'lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_1.15fr_1fr_auto]'}`}>
				<Field label="Firma ID" error={touched && !firmaValid}>
					<input type="number" min={0} step={1} class={inputClass(touched && !firmaValid)} value={firmaId.value} onInput={(e) => (firmaId.value = e.currentTarget.value)} />
				</Field>
				<Field label="Güzergâh kodu" error={touched && !guzergahValid}>
					<input type="number" min={0} step={1} class={inputClass(touched && !guzergahValid)} value={guzergahKodu.value} onInput={(e) => (guzergahKodu.value = e.currentTarget.value)} />
				</Field>
				{!general && (
					<>
						<Field label="Sefer tarihi" error={touched && !dateValid}>
							<input type="date" min="2023-01-01" class={`${inputClass(touched && !dateValid)} [color-scheme:light]`} value={seferTarihi.value} onInput={(e) => (seferTarihi.value = e.currentTarget.value)} />
						</Field>
						<Field label="Kalkış saati" error={touched && !timeValid} errorMessage="Saat 00–23, dakika ise 00–59 arasında olmalı.">
							<input type="time" aria-label="Kalkış saati" class={`${inputClass(touched && !timeValid)} [color-scheme:light]`} value={departureTime.value} onInput={(e) => (departureTime.value = e.currentTarget.value)} />
						</Field>
					</>
				)}
				<button type="submit" disabled={loading} class="inline-flex h-10 items-center justify-center gap-2 whitespace-nowrap rounded-md bg-primary px-4 text-sm font-medium text-white shadow-sm transition-colors hover:bg-slate-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-900 disabled:cursor-not-allowed disabled:opacity-60">
					{loading ? <LoaderCircle class="size-4 animate-spin" aria-hidden="true" /> : <ChartNoAxesCombined class="size-4" aria-hidden="true" />}
					{loading ? 'Hesaplanıyor...' : 'Tahmin et'}
				</button>
			</div>
		</form>
	);
}

function Field({ label, error, errorMessage, hint, children }: { label: string; error?: boolean; errorMessage?: string; hint?: string; children: preact.ComponentChildren }) {
	return <label class="grid gap-2 text-sm font-medium"><span class={error ? 'text-red-600' : ''}>{label}</span>{children}{error && errorMessage && <span class="text-xs font-normal leading-snug text-red-600">{errorMessage}</span>}{!error && hint && <span class="text-xs font-normal leading-snug text-muted-foreground">{hint}</span>}</label>;
}

function inputClass(error: boolean) {
	return `h-10 min-w-0 w-full rounded-md border bg-white px-3 text-sm shadow-sm outline-none transition-colors placeholder:text-muted-foreground focus:ring-2 focus:ring-slate-400/25 disabled:cursor-not-allowed disabled:bg-slate-50 disabled:text-muted-foreground ${error ? 'border-red-500 focus:border-red-500' : 'border-border focus:border-slate-400'}`;
}
