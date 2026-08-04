import type { ComponentChildren } from 'preact';

export type ModeOption<T extends string> = {
	value: T;
	label: string;
};

export const STANDARD_MODES: ModeOption<'trip' | 'general'>[] = [
	{ value: 'trip', label: 'Sefer bazlı' },
	{ value: 'general', label: 'Genel rota' },
];

export function ModeToggle<T extends string>({
	options,
	value,
	onChange,
	ariaLabel,
}: {
	options: ModeOption<T>[];
	value: T;
	onChange: (value: T) => void;
	ariaLabel: string;
}) {
	return (
		<div class="inline-flex shrink-0 rounded-md border border-border bg-muted p-0.5" role="group" aria-label={ariaLabel}>
			{options.map((item) => (
				<button
					key={item.value}
					type="button"
					aria-pressed={value === item.value}
					onClick={() => onChange(item.value)}
					class={`rounded px-3 py-1.5 text-sm font-medium transition-colors ${value === item.value ? 'bg-white text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}
				>
					{item.label}
				</button>
			))}
		</div>
	);
}

export function Field({
	label,
	error,
	errorMessage,
	hint,
	children,
}: {
	label: string;
	error?: boolean;
	errorMessage?: string;
	hint?: string;
	children: ComponentChildren;
}) {
	return (
		<label class="grid gap-2 text-sm font-medium">
			<span class={error ? 'text-red-600' : ''}>{label}</span>
			{children}
			{error && errorMessage && <span class="text-xs font-normal leading-snug text-red-600">{errorMessage}</span>}
			{!error && hint && <span class="text-xs font-normal leading-snug text-muted-foreground">{hint}</span>}
		</label>
	);
}

export function inputClass(error: boolean) {
	return `h-10 min-w-0 w-full rounded-md border bg-white px-3 text-sm shadow-sm outline-none transition-colors placeholder:text-muted-foreground focus:ring-2 focus:ring-slate-400/25 disabled:cursor-not-allowed disabled:bg-slate-50 disabled:text-muted-foreground ${error ? 'border-red-500 focus:border-red-500' : 'border-border focus:border-slate-400'}`;
}
