import { Building2 } from 'lucide-preact';
import { titleCase } from '../lib/display';

export function PredictionContext({
	heading,
	companyId,
	companyTitle,
	routeCode,
	date,
	time,
}: {
	heading: string;
	companyId: number;
	companyTitle?: string | null;
	routeCode: number;
	date?: string;
	time?: string;
}) {
	const title = companyTitle?.trim() ? titleCase(companyTitle) : `Firma ${companyId}`;

	return (
		<header>
			<div class="border-b border-border pb-3 flex min-w-0 flex-col gap-2 sm:flex-row sm:items-center sm:justify-between sm:gap-6">
				<div class="flex min-w-0 items-center gap-1.5 text-sm text-muted-foreground">
					<Building2 class="size-3.5 shrink-0" aria-hidden="true" />
					<p class="truncate" title={title}>{title}</p>
				</div>
				<p class="shrink-0 text-xs tabular-nums text-muted-foreground">
					Firma {companyId} <span class="px-1 text-slate-300">·</span>
					Güzergâh {routeCode}
					{date && time && (
						<>
							<span class="px-1 text-slate-300">·</span>
							{formatDate(date)} {time}
						</>
					)}
				</p>
			</div>
			<h2 class="mt-3 text-base font-semibold">{heading}</h2>
		</header>
	);
}

function formatDate(value: string) {
	const [year, month, day] = value.split('-');
	return `${day}.${month}.${year}`;
}
