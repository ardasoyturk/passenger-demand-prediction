import { lazy, Suspense } from 'preact/compat';
import { Bot, LoaderCircle, Maximize2, MessageCircle, Minus, X } from 'lucide-preact';
import { useEffect, useState } from 'preact/hooks';

const Chat = lazy(() => import('../pages/Chat/index').then((m) => m.Chat));

/** A persistent shortcut to the analyst assistant outside its dedicated page. */
export function FloatingAssistant() {
	const [open, setOpen] = useState(false);

	useEffect(() => {
		function onKeyDown(event: KeyboardEvent) {
			if (event.key === 'Escape') setOpen(false);
		}
		window.addEventListener('keydown', onKeyDown);
		return () => window.removeEventListener('keydown', onKeyDown);
	}, []);

	if (window.location.pathname === '/chat') return null;

	return (
		<div class="fixed bottom-4 right-4 z-50 sm:bottom-6 sm:right-6">
			{open && (
				<section
					class="floating-assistant animate-enter fixed inset-x-3 bottom-3 flex h-[calc(100dvh-1.5rem)] w-auto flex-col overflow-hidden rounded-xl border border-slate-200 bg-background shadow-2xl shadow-slate-900/15 sm:static sm:mb-3 sm:h-[min(38rem,calc(100dvh-7.5rem))] sm:w-[27rem]"
					aria-label="Yapay zekâ asistanı"
				>
					<header class="flex shrink-0 items-center gap-2.5 border-b border-border bg-white px-3 py-2.5">
						<span class="grid size-8 place-items-center rounded-md bg-primary text-white">
							<Bot class="size-4" aria-hidden="true" />
						</span>
						<div class="min-w-0 flex-1">
							<h2 class="text-sm font-semibold">Yapay Zekâ Asistanı</h2>
							<p class="text-xs text-muted-foreground">Güzergâh ve talep analizi</p>
						</div>
						<button type="button" onClick={() => setOpen(false)} class="grid size-8 place-items-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground" aria-label="Asistanı küçült" title="Küçült">
							<Minus class="size-4" aria-hidden="true" />
						</button>
						<a href="/chat" class="grid size-8 place-items-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground" aria-label="Asistanı tam sayfada aç" title="Tam sayfada aç">
							<Maximize2 class="size-4" aria-hidden="true" />
						</a>
					</header>
					<Suspense fallback={<div class="grid flex-1 place-items-center"><LoaderCircle class="size-5 animate-spin text-muted-foreground" /></div>}>
						<Chat compact />
					</Suspense>
				</section>
			)}
			<button
				type="button"
				onClick={() => setOpen((value) => !value)}
				class={`floating-assistant-trigger ml-auto size-12 place-items-center rounded-full bg-primary text-white shadow-lg shadow-slate-900/25 transition duration-200 hover:-translate-y-0.5 hover:bg-slate-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-900 ${open ? 'hidden sm:grid' : 'grid'}`}
				aria-label={open ? 'Asistanı küçült' : 'Yapay zekâ asistanını aç'}
				aria-expanded={open}
				title={open ? 'Asistanı küçült' : 'Yapay zekâ asistanına sor'}
			>
				{open ? <X class="size-5" aria-hidden="true" /> : <MessageCircle class="size-5" aria-hidden="true" />}
			</button>
		</div>
	);
}
