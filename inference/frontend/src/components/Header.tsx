import { useEffect, useState } from 'preact/hooks';
import { BusFront, ChartNoAxesCombined, LoaderCircle, Route, Sparkles } from 'lucide-preact';
import { getHealth } from '../api';

type BackendStatus = 'checking' | 'ready' | 'offline';

export function Header() {
	const [backendStatus, setBackendStatus] = useState<BackendStatus>('checking');
	const currentPath = window.location.pathname;

	useEffect(() => {
		let active = true;
		let currentRequest: AbortController | null = null;

		async function checkBackend() {
			currentRequest?.abort();
			const controller = new AbortController();
			currentRequest = controller;
			const timeout = window.setTimeout(() => controller.abort(), 3000);
			try {
				const health = await getHealth(controller.signal);
				if (active) {
					const ready = health.status === 'ok' && health.database === 'ok' && health.artifacts === 'loaded';
					setBackendStatus(ready ? 'ready' : 'offline');
				}
			} catch {
				if (active) setBackendStatus('offline');
			} finally {
				window.clearTimeout(timeout);
			}
		}

		void checkBackend();
		const interval = window.setInterval(checkBackend, 15000);
		return () => {
			active = false;
			currentRequest?.abort();
			window.clearInterval(interval);
		};
	}, []);

	return (
		<header class="border-b border-border bg-white">
			<div class="mx-auto flex min-h-14 max-w-7xl flex-wrap items-center gap-3 px-4 py-2 sm:px-6 lg:px-8">
				<a href="/" class="flex items-center gap-2.5 font-semibold tracking-tight" aria-label="Yolcu Talep Tahmini ana sayfa">
					<span class="grid size-8 place-items-center rounded-md bg-primary text-white">
						<BusFront class="size-4" aria-hidden="true" />
					</span>
					<span>Yolcu Talep Tahmini</span>
				</a>
				<nav class="order-3 flex w-full items-center gap-1 border-t border-border pt-2 sm:order-none sm:ml-5 sm:w-auto sm:border-0 sm:pt-0" aria-label="Ana menü">
					<NavLink href="/" active={currentPath === '/'} Icon={ChartNoAxesCombined}>Sefer Talep Tahmini</NavLink>
					<NavLink href="/proposed-routes" active={currentPath === '/proposed-routes'} Icon={Route}>Durak Ekleme Analizi</NavLink>
					<NavLink href="/chat" active={currentPath === '/chat'} Icon={Sparkles}>Yapay Zekâ Asistanı</NavLink>
				</nav>
				<div class="ml-auto flex items-center gap-2 text-xs">
					<BackendIndicator status={backendStatus} />
					<span class={`ml-2 hidden rounded-md px-2 py-1 font-medium sm:inline ${backendStatus === 'ready' ? 'bg-muted text-foreground' : 'bg-slate-100 text-muted-foreground'}`}>v4.2 + v4.4</span>
				</div>
			</div>
		</header>
	);
}

function NavLink({ href, active, Icon, children }: { href: string; active: boolean; Icon: typeof Route; children: preact.ComponentChildren }) {
	return <a href={href} aria-current={active ? 'page' : undefined} class={`inline-flex h-8 items-center gap-2 rounded-md px-3 text-sm font-medium transition-colors ${active ? 'bg-muted text-foreground' : 'text-muted-foreground hover:bg-muted/70 hover:text-foreground'}`}><Icon class="size-3.5" aria-hidden="true" />{children}</a>;
}

function BackendIndicator({ status }: { status: BackendStatus }) {
	if (status === 'checking') {
		return <span class="inline-flex items-center gap-2 text-muted-foreground" role="status"><LoaderCircle class="size-3.5 animate-spin" aria-hidden="true" /><span class="hidden sm:inline">Model kontrol ediliyor</span></span>;
	}
	if (status === 'ready') {
		return <span class="inline-flex items-center gap-2 text-muted-foreground" title="Backend, veritabanı ve model dosyaları hazır"><span class="size-2 rounded-full bg-emerald-500" /><span class="hidden sm:inline">Model hazır</span></span>;
	}
	return <span class="inline-flex items-center gap-2 font-medium text-red-600" role="alert" title="Backend sağlık kontrolüne yanıt vermiyor"><span class="size-2 rounded-full bg-red-500" /><span class="hidden sm:inline">Model bağlı değil</span></span>;
}
