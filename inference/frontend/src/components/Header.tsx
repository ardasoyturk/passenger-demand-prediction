import { useEffect, useState } from 'preact/hooks';
import { BusFront, LoaderCircle } from 'lucide-preact';
import { getHealth } from '../api';

type BackendStatus = 'checking' | 'ready' | 'offline';

export function Header() {
	const [backendStatus, setBackendStatus] = useState<BackendStatus>('checking');

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
			<div class="mx-auto flex h-14 max-w-7xl items-center px-4 sm:px-6 lg:px-8">
				<a href="/" class="flex items-center gap-2.5 font-semibold tracking-tight" aria-label="Yolcu Talep Tahmini ana sayfa">
					<span class="grid size-8 place-items-center rounded-md bg-primary text-white">
						<BusFront class="size-4" aria-hidden="true" />
					</span>
					<span>Yolcu Talep Tahmini</span>
				</a>
				<div class="ml-auto flex items-center gap-2 text-xs">
					<BackendIndicator status={backendStatus} />
					<span class={`ml-2 hidden rounded-md px-2 py-1 font-medium sm:inline ${backendStatus === 'ready' ? 'bg-muted text-foreground' : 'bg-slate-100 text-muted-foreground'}`}>v4.2 + v4.4</span>
				</div>
			</div>
		</header>
	);
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
