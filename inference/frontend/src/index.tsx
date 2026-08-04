import { render } from 'preact';
import { lazy, Suspense } from 'preact/compat';
import { LoaderCircle } from 'lucide-preact';

import { Header } from './components/Header';
import { FloatingAssistant } from './components/FloatingAssistant';
import { Home } from './pages/Home/index';
import { ProposedRoutes } from './pages/ProposedRoutes/index';
import './style.css';

// Sohbet sayfası (Markdown/LaTeX/AI SDK bağımlılıklarıyla birlikte) yalnızca
// /chat yolunda yüklensin diye ayrı bir parçaya bölünür.
const Chat = lazy(() => import('./pages/Chat/index').then((m) => m.Chat));

export function App() {
	const path = window.location.pathname;

	// Sohbet sayfası tam yükseklikte, kendi kaydırma alanına sahip bir düzen kullanır.
	if (path === '/chat') {
		return (
			<div class="flex h-dvh flex-col">
				<Header />
				<main id="main-content" class="flex min-h-0 flex-1 flex-col">
					<Suspense fallback={<ChatLoading />}>
						<Chat />
					</Suspense>
				</main>
			</div>
		);
	}

	const page = path === '/proposed-routes'
		? <ProposedRoutes />
		: <Home />;

	return (
		<>
			<Header />
			<main id="main-content">
				{page}
			</main>
			<FloatingAssistant />
		</>
	);
}

render(<App />, document.getElementById('app')!);

function ChatLoading() {
	return (
		<div class="grid flex-1 place-items-center" role="status">
			<LoaderCircle class="size-5 animate-spin text-muted-foreground" aria-hidden="true" />
			<span class="sr-only">Asistan yükleniyor</span>
		</div>
	);
}
