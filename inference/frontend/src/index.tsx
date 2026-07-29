import { render } from 'preact';

import { Header } from './components/Header';
import { Home } from './pages/Home/index';
import { ProposedRoutes } from './pages/ProposedRoutes/index';
import './style.css';

export function App() {
	const page = window.location.pathname === '/proposed-routes'
		? <ProposedRoutes />
		: <Home />;

	return (
		<>
			<Header />
			<main id="main-content">
				{page}
			</main>
		</>
	);
}

render(<App />, document.getElementById('app')!);
