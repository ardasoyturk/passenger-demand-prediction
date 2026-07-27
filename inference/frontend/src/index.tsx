import { render } from 'preact';

import { Header } from './components/Header';
import { Home } from './pages/Home/index';
import './style.css';

export function App() {
	return (
		<>
			<Header />
			<main id="main-content">
				<Home />
			</main>
		</>
	);
}

render(<App />, document.getElementById('app')!);
