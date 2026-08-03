import { defineConfig } from 'vite';
import preact from '@preact/preset-vite';
import tailwindcss from '@tailwindcss/vite';
import path from 'path';

const proxy = {
	'/api': {
		target: 'http://localhost:8000',
		changeOrigin: true,
		rewrite: (p: string) => p.replace(/^\/api/, ''),
	},
};

// https://vitejs.dev/config/
export default defineConfig({
	root: 'inference/frontend',
	publicDir: path.resolve(__dirname, 'inference/frontend/public'),
	build: {
		outDir: path.resolve(__dirname, 'inference/frontend/dist'),
	},
	plugins: [
		tailwindcss(),
		preact(),
	],
	server: {
		proxy,
	},
	// `vite preview` does not use `server.proxy`; keep preview behaviour equal
	// to local development so it cannot regress to a cross-origin request.
	preview: {
		proxy,
	},
});
