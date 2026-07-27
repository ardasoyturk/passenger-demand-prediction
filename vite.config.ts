import { defineConfig } from 'vite';
import preact from '@preact/preset-vite';
import tailwindcss from '@tailwindcss/vite';
import path from 'path';

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
		proxy: {
			'/api': {
				target: 'http://localhost:8000',
				changeOrigin: true,
				rewrite: (p) => p.replace(/^\/api/, ''),
			},
		},
	},
});
