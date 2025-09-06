import { defineConfig } from 'vite';
import { resolve } from 'path';
import react from '@vitejs/plugin-react';
import dts from 'vite-plugin-dts';
import tailwindcss from 'tailwindcss';

// https://vite.dev/config/
export default defineConfig({
    plugins: [
        react(),
        dts({
            tsconfigPath: resolve(__dirname, 'tsconfig.lib.json'),
        }),
    ],
    css: {
        postcss: {
            plugins: [tailwindcss],
        },
    },
    build: {
        lib: {
            entry: resolve(__dirname, 'lib/main.ts'),
            formats: ['es'],
        },
        rollupOptions: {
            external: [
                'react',
                'react/jsx-runtime',
                'tailwindcss',
                '@tensorflow/tfjs',
                'face-api.js',
            ],
        },
        sourcemap: true,
        copyPublicDir: false,
    },
});
