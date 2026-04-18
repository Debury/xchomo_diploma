import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath, URL } from 'node:url'

// Routes served by FastAPI that must be proxied when running `vite dev`
// (port 5173) back to the API running on :8000.
const BACKEND_ROUTES = [
  '/api',
  '/rag',
  '/embeddings',
  '/sources',
  '/auth',
  '/health',
  '/catalog',
  '/schedules',
  '/settings',
  '/logs',
  '/admin',
  '/qdrant',
  '/docs',
] as const

const proxy = Object.fromEntries(
  BACKEND_ROUTES.map(route => [
    route,
    {
      target: 'http://localhost:8000',
      changeOrigin: true,
      ...(route === '/api'
        ? { rewrite: (p: string) => p.replace(/^\/api/, '') }
        : {}),
    },
  ]),
)

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    port: 5173,
    proxy,
  },
  base: '/app/',  // must match the FastAPI route prefix
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    target: 'es2020',
    rollupOptions: {
      output: {
        // Keep Vue / Pinia / router in a separate vendor chunk so the main
        // app chunk invalidates less often on code-only changes.
        manualChunks: {
          vendor: ['vue', 'vue-router', 'pinia'],
        },
      },
    },
  },
})
