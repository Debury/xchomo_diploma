import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src')
    }
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      },
      '/rag': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/embeddings': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/sources': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/auth': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  },
  base: '/app/',  // Important: matches FastAPI route
  build: {
    outDir: 'dist',
    emptyOutDir: true
  }
})
