import { createApp } from 'vue'
import { createPinia } from 'pinia'
import router from './router'
import App from './App.vue'
import { useAuthStore } from './stores/auth'
import './style.css'

const app = createApp(App)

// Top-level error handler for anything the per-view ErrorBoundary misses
// (async work that isn't tied to a component render, for example).
// We just log — the boundary component handles anything render-scoped.
app.config.errorHandler = (err, _instance, info) => {
  console.error('[Vue errorHandler]', err, info)
}

// Surface uncaught promise rejections to the console so demo-day issues are
// visible without digging into sources.
if (typeof window !== 'undefined') {
  window.addEventListener('unhandledrejection', (event) => {
    console.error('[unhandledrejection]', event.reason)
  })
}

app.use(createPinia())
app.use(router)

// Verify any cached token against the server BEFORE mounting — prevents
// the protected UI from flashing while apiFetch catches a 401 on the first
// dashboard request. Costs one /auth/verify round-trip on boot; no call at
// all when the user isn't logged in.
const authStore = useAuthStore()
authStore.initialize().finally(() => {
  app.mount('#app')
})
