import { createApp } from 'vue'
import { createPinia } from 'pinia'
import router from './router'
import App from './App.vue'
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
app.mount('#app')
