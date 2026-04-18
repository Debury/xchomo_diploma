<template>
  <template v-if="error">
    <div class="min-h-[60vh] flex items-center justify-center p-6">
      <div class="card max-w-xl w-full">
        <div class="flex items-start gap-3 mb-4">
          <div class="w-10 h-10 rounded-lg bg-mendelu-alert/10 flex items-center justify-center flex-shrink-0">
            <svg class="w-5 h-5 text-mendelu-alert" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01M5 19h14a2 2 0 001.84-2.75L13.74 4a2 2 0 00-3.48 0L3.16 16.25A2 2 0 005 19z" />
            </svg>
          </div>
          <div class="flex-1 min-w-0">
            <h2 class="text-base font-semibold text-mendelu-black">Something went wrong on this page</h2>
            <p class="text-xs text-mendelu-gray-dark mt-1">
              The UI caught an error while rendering this view. You can try reloading this page or
              return to the dashboard.
            </p>
          </div>
        </div>

        <details class="mb-4">
          <summary class="text-xs text-mendelu-gray-dark cursor-pointer hover:text-mendelu-black">
            Show technical details
          </summary>
          <pre class="mt-2 text-[11px] font-mono bg-mendelu-gray-light text-mendelu-black p-3 rounded-lg overflow-x-auto whitespace-pre-wrap break-all">{{ error.message }}</pre>
          <pre v-if="info" class="mt-2 text-[10px] font-mono bg-mendelu-gray-light text-mendelu-gray-dark p-3 rounded-lg overflow-x-auto whitespace-pre-wrap break-all">{{ info }}</pre>
        </details>

        <div class="flex gap-3">
          <button type="button" @click="retry" class="btn-primary flex-1">Retry</button>
          <button type="button" @click="goHome" class="btn-secondary flex-1">Back to Dashboard</button>
        </div>
      </div>
    </div>
  </template>
  <slot v-else />
</template>

<script setup lang="ts">
import { ref, onErrorCaptured } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const error = ref(null)
const info = ref('')

onErrorCaptured((err, _instance, errorInfo) => {
  console.error('[ErrorBoundary]', err, errorInfo)
  error.value = err
  info.value = errorInfo || ''
  return false  // stop propagation
})

function retry() {
  error.value = null
  info.value = ''
}

function goHome() {
  error.value = null
  info.value = ''
  router.push('/').catch(() => {
    window.location.assign('/app/')
  })
}
</script>
