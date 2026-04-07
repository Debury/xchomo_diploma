<template>
  <div>
    <div class="flex items-center justify-between mb-4">
      <h3 class="text-sm font-semibold text-mendelu-black flex items-center gap-2">
        <svg class="w-4 h-4 text-mendelu-green" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
        Live Activity
        <span v-if="events.length" class="inline-flex items-center justify-center w-5 h-5 rounded-full text-[9px] font-bold bg-mendelu-green/10 text-mendelu-green" style="font-family: var(--font-mono);">{{ events.length }}</span>
      </h3>
      <div class="flex items-center gap-1.5">
        <div class="w-1.5 h-1.5 rounded-full bg-mendelu-green animate-pulse"></div>
        <span class="text-[10px] font-medium text-mendelu-gray-dark/60" style="font-family: var(--font-mono);">LIVE</span>
      </div>
    </div>

    <!-- Timeline -->
    <div class="relative">
      <!-- Vertical line -->
      <div class="absolute left-[7px] top-2 bottom-2 w-px bg-gradient-to-b from-mendelu-gray-semi via-mendelu-gray-semi/40 to-transparent"></div>

      <TransitionGroup name="feed">
        <div
          v-for="(event, idx) in events"
          :key="event.id"
          class="relative flex items-start gap-3 py-2 group"
          :style="{ animationDelay: `${idx * 30}ms` }"
        >
          <!-- Timeline dot -->
          <div class="relative z-10 mt-1 flex-shrink-0">
            <span
              class="block w-[15px] h-[15px] rounded-full border-2 transition-all duration-200"
              :class="dotClass(event.type)"
            ></span>
          </div>

          <!-- Content -->
          <div class="flex-1 min-w-0 pb-1">
            <p class="text-[13px] text-mendelu-black leading-snug group-hover:text-mendelu-green transition-colors duration-200">{{ event.message }}</p>
            <p class="text-[10px] text-mendelu-gray-dark/50 mt-1 font-medium" style="font-family: var(--font-mono);">{{ event.time }}</p>
          </div>

          <!-- Type badge -->
          <span
            class="flex-shrink-0 mt-0.5 px-1.5 py-0.5 rounded text-[9px] font-bold uppercase tracking-wider opacity-0 group-hover:opacity-100 transition-opacity duration-200"
            :class="badgeClass(event.type)"
            style="font-family: var(--font-mono);"
          >{{ event.type }}</span>
        </div>
      </TransitionGroup>

      <p v-if="!events.length" class="text-sm text-mendelu-gray-dark/50 py-6 text-center flex flex-col items-center gap-2">
        <svg class="w-8 h-8 text-mendelu-gray-semi" fill="none" stroke="currentColor" stroke-width="1" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        Waiting for events...
      </p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const events = ref([])
let intervalId = null
let eventCounter = 0
let prevProgress = null
let prevHealth = null

function dotClass(type) {
  return {
    'bg-mendelu-success/20 border-mendelu-success': type === 'success',
    'bg-mendelu-alert/20 border-mendelu-alert': type === 'error',
    'bg-mendelu-green/20 border-mendelu-green': type === 'info',
    'bg-mendelu-gray-semi border-mendelu-gray-dark/30': type === 'neutral',
  }
}

function badgeClass(type) {
  return {
    'bg-mendelu-success/10 text-mendelu-success': type === 'success',
    'bg-mendelu-alert/10 text-mendelu-alert': type === 'error',
    'bg-mendelu-green/10 text-mendelu-green': type === 'info',
    'bg-mendelu-gray-light text-mendelu-gray-dark': type === 'neutral',
  }
}

function addEvent(type, message) {
  const now = new Date()
  const time = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  events.value.unshift({ id: ++eventCounter, type, message, time })
  if (events.value.length > 10) events.value.pop()
}

async function poll() {
  try {
    const [healthResp, progressResp] = await Promise.all([
      fetch(`/health?t=${Date.now()}`).catch(() => null),
      fetch('/catalog/progress').catch(() => null),
    ])

    if (healthResp?.ok) {
      const h = await healthResp.json()
      if (prevHealth !== null) {
        if (h.dagster_available !== prevHealth.dagster_available) {
          addEvent(h.dagster_available ? 'success' : 'error', h.dagster_available ? 'Dagster connected' : 'Dagster disconnected')
        }
        if ((h.status === 'healthy') !== (prevHealth.status === 'healthy')) {
          addEvent(h.status === 'healthy' ? 'success' : 'error', h.status === 'healthy' ? 'Qdrant online' : 'Qdrant offline')
        }
      } else {
        if (h.status === 'healthy') addEvent('success', 'Qdrant online')
        if (h.dagster_available) addEvent('success', 'Dagster connected')
        addEvent('info', 'System health check OK')
      }
      prevHealth = h
    }

    if (progressResp?.ok) {
      const p = await progressResp.json()
      if (prevProgress !== null) {
        const newProcessed = (p.processed || 0) - (prevProgress.processed || 0)
        if (newProcessed > 0) addEvent('success', `Processed ${newProcessed} catalog entries`)
        if ((p.failed || 0) > (prevProgress.failed || 0)) addEvent('error', `${p.failed - prevProgress.failed} entries failed`)
        if (p.thread_alive && !prevProgress.thread_alive) addEvent('info', `Processing started (Phase ${p.current_phase})`)
        if (!p.thread_alive && prevProgress.thread_alive) addEvent('success', 'Processing completed')
      }
      prevProgress = p
    }
  } catch (e) {}
}

onMounted(() => {
  poll()
  intervalId = setInterval(poll, 15000)
})

onUnmounted(() => {
  if (intervalId) clearInterval(intervalId)
})
</script>

<style scoped>
.feed-enter-active {
  transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}
.feed-enter-from {
  opacity: 0;
  transform: translateX(-8px);
}
.feed-leave-active {
  transition: all 0.2s ease-in;
}
.feed-leave-to {
  opacity: 0;
  transform: translateX(8px);
}
</style>
