<template>
  <div>
    <h3 class="text-sm font-medium text-mendelu-black mb-3">Activity</h3>
    <div class="space-y-0">
      <TransitionGroup name="feed">
        <div
          v-for="event in events"
          :key="event.id"
          class="flex items-start gap-3 py-2.5 border-b border-mendelu-gray-semi/40 last:border-0"
        >
          <span
            class="mt-1.5 w-2 h-2 rounded-full flex-shrink-0"
            :class="{
              'bg-mendelu-success': event.type === 'success',
              'bg-mendelu-alert': event.type === 'error',
              'bg-mendelu-green': event.type === 'info',
              'bg-mendelu-gray-semi': event.type === 'neutral',
            }"
          ></span>
          <div class="min-w-0 flex-1">
            <p class="text-sm text-mendelu-black leading-snug">{{ event.message }}</p>
            <p class="text-[10px] text-mendelu-gray-dark mt-0.5">{{ event.time }}</p>
          </div>
        </div>
      </TransitionGroup>
      <p v-if="!events.length" class="text-sm text-mendelu-gray-dark py-4 text-center">Waiting for events...</p>
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
        if (h.qdrant !== prevHealth.qdrant) {
          addEvent(h.qdrant ? 'success' : 'error', h.qdrant ? 'Qdrant online' : 'Qdrant offline')
        }
      } else {
        if (h.qdrant) addEvent('success', 'Qdrant online')
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
  transition: all 0.3s ease-out;
}
.feed-enter-from {
  opacity: 0;
  transform: translateY(-8px);
}
.feed-leave-active {
  transition: all 0.2s ease-in;
}
.feed-leave-to {
  opacity: 0;
}
</style>
