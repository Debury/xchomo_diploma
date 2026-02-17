<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-xl font-semibold text-white">ETL Monitor</h1>
        <p class="text-sm text-gray-500">Live view of running ETL jobs and batch processing</p>
      </div>
      <div class="flex gap-3">
        <button
          @click="retryFailed"
          :disabled="!progress || progress.failed === 0"
          class="px-3 py-1.5 text-sm bg-dark-hover text-gray-300 rounded-md hover:bg-gray-600 transition-colors disabled:opacity-50"
        >
          Retry Failed
        </button>
        <button
          @click="refreshAll"
          :disabled="loading"
          class="px-3 py-1.5 text-sm bg-dark-hover text-gray-300 rounded-md hover:bg-gray-600 transition-colors disabled:opacity-50"
        >
          Refresh
        </button>
      </div>
    </div>

    <!-- Status Cards -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
      <div class="card !p-4">
        <h3 class="text-gray-500 text-xs mb-1">Total Sources</h3>
        <p class="text-xl font-semibold text-white">{{ progress?.total || 0 }}</p>
      </div>
      <div class="card !p-4">
        <h3 class="text-gray-500 text-xs mb-1">Processed</h3>
        <p class="text-xl font-semibold text-green-400">{{ progress?.processed || 0 }}</p>
      </div>
      <div class="card !p-4">
        <h3 class="text-gray-500 text-xs mb-1">Failed</h3>
        <p class="text-xl font-semibold text-red-400">{{ progress?.failed || 0 }}</p>
      </div>
      <div class="card !p-4">
        <h3 class="text-gray-500 text-xs mb-1">Pending</h3>
        <p class="text-xl font-semibold text-yellow-400">{{ progress?.pending || 0 }}</p>
      </div>
    </div>

    <!-- Progress Bar -->
    <div v-if="progress && progress.total > 0" class="card !p-4">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs text-gray-500">
          Phase {{ progress.current_phase ?? '—' }}
          <span v-if="progress.current_source" class="text-blue-400 ml-2">{{ progress.current_source }}</span>
        </span>
        <span class="text-xs text-white font-mono">
          {{ Math.round((progress.processed / progress.total) * 100) }}%
        </span>
      </div>
      <div class="w-full bg-gray-700 rounded-full h-2">
        <div class="flex h-2 rounded-full overflow-hidden">
          <div
            class="bg-green-500 transition-all duration-500"
            :style="{ width: `${(progress.processed / progress.total) * 100}%` }"
          ></div>
          <div
            class="bg-red-500 transition-all duration-500"
            :style="{ width: `${(progress.failed / progress.total) * 100}%` }"
          ></div>
        </div>
      </div>
      <div class="flex justify-between mt-2 text-xs text-gray-500">
        <span>Started: {{ formatTime(progress.started_at) }}</span>
        <span>Updated: {{ formatTime(progress.updated_at) }}</span>
      </div>
    </div>

    <!-- Log Output -->
    <div class="card !p-4">
      <div class="flex items-center justify-between mb-3">
        <h3 class="text-sm font-semibold text-white">ETL Logs</h3>
        <div class="flex gap-2">
          <select v-model="logLines" class="bg-dark-hover border border-dark-border rounded px-3 py-1 text-sm text-white">
            <option :value="50">50 lines</option>
            <option :value="100">100 lines</option>
            <option :value="500">500 lines</option>
          </select>
          <button @click="fetchLogs" class="px-3 py-1.5 text-sm bg-dark-hover text-gray-300 rounded-md hover:bg-gray-600">
            Refresh Logs
          </button>
        </div>
      </div>
      <div class="bg-black rounded-lg p-4 max-h-96 overflow-y-auto font-mono text-xs">
        <pre class="text-green-300 whitespace-pre-wrap">{{ logs || 'No logs available. Start a processing job to see output.' }}</pre>
      </div>
    </div>

    <!-- Per-Phase Breakdown -->
    <div v-if="progress && progress.phases" class="card !p-4">
      <h3 class="text-sm font-semibold text-white mb-3">Per-Phase Breakdown</h3>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        <div v-for="(info, phase) in progress.phases" :key="phase" class="bg-dark-hover rounded-lg p-4">
          <div class="flex items-center justify-between mb-2">
            <span class="text-white font-medium">Phase {{ phase }}</span>
            <span class="text-xs text-gray-400">{{ phaseLabel(phase) }}</span>
          </div>
          <div class="flex gap-3 text-sm">
            <span class="text-green-400">{{ info.completed }} done</span>
            <span class="text-red-400">{{ info.failed }} failed</span>
            <span class="text-gray-400">{{ info.total }} total</span>
          </div>
          <div class="w-full bg-gray-700 rounded-full h-2 mt-2">
            <div
              class="bg-green-500 h-2 rounded-full transition-all"
              :style="{ width: info.total > 0 ? `${(info.completed / info.total) * 100}%` : '0%' }"
            ></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const progress = ref(null)
const logs = ref('')
const loading = ref(false)
const logLines = ref(100)

let pollTimer = null

function formatTime(ts) {
  if (!ts) return '—'
  try {
    return new Date(ts).toLocaleString()
  } catch {
    return ts
  }
}

function phaseLabel(phase) {
  const labels = { '0': 'Metadata', '1': 'Direct download', '2': 'Registration', '3': 'API portals', '4': 'Manual' }
  return labels[String(phase)] || ''
}

async function fetchProgress() {
  try {
    const resp = await fetch('/catalog/progress')
    if (resp.ok) progress.value = await resp.json()
  } catch (e) {
    console.error('Failed to fetch progress:', e)
  }
}

async function fetchLogs() {
  try {
    const resp = await fetch(`/logs/etl?lines=${logLines.value}`)
    if (resp.ok) {
      const data = await resp.json()
      logs.value = data.content || 'No logs available'
    }
  } catch (e) {
    console.error('Failed to fetch logs:', e)
  }
}

async function retryFailed() {
  try {
    const resp = await fetch('/catalog/retry-failed', { method: 'POST' })
    if (resp.ok) {
      console.log('Retry started')
      fetchProgress()
    }
  } catch (e) {
    console.error('Failed to retry:', e)
  }
}

async function refreshAll() {
  loading.value = true
  try {
    await Promise.all([fetchProgress(), fetchLogs()])
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  refreshAll()
  // Poll progress every 5 seconds
  pollTimer = setInterval(fetchProgress, 5000)
})

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
})
</script>
