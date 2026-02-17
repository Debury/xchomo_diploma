<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-2xl font-bold text-white">ETL Monitor</h1>
        <p class="text-gray-400">Live view of running ETL jobs and batch processing</p>
      </div>
      <div class="flex gap-3">
        <button
          @click="retryFailed"
          :disabled="!progress || progress.failed === 0"
          class="px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors disabled:opacity-50"
        >
          Retry Failed
        </button>
        <button
          @click="refreshAll"
          :disabled="loading"
          class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
        >
          Refresh
        </button>
      </div>
    </div>

    <!-- Status Cards -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
      <div class="card">
        <h3 class="text-gray-400 text-sm mb-2">Total Sources</h3>
        <p class="text-3xl font-bold text-white">{{ progress?.total || 0 }}</p>
      </div>
      <div class="card">
        <h3 class="text-gray-400 text-sm mb-2">Processed</h3>
        <p class="text-3xl font-bold text-green-400">{{ progress?.processed || 0 }}</p>
      </div>
      <div class="card">
        <h3 class="text-gray-400 text-sm mb-2">Failed</h3>
        <p class="text-3xl font-bold text-red-400">{{ progress?.failed || 0 }}</p>
      </div>
      <div class="card">
        <h3 class="text-gray-400 text-sm mb-2">Pending</h3>
        <p class="text-3xl font-bold text-yellow-400">{{ progress?.pending || 0 }}</p>
      </div>
    </div>

    <!-- Progress Bar -->
    <div v-if="progress && progress.total > 0" class="card">
      <div class="flex items-center justify-between mb-3">
        <span class="text-sm text-gray-400">
          Phase {{ progress.current_phase ?? '—' }}
          <span v-if="progress.current_source" class="text-blue-400 ml-2">{{ progress.current_source }}</span>
        </span>
        <span class="text-sm text-white font-mono">
          {{ Math.round((progress.processed / progress.total) * 100) }}%
        </span>
      </div>
      <div class="w-full bg-gray-700 rounded-full h-4">
        <div class="flex h-4 rounded-full overflow-hidden">
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
    <div class="card">
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-lg font-semibold text-white">ETL Logs</h3>
        <div class="flex gap-2">
          <select v-model="logLines" class="bg-dark-hover border border-dark-border rounded px-3 py-1 text-sm text-white">
            <option :value="50">50 lines</option>
            <option :value="100">100 lines</option>
            <option :value="500">500 lines</option>
          </select>
          <button @click="fetchLogs" class="px-3 py-1 bg-gray-600 text-white rounded text-sm hover:bg-gray-500">
            Refresh Logs
          </button>
        </div>
      </div>
      <div class="bg-black rounded-lg p-4 max-h-96 overflow-y-auto font-mono text-xs">
        <pre class="text-green-300 whitespace-pre-wrap">{{ logs || 'No logs available. Start a processing job to see output.' }}</pre>
      </div>
    </div>

    <!-- Dagster Runs -->
    <div class="card">
      <h3 class="text-lg font-semibold text-white mb-4">Recent Dagster Runs</h3>
      <div v-if="runs.length === 0" class="text-gray-500 text-sm">No recent runs found</div>
      <div v-for="run in runs" :key="run.run_id" class="flex items-center justify-between py-3 border-b border-dark-border/50 last:border-0">
        <div>
          <span class="text-white font-medium">{{ run.job_name }}</span>
          <span class="text-gray-500 text-xs ml-3">{{ run.run_id.slice(0, 8) }}</span>
        </div>
        <div class="flex items-center gap-3">
          <span v-if="run.start_time" class="text-gray-500 text-xs">{{ formatTime(run.start_time) }}</span>
          <span class="px-2 py-1 rounded-full text-xs font-medium" :class="runStatusClass(run.status)">
            {{ run.status }}
          </span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const progress = ref(null)
const logs = ref('')
const runs = ref([])
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

function runStatusClass(status) {
  const map = {
    SUCCESS: 'bg-green-500/20 text-green-400',
    FAILURE: 'bg-red-500/20 text-red-400',
    STARTED: 'bg-blue-500/20 text-blue-400',
    QUEUED: 'bg-yellow-500/20 text-yellow-400',
    CANCELED: 'bg-gray-500/20 text-gray-400',
  }
  return map[status] || 'bg-gray-500/20 text-gray-400'
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

async function fetchRuns() {
  try {
    const resp = await fetch('/runs')
    if (resp.ok) runs.value = await resp.json()
  } catch (e) {
    console.error('Failed to fetch runs:', e)
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
    await Promise.all([fetchProgress(), fetchLogs(), fetchRuns()])
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
