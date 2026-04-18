<template>
  <div class="space-y-6">
    <PageHeader title="ETL Monitor" subtitle="Live view of running ETL jobs and batch processing">
      <template #actions>
        <button @click="retryFailed" :disabled="!progress || progress.failed === 0" class="btn-ghost disabled:opacity-50">
          Retry Failed
        </button>
        <button @click="refreshAll" :disabled="loading" class="btn-secondary disabled:opacity-50">
          Refresh
        </button>
      </template>
    </PageHeader>

    <!-- Status Cards -->
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
      <StatCard label="Total Sources" :value="String(progress?.total || 0)" :loading="loading" />
      <StatCard label="Processed" :value="String(progress?.processed || 0)" :loading="loading" />
      <StatCard label="Metadata Only" :value="String(progress?.metadata_only || 0)" :loading="loading" />
      <StatCard label="Failed" :value="String(progress?.failed || 0)" :loading="loading" />
    </div>

    <!-- Idle banner — when no batch is running, make that obvious so the
         "X pending" count from historical catalog rows doesn't read as live. -->
    <div v-if="progress && !progress.thread_alive" class="card !py-3 flex items-center gap-3 text-sm">
      <span class="inline-block w-2 h-2 rounded-full bg-mendelu-gray-dark"></span>
      <span class="text-mendelu-gray-dark">
        No ETL batch is currently running.
        <span v-if="progress.thread_crashed" class="text-mendelu-alert font-medium ml-1">
          Last run crashed — check the logs below.
        </span>
      </span>
    </div>

    <!-- Progress Bar — only shown while a batch is actually running. -->
    <div v-if="progress && progress.thread_alive && progress.total > 0" class="card">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs text-mendelu-gray-dark">
          <span v-if="progress.current_source" class="text-mendelu-green font-medium">Processing {{ progress.current_source }}</span>
          <span v-else>Working…</span>
        </span>
        <span class="text-xs text-mendelu-black font-mono tabular-nums">
          {{ pctOf(progress.processed + (progress.metadata_only || 0)) }}%
        </span>
      </div>
      <div class="w-full bg-mendelu-gray-semi rounded-full h-2">
        <div class="flex h-2 rounded-full overflow-hidden">
          <div class="bg-mendelu-success transition-all duration-500" :style="{ width: `${pctOf(progress.processed)}%` }"></div>
          <div v-if="progress.metadata_only" class="bg-mendelu-green/40 transition-all duration-500" :style="{ width: `${pctOf(progress.metadata_only)}%` }"></div>
          <div class="bg-mendelu-alert transition-all duration-500" :style="{ width: `${pctOf(progress.failed)}%` }"></div>
        </div>
      </div>
      <div class="flex justify-between mt-2 text-xs text-mendelu-gray-dark">
        <span>Started: {{ formatTime(progress.started_at) }}</span>
        <span>Updated: {{ formatTime(progress.updated_at) }}</span>
      </div>
    </div>

    <!-- Log Output -->
    <div class="card">
      <div class="flex items-center justify-between mb-3">
        <h3 class="text-sm font-medium text-mendelu-black">ETL Logs</h3>
        <div class="flex gap-2">
          <select v-model="logLines" class="input-field !w-auto !py-1.5 text-xs">
            <option :value="50">50 lines</option>
            <option :value="100">100 lines</option>
            <option :value="500">500 lines</option>
          </select>
          <button @click="copyLogs" class="btn-ghost text-xs">Copy</button>
          <button @click="fetchLogs" class="btn-ghost text-xs">Refresh</button>
        </div>
      </div>
      <div ref="logContainer" class="bg-mendelu-black rounded-lg p-4 max-h-96 overflow-y-auto font-mono text-xs">
        <pre class="text-mendelu-green whitespace-pre-wrap">{{ logs || 'No logs available. Start a processing job to see output.' }}</pre>
      </div>
    </div>

    <!-- Per-Phase Breakdown — only shown if an *active* catalog batch is
         running. Phases are a catalog-specific concept; they're meaningless
         for user-added sources, so don't clutter the idle view with them. -->
    <div v-if="progress && progress.thread_alive && progress.phases && Object.keys(progress.phases).length" class="card">
      <h3 class="text-sm font-medium text-mendelu-black mb-3">Catalog batch — per-phase breakdown</h3>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        <div v-for="(info, phase) in progress.phases" :key="phase" class="bg-mendelu-gray-light rounded-lg p-4">
          <div class="flex items-center justify-between mb-2">
            <span class="text-mendelu-black font-medium text-sm">Phase {{ phase }}</span>
            <span class="text-xs text-mendelu-gray-dark">{{ phaseLabel(phase) }}</span>
          </div>
          <div class="flex gap-3 text-xs">
            <span class="text-mendelu-success">{{ info.completed }} done</span>
            <span class="text-mendelu-alert">{{ info.failed }} failed</span>
            <span class="text-mendelu-gray-dark">{{ info.total }} total</span>
          </div>
          <div class="w-full bg-mendelu-gray-semi rounded-full h-1.5 mt-2">
            <div
              class="bg-mendelu-green h-1.5 rounded-full transition-all duration-300"
              :style="{ width: info.total > 0 ? `${(info.completed / info.total) * 100}%` : '0%' }"
            ></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import PageHeader from '../components/PageHeader.vue'
import StatCard from '../components/StatCard.vue'
import { apiFetch } from '../api'

const progress = ref(null)
const logs = ref('')
const loading = ref(false)
const logLines = ref(100)
const logContainer = ref(null)
let pollTimer: ReturnType<typeof setInterval> | null = null

function stopPolling(): void {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

function startPolling(): void {
  stopPolling()
  pollTimer = setInterval(fetchProgress, 5000)
}

function onVisibilityChange(): void {
  if (document.hidden) {
    stopPolling()
  } else {
    fetchProgress()
    startPolling()
  }
}

function pctOf(value: number): number {
  const total = progress.value?.total || 1
  return Math.min(100, Math.round((value / total) * 100))
}

function formatTime(ts) {
  if (!ts) return '---'
  try { return new Date(ts).toLocaleString() } catch { return ts }
}

function phaseLabel(phase) {
  const labels = { '0': 'Metadata', '1': 'Direct download', '2': 'Registration', '3': 'API portals', '4': 'Manual' }
  return labels[String(phase)] || ''
}

async function fetchProgress() {
  try {
    const resp = await apiFetch('/catalog/progress')
    if (resp.ok) progress.value = await resp.json()
  } catch (e) { console.error('Failed to fetch progress:', e) }
}

async function fetchLogs() {
  try {
    const resp = await apiFetch(`/logs/etl?lines=${logLines.value}`)
    if (resp.ok) {
      const data = await resp.json()
      logs.value = data.content || 'No logs available'
      await nextTick()
      if (logContainer.value) logContainer.value.scrollTop = logContainer.value.scrollHeight
    }
  } catch (e) { console.error('Failed to fetch logs:', e) }
}

function copyLogs() {
  if (logs.value) {
    navigator.clipboard.writeText(logs.value).catch(() => {})
  }
}

async function retryFailed() {
  try {
    const resp = await apiFetch('/catalog/retry-failed', { method: 'POST' })
    if (resp.ok) fetchProgress()
  } catch (e) { console.error('Failed to retry:', e) }
}

async function refreshAll() {
  loading.value = true
  try { await Promise.all([fetchProgress(), fetchLogs()]) }
  finally { loading.value = false }
}

onMounted(() => {
  refreshAll()
  startPolling()
  document.addEventListener('visibilitychange', onVisibilityChange)
})
onUnmounted(() => {
  stopPolling()
  document.removeEventListener('visibilitychange', onVisibilityChange)
})
</script>
