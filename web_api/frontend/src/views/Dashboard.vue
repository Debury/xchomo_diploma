<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-xl font-semibold text-white">Dashboard</h1>
        <p class="text-sm text-gray-500">System overview</p>
      </div>
      <button
        @click="refreshAll"
        :disabled="loading"
        class="px-3 py-1.5 text-sm bg-dark-hover text-gray-300 rounded-md hover:bg-gray-600 transition-colors disabled:opacity-50"
      >
        {{ loading ? 'Loading...' : 'Refresh' }}
      </button>
    </div>

    <!-- Stats Cards -->
    <div class="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <div class="card">
        <h3 class="text-xs text-gray-500 uppercase tracking-wider mb-1">Vectors</h3>
        <p class="text-2xl font-semibold text-white tabular-nums">{{ stats.total_embeddings?.toLocaleString() || '0' }}</p>
      </div>
      <div class="card">
        <h3 class="text-xs text-gray-500 uppercase tracking-wider mb-1">Variables</h3>
        <p class="text-2xl font-semibold text-white tabular-nums">{{ stats.variables?.length || '0' }}</p>
      </div>
      <div class="card">
        <h3 class="text-xs text-gray-500 uppercase tracking-wider mb-1">Sources</h3>
        <p class="text-2xl font-semibold text-white tabular-nums">{{ stats.sources?.length || '0' }}</p>
      </div>
      <div class="card">
        <h3 class="text-xs text-gray-500 uppercase tracking-wider mb-1">LLM</h3>
        <p class="text-lg font-medium" :class="health.llmOnline ? 'text-green-400' : 'text-gray-500'">
          {{ health.llmOnline ? 'Connected' : 'Offline' }}
        </p>
      </div>
    </div>

    <!-- Catalog Progress -->
    <div v-if="catalogProgress && catalogProgress.total > 0" class="card">
      <h3 class="text-sm font-medium text-gray-300 mb-3">Catalog Processing</h3>
      <div v-if="catalogProgress.phases && Object.keys(catalogProgress.phases).length" class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-3">
        <div v-for="(info, phase) in catalogProgress.phases" :key="phase" class="bg-dark-bg rounded-lg p-3">
          <div class="flex items-center justify-between mb-1">
            <span class="text-xs text-white font-medium">Phase {{ phase }}</span>
            <span class="text-[10px] text-gray-500">{{ phaseLabel(phase) }}</span>
          </div>
          <div class="flex gap-2 text-xs mb-1.5">
            <span class="text-green-400">{{ info.completed }}/{{ info.total }}</span>
            <span v-if="info.failed > 0" class="text-red-400">{{ info.failed }} failed</span>
          </div>
          <div class="w-full bg-gray-700 rounded-full h-1">
            <div
              class="bg-green-500 h-1 rounded-full transition-all"
              :style="{ width: info.total > 0 ? `${(info.completed / info.total) * 100}%` : '0%' }"
            ></div>
          </div>
        </div>
      </div>
      <div class="flex gap-4 text-xs text-gray-500">
        <span v-if="catalogProgress.current_phase != null" class="text-blue-400">Phase {{ catalogProgress.current_phase }}</span>
        <span>{{ catalogProgress.processed }} processed</span>
        <span v-if="catalogProgress.failed" class="text-red-400">{{ catalogProgress.failed }} failed</span>
        <span>{{ catalogProgress.pending }} pending</span>
      </div>
    </div>

    <!-- System Health -->
    <div class="card">
      <h3 class="text-sm font-medium text-gray-300 mb-3">Services</h3>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div class="flex items-center gap-2">
          <span class="w-2 h-2 rounded-full" :class="health.qdrant ? 'bg-green-500' : 'bg-red-500'"></span>
          <span class="text-sm text-gray-400">Qdrant</span>
        </div>
        <div class="flex items-center gap-2">
          <span class="w-2 h-2 rounded-full" :class="health.dagster ? 'bg-green-500' : 'bg-red-500'"></span>
          <span class="text-sm text-gray-400">Dagster</span>
        </div>
        <div class="flex items-center gap-2">
          <span class="w-2 h-2 rounded-full" :class="health.llmOnline ? 'bg-green-500' : 'bg-yellow-500'"></span>
          <span class="text-sm text-gray-400">LLM</span>
        </div>
        <div class="flex items-center gap-2">
          <span class="w-2 h-2 rounded-full bg-green-500"></span>
          <span class="text-sm text-gray-400">API</span>
        </div>
      </div>
    </div>

    <!-- Hazard Types -->
    <div class="card">
      <h3 class="text-sm font-medium text-gray-300 mb-3">Datasets by Hazard</h3>
      <div class="flex flex-wrap gap-1.5">
        <span
          v-for="h in hazardCounts"
          :key="h.name"
          class="px-2.5 py-1 rounded text-xs font-medium"
          :class="hazardColor(h.name)"
        >
          {{ h.name }} <span class="opacity-60">{{ h.count }}</span>
        </span>
        <span v-if="!hazardCounts.length" class="text-gray-600 text-sm">No catalog data loaded</span>
      </div>
    </div>

    <!-- Quick Actions -->
    <div class="card">
      <h3 class="text-sm font-medium text-gray-300 mb-3">Quick Actions</h3>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
        <router-link to="/chat" class="p-3 bg-dark-bg rounded-lg hover:bg-dark-hover transition-colors">
          <h4 class="text-sm font-medium text-white mb-0.5">Ask a Question</h4>
          <p class="text-xs text-gray-500">Query climate data with AI</p>
        </router-link>
        <router-link to="/catalog" class="p-3 bg-dark-bg rounded-lg hover:bg-dark-hover transition-colors">
          <h4 class="text-sm font-medium text-white mb-0.5">View Catalog</h4>
          <p class="text-xs text-gray-500">Browse 233 sources</p>
        </router-link>
        <button @click="runPhase0" :disabled="runningPhase0" class="p-3 bg-dark-bg rounded-lg hover:bg-dark-hover transition-colors text-left disabled:opacity-50">
          <h4 class="text-sm font-medium text-white mb-0.5">{{ runningPhase0 ? 'Processing...' : 'Run Phase 0' }}</h4>
          <p class="text-xs text-gray-500">Embed catalog metadata</p>
        </button>
        <a href="/docs" target="_blank" class="p-3 bg-dark-bg rounded-lg hover:bg-dark-hover transition-colors">
          <h4 class="text-sm font-medium text-white mb-0.5">API Docs</h4>
          <p class="text-xs text-gray-500">REST API reference</p>
        </a>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const stats = ref({})
const health = ref({ llm: 'Checking...', qdrant: false, dagster: false, llmOnline: false })
const catalogProgress = ref(null)
const hazardCounts = ref([])
const loading = ref(false)
const runningPhase0 = ref(false)

const HAZARD_COLORS = {
  'Drought': 'bg-amber-500/15 text-amber-400',
  'Temperature': 'bg-red-500/15 text-red-400',
  'Precipitation': 'bg-blue-500/15 text-blue-400',
  'Sea Level Rise': 'bg-cyan-500/15 text-cyan-400',
  'Flood': 'bg-indigo-500/15 text-indigo-400',
  'Wind': 'bg-teal-500/15 text-teal-400',
}

function hazardColor(name) {
  return HAZARD_COLORS[name] || 'bg-gray-500/15 text-gray-400'
}

function phaseLabel(phase) {
  const labels = { '0': 'Metadata', '1': 'Direct download', '2': 'Registration', '3': 'API portals', '4': 'Manual' }
  return labels[String(phase)] || ''
}

async function loadStats() {
  try {
    const resp = await fetch(`/rag/info?t=${Date.now()}`)
    if (resp.ok) {
      stats.value = await resp.json()
      health.value.qdrant = true
    }
  } catch (e) {
    health.value.qdrant = false
  }
}

async function checkHealth() {
  try {
    const resp = await fetch(`/health?t=${Date.now()}`)
    if (resp.ok) {
      const data = await resp.json()
      health.value.dagster = data.dagster_available
      health.value.llmOnline = data.dagster_available
    }
  } catch (e) {
    health.value.dagster = false
  }
}

async function loadCatalogProgress() {
  try {
    const resp = await fetch('/catalog/progress')
    if (resp.ok) catalogProgress.value = await resp.json()
  } catch (e) {}
}

async function loadCatalogHazards() {
  try {
    const resp = await fetch('/catalog')
    if (resp.ok) {
      const entries = await resp.json()
      const counts = {}
      for (const entry of entries) {
        const h = entry.hazard || 'Other'
        counts[h] = (counts[h] || 0) + 1
      }
      hazardCounts.value = Object.entries(counts)
        .map(([name, count]) => ({ name, count }))
        .sort((a, b) => b.count - a.count)
    }
  } catch (e) {}
}

async function runPhase0() {
  runningPhase0.value = true
  try {
    const resp = await fetch('/catalog/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ phases: [0] }),
    })
    if (resp.ok) {
      const poll = setInterval(async () => {
        await loadCatalogProgress()
        if (catalogProgress.value && catalogProgress.value.pending === 0) {
          clearInterval(poll)
          runningPhase0.value = false
          refreshAll()
        }
      }, 3000)
    }
  } catch (e) {
    runningPhase0.value = false
  }
}

async function refreshAll() {
  loading.value = true
  try {
    await Promise.all([loadStats(), checkHealth(), loadCatalogProgress(), loadCatalogHazards()])
  } finally {
    loading.value = false
  }
}

onMounted(refreshAll)
</script>
