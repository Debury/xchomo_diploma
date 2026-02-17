<template>
  <div class="space-y-6">
    <!-- Header with Refresh -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-2xl font-bold text-white">Dashboard</h1>
        <p class="text-gray-400">Overview of your climate data system</p>
      </div>
      <button
        @click="refreshAll"
        :disabled="loading"
        class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
      >
        <span v-if="loading">⏳</span>
        <span v-else>🔄</span>
        {{ loading ? 'Loading...' : 'Refresh' }}
      </button>
    </div>

    <!-- Stats Cards -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      <div class="card">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-gray-400 text-sm font-medium">Total Embeddings</h3>
          <div class="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center">
            <span class="text-blue-400">📊</span>
          </div>
        </div>
        <p class="text-3xl font-bold text-blue-400">{{ stats.total_embeddings?.toLocaleString() || '—' }}</p>
      </div>

      <div class="card">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-gray-400 text-sm font-medium">Variables</h3>
          <div class="w-10 h-10 rounded-lg bg-purple-500/20 flex items-center justify-center">
            <span class="text-purple-400">🔢</span>
          </div>
        </div>
        <p class="text-3xl font-bold text-purple-400">{{ stats.variables?.length || '—' }}</p>
      </div>

      <div class="card">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-gray-400 text-sm font-medium">Sources</h3>
          <div class="w-10 h-10 rounded-lg bg-green-500/20 flex items-center justify-center">
            <span class="text-green-400">📁</span>
          </div>
        </div>
        <p class="text-3xl font-bold text-green-400">{{ stats.sources?.length || '—' }}</p>
      </div>

      <div class="card">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-gray-400 text-sm font-medium">LLM Status</h3>
          <div class="w-10 h-10 rounded-lg bg-yellow-500/20 flex items-center justify-center">
            <span class="text-yellow-400">🤖</span>
          </div>
        </div>
        <p class="text-lg font-bold text-yellow-400">{{ health.llm || 'Checking...' }}</p>
      </div>
    </div>

    <!-- Catalog Progress -->
    <div v-if="catalogProgress && catalogProgress.total > 0" class="card">
      <h3 class="text-lg font-semibold text-white mb-4">Catalog Processing</h3>
      <!-- Per-phase breakdown -->
      <div v-if="catalogProgress.phases && Object.keys(catalogProgress.phases).length" class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
        <div v-for="(info, phase) in catalogProgress.phases" :key="phase" class="bg-dark-hover rounded-lg p-3">
          <div class="flex items-center justify-between mb-1">
            <span class="text-sm text-white font-medium">Phase {{ phase }}</span>
            <span class="text-xs text-gray-500">{{ phaseLabel(phase) }}</span>
          </div>
          <div class="flex gap-2 text-xs mb-2">
            <span class="text-green-400">{{ info.completed }}/{{ info.total }}</span>
            <span v-if="info.failed > 0" class="text-red-400">{{ info.failed }} failed</span>
          </div>
          <div class="w-full bg-gray-700 rounded-full h-1.5">
            <div
              class="bg-green-500 h-1.5 rounded-full transition-all"
              :style="{ width: info.total > 0 ? `${(info.completed / info.total) * 100}%` : '0%' }"
            ></div>
          </div>
        </div>
      </div>
      <!-- Overall summary -->
      <div class="flex gap-4 text-xs text-gray-500">
        <span v-if="catalogProgress.current_phase != null" class="text-blue-400">Current: Phase {{ catalogProgress.current_phase }}</span>
        <span class="text-green-400">{{ catalogProgress.processed }} processed</span>
        <span class="text-red-400">{{ catalogProgress.failed }} failed</span>
        <span class="text-gray-400">{{ catalogProgress.pending }} pending</span>
      </div>
    </div>

    <!-- System Health -->
    <div class="card">
      <h3 class="text-lg font-semibold text-white mb-4">System Health</h3>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div class="flex items-center gap-3">
          <span class="w-3 h-3 rounded-full" :class="health.qdrant ? 'bg-green-500' : 'bg-red-500'"></span>
          <span class="text-gray-300 text-sm">Qdrant</span>
        </div>
        <div class="flex items-center gap-3">
          <span class="w-3 h-3 rounded-full" :class="health.dagster ? 'bg-green-500' : 'bg-red-500'"></span>
          <span class="text-gray-300 text-sm">Dagster</span>
        </div>
        <div class="flex items-center gap-3">
          <span class="w-3 h-3 rounded-full" :class="health.llmOnline ? 'bg-green-500' : 'bg-yellow-500'"></span>
          <span class="text-gray-300 text-sm">LLM</span>
        </div>
        <div class="flex items-center gap-3">
          <span class="w-3 h-3 rounded-full bg-green-500"></span>
          <span class="text-gray-300 text-sm">API</span>
        </div>
      </div>
    </div>

    <!-- Datasets by Hazard Type -->
    <div class="card">
      <h3 class="text-lg font-semibold text-white mb-4">Datasets by Hazard Type</h3>
      <div class="flex flex-wrap gap-2">
        <span
          v-for="h in hazardCounts"
          :key="h.name"
          class="px-3 py-1.5 rounded-lg text-sm font-medium"
          :class="hazardColor(h.name)"
        >
          {{ h.name }} <span class="opacity-70">{{ h.count }}</span>
        </span>
        <span v-if="!hazardCounts.length" class="text-gray-500">No catalog data loaded</span>
      </div>
    </div>

    <!-- Quick Actions -->
    <div class="card">
      <h3 class="text-lg font-semibold text-white mb-4">Quick Actions</h3>
      <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
        <router-link to="/chat" class="p-4 bg-dark-hover rounded-lg hover:bg-gray-600 transition-colors">
          <div class="text-2xl mb-2">💬</div>
          <h4 class="font-medium text-white">Ask a Question</h4>
          <p class="text-sm text-gray-400">Query your climate data with AI</p>
        </router-link>

        <router-link to="/catalog" class="p-4 bg-dark-hover rounded-lg hover:bg-gray-600 transition-colors">
          <div class="text-2xl mb-2">📋</div>
          <h4 class="font-medium text-white">View Catalog</h4>
          <p class="text-sm text-gray-400">Browse 234 climate data sources</p>
        </router-link>

        <button @click="runPhase0" :disabled="runningPhase0" class="p-4 bg-dark-hover rounded-lg hover:bg-gray-600 transition-colors text-left disabled:opacity-50">
          <div class="text-2xl mb-2">⚡</div>
          <h4 class="font-medium text-white">{{ runningPhase0 ? 'Processing...' : 'Run Phase 0' }}</h4>
          <p class="text-sm text-gray-400">Embed all catalog metadata</p>
        </button>

        <a href="/docs" target="_blank" class="p-4 bg-dark-hover rounded-lg hover:bg-gray-600 transition-colors">
          <div class="text-2xl mb-2">📚</div>
          <h4 class="font-medium text-white">API Documentation</h4>
          <p class="text-sm text-gray-400">Explore the REST API</p>
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
  'Drought': 'bg-amber-500/20 text-amber-400',
  'Temperature': 'bg-red-500/20 text-red-400',
  'Precipitation': 'bg-blue-500/20 text-blue-400',
  'Sea Level Rise': 'bg-cyan-500/20 text-cyan-400',
  'Flood': 'bg-indigo-500/20 text-indigo-400',
  'Wind': 'bg-teal-500/20 text-teal-400',
}

function hazardColor(name) {
  return HAZARD_COLORS[name] || 'bg-gray-500/20 text-gray-300'
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
    console.error('Failed to load stats:', e)
    health.value.qdrant = false
  }
}

async function checkHealth() {
  try {
    const resp = await fetch(`/health?t=${Date.now()}`)
    if (resp.ok) {
      const data = await resp.json()
      health.value.dagster = data.dagster_available
      health.value.llm = data.dagster_available ? 'Online' : 'Offline'
      health.value.llmOnline = data.dagster_available
    }
  } catch (e) {
    health.value.llm = 'Error'
    health.value.dagster = false
  }
}

async function loadCatalogProgress() {
  try {
    const resp = await fetch('/catalog/progress')
    if (resp.ok) catalogProgress.value = await resp.json()
  } catch (e) {
    // Catalog endpoints may not be available yet
  }
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
  } catch (e) {
    // Catalog endpoint may not be available
  }
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
      // Start polling progress
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
    console.error('Failed to run Phase 0:', e)
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

onMounted(() => {
  refreshAll()
})
</script>
