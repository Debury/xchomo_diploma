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
      <div class="flex items-center justify-between mb-2">
        <span class="text-sm text-gray-400">{{ catalogProgress.processed }} / {{ catalogProgress.total }} sources processed</span>
        <span class="text-sm text-white font-mono">{{ Math.round((catalogProgress.processed / catalogProgress.total) * 100) }}%</span>
      </div>
      <div class="w-full bg-gray-700 rounded-full h-3 mb-3">
        <div class="flex h-3 rounded-full overflow-hidden">
          <div
            class="bg-green-500 transition-all duration-500"
            :style="{ width: `${(catalogProgress.processed / catalogProgress.total) * 100}%` }"
          ></div>
          <div
            class="bg-red-500 transition-all duration-500"
            :style="{ width: `${(catalogProgress.failed / catalogProgress.total) * 100}%` }"
          ></div>
        </div>
      </div>
      <div class="flex gap-4 text-xs text-gray-500">
        <span class="text-green-400">{{ catalogProgress.processed }} completed</span>
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

    <!-- Variables List -->
    <div class="card">
      <h3 class="text-lg font-semibold text-white mb-4">Available Variables</h3>
      <div class="flex flex-wrap gap-2">
        <span
          v-for="v in stats.variables"
          :key="v"
          class="px-3 py-1.5 bg-dark-hover rounded-lg text-sm text-gray-300"
        >
          {{ v }}
        </span>
        <span v-if="!stats.variables?.length" class="text-gray-500">No variables found</span>
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
const loading = ref(false)
const runningPhase0 = ref(false)

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
    await Promise.all([loadStats(), checkHealth(), loadCatalogProgress()])
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  refreshAll()
})
</script>
