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
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <router-link to="/chat" class="p-4 bg-dark-hover rounded-lg hover:bg-gray-600 transition-colors">
          <div class="text-2xl mb-2">💬</div>
          <h4 class="font-medium text-white">Ask a Question</h4>
          <p class="text-sm text-gray-400">Query your climate data with AI</p>
        </router-link>
        
        <router-link to="/sources/create" class="p-4 bg-dark-hover rounded-lg hover:bg-gray-600 transition-colors">
          <div class="text-2xl mb-2">➕</div>
          <h4 class="font-medium text-white">Add Source</h4>
          <p class="text-sm text-gray-400">Configure a new data source</p>
        </router-link>
        
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
const health = ref({ llm: 'Checking...' })
const loading = ref(false)

async function loadStats() {
  try {
    // Force refresh by adding timestamp to bypass cache
    const resp = await fetch(`/rag/info?t=${Date.now()}`)
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`)
    }
    stats.value = await resp.json()
    console.log('✅ Stats loaded:', stats.value)
  } catch (e) {
    console.error('Failed to load stats:', e)
    alert(`❌ Error loading stats: ${e.message}`)
  }
}

async function checkHealth() {
  try {
    const resp = await fetch(`/health?t=${Date.now()}`)
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`)
    }
    const data = await resp.json()
    health.value.llm = data.dagster_available ? 'Online' : 'Offline'
  } catch (e) {
    health.value.llm = 'Error'
    console.error('Failed to check health:', e)
  }
}

async function refreshAll() {
  loading.value = true
  try {
    await Promise.all([loadStats(), checkHealth()])
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  refreshAll()
})
</script>
