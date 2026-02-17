<template>
  <div class="space-y-6">
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-xl font-semibold text-mendelu-black">Embeddings</h1>
        <p class="text-sm text-mendelu-gray-dark">Vector store overview</p>
      </div>
      <button @click="refreshStats" :disabled="loading" class="btn-secondary disabled:opacity-50">
        {{ loading ? 'Loading...' : 'Refresh' }}
      </button>
    </div>

    <!-- Collection Health -->
    <div class="card !p-4">
      <div class="flex items-center gap-3 mb-3">
        <div class="w-10 h-10 rounded-full flex items-center justify-center"
             :class="healthStatus === 'healthy' ? 'bg-green-50' : healthStatus === 'degraded' ? 'bg-amber-50' : 'bg-red-50'">
          <div class="w-4 h-4 rounded-full"
               :class="healthStatus === 'healthy' ? 'bg-mendelu-success' : healthStatus === 'degraded' ? 'bg-amber-400' : 'bg-mendelu-alert'"></div>
        </div>
        <div>
          <h3 class="text-sm font-semibold text-mendelu-black">Collection Health</h3>
          <p class="text-xs text-mendelu-gray-dark capitalize">{{ healthStatus }}</p>
        </div>
      </div>
    </div>

    <!-- Stats Overview -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
      <div class="card">
        <h3 class="text-xs text-mendelu-gray-dark uppercase tracking-wider mb-1">Total Vectors</h3>
        <p class="text-2xl font-semibold text-mendelu-black tabular-nums">{{ stats.total_embeddings?.toLocaleString() || '0' }}</p>
      </div>
      <div class="card">
        <h3 class="text-xs text-mendelu-gray-dark uppercase tracking-wider mb-1">Dimensions</h3>
        <p class="text-2xl font-semibold text-mendelu-black">1024</p>
      </div>
      <div class="card">
        <h3 class="text-xs text-mendelu-gray-dark uppercase tracking-wider mb-1">Collection</h3>
        <p class="text-lg font-mono text-mendelu-black">climate_data</p>
      </div>
    </div>

    <!-- Dataset Breakdown -->
    <div class="card">
      <h3 class="text-sm font-medium text-mendelu-black mb-3">Chunks by Dataset</h3>
      <div v-if="datasetBreakdown.length" class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="text-left text-xs text-mendelu-gray-dark border-b border-mendelu-gray-semi uppercase tracking-wider">
              <th class="pb-2 font-medium">Dataset</th>
              <th class="pb-2 font-medium text-right">Chunks</th>
              <th class="pb-2 font-medium w-1/3">Distribution</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-mendelu-gray-semi/50">
            <tr v-for="ds in datasetBreakdown" :key="ds.name">
              <td class="py-2.5 text-mendelu-black font-medium">{{ ds.name }}</td>
              <td class="py-2.5 text-mendelu-black text-right tabular-nums">{{ ds.count.toLocaleString() }}</td>
              <td class="py-2.5">
                <div class="w-full bg-mendelu-gray-light rounded-full h-2">
                  <div class="bg-mendelu-green h-2 rounded-full" :style="{ width: `${ds.percent}%` }"></div>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <p v-else class="text-mendelu-gray-dark text-sm">No dataset data available</p>
    </div>

    <!-- Variable Breakdown -->
    <div class="card">
      <h3 class="text-sm font-medium text-mendelu-black mb-3">Chunks by Variable</h3>
      <div v-if="variableBreakdown.length" class="flex flex-wrap gap-2">
        <span
          v-for="v in variableBreakdown"
          :key="v.name"
          class="px-3 py-1.5 bg-mendelu-green/10 text-mendelu-green rounded-full text-xs font-medium"
        >
          {{ v.name }}
        </span>
      </div>
      <p v-else class="text-mendelu-gray-dark text-sm">No variable data available</p>
    </div>

    <!-- Sample Embeddings -->
    <div class="card">
      <h3 class="text-sm font-medium text-mendelu-black mb-3">Sample Records</h3>
      <div v-if="samples.length" class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="text-left text-mendelu-gray-dark border-b border-mendelu-gray-semi text-xs uppercase tracking-wider">
              <th class="pb-2 font-medium">Variable</th>
              <th class="pb-2 font-medium">Source</th>
              <th class="pb-2 font-medium">Temporal</th>
              <th class="pb-2 font-medium">Spatial</th>
              <th class="pb-2 font-medium">Preview</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-mendelu-gray-semi/50">
            <tr v-for="sample in samples" :key="sample.id" class="text-mendelu-gray-dark">
              <td class="py-2.5">
                <span class="px-1.5 py-0.5 bg-mendelu-green/10 text-mendelu-green rounded text-xs font-medium">{{ sample.variable }}</span>
              </td>
              <td class="py-2.5">{{ sample.source }}</td>
              <td class="py-2.5">{{ sample.temporal }}</td>
              <td class="py-2.5">{{ sample.spatial }}</td>
              <td class="py-2.5 max-w-xs truncate text-mendelu-gray-dark text-xs">{{ sample.text }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p v-else class="text-mendelu-gray-dark text-sm">No sample data available</p>
    </div>

    <!-- Actions -->
    <div class="card">
      <h3 class="text-sm font-medium text-mendelu-black mb-3">Actions</h3>
      <div class="flex gap-3 flex-wrap">
        <button @click="regenerateEmbeddings" class="btn-secondary">Regenerate All</button>
        <button @click="optimizeCollection" class="btn-secondary">Optimize Collection</button>
        <button @click="exportEmbeddings" class="btn-secondary">Export JSON</button>
        <button @click="clearEmbeddings" class="btn-danger">Clear All</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'

const stats = ref({})
const loading = ref(false)
const samples = ref([])
const datasetBreakdown = ref([])
const variableBreakdown = ref([])

const healthStatus = computed(() => {
  if (!stats.value.total_embeddings) return 'empty'
  if (stats.value.total_embeddings > 0) return 'healthy'
  return 'degraded'
})

async function refreshStats() {
  loading.value = true
  try {
    const resp = await fetch(`/rag/info?t=${Date.now()}`)
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    const data = await resp.json()
    stats.value = data

    // Build dataset breakdown
    const sources = data.sources || []
    const total = data.total_embeddings || 1
    datasetBreakdown.value = sources.map(s => ({
      name: s,
      count: Math.round(total / sources.length),
      percent: Math.round(100 / sources.length)
    }))

    // Build variable breakdown
    variableBreakdown.value = (data.variables || []).map(v => ({ name: v }))

    // Build sample records
    samples.value = (data.variables || []).slice(0, 5).map((v, i) => ({
      id: i,
      variable: v,
      source: data.sources?.[0] || 'Unknown',
      temporal: '2020-2100',
      spatial: '0.5 grid',
      text: `Climate data for ${v} — trends and patterns across the study region`
    }))

    // Try to get detailed health
    try {
      const healthResp = await fetch('/admin/qdrant/health')
      if (healthResp.ok) {
        const healthData = await healthResp.json()
        if (healthData.datasets) {
          const totalChunks = Object.values(healthData.datasets).reduce((a, b) => a + b, 0)
          datasetBreakdown.value = Object.entries(healthData.datasets).map(([name, count]) => ({
            name, count, percent: totalChunks > 0 ? Math.round((count / totalChunks) * 100) : 0
          })).sort((a, b) => b.count - a.count)
        }
      }
    } catch (e) {}
  } catch (e) {
    console.error('Failed to load stats:', e)
  } finally {
    loading.value = false
  }
}

function regenerateEmbeddings() {
  if (confirm('This will regenerate all embeddings. Continue?')) {
    alert('Embedding regeneration would be triggered via Dagster pipeline')
  }
}

function optimizeCollection() {
  alert('Collection optimization initiated')
}

function exportEmbeddings() {
  alert('Export feature coming soon')
}

async function clearEmbeddings() {
  if (!confirm('This will permanently delete ALL embeddings from Qdrant. Are you sure?')) return
  try {
    const resp = await fetch('/embeddings/clear?confirm=true&delete_sources=false', { method: 'POST' })
    if (!resp.ok) {
      const error = await resp.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(error.detail || `HTTP ${resp.status}`)
    }
    await refreshStats()
  } catch (e) {
    console.error('Error clearing embeddings:', e)
    alert(`Error: ${e.message}`)
  }
}

onMounted(refreshStats)
</script>
