<template>
  <div class="space-y-6">
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-xl font-semibold text-white">Embeddings</h1>
        <p class="text-sm text-gray-500">Vector store overview</p>
      </div>
      <button
        @click="refreshStats"
        :disabled="loading"
        class="px-3 py-1.5 text-sm bg-dark-hover text-gray-300 rounded-md hover:bg-gray-600 transition-colors disabled:opacity-50"
      >
        {{ loading ? 'Loading...' : 'Refresh' }}
      </button>
    </div>

    <!-- Stats Overview -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
      <div class="card">
        <h3 class="text-xs text-gray-500 uppercase tracking-wider mb-1">Total Vectors</h3>
        <p class="text-2xl font-semibold text-white tabular-nums">{{ stats.total_embeddings?.toLocaleString() || '0' }}</p>
      </div>
      <div class="card">
        <h3 class="text-xs text-gray-500 uppercase tracking-wider mb-1">Dimensions</h3>
        <p class="text-2xl font-semibold text-white">1024</p>
      </div>
      <div class="card">
        <h3 class="text-xs text-gray-500 uppercase tracking-wider mb-1">Collection</h3>
        <p class="text-lg font-mono text-white">climate_data</p>
      </div>
    </div>

    <!-- Sample Embeddings -->
    <div class="card">
      <h3 class="text-sm font-medium text-gray-300 mb-3">Sample Records</h3>
      <div v-if="samples.length" class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="text-left text-gray-500 border-b border-dark-border text-xs">
              <th class="pb-2 font-medium">Variable</th>
              <th class="pb-2 font-medium">Source</th>
              <th class="pb-2 font-medium">Temporal</th>
              <th class="pb-2 font-medium">Spatial</th>
              <th class="pb-2 font-medium">Preview</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-dark-border/50">
            <tr v-for="sample in samples" :key="sample.id" class="text-gray-400">
              <td class="py-2.5">
                <span class="px-1.5 py-0.5 bg-blue-500/10 text-blue-400 rounded text-xs">{{ sample.variable }}</span>
              </td>
              <td class="py-2.5">{{ sample.source }}</td>
              <td class="py-2.5">{{ sample.temporal }}</td>
              <td class="py-2.5">{{ sample.spatial }}</td>
              <td class="py-2.5 max-w-xs truncate text-gray-500 text-xs">{{ sample.text }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p v-else class="text-gray-600 text-sm">No sample data available</p>
    </div>

    <!-- Actions -->
    <div class="card">
      <h3 class="text-sm font-medium text-gray-300 mb-3">Actions</h3>
      <div class="flex gap-3">
        <button
          @click="regenerateEmbeddings"
          class="px-4 py-2 bg-dark-hover text-gray-300 rounded-md hover:bg-gray-600 transition-colors text-sm"
        >
          Regenerate All
        </button>
        <button
          @click="exportEmbeddings"
          class="px-4 py-2 bg-dark-hover text-gray-300 rounded-md hover:bg-gray-600 transition-colors text-sm"
        >
          Export JSON
        </button>
        <button
          @click="clearEmbeddings"
          class="px-4 py-2 bg-red-500/10 text-red-400 rounded-md hover:bg-red-500/20 transition-colors text-sm"
        >
          Clear All
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const stats = ref({})
const loading = ref(false)
const samples = ref([])

async function refreshStats() {
  loading.value = true
  try {
    const resp = await fetch(`/rag/info?t=${Date.now()}`)
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    const data = await resp.json()
    stats.value = data

    samples.value = (data.variables || []).slice(0, 5).map((v, i) => ({
      id: i,
      variable: v,
      source: data.sources?.[0] || 'Unknown',
      temporal: '2020-2100',
      spatial: '0.5 grid',
      text: `Climate data for ${v} — trends and patterns across the study region`
    }))
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
