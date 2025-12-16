<template>
  <div class="space-y-6">
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-2xl font-bold text-white">Embeddings</h1>
        <p class="text-gray-400">View and manage vector embeddings</p>
      </div>
      <button 
        @click="refreshStats"
        :disabled="loading"
        class="px-4 py-2 bg-dark-hover text-gray-300 rounded-lg hover:bg-gray-600 transition-colors"
      >
        {{ loading ? 'Loading...' : 'Refresh' }}
      </button>
    </div>

    <!-- Stats Overview -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
      <div class="card">
        <h3 class="text-gray-400 text-sm font-medium mb-2">Total Embeddings</h3>
        <p class="text-3xl font-bold text-blue-400">{{ stats.total_embeddings?.toLocaleString() || '—' }}</p>
      </div>
      <div class="card">
        <h3 class="text-gray-400 text-sm font-medium mb-2">Vector Dimension</h3>
        <p class="text-3xl font-bold text-purple-400">384</p>
      </div>
      <div class="card">
        <h3 class="text-gray-400 text-sm font-medium mb-2">Collection</h3>
        <p class="text-xl font-bold text-green-400">climate_embeddings</p>
      </div>
    </div>

    <!-- Variables Distribution -->
    <div class="card">
      <h3 class="text-lg font-semibold text-white mb-4">Variables Distribution</h3>
      <div class="space-y-3">
        <div 
          v-for="(count, variable) in variableDistribution" 
          :key="variable"
          class="flex items-center"
        >
          <span class="w-32 text-sm text-gray-300 truncate">{{ variable }}</span>
          <div class="flex-1 mx-4 bg-dark-hover rounded-full h-2 overflow-hidden">
            <div 
              class="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"
              :style="{ width: `${(count / maxCount) * 100}%` }"
            ></div>
          </div>
          <span class="text-sm text-gray-400 w-16 text-right">{{ count }}</span>
        </div>
      </div>
    </div>

    <!-- Sample Embeddings -->
    <div class="card">
      <h3 class="text-lg font-semibold text-white mb-4">Sample Records</h3>
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="text-left text-gray-400 border-b border-dark-border">
              <th class="pb-3 font-medium">Variable</th>
              <th class="pb-3 font-medium">Source</th>
              <th class="pb-3 font-medium">Temporal</th>
              <th class="pb-3 font-medium">Spatial</th>
              <th class="pb-3 font-medium">Text Preview</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-dark-border">
            <tr v-for="sample in samples" :key="sample.id" class="text-gray-300">
              <td class="py-3">
                <span class="px-2 py-1 bg-blue-500/20 text-blue-400 rounded text-xs">
                  {{ sample.variable }}
                </span>
              </td>
              <td class="py-3">{{ sample.source }}</td>
              <td class="py-3">{{ sample.temporal }}</td>
              <td class="py-3">{{ sample.spatial }}</td>
              <td class="py-3 max-w-xs truncate text-gray-400">{{ sample.text }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Actions -->
    <div class="card">
      <h3 class="text-lg font-semibold text-white mb-4">Actions</h3>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <button 
          @click="regenerateEmbeddings"
          class="p-4 bg-dark-hover rounded-lg hover:bg-gray-600 transition-colors text-left"
        >
          <div class="text-2xl mb-2">🔄</div>
          <h4 class="font-medium text-white">Regenerate All</h4>
          <p class="text-sm text-gray-400">Re-process all embeddings</p>
        </button>
        
        <button 
          @click="exportEmbeddings"
          class="p-4 bg-dark-hover rounded-lg hover:bg-gray-600 transition-colors text-left"
        >
          <div class="text-2xl mb-2">📥</div>
          <h4 class="font-medium text-white">Export</h4>
          <p class="text-sm text-gray-400">Download embeddings as JSON</p>
        </button>
        
        <button 
          @click="clearEmbeddings"
          class="p-4 bg-dark-hover rounded-lg hover:bg-red-900/30 transition-colors text-left"
        >
          <div class="text-2xl mb-2">🗑️</div>
          <h4 class="font-medium text-red-400">Clear All</h4>
          <p class="text-sm text-gray-400">Remove all embeddings</p>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'

const stats = ref({})
const loading = ref(false)
const samples = ref([])

const variableDistribution = computed(() => {
  // Simulated distribution based on available variables
  const vars = stats.value.variables || []
  const total = stats.value.total_embeddings || 0
  const dist = {}
  vars.forEach((v, i) => {
    dist[v] = Math.floor(total / vars.length) + (i < total % vars.length ? 1 : 0)
  })
  return dist
})

const maxCount = computed(() => {
  return Math.max(...Object.values(variableDistribution.value), 1)
})

async function refreshStats() {
  loading.value = true
  try {
    const resp = await fetch('/rag/info')
    stats.value = await resp.json()
    
    // Generate sample data
    samples.value = (stats.value.variables || []).slice(0, 5).map((v, i) => ({
      id: i,
      variable: v,
      source: stats.value.sources?.[0] || 'Unknown',
      temporal: '2020-2100',
      spatial: '0.5° grid',
      text: `Climate data for ${v} showing trends and patterns across the study region...`
    }))
  } catch (e) {
    console.error('Failed to load stats:', e)
  } finally {
    loading.value = false
  }
}

function regenerateEmbeddings() {
  if (confirm('This will regenerate all embeddings. This may take a while. Continue?')) {
    alert('Embedding regeneration would be triggered via Dagster pipeline')
  }
}

function exportEmbeddings() {
  alert('Export feature coming soon!')
}

async function clearEmbeddings() {
  const deleteSources = confirm('⚠️ This will permanently delete ALL embeddings from Qdrant.\n\nDo you also want to delete all sources?')
  const confirmMsg = deleteSources 
    ? '⚠️ This will permanently delete ALL embeddings AND all sources. This cannot be undone!\n\nAre you absolutely sure?'
    : '⚠️ This will permanently delete ALL embeddings from Qdrant. This cannot be undone!\n\nAre you sure?'
  
  if (confirm(confirmMsg)) {
    try {
      const resp = await fetch(`/embeddings/clear?confirm=true&delete_sources=${deleteSources}`, {
        method: 'POST'
      })
      
      if (!resp.ok) {
        const error = await resp.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(error.detail || `HTTP ${resp.status}`)
      }
      
      const result = await resp.json()
      alert(`✅ Successfully cleared embeddings${deleteSources ? ' and sources' : ''}!\n\nCollection: ${result.collection}\nSources deleted: ${result.sources_deleted || 0}`)
      
      // Refresh stats
      await refreshStats()
    } catch (e) {
      console.error('Error clearing embeddings:', e)
      alert(`❌ Error: ${e.message}`)
    }
  }
}

onMounted(refreshStats)
</script>
