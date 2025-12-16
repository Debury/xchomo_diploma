<template>
  <div class="space-y-6">
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-2xl font-bold text-white">Data Sources</h1>
        <p class="text-gray-400">Manage your climate data sources</p>
      </div>
      <router-link 
        to="/sources/create"
        class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
      >
        + Add Source
      </router-link>
    </div>

    <!-- Sources Grid -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      <div 
        v-for="source in sources" 
        :key="source.name"
        class="card hover:border-blue-500/50 transition-colors"
      >
        <div class="flex items-start justify-between mb-4">
          <div class="w-12 h-12 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-2xl">
            {{ getSourceIcon(source.name) }}
          </div>
          <span 
            class="px-2 py-1 rounded text-xs font-medium"
            :class="source.enabled ? 'bg-green-500/20 text-green-400' : 'bg-gray-500/20 text-gray-400'"
          >
            {{ source.enabled ? 'Active' : 'Inactive' }}
          </span>
        </div>
        
        <h3 class="text-lg font-semibold text-white mb-2">{{ source.name }}</h3>
        <p class="text-gray-400 text-sm mb-4">{{ source.description || 'No description' }}</p>
        
        <div class="space-y-2 text-sm">
          <div class="flex justify-between">
            <span class="text-gray-500">Type:</span>
            <span class="text-gray-300">{{ source.type || 'NetCDF' }}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-500">Embeddings:</span>
            <span class="text-gray-300">{{ source.embedding_count?.toLocaleString() || '—' }}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-500">Variables:</span>
            <span class="text-gray-300">{{ source.variables?.length || '—' }}</span>
          </div>
        </div>
        
        <div class="mt-4 pt-4 border-t border-dark-border flex space-x-2">
          <button 
            @click="viewDetails(source)"
            class="flex-1 px-3 py-2 bg-dark-hover text-gray-300 rounded hover:bg-gray-600 transition-colors text-sm"
          >
            Details
          </button>
          <button 
            @click="refreshSource(source)"
            class="flex-1 px-3 py-2 bg-dark-hover text-gray-300 rounded hover:bg-gray-600 transition-colors text-sm"
          >
            Refresh
          </button>
        </div>
      </div>
    </div>

    <!-- Empty State -->
    <div v-if="!loading && sources.length === 0" class="card text-center py-12">
      <div class="text-4xl mb-4">📁</div>
      <h3 class="text-lg font-medium text-white mb-2">No Sources Configured</h3>
      <p class="text-gray-400 mb-6">Add your first data source to get started</p>
      <router-link 
        to="/sources/create"
        class="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors inline-block"
      >
        Add Source
      </router-link>
    </div>

    <!-- Source Detail Modal -->
    <div 
      v-if="selectedSource" 
      class="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
      @click.self="selectedSource = null"
    >
      <div class="bg-dark-card border border-dark-border rounded-xl p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto">
        <div class="flex items-center justify-between mb-6">
          <h2 class="text-xl font-bold text-white">{{ selectedSource.name }}</h2>
          <button @click="selectedSource = null" class="text-gray-400 hover:text-white">✕</button>
        </div>
        
        <div class="space-y-4">
          <div>
            <h4 class="text-sm font-medium text-gray-400 mb-2">Variables</h4>
            <div class="flex flex-wrap gap-2">
              <span 
                v-for="v in selectedSource.variables" 
                :key="v"
                class="px-3 py-1 bg-dark-hover rounded text-sm text-gray-300"
              >
                {{ v }}
              </span>
            </div>
          </div>
          
          <div>
            <h4 class="text-sm font-medium text-gray-400 mb-2">Configuration</h4>
            <pre class="bg-dark-hover p-4 rounded text-sm text-gray-300 overflow-x-auto">{{ JSON.stringify(selectedSource.config, null, 2) }}</pre>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const sources = ref([])
const loading = ref(true)
const selectedSource = ref(null)

function getSourceIcon(name) {
  const icons = {
    'ISIMP': '🌍',
    'ERA5': '🌤️',
    'CMIP': '📊',
    'default': '📁'
  }
  return icons[name] || icons.default
}

async function loadSources() {
  loading.value = true
  try {
    // Get sources from the sources endpoint
    const resp = await fetch('/sources?active_only=false')
    
    if (!resp.ok) {
      throw new Error(`Failed to load sources: ${resp.statusText}`)
    }
    
    const data = await resp.json()
    
    // Transform API response to match component expectations
    sources.value = data.map(source => ({
      name: source.source_id || 'Unknown',
      enabled: source.is_active !== false,
      description: source.description || `Data source: ${source.source_id}`,
      type: source.format || 'NetCDF',
      variables: source.variables || [],
      embedding_count: 0, // Will be calculated separately if needed
      url: source.url,
      source_id: source.source_id,
      processing_status: source.processing_status || 'pending'
    }))
  } catch (e) {
    console.error('Failed to load sources:', e)
    // Show error to user
    alert(`Failed to load sources: ${e.message}`)
  } finally {
    loading.value = false
  }
}

function viewDetails(source) {
  selectedSource.value = {
    ...source,
    config: {
      base_url: 'https://data.isimip.org',
      collection: 'climate_embeddings',
      chunk_strategy: 'temporal_spatial'
    }
  }
}

async function refreshSource(source) {
  alert(`Refreshing ${source.name}... (Dagster pipeline would be triggered)`)
}

onMounted(loadSources)
</script>
