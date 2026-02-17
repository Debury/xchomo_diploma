<template>
  <div class="space-y-6">
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-xl font-semibold text-white">Data Sources</h1>
        <p class="text-sm text-gray-500">Manage your climate data sources</p>
      </div>
      <div class="flex gap-2">
        <button
          @click="loadSources()"
          :disabled="loading"
          class="px-3 py-1.5 text-sm bg-dark-hover text-gray-300 rounded-md hover:bg-gray-600 transition-colors disabled:opacity-50"
          title="Refresh sources list"
        >
          Refresh
        </button>
      <div class="flex gap-2">
        <button
          @click="deleteAllSources"
          class="px-3 py-1.5 text-sm bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
          title="Delete all sources"
        >
          Clear All
        </button>
        <router-link 
          to="/sources/create"
          class="px-3 py-1.5 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
        >
          Add Source
        </router-link>
      </div>
      </div>
    </div>

    <!-- Sources Grid -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      <div
        v-for="source in sources"
        :key="source.name"
        class="card !p-4 hover:border-blue-500/50 transition-colors"
      >
        <div class="flex items-start justify-between mb-3">
          <div class="flex flex-col gap-1 items-end">
            <span 
              class="px-2 py-1 rounded text-xs font-medium"
              :class="source.enabled ? 'bg-green-500/20 text-green-400' : 'bg-gray-500/20 text-gray-400'"
            >
              {{ source.enabled ? 'Active' : 'Inactive' }}
            </span>
            <span 
              class="px-2 py-1 rounded text-xs font-medium"
              :class="getStatusClass(source.processing_status)"
            >
              {{ getStatusLabel(source.processing_status) }}
            </span>
          </div>
        </div>
        
        <div class="flex items-center gap-2 mb-1">
          <h3 class="text-sm font-semibold text-white">{{ source.name }}</h3>
          <span
            v-if="isCatalogSource(source)"
            class="px-1.5 py-0.5 rounded text-[10px] font-bold bg-purple-500/20 text-purple-400 border border-purple-500/30"
          >
            D1.1
          </span>
        </div>
        <p class="text-gray-500 text-xs mb-2">{{ source.description || 'No description' }}</p>
        
        <!-- Error message if processing failed -->
        <div v-if="source.processing_status === 'failed' && source.error_message"
             class="mb-2 p-2 bg-red-500/20 border border-red-500/30 rounded text-xs text-red-300">
          {{ source.error_message }}
        </div>
        
        <div class="space-y-1 text-xs">
          <div class="flex justify-between">
            <span class="text-gray-500">Type:</span>
            <span class="text-gray-300">{{ source.type || 'Unknown' }}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-500">Embeddings:</span>
            <span class="text-gray-300">{{ source.embedding_count?.toLocaleString() || '—' }}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-500">Variables:</span>
            <span class="text-gray-300">{{ source.variables?.length || '—' }}</span>
          </div>
          <div v-if="source.last_processed" class="flex justify-between">
            <span class="text-gray-500">Last Processed:</span>
            <span class="text-gray-300 text-xs">{{ formatDate(source.last_processed) }}</span>
          </div>
        </div>
        
        <div class="mt-3 pt-3 border-t border-dark-border flex space-x-2">
          <button 
            @click="viewDetails(source)"
            class="flex-1 px-3 py-1.5 bg-dark-hover text-gray-300 rounded-md hover:bg-gray-600 transition-colors text-xs"
          >
            Details
          </button>
          <button 
            @click="refreshSource(source)"
            :disabled="source.refreshing || source.processing_status === 'processing'"
            class="flex-1 px-3 py-1.5 bg-dark-hover text-gray-300 rounded-md hover:bg-gray-600 transition-colors text-xs disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {{ source.refreshing ? 'Triggering...' : 'Refresh' }}
          </button>
          <button 
            @click="deleteSource(source)"
            class="px-3 py-1.5 bg-red-600/20 text-red-400 rounded-md hover:bg-red-600/30 transition-colors text-xs"
            title="Delete source"
          >
            Delete
          </button>
        </div>
      </div>
    </div>

    <!-- Empty State -->
    <div v-if="!loading && sources.length === 0" class="card text-center py-8">
      <h3 class="text-sm font-medium text-white mb-2">No Sources Configured</h3>
      <p class="text-gray-500 text-xs mb-4">Add your first data source to get started</p>
      <router-link
        to="/sources/create"
        class="px-4 py-2 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors inline-block"
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
          <h2 class="text-lg font-semibold text-white">{{ selectedSource.name }}</h2>
          <button @click="selectedSource = null" class="text-gray-400 hover:text-white text-sm">Close</button>
        </div>
        
        <div class="space-y-4">
          <!-- Processing Status -->
          <div>
            <h4 class="text-sm font-medium text-gray-400 mb-2">Processing Status</h4>
            <div class="space-y-2">
              <div class="flex items-center gap-2">
                <span 
                  class="px-3 py-1 rounded text-sm font-medium"
                  :class="getStatusClass(selectedSource.processing_status)"
                >
                  {{ getStatusLabel(selectedSource.processing_status) }}
                </span>
                <span v-if="selectedSource.last_processed" class="text-xs text-gray-500">
                  Last: {{ formatDate(selectedSource.last_processed) }}
                </span>
              </div>
              <div v-if="selectedSource.error_message" 
                   class="p-3 bg-red-500/20 border border-red-500/30 rounded text-sm text-red-300">
                <strong>Error:</strong> {{ selectedSource.error_message }}
              </div>
            </div>
          </div>
          
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
              <span v-if="!selectedSource.variables || selectedSource.variables.length === 0" 
                    class="text-gray-500 text-sm">
                No variables configured
              </span>
            </div>
          </div>
          
          <div>
            <h4 class="text-sm font-medium text-gray-400 mb-2">Source Details</h4>
            <div class="bg-dark-hover p-4 rounded text-sm space-y-2">
              <div class="flex justify-between">
                <span class="text-gray-500">Source ID:</span>
                <span class="text-gray-300">{{ selectedSource.source_id }}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-500">Format:</span>
                <span class="text-gray-300">{{ selectedSource.type || 'Unknown' }}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-500">URL:</span>
                <span class="text-gray-300 text-xs break-all">{{ selectedSource.url }}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-500">Active:</span>
                <span class="text-gray-300">{{ selectedSource.enabled ? 'Yes' : 'No' }}</span>
              </div>
            </div>
          </div>
          
          <div class="pt-4 border-t border-dark-border">
            <h4 class="text-sm font-medium text-gray-400 mb-3">Danger Zone</h4>
            <div class="space-y-2">
              <button
                @click="deleteSourceEmbeddings(selectedSource)"
                class="w-full px-4 py-2 bg-yellow-600/20 text-yellow-400 rounded hover:bg-yellow-600/30 transition-colors text-sm"
              >
                Delete Embeddings for This Source
              </button>
              <button
                @click="deleteSource(selectedSource)"
                class="w-full px-4 py-2 bg-red-600/20 text-red-400 rounded hover:bg-red-600/30 transition-colors text-sm"
              >
                Delete Source
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const sources = ref([])
const loading = ref(true)
const selectedSource = ref(null)
let statusPollInterval = null

function getSourceIcon(name) {
  const icons = {
    'ISIMP': 'IS',
    'ERA5': 'E5',
    'CMIP': 'CM',
    'default': 'DS'
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
      type: source.format || 'Unknown',
      variables: source.variables || [],
      embedding_count: 0,
      url: source.url,
      source_id: source.source_id,
      processing_status: source.processing_status || 'pending',
      error_message: source.error_message || null,
      last_processed: source.last_processed || null,
      tags: source.tags || [],
      refreshing: false
    }))
  } catch (e) {
    console.error('Failed to load sources:', e)
    // Show error to user
    console.log(`Failed to load sources: ${e.message}`)
  } finally {
    loading.value = false
  }
}

function viewDetails(source) {
  selectedSource.value = {
    ...source
  }
}

function isCatalogSource(source) {
  return source.tags && source.tags.some(t => t === 'catalog:D1.1')
}

function getStatusClass(status) {
  const classes = {
    'success': 'bg-green-500/20 text-green-400',
    'completed': 'bg-green-500/20 text-green-400',
    'metadata_only': 'bg-amber-500/20 text-amber-400',
    'failed': 'bg-red-500/20 text-red-400',
    'error': 'bg-red-500/20 text-red-400',
    'processing': 'bg-yellow-500/20 text-yellow-400',
    'pending': 'bg-gray-500/20 text-gray-400'
  }
  return classes[status] || classes['pending']
}

function getStatusLabel(status) {
  const labels = {
    'success': 'Success',
    'completed': 'Completed',
    'metadata_only': 'Metadata Only',
    'failed': 'Failed',
    'error': 'Error',
    'processing': 'Processing',
    'pending': 'Pending'
  }
  return labels[status] || 'Pending'
}

function formatDate(dateString) {
  if (!dateString) return '—'
  try {
    const date = new Date(dateString)
    return date.toLocaleString()
  } catch {
    return dateString
  }
}

async function refreshSource(source) {
  if (source.refreshing || source.processing_status === 'processing') {
    return
  }

  source.refreshing = true
  
  try {
    const resp = await fetch(`/sources/${source.source_id}/trigger`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      }
    })
    
    if (!resp.ok) {
      const errorData = await resp.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(errorData.detail || `HTTP ${resp.status}: ${resp.statusText}`)
    }
    
    const result = await resp.json()
    console.log('Pipeline triggered:', result)
    
    // Update source status to processing
    source.processing_status = 'processing'
    source.error_message = null
    
    // Show success message
    console.log(`Pipeline triggered for ${source.name}, run ID: ${result.run_id}, status: ${result.status}`)
    
    // Reload sources after a short delay to get updated status
    setTimeout(() => {
      loadSources()
    }, 2000)
    
  } catch (e) {
    console.error('Error triggering pipeline:', e)
    console.log(`Error triggering pipeline: ${e.message}`)
  } finally {
    source.refreshing = false
  }
}

async function deleteSource(source) {
  if (!confirm(`Delete source "${source.name}"?\n\nThis will permanently delete the source configuration.`)) {
    return
  }
  
  try {
    const resp = await fetch(`/sources/${source.source_id}`, {
      method: 'DELETE'
    })
    
    if (!resp.ok && resp.status !== 204) {
      const errorData = await resp.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(errorData.detail || `HTTP ${resp.status}: ${resp.statusText}`)
    }
    
    console.log(`Source "${source.name}" deleted successfully`)
    await loadSources()
    if (selectedSource.value && selectedSource.value.source_id === source.source_id) {
      selectedSource.value = null
    }
  } catch (e) {
    console.error('Error deleting source:', e)
    console.log(`Error: ${e.message}`)
  }
}

async function deleteSourceEmbeddings(source) {
  if (!confirm(`Delete all embeddings for source "${source.name}"?\n\nThis will permanently delete all embeddings from Qdrant for this source. This cannot be undone.`)) {
    return
  }
  
  try {
    const resp = await fetch(`/sources/${source.source_id}/embeddings?confirm=true`, {
      method: 'DELETE'
    })
    
    if (!resp.ok) {
      const errorData = await resp.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(errorData.detail || `HTTP ${resp.status}: ${resp.statusText}`)
    }
    
    const result = await resp.json()
    console.log(result.message)
    await loadSources()
  } catch (e) {
    console.error('Error deleting embeddings:', e)
    console.log(`Error: ${e.message}`)
  }
}

function startStatusPolling() {
  // Poll for status updates every 5 seconds if any source is processing
  statusPollInterval = setInterval(() => {
    const hasProcessing = sources.value.some(s => s.processing_status === 'processing')
    if (hasProcessing) {
      loadSources()
    }
  }, 5000) // Poll every 5 seconds
}

function stopStatusPolling() {
  if (statusPollInterval) {
    clearInterval(statusPollInterval)
    statusPollInterval = null
  }
}

async function deleteAllSources() {
  const deleteEmbeddings = confirm('This will delete ALL sources.\n\nDo you also want to delete embeddings from Qdrant?')
  const confirmMsg = deleteEmbeddings
    ? 'This will permanently delete ALL sources AND their embeddings. This cannot be undone.\n\nAre you sure?'
    : 'This will permanently delete ALL sources. This cannot be undone.\n\nAre you sure?'
  
  if (confirm(confirmMsg)) {
    try {
      const resp = await fetch(`/sources?confirm=true&delete_embeddings=${deleteEmbeddings}`, {
        method: 'DELETE'
      })
      
      if (!resp.ok) {
        const error = await resp.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(error.detail || `HTTP ${resp.status}`)
      }
      
      const result = await resp.json()
      console.log(`Deleted ${result.sources_deleted} source(s)${deleteEmbeddings ? ' and embeddings' : ''}`)
      
      // Refresh sources list
      await loadSources()
    } catch (e) {
      console.error('Error deleting sources:', e)
      console.log(`Error: ${e.message}`)
    }
  }
}

onMounted(() => {
  loadSources()
  startStatusPolling()
})

onUnmounted(() => {
  stopStatusPolling()
})
</script>
