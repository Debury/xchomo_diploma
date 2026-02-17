<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-xl font-semibold text-white">Settings</h1>
        <p class="text-sm text-gray-500">System configuration and status</p>
      </div>
      <button
        @click="refreshSettings"
        :disabled="loading"
        class="px-3 py-1.5 text-sm bg-dark-hover text-gray-300 rounded-md hover:bg-gray-600 transition-colors disabled:opacity-50"
      >
        Refresh
      </button>
    </div>

    <!-- LLM Configuration -->
    <div class="card !p-4">
      <h3 class="text-sm font-semibold text-white mb-3">LLM Providers</h3>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div class="bg-dark-hover rounded-lg p-4">
          <div class="flex items-center justify-between mb-2">
            <span class="text-white font-medium">OpenRouter</span>
            <span class="w-3 h-3 rounded-full" :class="settings?.llm?.providers?.openrouter ? 'bg-green-500' : 'bg-red-500'"></span>
          </div>
          <p class="text-xs text-gray-500">{{ settings?.llm?.providers?.openrouter ? 'API key configured' : 'No API key set' }}</p>
        </div>
        <div class="bg-dark-hover rounded-lg p-4">
          <div class="flex items-center justify-between mb-2">
            <span class="text-white font-medium">Groq</span>
            <span class="w-3 h-3 rounded-full" :class="settings?.llm?.providers?.groq ? 'bg-green-500' : 'bg-red-500'"></span>
          </div>
          <p class="text-xs text-gray-500">{{ settings?.llm?.providers?.groq ? 'API key configured' : 'No API key set' }}</p>
        </div>
        <div class="bg-dark-hover rounded-lg p-4">
          <div class="flex items-center justify-between mb-2">
            <span class="text-white font-medium">Ollama</span>
            <span class="w-3 h-3 rounded-full bg-blue-500"></span>
          </div>
          <p class="text-xs text-gray-500">{{ settings?.llm?.providers?.ollama || 'localhost:11434' }}</p>
        </div>
      </div>
    </div>

    <!-- Embedding Model -->
    <div class="card !p-4">
      <h3 class="text-sm font-semibold text-white mb-3">Embedding Model</h3>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div>
          <span class="text-gray-500 text-xs block">Model</span>
          <span class="text-white font-mono text-sm">{{ settings?.embedding_model?.name || 'BAAI/bge-large-en-v1.5' }}</span>
        </div>
        <div>
          <span class="text-gray-500 text-xs block">Dimensions</span>
          <span class="text-white font-mono text-sm">{{ settings?.embedding_model?.dimensions || 1024 }}</span>
        </div>
        <div>
          <span class="text-gray-500 text-xs block">Distance Metric</span>
          <span class="text-white font-mono text-sm">{{ settings?.embedding_model?.distance || 'COSINE' }}</span>
        </div>
        <div>
          <span class="text-gray-500 text-xs block">Status</span>
          <span class="text-green-400 text-sm">Active</span>
        </div>
      </div>
    </div>

    <!-- Qdrant -->
    <div class="card !p-4">
      <h3 class="text-sm font-semibold text-white mb-3">Vector Database (Qdrant)</h3>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div>
          <span class="text-gray-500 text-xs block">Host</span>
          <span class="text-white font-mono text-sm">{{ settings?.qdrant?.host || 'localhost' }}</span>
        </div>
        <div>
          <span class="text-gray-500 text-xs block">Port</span>
          <span class="text-white font-mono text-sm">{{ settings?.qdrant?.port || 6333 }}</span>
        </div>
        <div>
          <span class="text-gray-500 text-xs block">Embeddings</span>
          <span class="text-white font-mono text-sm">{{ embeddingStats?.total_embeddings?.toLocaleString() || '—' }}</span>
        </div>
        <div>
          <span class="text-gray-500 text-xs block">Collection</span>
          <span class="text-white font-mono text-sm">{{ embeddingStats?.collection_name || 'climate_data' }}</span>
        </div>
      </div>
      <div class="mt-4 flex gap-3">
        <button
          @click="clearEmbeddings"
          class="px-3 py-1.5 text-sm bg-red-600/20 text-red-400 rounded-md hover:bg-red-600/40 transition-colors"
        >
          Clear Collection
        </button>
      </div>
    </div>

    <!-- System Resources -->
    <div class="card !p-4">
      <h3 class="text-sm font-semibold text-white mb-3">System Resources</h3>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <span class="text-gray-500 text-xs block mb-2">Disk Usage</span>
          <div v-if="settings?.disk" class="space-y-1">
            <div class="w-full bg-gray-700 rounded-full h-3">
              <div
                class="bg-blue-500 h-3 rounded-full"
                :style="{ width: `${diskPercent}%` }"
              ></div>
            </div>
            <span class="text-xs text-gray-500">
              {{ settings.disk.used_gb }} GB / {{ settings.disk.total_gb }} GB
              ({{ settings.disk.free_gb }} GB free)
            </span>
          </div>
          <span v-else class="text-gray-500">—</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'

const settings = ref(null)
const embeddingStats = ref(null)
const loading = ref(false)

const diskPercent = computed(() => {
  if (!settings.value?.disk) return 0
  return Math.round((settings.value.disk.used_gb / settings.value.disk.total_gb) * 100)
})

async function refreshSettings() {
  loading.value = true
  try {
    const [sysResp, embResp] = await Promise.all([
      fetch('/settings/system'),
      fetch('/embeddings/stats'),
    ])
    if (sysResp.ok) settings.value = await sysResp.json()
    if (embResp.ok) embeddingStats.value = await embResp.json()
  } catch (e) {
    console.error('Failed to load settings:', e)
  } finally {
    loading.value = false
  }
}

async function clearEmbeddings() {
  if (!confirm('Are you sure you want to clear ALL embeddings? This cannot be undone.')) return
  try {
    const resp = await fetch('/embeddings/clear?confirm=true', { method: 'POST' })
    if (resp.ok) {
      console.log('Embeddings cleared successfully')
      refreshSettings()
    }
  } catch (e) {
    console.error('Failed to clear embeddings:', e)
  }
}

onMounted(() => {
  refreshSettings()
})
</script>
