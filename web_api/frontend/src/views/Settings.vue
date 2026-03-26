<template>
  <div class="space-y-6">
    <PageHeader title="Settings" subtitle="System configuration and status">
      <template #actions>
        <button @click="refreshSettings" :disabled="loading" class="btn-secondary disabled:opacity-50">Refresh</button>
      </template>
    </PageHeader>

    <!-- LLM Configuration -->
    <div class="card">
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-sm font-medium text-mendelu-black">LLM Configuration</h3>
        <button v-if="hasChanges" @click="saveSettings" :disabled="saving" class="btn-primary !py-1.5 !text-xs disabled:opacity-50">
          {{ saving ? 'Saving...' : 'Save Changes' }}
        </button>
      </div>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">LLM Model</label>
          <select v-model="editableSettings.model" class="input-field">
            <option value="google/gemini-2.0-flash-001">Gemini 2.0 Flash</option>
            <option value="anthropic/claude-3.5-sonnet">Claude 3.5 Sonnet</option>
            <option value="openai/gpt-4o-mini">GPT-4o Mini</option>
            <option value="meta-llama/llama-3.1-70b-instruct">Llama 3.1 70B</option>
            <option value="mistralai/mistral-large-latest">Mistral Large</option>
          </select>
        </div>
        <div>
          <label class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Temperature: {{ editableSettings.temperature }}</label>
          <input type="range" v-model.number="editableSettings.temperature" min="0" max="1" step="0.1" class="w-full h-2 bg-mendelu-gray-semi rounded-lg appearance-none cursor-pointer accent-mendelu-green" />
          <div class="flex justify-between text-[10px] text-mendelu-gray-dark mt-1">
            <span>Precise (0)</span>
            <span>Creative (1)</span>
          </div>
        </div>
        <div>
          <label class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Top-K Results</label>
          <input type="number" v-model.number="editableSettings.top_k" min="1" max="50" class="input-field" />
        </div>
        <div>
          <label class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Embedding Batch Size</label>
          <input type="number" v-model.number="editableSettings.batch_size" min="1" max="1000" class="input-field" />
        </div>
      </div>
    </div>

    <!-- Portal Credentials -->
    <div class="card">
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-sm font-medium text-mendelu-black">Portal Credentials</h3>
        <button v-if="hasCredentialChanges" @click="saveCredentials" :disabled="savingCredentials" class="btn-primary !py-1.5 !text-xs disabled:opacity-50">
          {{ savingCredentials ? 'Saving...' : 'Save Credentials' }}
        </button>
      </div>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div v-for="field in credentialFields" :key="field.key">
          <div class="flex items-center gap-2 mb-1">
            <span class="w-2 h-2 rounded-full" :class="credentials[field.key]?.configured ? 'bg-mendelu-success' : 'bg-mendelu-gray-semi'"></span>
            <label class="text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider">{{ field.label }}</label>
          </div>
          <div class="relative">
            <input
              :type="field.visible ? 'text' : 'password'"
              v-model="credentialEdits[field.key]"
              :placeholder="credentials[field.key]?.configured ? credentials[field.key]?.masked : 'Not configured'"
              class="input-field pr-14"
            />
            <button
              type="button"
              @click="field.visible = !field.visible"
              class="absolute right-2 top-1/2 -translate-y-1/2 btn-ghost !px-2 !py-0.5 text-xs"
            >
              {{ field.visible ? 'Hide' : 'Show' }}
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- LLM Providers -->
    <div class="card">
      <h3 class="text-sm font-medium text-mendelu-black mb-3">LLM Providers</h3>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div class="bg-mendelu-gray-light rounded-lg p-4 hover:bg-mendelu-gray-semi/50 transition-all duration-150">
          <div class="flex items-center justify-between mb-2">
            <span class="text-mendelu-black font-medium text-sm">OpenRouter</span>
            <span class="w-2.5 h-2.5 rounded-full" :class="settings?.llm?.providers?.openrouter ? 'bg-mendelu-success' : 'bg-mendelu-gray-semi'"></span>
          </div>
          <p class="text-xs text-mendelu-gray-dark">{{ settings?.llm?.providers?.openrouter ? 'API key configured' : 'No API key set' }}</p>
        </div>
        <div class="bg-mendelu-gray-light rounded-lg p-4 hover:bg-mendelu-gray-semi/50 transition-all duration-150">
          <div class="flex items-center justify-between mb-2">
            <span class="text-mendelu-black font-medium text-sm">Groq</span>
            <span class="w-2.5 h-2.5 rounded-full" :class="settings?.llm?.providers?.groq ? 'bg-mendelu-success' : 'bg-mendelu-gray-semi'"></span>
          </div>
          <p class="text-xs text-mendelu-gray-dark">{{ settings?.llm?.providers?.groq ? 'API key configured' : 'No API key set' }}</p>
        </div>
        <div class="bg-mendelu-gray-light rounded-lg p-4 hover:bg-mendelu-gray-semi/50 transition-all duration-150">
          <div class="flex items-center justify-between mb-2">
            <span class="text-mendelu-black font-medium text-sm">Ollama</span>
            <span class="w-2.5 h-2.5 rounded-full bg-mendelu-green/50"></span>
          </div>
          <p class="text-xs text-mendelu-gray-dark">{{ settings?.llm?.providers?.ollama || 'localhost:11434' }}</p>
        </div>
      </div>
    </div>

    <!-- Embedding Model -->
    <div class="card">
      <h3 class="text-sm font-medium text-mendelu-black mb-3">Embedding Model</h3>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div>
          <span class="text-xs text-mendelu-gray-dark uppercase tracking-wider block mb-0.5">Model</span>
          <span class="text-mendelu-black font-mono text-sm">{{ settings?.embedding_model?.name || 'BAAI/bge-large-en-v1.5' }}</span>
        </div>
        <div>
          <span class="text-xs text-mendelu-gray-dark uppercase tracking-wider block mb-0.5">Dimensions</span>
          <span class="text-mendelu-black font-mono text-sm">{{ settings?.embedding_model?.dimensions || 1024 }}</span>
        </div>
        <div>
          <span class="text-xs text-mendelu-gray-dark uppercase tracking-wider block mb-0.5">Distance Metric</span>
          <span class="text-mendelu-black font-mono text-sm">{{ settings?.embedding_model?.distance || 'COSINE' }}</span>
        </div>
        <div>
          <span class="text-xs text-mendelu-gray-dark uppercase tracking-wider block mb-0.5">Status</span>
          <span class="text-mendelu-success text-sm font-medium">Active</span>
        </div>
      </div>
    </div>

    <!-- Qdrant -->
    <div class="card">
      <h3 class="text-sm font-medium text-mendelu-black mb-3">Vector Database (Qdrant)</h3>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div>
          <span class="text-xs text-mendelu-gray-dark uppercase tracking-wider block mb-0.5">Host</span>
          <span class="text-mendelu-black font-mono text-sm">{{ settings?.qdrant?.host || 'localhost' }}</span>
        </div>
        <div>
          <span class="text-xs text-mendelu-gray-dark uppercase tracking-wider block mb-0.5">Port</span>
          <span class="text-mendelu-black font-mono text-sm">{{ settings?.qdrant?.port || 6333 }}</span>
        </div>
        <div>
          <span class="text-xs text-mendelu-gray-dark uppercase tracking-wider block mb-0.5">Embeddings</span>
          <span class="text-mendelu-black font-mono text-sm tabular-nums">{{ embeddingStats?.total_embeddings?.toLocaleString() || '---' }}</span>
        </div>
        <div>
          <span class="text-xs text-mendelu-gray-dark uppercase tracking-wider block mb-0.5">Collection</span>
          <span class="text-mendelu-black font-mono text-sm">{{ embeddingStats?.collection_name || 'climate_data' }}</span>
        </div>
      </div>
    </div>

    <!-- System Resources -->
    <div class="card">
      <h3 class="text-sm font-medium text-mendelu-black mb-3">System Resources</h3>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <span class="text-xs text-mendelu-gray-dark uppercase tracking-wider block mb-2">Disk Usage</span>
          <div v-if="settings?.disk" class="space-y-1">
            <div class="w-full bg-mendelu-gray-semi rounded-full h-2.5">
              <div
                class="bg-mendelu-green h-2.5 rounded-full transition-all duration-300"
                :style="{ width: `${diskPercent}%` }"
              ></div>
            </div>
            <span class="text-xs text-mendelu-gray-dark tabular-nums">
              {{ settings.disk.used_gb }} GB / {{ settings.disk.total_gb }} GB
              ({{ settings.disk.free_gb }} GB free)
            </span>
          </div>
          <span v-else class="text-mendelu-gray-dark text-sm">---</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, reactive, onMounted } from 'vue'
import PageHeader from '../components/PageHeader.vue'

const settings = ref(null)
const embeddingStats = ref(null)
const loading = ref(false)
const saving = ref(false)
const savingCredentials = ref(false)

const editableSettings = reactive({
  model: 'google/gemini-2.0-flash-001',
  temperature: 0.3,
  top_k: 5,
  batch_size: 100
})

const originalSettings = ref({})
const hasChanges = computed(() => {
  return editableSettings.model !== originalSettings.value.model ||
    editableSettings.temperature !== originalSettings.value.temperature ||
    editableSettings.top_k !== originalSettings.value.top_k ||
    editableSettings.batch_size !== originalSettings.value.batch_size
})

const credentials = ref({})
const credentialEdits = reactive({
  openrouter_api_key: '', groq_api_key: '', cds_api_key: '',
  nasa_earthdata_user: '', nasa_earthdata_password: '',
  cmems_username: '', cmems_password: ''
})
const credentialFields = reactive([
  { key: 'openrouter_api_key', label: 'OpenRouter API Key', visible: false },
  { key: 'groq_api_key', label: 'Groq API Key', visible: false },
  { key: 'cds_api_key', label: 'CDS API Key (Copernicus)', visible: false },
  { key: 'nasa_earthdata_user', label: 'NASA Earthdata Username', visible: true },
  { key: 'nasa_earthdata_password', label: 'NASA Earthdata Password', visible: false },
  { key: 'cmems_username', label: 'CMEMS Username', visible: true },
  { key: 'cmems_password', label: 'CMEMS Password', visible: false }
])
const hasCredentialChanges = computed(() => {
  return Object.values(credentialEdits).some(v => v !== '')
})

const diskPercent = computed(() => {
  if (!settings.value?.disk) return 0
  return Math.round((settings.value.disk.used_gb / settings.value.disk.total_gb) * 100)
})

async function refreshSettings() {
  loading.value = true
  try {
    const [sysResp, embResp, credResp] = await Promise.all([
      fetch('/settings/system'), fetch('/embeddings/stats'), fetch('/settings/credentials'),
    ])
    if (sysResp.ok) {
      settings.value = await sysResp.json()
      if (settings.value.llm) {
        editableSettings.model = settings.value.llm.model || editableSettings.model
        editableSettings.temperature = settings.value.llm.temperature ?? editableSettings.temperature
        editableSettings.top_k = settings.value.llm.top_k ?? editableSettings.top_k
        editableSettings.batch_size = settings.value.llm.batch_size ?? editableSettings.batch_size
      }
      originalSettings.value = { ...editableSettings }
    }
    if (embResp.ok) embeddingStats.value = await embResp.json()
    if (credResp.ok) {
      credentials.value = await credResp.json()
      Object.keys(credentialEdits).forEach(k => credentialEdits[k] = '')
    }
  } catch (e) {
    console.error('Failed to load settings:', e)
  } finally {
    loading.value = false
  }
}

async function saveSettings() {
  saving.value = true
  try {
    const resp = await fetch('/settings/system', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: editableSettings.model, temperature: editableSettings.temperature,
        top_k: editableSettings.top_k, batch_size: editableSettings.batch_size
      })
    })
    if (resp.ok) {
      originalSettings.value = { ...editableSettings }
    } else {
      const err = await resp.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(err.detail || `HTTP ${resp.status}`)
    }
  } catch (e) {
    console.error('Failed to save settings:', e)
    alert(`Error: ${e.message}`)
  } finally {
    saving.value = false
  }
}

async function saveCredentials() {
  savingCredentials.value = true
  try {
    const payload = {}
    Object.entries(credentialEdits).forEach(([k, v]) => { if (v !== '') payload[k] = v })
    if (Object.keys(payload).length === 0) return
    const resp = await fetch('/settings/credentials', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    if (resp.ok) {
      await refreshSettings()
    } else {
      const err = await resp.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(err.detail || `HTTP ${resp.status}`)
    }
  } catch (e) {
    console.error('Failed to save credentials:', e)
    alert(`Error: ${e.message}`)
  } finally {
    savingCredentials.value = false
  }
}

onMounted(() => { refreshSettings() })
</script>
