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
        <span v-if="saveSuccess" class="text-xs text-mendelu-success font-medium">Saved!</span>
        <button v-if="hasChanges" @click="saveSettings" :disabled="saving" class="btn-primary !py-1.5 !text-xs disabled:opacity-50">
          {{ saving ? 'Saving...' : 'Save Changes' }}
        </button>
      </div>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label for="settings-openrouter-key" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">OpenRouter API Key</label>
          <div class="relative">
            <input
              id="settings-openrouter-key"
              :type="revealedKeys.openrouter_api_key ? 'text' : 'password'"
              v-model="credentialEdits.openrouter_api_key"
              :placeholder="credentials.openrouter_api_key?.configured ? credentials.openrouter_api_key?.masked : 'sk-or-v1-...'"
              class="input-field pr-14"
            />
            <button
              v-if="credentials.openrouter_api_key?.configured || credentialEdits.openrouter_api_key"
              type="button"
              @click="toggleReveal('openrouter_api_key')"
              class="absolute right-2 top-1/2 -translate-y-1/2 btn-ghost !px-2 !py-0.5 text-xs"
            >
              {{ revealedKeys.openrouter_api_key ? 'Hide' : 'Show' }}
            </button>
          </div>
          <div class="flex items-center gap-1.5 mt-1">
            <span class="w-1.5 h-1.5 rounded-full" :class="credentials.openrouter_api_key?.configured ? 'bg-mendelu-success' : 'bg-mendelu-gray-semi'"></span>
            <span class="text-[10px] text-mendelu-gray-dark">{{ credentials.openrouter_api_key?.configured ? 'Configured' : 'Not configured' }}</span>
          </div>
        </div>
        <div>
          <label for="settings-llm-model" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">LLM Model</label>
          <div class="flex gap-2">
            <input id="settings-llm-model" type="text" v-model="editableSettings.model" placeholder="anthropic/claude-sonnet-4.6" class="input-field flex-1" />
            <button @click="testConnection" :disabled="testingConnection" class="btn-secondary !py-1.5 !text-xs whitespace-nowrap disabled:opacity-50">
              {{ testingConnection ? 'Testing...' : connectionStatus === 'ok' ? '✓ OK' : connectionStatus === 'fail' ? '✗ Fail' : 'Test' }}
            </button>
          </div>
          <div class="flex flex-wrap gap-1 mt-2">
            <button v-for="m in quickModels" :key="m.id" @click="editableSettings.model = m.id"
              class="text-[10px] px-2 py-0.5 rounded-full transition-colors duration-150"
              :class="editableSettings.model === m.id ? 'bg-mendelu-green text-white' : 'bg-mendelu-gray-light text-mendelu-gray-dark hover:bg-mendelu-gray-semi'">
              {{ m.label }}
            </button>
          </div>
        </div>
        <div>
          <label for="settings-temperature" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Temperature: {{ editableSettings.temperature }}</label>
          <input id="settings-temperature" type="range" v-model.number="editableSettings.temperature" min="0" max="1" step="0.1" class="w-full h-2 bg-mendelu-gray-semi rounded-lg appearance-none cursor-pointer accent-mendelu-green" />
          <div class="flex justify-between text-[10px] text-mendelu-gray-dark mt-1">
            <span>Precise (0)</span>
            <span>Creative (1)</span>
          </div>
        </div>
        <div>
          <label for="settings-top-k" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Top-K Results</label>
          <input id="settings-top-k" type="number" v-model.number="editableSettings.top_k" min="1" max="50" class="input-field" />
        </div>
        <div>
          <label for="settings-reranker" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Cross-encoder reranker</label>
          <div class="flex items-start gap-2 mt-2">
            <input id="settings-reranker" type="checkbox" v-model="editableSettings.use_reranker" class="w-4 h-4 accent-mendelu-green mt-0.5" />
            <div class="text-sm">
              <div class="text-mendelu-black">Re-rank retrieved chunks with <span class="font-mono text-xs">BAAI/bge-reranker-v2-m3</span></div>
              <p class="text-xs text-mendelu-gray-dark mt-1 leading-snug">
                <span class="font-medium text-mendelu-alert">Off by default — it is slower.</span>
                Turning it on adds roughly 2–4&nbsp;seconds per query (cross-encoder
                inference runs on top of the vector search). On our golden-query
                evaluation (see <span class="font-mono">docs/rag_eval_v2_*.md</span>) the no-reranker
                pipeline already matches or beats the reranked one on Faithfulness
                (95–100%) and Answer Correctness (78–85%), so the trade-off rarely pays off.
                Enable if you notice the top chunks look noisy on a specific query.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Portal Adapters & Credentials -->
    <div class="card">
      <div class="flex items-center justify-between mb-4">
        <div>
          <h3 class="text-sm font-medium text-mendelu-black">Portal Adapters & Credentials</h3>
          <p class="text-[10px] text-mendelu-gray-dark mt-0.5">Phase 3 data portal configurations</p>
        </div>
        <div class="flex items-center gap-2">
          <span class="text-[10px] px-1.5 py-0.5 rounded-full bg-mendelu-gray-light text-mendelu-gray-dark" style="font-family: var(--font-mono);">
            {{ configuredAdapterCount }}/{{ adapterGroups.length }} configured
          </span>
          <button
            @click="showAdapterForm = !showAdapterForm"
            class="btn-ghost !py-1.5 !text-xs"
            :class="showAdapterForm ? 'text-mendelu-gray-dark' : 'text-mendelu-green'"
          >
            {{ showAdapterForm ? 'Cancel' : '+ Add adapter' }}
          </button>
          <button v-if="hasCredentialChanges" @click="saveCredentials" :disabled="savingCredentials" class="btn-primary !py-1.5 !text-xs disabled:opacity-50">
            {{ savingCredentials ? 'Saving...' : 'Save Credentials' }}
          </button>
        </div>
      </div>

      <!-- Add-adapter form. Collapsed by default. When expanded, lets the
           operator register a new portal config without a code deploy — useful
           for a portal whose Python adapter isn't yet shipped but whose data
           you still want to collect credentials for. -->
      <div v-if="showAdapterForm" class="mb-4 border border-mendelu-green/30 bg-mendelu-green/[0.03] rounded-xl p-4 space-y-3">
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label class="block text-[10px] font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Name *</label>
            <input v-model="adapterDraft.name" type="text" placeholder="e.g. CEDA CMIP6" class="input-field !py-1.5 text-xs" />
          </div>
          <div>
            <label class="block text-[10px] font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Datasets (hint)</label>
            <input v-model="adapterDraft.datasets" type="text" placeholder="e.g. CMIP6, CORDEX" class="input-field !py-1.5 text-xs" />
          </div>
        </div>
        <div>
          <label class="block text-[10px] font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Description</label>
          <input v-model="adapterDraft.description" type="text" placeholder="One-liner shown next to the adapter name" class="input-field !py-1.5 text-xs" />
        </div>

        <div>
          <div class="flex items-center justify-between mb-2">
            <span class="text-[10px] font-medium text-mendelu-gray-dark uppercase tracking-wider">Credential fields</span>
            <button @click="addDraftField" type="button" class="btn-ghost !px-2 !py-0.5 text-xs text-mendelu-green">+ Field</button>
          </div>
          <div v-for="(f, idx) in adapterDraft.fields" :key="idx" class="grid grid-cols-12 gap-2 mb-2 items-start">
            <input v-model="f.key" type="text" placeholder="key (e.g. ceda_token)" class="input-field !py-1.5 text-xs col-span-3" />
            <input v-model="f.label" type="text" placeholder="Display label" class="input-field !py-1.5 text-xs col-span-4" />
            <input v-model="f.hint" type="text" placeholder="Hint (optional)" class="input-field !py-1.5 text-xs col-span-4" />
            <button
              v-if="adapterDraft.fields.length > 1"
              @click="removeDraftField(idx)"
              type="button"
              class="btn-ghost !px-2 !py-1 text-xs text-mendelu-alert col-span-1"
              title="Remove field"
            >×</button>
          </div>
          <p class="text-[10px] text-mendelu-gray-dark/70">
            Leave this list empty for a public (no-auth) adapter. Keys are normalised to lowercase/underscore.
          </p>
        </div>

        <div class="flex justify-end gap-2 pt-2">
          <button @click="resetAdapterDraft" type="button" class="btn-ghost text-xs">Cancel</button>
          <button
            @click="submitAdapter"
            :disabled="adapterSubmitting || !adapterDraft.name.trim()"
            class="btn-primary !py-1.5 !text-xs disabled:opacity-50"
          >{{ adapterSubmitting ? 'Saving…' : 'Add adapter' }}</button>
        </div>
      </div>

      <div class="space-y-2">
        <div v-for="adapter in adapterGroups" :key="adapter.id" class="border rounded-xl overflow-hidden transition-colors duration-150" :class="adapter.expanded ? 'border-mendelu-green/30 bg-mendelu-green/[0.02]' : 'border-mendelu-gray-semi/60'">
          <!-- Adapter header (clickable) -->
          <button
            @click="adapter.expanded = !adapter.expanded"
            class="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-mendelu-gray-light/50 transition-colors duration-150"
          >
            <!-- Status dot -->
            <span class="w-2.5 h-2.5 rounded-full flex-shrink-0" :class="adapterConfigured(adapter) ? 'bg-mendelu-success' : adapter.public ? 'bg-blue-400' : 'bg-mendelu-gray-semi'"></span>

            <!-- Name + description -->
            <div class="flex-1 min-w-0">
              <div class="flex items-center gap-2">
                <span class="text-sm font-medium text-mendelu-black">{{ adapter.name }}</span>
                <span v-if="adapter.public" class="text-[9px] px-1.5 py-0.5 rounded bg-blue-50 text-blue-500 font-medium uppercase tracking-wider" style="font-family: var(--font-mono);">Public</span>
                <span v-else-if="adapterConfigured(adapter)" class="text-[9px] px-1.5 py-0.5 rounded bg-mendelu-success/10 text-mendelu-success font-medium uppercase tracking-wider" style="font-family: var(--font-mono);">Ready</span>
                <span v-else class="text-[9px] px-1.5 py-0.5 rounded bg-amber-50 text-amber-500 font-medium uppercase tracking-wider" style="font-family: var(--font-mono);">Needs Setup</span>
              </div>
              <p class="text-[10px] text-mendelu-gray-dark mt-0.5">{{ adapter.description }}</p>
            </div>

            <!-- Datasets badge -->
            <span v-if="adapter.datasets" class="text-[10px] text-mendelu-gray-dark hidden sm:block">{{ adapter.datasets }}</span>

            <!-- Custom badge -->
            <span v-if="!adapter.builtin" class="text-[9px] px-1.5 py-0.5 rounded bg-mendelu-green/10 text-mendelu-green font-medium uppercase tracking-wider" style="font-family: var(--font-mono);">Custom</span>

            <!-- Chevron -->
            <svg class="w-4 h-4 text-mendelu-gray-dark transition-transform duration-150 flex-shrink-0" :class="{ 'rotate-180': adapter.expanded }" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          <!-- Expanded content -->
          <div v-if="adapter.expanded" class="px-4 pb-4 pt-1 border-t border-mendelu-gray-semi/40">
            <!-- Public adapters - no credentials needed -->
            <div v-if="adapter.public" class="flex items-center gap-2 py-2">
              <svg class="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span class="text-xs text-mendelu-gray-dark">No credentials required — publicly accessible data.</span>
            </div>

            <!-- Credential fields -->
            <div v-else class="space-y-3 pt-2">
              <div v-for="field in adapter.fields" :key="field.key">
                <div class="flex items-center gap-2 mb-1">
                  <span class="w-1.5 h-1.5 rounded-full" :class="credentials[field.key]?.configured ? 'bg-mendelu-success' : 'bg-mendelu-gray-semi'"></span>
                  <label :for="`settings-cred-${field.key}`" class="text-xs font-medium text-mendelu-gray-dark">{{ field.label }}</label>
                  <span v-if="credentials[field.key]?.configured" class="text-[9px] text-mendelu-success font-medium ml-auto" style="font-family: var(--font-mono);">configured</span>
                </div>
                <div class="relative">
                  <input
                    :id="`settings-cred-${field.key}`"
                    :type="revealedKeys[field.key] ? 'text' : 'password'"
                    v-model="credentialEdits[field.key]"
                    :placeholder="credentials[field.key]?.configured ? credentials[field.key]?.masked : 'Not configured'"
                    class="input-field pr-14"
                  />
                  <button
                    v-if="credentials[field.key]?.configured || credentialEdits[field.key]"
                    type="button"
                    @click="toggleReveal(field.key)"
                    class="absolute right-2 top-1/2 -translate-y-1/2 btn-ghost !px-2 !py-0.5 text-xs"
                  >
                    {{ revealedKeys[field.key] ? 'Hide' : 'Show' }}
                  </button>
                </div>
                <p v-if="field.hint" class="text-[10px] text-mendelu-gray-dark/60 mt-1">{{ field.hint }}</p>
              </div>
            </div>

            <!-- Custom-adapter controls: only shown on non-builtin entries.
                 The built-in adapters are immutable from the UI. -->
            <div v-if="!adapter.builtin" class="flex justify-end pt-3 mt-2 border-t border-mendelu-gray-semi/30">
              <button
                type="button"
                @click="removeAdapter(adapter)"
                class="btn-ghost !py-1 !px-2 text-xs text-mendelu-alert hover:underline"
              >Remove adapter</button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- LLM Provider -->
    <div class="card">
      <h3 class="text-sm font-medium text-mendelu-black mb-3">LLM Provider</h3>
      <div class="bg-mendelu-gray-light rounded-lg p-4">
        <div class="flex items-center justify-between mb-2">
          <span class="text-mendelu-black font-medium text-sm">OpenRouter</span>
          <span class="w-2.5 h-2.5 rounded-full" :class="settings?.llm?.providers?.openrouter ? 'bg-mendelu-success' : 'bg-mendelu-gray-semi'"></span>
        </div>
        <div class="grid grid-cols-2 gap-4 mt-3">
          <div>
            <span class="text-xs text-mendelu-gray-dark uppercase tracking-wider block mb-0.5">Main Model</span>
            <span class="text-mendelu-black font-mono text-sm">{{ settings?.llm?.model || '—' }}</span>
          </div>
          <div>
            <span class="text-xs text-mendelu-gray-dark uppercase tracking-wider block mb-0.5">Fast Model (Query Expansion)</span>
            <span class="text-mendelu-black font-mono text-sm">{{ settings?.llm?.fast_model || '—' }}</span>
          </div>
        </div>
        <p class="text-xs text-mendelu-gray-dark mt-2">{{ settings?.llm?.providers?.openrouter ? 'API key configured' : 'No API key set — add OPENROUTER_API_KEY to .env' }}</p>
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

<script setup lang="ts">
import { ref, computed, reactive, onMounted } from 'vue'
import PageHeader from '../components/PageHeader.vue'
import { apiFetch } from '../api'
import { useToast } from '../composables/useToast'

const toast = useToast()

const settings = ref(null)
const embeddingStats = ref(null)
const loading = ref(false)
const saving = ref(false)
const savingCredentials = ref(false)
const saveSuccess = ref(false)

const testingConnection = ref(false)
const connectionStatus = ref('')  // '', 'ok', 'fail'

const quickModels = [
  { id: 'anthropic/claude-sonnet-4.6', label: 'Sonnet 4.6' },
  { id: 'anthropic/claude-opus-4-6', label: 'Opus 4.6' },
  { id: 'anthropic/claude-haiku-4-5', label: 'Haiku 4.5' },
  { id: 'x-ai/grok-4.1-fast', label: 'Grok 4.1' },
  { id: 'openai/gpt-4o-mini', label: 'GPT-4o Mini' },
]

interface EditableSettings {
  model: string
  temperature: number
  top_k: number
  use_reranker: boolean
  batch_size?: number
}

const editableSettings = reactive<EditableSettings>({
  model: '',
  temperature: 0.1,
  top_k: 10,
  use_reranker: false,
})

async function testConnection() {
  testingConnection.value = true
  connectionStatus.value = ''
  // Use a 30s abort controller so the test button can't hang indefinitely
  // if the LLM provider is down. 30s covers our observed p99 (~22s end-to-end).
  const controller = new AbortController()
  const abortTimer = setTimeout(() => controller.abort(), 30_000)
  try {
    const resp = await apiFetch('/rag/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: 'test connection', top_k: 1, use_llm: true, use_reranker: false }),
      signal: controller.signal,
    })
    connectionStatus.value = resp.ok ? 'ok' : 'fail'
  } catch {
    connectionStatus.value = 'fail'
  } finally {
    clearTimeout(abortTimer)
    testingConnection.value = false
    setTimeout(() => { connectionStatus.value = '' }, 5000)
  }
}

const originalSettings = ref<any>({})
const hasChanges = computed(() => {
  return editableSettings.model !== originalSettings.value.model ||
    editableSettings.temperature !== originalSettings.value.temperature ||
    editableSettings.top_k !== originalSettings.value.top_k ||
    editableSettings.use_reranker !== originalSettings.value.use_reranker
})

const credentials = ref<Record<string, { configured?: boolean; masked?: string }>>({})
const revealedKeys = reactive<Record<string, boolean>>({})

async function toggleReveal(key) {
  if (revealedKeys[key]) {
    // Hide — clear the fetched value, restore to empty edit
    revealedKeys[key] = false
    credentialEdits[key] = ''
    return
  }
  // If user already typed something, just toggle visibility
  if (credentialEdits[key]) {
    revealedKeys[key] = true
    return
  }
  // Fetch full value from backend
  try {
    const resp = await apiFetch(`/settings/credentials/${key}`)
    if (resp.ok) {
      const data = await resp.json()
      credentialEdits[key] = data.value
      revealedKeys[key] = true
    }
  } catch (e) {
    console.error('Failed to reveal credential:', e)
  }
}

const credentialEdits = reactive({
  openrouter_api_key: '', cds_api_key: '',
  nasa_earthdata_user: '', nasa_earthdata_password: '',
  cmems_username: '', cmems_password: ''
})

// Adapters grouped by portal with their credential fields
const builtinAdapters = reactive([
  {
    id: 'cds', builtin: true,
    name: 'Copernicus CDS',
    description: 'Climate Data Store — ERA5, CERRA, seasonal forecasts',
    datasets: 'ERA5, CERRA, SEAS5',
    public: false,
    expanded: false,
    fields: [
      { key: 'cds_api_key', label: 'Personal Access Token', hint: 'CDS v2024+ format — get at cds.climate.copernicus.eu/profile' },
    ],
  },
  {
    id: 'nasa', builtin: true,
    name: 'NASA Earthdata',
    description: 'GES DISC, PO.DAAC — MERRA-2, GPM, sea level data',
    datasets: 'MERRA-2, GPM, GRACE',
    public: false,
    expanded: false,
    fields: [
      { key: 'nasa_earthdata_user', label: 'Username', hint: null },
      { key: 'nasa_earthdata_password', label: 'Password', hint: 'Register at urs.earthdata.nasa.gov' },
    ],
  },
  {
    id: 'cmems', builtin: true,
    name: 'Marine Copernicus (CMEMS)',
    description: 'Copernicus Marine — ocean temperature, sea level, ice',
    datasets: 'SST, sea level, sea ice',
    public: false,
    expanded: false,
    fields: [
      { key: 'cmems_username', label: 'Username', hint: null },
      { key: 'cmems_password', label: 'Password', hint: 'Register at data.marine.copernicus.eu' },
    ],
  },
  {
    id: 'esgf', builtin: true,
    name: 'ESGF',
    description: 'Earth System Grid Federation — CMIP6, CORDEX projections',
    datasets: 'CMIP6, CORDEX',
    public: true,
    expanded: false,
    fields: [],
  },
  {
    id: 'noaa', builtin: true,
    name: 'NOAA',
    description: 'PSL, NCEI — reanalysis, observational datasets',
    datasets: '20CRv3, NCEP/NCAR',
    public: true,
    expanded: false,
    fields: [],
  },
  {
    id: 'eidc', builtin: true,
    name: 'EIDC / CEDA',
    description: 'Hydro-JULES, UK environmental data via OpenDAP',
    datasets: 'Hydro-JULES, CHESS',
    public: true,
    expanded: false,
    fields: [],
  },
])

// Custom adapters registered through POST /settings/adapters. Fetched on
// mount and re-fetched whenever the user adds/removes one.
const customAdapters = ref<any[]>([])

// Unified list rendered by the template. Built-in first, then custom.
const adapterGroups = computed(() => [...builtinAdapters, ...customAdapters.value])

function adapterConfigured(adapter) {
  if (adapter.public) return true
  return adapter.fields.every(f => credentials.value[f.key]?.configured)
}

const configuredAdapterCount = computed(() => {
  return adapterGroups.value.filter(a => adapterConfigured(a)).length
})

// "Add adapter" form state.
const showAdapterForm = ref(false)
const adapterDraft = ref({
  name: '',
  description: '',
  datasets: '',
  fields: [{ key: '', label: '', hint: '' }],
})
const adapterSubmitting = ref(false)

function addDraftField() {
  adapterDraft.value.fields.push({ key: '', label: '', hint: '' })
}
function removeDraftField(idx: number) {
  if (adapterDraft.value.fields.length > 1) adapterDraft.value.fields.splice(idx, 1)
}
function resetAdapterDraft() {
  adapterDraft.value = { name: '', description: '', datasets: '', fields: [{ key: '', label: '', hint: '' }] }
  showAdapterForm.value = false
}

async function loadCustomAdapters() {
  try {
    const resp = await apiFetch('/settings/adapters')
    if (!resp.ok) return
    const data = await resp.json()
    // Keep the same shape as builtinAdapters so the template needs no branching.
    customAdapters.value = (data.adapters || []).map((a: any) => ({
      ...a,
      builtin: false,
      expanded: false,
    }))
  } catch (e) {
    console.error('Failed to load custom adapters:', e)
  }
}

async function submitAdapter() {
  const draft = adapterDraft.value
  if (!draft.name.trim()) {
    toast.error('Adapter name is required')
    return
  }
  const validFields = draft.fields.filter(f => f.key.trim() && f.label.trim())
  // An adapter with zero fields would be a public one — if the user actually
  // wanted public, they can still submit an empty field list.
  const payload = {
    name: draft.name.trim(),
    description: draft.description.trim(),
    datasets: draft.datasets.trim(),
    fields: validFields.map(f => ({ key: f.key.trim(), label: f.label.trim(), hint: f.hint?.trim() || null })),
  }
  adapterSubmitting.value = true
  try {
    const resp = await apiFetch('/settings/adapters', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    const body = await resp.json().catch(() => ({}))
    if (!resp.ok) {
      const msg = Array.isArray(body.detail)
        ? body.detail.map((d: any) => d.msg || JSON.stringify(d)).join('; ')
        : (body.detail || `HTTP ${resp.status}`)
      throw new Error(msg)
    }
    toast.success(`Adapter "${body.name}" added`)
    resetAdapterDraft()
    await Promise.all([loadCustomAdapters(), refreshSettings()])
  } catch (e: any) {
    toast.error(`Could not add adapter: ${e?.message || e}`)
  } finally {
    adapterSubmitting.value = false
  }
}

async function removeAdapter(adapter: any) {
  if (!adapter?.id || adapter.builtin) return
  if (!window.confirm(`Remove adapter "${adapter.name}"? Stored credentials for its fields stay behind and can be cleared separately.`)) return
  try {
    const resp = await apiFetch(`/settings/adapters/${adapter.id}`, { method: 'DELETE' })
    if (!resp.ok && resp.status !== 204) {
      const err = await resp.json().catch(() => ({}))
      throw new Error(err.detail || `HTTP ${resp.status}`)
    }
    toast.success(`Removed "${adapter.name}"`)
    await Promise.all([loadCustomAdapters(), refreshSettings()])
  } catch (e: any) {
    toast.error(`Delete failed: ${e?.message || e}`)
  }
}

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
      apiFetch('/settings/system'), apiFetch('/embeddings/stats'), apiFetch('/settings/credentials'),
    ])
    if (sysResp.ok) {
      settings.value = await sysResp.json()
      if (settings.value.llm) {
        editableSettings.model = settings.value.llm.model || 'anthropic/claude-sonnet-4.6'
        editableSettings.temperature = settings.value.llm.temperature ?? 0.1
        editableSettings.top_k = settings.value.llm.top_k ?? 10
        editableSettings.use_reranker = settings.value.llm.use_reranker ?? false
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
    const resp = await apiFetch('/settings/system', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: editableSettings.model, temperature: editableSettings.temperature,
        top_k: editableSettings.top_k, batch_size: editableSettings.batch_size,
        use_reranker: editableSettings.use_reranker,
      })
    })
    if (resp.ok) {
      originalSettings.value = { ...editableSettings }
      saveSuccess.value = true
      setTimeout(() => { saveSuccess.value = false }, 2000)
    } else {
      const err = await resp.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(err.detail || `HTTP ${resp.status}`)
    }
  } catch (e) {
    console.error('Failed to save settings:', e)
    toast.error(`Error: ${e.message}`)
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
    const resp = await apiFetch('/settings/credentials', {
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
    toast.error(`Error: ${e.message}`)
  } finally {
    savingCredentials.value = false
  }
}

onMounted(() => {
  refreshSettings()
  loadCustomAdapters()
})
</script>
