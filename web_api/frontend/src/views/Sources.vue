<template>
  <div class="space-y-6">
    <PageHeader title="Data Sources" subtitle="Manage your climate data sources">
      <template #actions>
        <button @click="loadSources()" :disabled="loading" class="btn-ghost disabled:opacity-50">Refresh</button>
        <button @click="deleteAllSources" class="btn-ghost text-mendelu-alert hover:bg-mendelu-alert/10">Clear All</button>
        <router-link to="/sources/create" class="btn-primary">Add Source</router-link>
      </template>
    </PageHeader>

    <!-- Sources Grid -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      <div
        v-for="source in sources"
        :key="source.name"
        class="card hover:shadow-md hover:border-mendelu-green/30 transition-all duration-150 cursor-pointer"
        @click="viewDetails(source)"
      >
        <div class="flex items-start justify-between mb-3">
          <div class="flex items-center gap-2 min-w-0">
            <span
              class="w-2 h-2 rounded-full flex-shrink-0"
              :class="getFreshnessDot(source)"
              :title="getFreshnessTitle(source)"
            ></span>
            <h3 class="text-sm font-medium text-mendelu-black truncate">{{ source.name }}</h3>
            <span v-if="isCatalogSource(source)" class="badge-neutral">D1.1</span>
            <span v-if="source.schedule" class="badge-info">Scheduled</span>
          </div>
          <div class="flex flex-col gap-1 items-end flex-shrink-0">
            <span :class="source.enabled ? 'badge-success' : 'badge-neutral'">
              {{ source.enabled ? 'Active' : 'Inactive' }}
            </span>
            <span :class="getStatusBadge(source.processing_status)">
              {{ getStatusLabel(source.processing_status) }}
            </span>
          </div>
        </div>

        <p class="text-mendelu-gray-dark text-xs mb-2 line-clamp-2">{{ source.description || 'No description' }}</p>

        <div v-if="source.processing_status === 'failed' && source.error_message"
             class="mb-2 p-2 border-l-2 border-mendelu-alert bg-mendelu-alert/5 rounded text-xs text-mendelu-alert">
          {{ source.error_message }}
        </div>

        <div class="space-y-1 text-xs">
          <div class="flex justify-between">
            <span class="text-mendelu-gray-dark">Type</span>
            <span class="text-mendelu-black">{{ source.type || 'Unknown' }}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-mendelu-gray-dark">Embeddings</span>
            <span class="text-mendelu-black tabular-nums">{{ source.embedding_count?.toLocaleString() || '---' }}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-mendelu-gray-dark">Variables</span>
            <span class="text-mendelu-black tabular-nums">{{ source.variables?.length || '---' }}</span>
          </div>
          <div v-if="source.last_processed" class="flex justify-between">
            <span class="text-mendelu-gray-dark">Last Processed</span>
            <span class="text-mendelu-black">{{ formatDate(source.last_processed) }}</span>
          </div>
        </div>

        <div class="mt-3 pt-3 border-t border-mendelu-gray-semi flex gap-2" @click.stop>
          <button @click="openEditModal(source)" class="btn-ghost flex-1 text-xs">Edit</button>
          <button
            @click="refreshSource(source)"
            :disabled="source.refreshing || source.processing_status === 'processing'"
            class="btn-ghost flex-1 text-xs text-mendelu-green hover:bg-mendelu-green/10 disabled:opacity-50"
          >
            {{ source.refreshing ? 'Triggering...' : 'Reprocess' }}
          </button>
          <button @click="deleteSource(source)" class="btn-ghost text-xs text-mendelu-alert hover:bg-mendelu-alert/10">
            Delete
          </button>
        </div>
      </div>
    </div>

    <!-- Empty State -->
    <div v-if="!loading && sources.length === 0" class="card text-center py-12">
      <h3 class="text-sm font-medium text-mendelu-black mb-2">No Sources Configured</h3>
      <p class="text-mendelu-gray-dark text-xs mb-4">Add your first data source to get started</p>
      <router-link to="/sources/create" class="btn-primary inline-block">Add Source</router-link>
    </div>

    <!-- Source Detail Modal -->
    <div
      v-if="selectedSource"
      class="fixed inset-0 bg-black/40 flex items-center justify-center z-50"
      @click.self="selectedSource = null"
    >
      <div class="bg-white border border-mendelu-gray-semi rounded-xl p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto shadow-lg">
        <div class="flex items-center justify-between mb-6">
          <h2 class="text-lg font-semibold text-mendelu-black">{{ selectedSource.name }}</h2>
          <button @click="selectedSource = null" class="btn-ghost !px-2 !py-1">&times;</button>
        </div>

        <div class="space-y-5">
          <!-- Processing Status -->
          <div>
            <h4 class="text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-2">Processing Status</h4>
            <div class="space-y-2">
              <div class="flex items-center gap-2">
                <span :class="getStatusBadge(selectedSource.processing_status)">
                  {{ getStatusLabel(selectedSource.processing_status) }}
                </span>
                <span v-if="selectedSource.last_processed" class="text-xs text-mendelu-gray-dark">
                  Last: {{ formatDate(selectedSource.last_processed) }}
                </span>
              </div>
              <div v-if="selectedSource.error_message"
                   class="p-3 border-l-2 border-mendelu-alert bg-mendelu-alert/5 rounded text-sm text-mendelu-alert">
                {{ selectedSource.error_message }}
              </div>
            </div>
          </div>

          <div>
            <h4 class="text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-2">Variables</h4>
            <div class="flex flex-wrap gap-2">
              <span v-for="v in selectedSource.variables" :key="v" class="badge-info">{{ v }}</span>
              <span v-if="!selectedSource.variables || selectedSource.variables.length === 0" class="text-mendelu-gray-dark text-sm">
                No variables configured
              </span>
            </div>
          </div>

          <div>
            <h4 class="text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-2">Source Details</h4>
            <div class="bg-mendelu-gray-light p-4 rounded-lg text-sm space-y-2">
              <div class="flex justify-between"><span class="text-mendelu-gray-dark">Source ID</span><span class="text-mendelu-black">{{ selectedSource.source_id }}</span></div>
              <div class="flex justify-between"><span class="text-mendelu-gray-dark">Format</span><span class="text-mendelu-black">{{ selectedSource.type || 'Unknown' }}</span></div>
              <div class="flex justify-between"><span class="text-mendelu-gray-dark">URL</span><span class="text-mendelu-black text-xs break-all">{{ selectedSource.url }}</span></div>
              <div class="flex justify-between"><span class="text-mendelu-gray-dark">Active</span><span class="text-mendelu-black">{{ selectedSource.enabled ? 'Yes' : 'No' }}</span></div>
              <div v-if="selectedSource.auth_method" class="flex justify-between"><span class="text-mendelu-gray-dark">Auth Method</span><span class="text-mendelu-black">{{ selectedSource.auth_method }}</span></div>
              <div v-if="selectedSource.portal" class="flex justify-between"><span class="text-mendelu-gray-dark">Portal</span><span class="text-mendelu-black">{{ selectedSource.portal }}</span></div>
            </div>
          </div>

          <!-- Schedule -->
          <div class="pt-4 border-t border-mendelu-gray-semi">
            <h4 class="text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-2">Schedule</h4>
            <div v-if="selectedSource.schedule" class="bg-mendelu-gray-light p-3 rounded-lg text-xs space-y-1">
              <div class="flex justify-between"><span class="text-mendelu-gray-dark">Cron</span><code class="text-mendelu-black font-mono">{{ selectedSource.schedule.cron_expression }}</code></div>
              <div class="flex justify-between"><span class="text-mendelu-gray-dark">Enabled</span><span :class="selectedSource.schedule.is_enabled ? 'text-mendelu-success' : 'text-mendelu-gray-dark'">{{ selectedSource.schedule.is_enabled ? 'Yes' : 'No' }}</span></div>
              <div v-if="selectedSource.schedule.next_run_at" class="flex justify-between"><span class="text-mendelu-gray-dark">Next run</span><span class="text-mendelu-black">{{ formatDate(selectedSource.schedule.next_run_at) }}</span></div>
            </div>
            <p v-else class="text-xs text-mendelu-gray-dark">No schedule configured</p>
          </div>

          <!-- Processing History -->
          <div class="pt-4 border-t border-mendelu-gray-semi">
            <ProcessingHistory :source-id="selectedSource.source_id" />
          </div>

          <!-- Connection Test -->
          <div class="pt-4 border-t border-mendelu-gray-semi">
            <button
              @click="testConnection(selectedSource)"
              :disabled="testingConnection"
              class="btn-secondary w-full disabled:opacity-50"
            >
              {{ testingConnection ? 'Testing...' : 'Test Connection' }}
            </button>
            <div v-if="connectionResult" class="mt-2 p-3 rounded-lg text-xs" :class="connectionResult.reachable ? 'bg-mendelu-success/10 text-mendelu-success' : 'bg-mendelu-alert/10 text-mendelu-alert'">
              <p><strong>{{ connectionResult.reachable ? 'Reachable' : 'Unreachable' }}</strong></p>
              <p v-if="connectionResult.latency_ms">Latency: {{ connectionResult.latency_ms }}ms</p>
              <p v-if="connectionResult.content_type">Content-Type: {{ connectionResult.content_type }}</p>
              <p v-if="connectionResult.error">Error: {{ connectionResult.error }}</p>
            </div>
          </div>

          <div class="pt-4 border-t border-mendelu-gray-semi">
            <h4 class="text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-3">Danger Zone</h4>
            <div class="space-y-2">
              <button @click="deleteSourceEmbeddings(selectedSource)" class="btn-ghost w-full text-mendelu-alert hover:bg-mendelu-alert/10 border border-mendelu-alert/20">
                Delete Embeddings for This Source
              </button>
              <button @click="deleteSource(selectedSource)" class="btn-danger w-full">
                Delete Source
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Edit Modal -->
    <div
      v-if="editingSource"
      class="fixed inset-0 bg-black/40 flex items-center justify-center z-50"
      @click.self="editingSource = null"
    >
      <div class="bg-white border border-mendelu-gray-semi rounded-xl p-6 max-w-lg w-full mx-4 shadow-lg">
        <div class="flex items-center justify-between mb-6">
          <h2 class="text-lg font-semibold text-mendelu-black">Edit Source</h2>
          <button @click="editingSource = null" class="btn-ghost !px-2 !py-1">&times;</button>
        </div>

        <form @submit.prevent="saveEdit" class="space-y-4">
          <div>
            <label class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Source Name</label>
            <input v-model="editForm.name" type="text" class="input-field" disabled />
          </div>
          <div>
            <label class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">URL</label>
            <input v-model="editForm.url" type="text" class="input-field" />
          </div>
          <div>
            <label class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Format</label>
            <select v-model="editForm.format" class="input-field">
              <option value="netcdf">NetCDF</option>
              <option value="csv">CSV</option>
              <option value="api">REST API</option>
              <option value="geotiff">GeoTIFF</option>
            </select>
          </div>
          <div>
            <label class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Description</label>
            <textarea v-model="editForm.description" rows="2" class="input-field resize-none"></textarea>
          </div>
          <div class="flex items-center gap-3">
            <label class="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" v-model="editForm.is_active" class="sr-only peer">
              <div class="w-9 h-5 bg-mendelu-gray-semi peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-mendelu-green"></div>
            </label>
            <span class="text-sm text-mendelu-black">Active</span>
          </div>

          <!-- Auth fields -->
          <div class="p-4 bg-mendelu-gray-light rounded-lg space-y-3">
            <h4 class="text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider">Authentication</h4>
            <div>
              <label class="block text-xs font-medium text-mendelu-gray-dark mb-1">Portal Preset</label>
              <select v-model="editForm.portal" @change="onEditPortalChange" class="input-field">
                <option value="">Custom / None</option>
                <option value="CDS">Copernicus CDS</option>
                <option value="NASA">NASA Earthdata</option>
                <option value="MARINE">Marine Copernicus</option>
                <option value="ESGF">ESGF (CMIP6/CORDEX)</option>
                <option value="NOAA">NOAA PSL</option>
              </select>
              <p v-if="editForm.portal" class="mt-1 text-xs text-mendelu-green">Uses global credentials from Settings</p>
            </div>
            <div>
              <label class="block text-xs font-medium text-mendelu-gray-dark mb-1">Auth Method</label>
              <select v-model="editForm.auth_method" class="input-field">
                <option value="none">None (Open Access)</option>
                <option value="api_key">API Key</option>
                <option value="bearer_token">Bearer Token</option>
                <option value="basic">Username &amp; Password</option>
              </select>
            </div>
            <div v-if="editForm.auth_method === 'api_key'">
              <label class="block text-xs font-medium text-mendelu-gray-dark mb-1">API Key</label>
              <input type="password" v-model="editForm.auth_api_key" class="input-field" placeholder="Leave empty to keep current" />
            </div>
            <div v-if="editForm.auth_method === 'bearer_token'">
              <label class="block text-xs font-medium text-mendelu-gray-dark mb-1">Bearer Token</label>
              <input type="password" v-model="editForm.auth_token" class="input-field" placeholder="Leave empty to keep current" />
            </div>
            <div v-if="editForm.auth_method === 'basic'" class="space-y-2">
              <div>
                <label class="block text-xs font-medium text-mendelu-gray-dark mb-1">Username</label>
                <input type="text" v-model="editForm.auth_username" class="input-field" placeholder="Leave empty to keep current" />
              </div>
              <div>
                <label class="block text-xs font-medium text-mendelu-gray-dark mb-1">Password</label>
                <input type="password" v-model="editForm.auth_password" class="input-field" placeholder="Leave empty to keep current" />
              </div>
            </div>
          </div>

          <div class="flex gap-3 pt-2">
            <button type="submit" :disabled="saving" class="btn-primary flex-1 disabled:opacity-50">
              {{ saving ? 'Saving...' : 'Save Changes' }}
            </button>
            <button type="button" @click="editingSource = null" class="btn-secondary flex-1">Cancel</button>
          </div>
        </form>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import PageHeader from '../components/PageHeader.vue'
import ProcessingHistory from '../components/ProcessingHistory.vue'

const sources = ref([])
const loading = ref(true)
const selectedSource = ref(null)
const editingSource = ref(null)
const saving = ref(false)
const testingConnection = ref(false)
const connectionResult = ref(null)
const editForm = ref({ name: '', url: '', format: '', description: '', is_active: true, auth_method: 'none', auth_api_key: '', auth_token: '', auth_username: '', auth_password: '', portal: '' })
let statusPollInterval = null

async function loadSources() {
  loading.value = true
  try {
    const resp = await fetch('/sources?active_only=false')
    if (!resp.ok) throw new Error(`Failed to load sources: ${resp.statusText}`)
    const data = await resp.json()
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
      auth_method: source.auth_method || null,
      portal: source.portal || null,
      schedule: source.schedule || null,
      refreshing: false
    }))
  } catch (e) {
    console.error('Failed to load sources:', e)
  } finally {
    loading.value = false
  }
}

const PORTAL_AUTH_MAP = { CDS: 'api_key', NASA: 'bearer_token', MARINE: 'basic', ESGF: 'api_key', NOAA: 'none' }

function onEditPortalChange() {
  if (editForm.value.portal && PORTAL_AUTH_MAP[editForm.value.portal]) {
    editForm.value.auth_method = PORTAL_AUTH_MAP[editForm.value.portal]
  }
}

function openEditModal(source) {
  editForm.value = {
    name: source.source_id, url: source.url || '', format: source.type || 'netcdf',
    description: source.description || '', is_active: source.enabled,
    auth_method: source.auth_method || 'none', auth_api_key: '', auth_token: '',
    auth_username: '', auth_password: '', portal: source.portal || ''
  }
  editingSource.value = source
}

async function saveEdit() {
  saving.value = true
  try {
    let authCredentials = null
    if (editForm.value.auth_method === 'api_key' && editForm.value.auth_api_key) {
      authCredentials = { api_key: editForm.value.auth_api_key }
    } else if (editForm.value.auth_method === 'bearer_token' && editForm.value.auth_token) {
      authCredentials = { token: editForm.value.auth_token }
    } else if (editForm.value.auth_method === 'basic' && editForm.value.auth_username) {
      authCredentials = { username: editForm.value.auth_username, password: editForm.value.auth_password }
    }

    const payload = {
      url: editForm.value.url, format: editForm.value.format, description: editForm.value.description,
      is_active: editForm.value.is_active,
      auth_method: editForm.value.auth_method !== 'none' ? editForm.value.auth_method : null,
      portal: editForm.value.portal || null
    }
    if (authCredentials) payload.auth_credentials = authCredentials

    const resp = await fetch(`/sources/${editForm.value.name}`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(err.detail || `HTTP ${resp.status}`)
    }
    editingSource.value = null
    await loadSources()
  } catch (e) {
    console.error('Failed to save source:', e)
    alert(`Error: ${e.message}`)
  } finally {
    saving.value = false
  }
}

function isCatalogSource(source) {
  return source.tags && source.tags.some(t => t === 'catalog:D1.1')
}

function getStatusBadge(status) {
  return { 'success': 'badge-success', 'completed': 'badge-success', 'metadata_only': 'badge-warning', 'failed': 'badge-danger', 'error': 'badge-danger', 'processing': 'badge-info', 'pending': 'badge-neutral' }[status] || 'badge-neutral'
}

function getStatusLabel(status) {
  return { 'success': 'Success', 'completed': 'Completed', 'metadata_only': 'Metadata Only', 'failed': 'Failed', 'error': 'Error', 'processing': 'Processing', 'pending': 'Pending' }[status] || 'Pending'
}

function formatDate(dateString) {
  if (!dateString) return '---'
  try { return new Date(dateString).toLocaleString() } catch { return dateString }
}

async function refreshSource(source) {
  if (source.refreshing || source.processing_status === 'processing') return
  source.refreshing = true
  try {
    const resp = await fetch(`/sources/${source.source_id}/trigger`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }
    })
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(err.detail || `HTTP ${resp.status}`)
    }
    source.processing_status = 'processing'
    source.error_message = null
    setTimeout(() => loadSources(), 2000)
  } catch (e) {
    console.error('Error triggering pipeline:', e)
  } finally {
    source.refreshing = false
  }
}

async function deleteSource(source) {
  if (!confirm(`Delete source "${source.name}"?\n\nThis will permanently delete the source configuration.`)) return
  try {
    const resp = await fetch(`/sources/${source.source_id}`, { method: 'DELETE' })
    if (!resp.ok && resp.status !== 204) {
      const err = await resp.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(err.detail || `HTTP ${resp.status}`)
    }
    await loadSources()
    if (selectedSource.value?.source_id === source.source_id) selectedSource.value = null
  } catch (e) {
    console.error('Error deleting source:', e)
  }
}

async function deleteSourceEmbeddings(source) {
  if (!confirm(`Delete all embeddings for "${source.name}"? This cannot be undone.`)) return
  try {
    const resp = await fetch(`/sources/${source.source_id}/embeddings?confirm=true`, { method: 'DELETE' })
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(err.detail || `HTTP ${resp.status}`)
    }
    await loadSources()
  } catch (e) {
    console.error('Error deleting embeddings:', e)
  }
}

async function deleteAllSources() {
  const deleteEmbeddings = confirm('This will delete ALL sources.\n\nDo you also want to delete embeddings from Qdrant?')
  const confirmMsg = deleteEmbeddings
    ? 'This will permanently delete ALL sources AND their embeddings. Are you sure?'
    : 'This will permanently delete ALL sources. Are you sure?'
  if (confirm(confirmMsg)) {
    try {
      const resp = await fetch(`/sources?confirm=true&delete_embeddings=${deleteEmbeddings}`, { method: 'DELETE' })
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(err.detail || `HTTP ${resp.status}`)
      }
      await loadSources()
    } catch (e) {
      console.error('Error deleting sources:', e)
    }
  }
}

function getFreshnessDot(source) {
  if (source.processing_status === 'processing') return 'bg-mendelu-green animate-pulse'
  if (source.processing_status === 'failed') return 'bg-mendelu-alert'
  if (!source.last_processed) return 'bg-mendelu-gray-dark'
  const daysSince = (Date.now() - new Date(source.last_processed).getTime()) / (1000 * 60 * 60 * 24)
  if (daysSince < 7) return 'bg-mendelu-success'
  if (daysSince < 30) return 'bg-mendelu-green/50'
  return 'bg-mendelu-alert'
}

function getFreshnessTitle(source) {
  if (source.processing_status === 'processing') return 'Currently processing'
  if (source.processing_status === 'failed') return 'Last processing failed'
  if (!source.last_processed) return 'Never processed'
  const daysSince = Math.round((Date.now() - new Date(source.last_processed).getTime()) / (1000 * 60 * 60 * 24))
  return `Last processed ${daysSince} day(s) ago`
}

async function testConnection(source) {
  testingConnection.value = true
  connectionResult.value = null
  try {
    const resp = await fetch(`/sources/${source.source_id}/test-connection`, { method: 'POST' })
    if (resp.ok) connectionResult.value = await resp.json()
  } catch (e) {
    connectionResult.value = { reachable: false, error: e.message }
  } finally {
    testingConnection.value = false
  }
}

function viewDetails(source) {
  selectedSource.value = { ...source }
  connectionResult.value = null
}

function startStatusPolling() {
  statusPollInterval = setInterval(() => {
    if (sources.value.some(s => s.processing_status === 'processing')) loadSources()
  }, 5000)
}

onMounted(() => { loadSources(); startStatusPolling() })
onUnmounted(() => { if (statusPollInterval) clearInterval(statusPollInterval) })
</script>
