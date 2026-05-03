<template>
  <div class="space-y-5">
    <PageHeader title="Data Sources" subtitle="All climate datasets — manage, schedule, and reprocess">
      <template #actions>
        <button @click="loadSources()" :disabled="loading" class="btn-ghost disabled:opacity-50">Refresh</button>
        <router-link to="/sources/create" class="btn-primary">Add Source</router-link>
      </template>
    </PageHeader>

    <!-- Summary -->
    <div class="grid grid-cols-4 gap-3">
      <div class="stat-card !p-3">
        <span class="text-[11px] text-mendelu-gray-dark uppercase tracking-wider">Total</span>
        <span class="text-lg font-semibold text-mendelu-black tabular-nums block">{{ sources.length }}</span>
      </div>
      <div class="stat-card !p-3">
        <span class="text-[11px] text-mendelu-gray-dark uppercase tracking-wider">With Data</span>
        <span class="text-lg font-semibold text-mendelu-success tabular-nums block">{{ withDataCount }}</span>
      </div>
      <div class="stat-card !p-3">
        <span class="text-[11px] text-mendelu-gray-dark uppercase tracking-wider">Metadata Only</span>
        <span class="text-lg font-semibold text-amber-500 tabular-nums block">{{ metadataOnlyCount }}</span>
      </div>
      <div class="stat-card !p-3">
        <span class="text-[11px] text-mendelu-gray-dark uppercase tracking-wider">Scheduled</span>
        <span class="text-lg font-semibold text-mendelu-black tabular-nums block">{{ scheduledCount }}</span>
      </div>
    </div>

    <!-- Filters — fixed widths so a long option label (e.g.
         "Heavy precipitation and pluvial floods") can't blow out
         the row width. Selected text truncates with ellipsis; the
         full label is still visible in the open dropdown. -->
    <div class="flex gap-3 flex-wrap">
      <input v-model="searchInput" type="text" placeholder="Search sources..." class="input-field !w-56" />
      <select v-model="filters.status" class="input-field !w-40 truncate">
        <option value="">All Statuses</option>
        <option value="completed">With Data</option>
        <option value="metadata_only">Metadata Only</option>
        <option value="processing">Processing</option>
        <option value="failed">Failed</option>
        <option value="pending">Pending</option>
      </select>
      <select v-model="filters.hazard" class="input-field !w-48 truncate">
        <option value="">All Hazards</option>
        <option v-for="h in hazardTypes" :key="h" :value="h">{{ h }}</option>
      </select>
      <select v-model="filters.region" class="input-field !w-48 truncate">
        <option value="">All Regions</option>
        <option v-for="r in regions" :key="r" :value="r">{{ r }}</option>
      </select>
      <select v-model="filters.origin" class="input-field !w-36 truncate">
        <option value="">All Origins</option>
        <option value="catalog">From Catalog</option>
        <option value="user">User-added</option>
      </select>
    </div>

    <!-- Table -->
    <div class="card !p-0 overflow-hidden">
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead class="sticky top-0 bg-white z-10">
            <tr class="border-b border-mendelu-gray-semi">
              <th class="table-header cursor-pointer hover:text-mendelu-black" @click="toggleSort('dataset_name')">
                Dataset {{ sortIcon('dataset_name') }}
              </th>
              <th class="table-header">Hazard</th>
              <th class="table-header">Region</th>
              <th class="table-header cursor-pointer hover:text-mendelu-black" @click="toggleSort('embedding_count')">
                Chunks {{ sortIcon('embedding_count') }}
              </th>
              <th class="table-header">Variables</th>
              <th class="table-header">Status</th>
              <th class="table-header">Schedule</th>
              <th class="table-header">Link</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="source in filteredSources"
              :key="source.source_id"
              class="border-b border-mendelu-gray-semi/50 hover:bg-mendelu-gray-light cursor-pointer transition-all duration-150"
              @click="selectedSource = source"
            >
              <td class="px-4 py-2.5">
                <div class="flex items-center gap-2">
                  <span class="text-mendelu-black text-sm font-medium">{{ source.dataset_name || source.source_id }}</span>
                  <span v-if="source.from_catalog" class="badge-neutral !text-[9px] !py-0" title="From the D1.1.xlsx catalog">catalog</span>
                  <span v-else class="badge-info !text-[9px] !py-0" title="Added by user via Add Sources / upload">user</span>
                  <span v-if="source.schedule" class="badge-info !text-[9px] !py-0">cron</span>
                </div>
              </td>
              <td class="px-4 py-2.5 text-mendelu-gray-dark text-xs">{{ source.hazard_type || '--' }}</td>
              <td class="px-4 py-2.5 text-mendelu-gray-dark text-xs">{{ source.location_name || '--' }}</td>
              <td class="px-4 py-2.5 tabular-nums text-xs">
                <span v-if="source.embedding_count > 10" class="text-mendelu-success font-medium">{{ source.embedding_count.toLocaleString() }}</span>
                <span v-else-if="source.embedding_count > 0" class="text-amber-500">{{ source.embedding_count }}</span>
                <span v-else class="text-mendelu-gray-dark">--</span>
              </td>
              <td class="px-4 py-2.5 text-xs">
                <div class="flex flex-wrap gap-1">
                  <span v-for="v in (source.variables || []).slice(0, 3)" :key="v" class="badge-info !py-0 !text-[10px]">{{ v }}</span>
                  <span v-if="(source.variables || []).length > 3" class="text-mendelu-gray-dark">+{{ source.variables.length - 3 }}</span>
                </div>
              </td>
              <td class="px-4 py-2.5">
                <span :class="statusBadge(source.processing_status)">
                  {{ statusLabel(source.processing_status) }}
                </span>
              </td>
              <td class="px-4 py-2.5 text-xs text-mendelu-gray-dark">
                <span v-if="source.schedule" class="font-mono">{{ source.schedule.cron_expression }}</span>
                <span v-else>--</span>
              </td>
              <td class="px-4 py-2.5" @click.stop>
                <a v-if="source.url" :href="source.url" target="_blank"
                   class="text-mendelu-green hover:text-mendelu-green-hover text-xs">Open</a>
                <span v-else class="text-mendelu-gray-dark text-xs">--</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <div class="text-center text-mendelu-gray-dark text-xs py-3 border-t border-mendelu-gray-semi/50">
        {{ filteredSources.length }} of {{ sources.length }} sources
      </div>
    </div>

    <!-- Empty State -->
    <div v-if="!loading && sources.length === 0" class="card text-center py-12">
      <h3 class="text-sm font-medium text-mendelu-black mb-2">No Sources Found</h3>
      <p class="text-mendelu-gray-dark text-xs mb-4">Add your first data source to get started</p>
      <router-link to="/sources/create" class="btn-primary inline-block">Add Source</router-link>
    </div>

    <!-- Detail Modal -->
    <Modal
      :open="!!selectedSource"
      :title="selectedSource?.dataset_name || selectedSource?.source_id || ''"
      :subtitle="selectedSource?.hazard_type || ''"
      max-width="2xl"
      @close="selectedSource = null"
    >
      <template v-if="selectedSource">
        <div class="space-y-5">
          <!-- Qdrant Stats -->
          <div v-if="selectedSource.embedding_count > 0"
               class="p-4 rounded-lg"
               :class="selectedSource.embedding_count > 10 ? 'bg-mendelu-success/5 border border-mendelu-success/20' : 'bg-amber-50 border border-amber-200'">
            <div class="flex items-center justify-between mb-2">
              <span class="text-xs font-medium uppercase tracking-wider"
                    :class="selectedSource.embedding_count > 10 ? 'text-mendelu-success' : 'text-amber-600'">
                {{ selectedSource.embedding_count > 10 ? 'Full Data' : 'Metadata Only' }}
              </span>
              <span class="text-sm font-semibold tabular-nums"
                    :class="selectedSource.embedding_count > 10 ? 'text-mendelu-success' : 'text-amber-600'">
                {{ selectedSource.embedding_count.toLocaleString() }} chunks
              </span>
            </div>
            <div v-if="selectedSource.variables && selectedSource.variables.length" class="flex flex-wrap gap-1">
              <span v-for="v in selectedSource.variables" :key="v" class="badge-info !text-[10px]">{{ v }}</span>
            </div>
          </div>

          <!-- Details Grid -->
          <div>
            <h4 class="text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-2">Details</h4>
            <div class="bg-mendelu-gray-light p-4 rounded-lg text-sm space-y-2">
              <div class="flex justify-between"><span class="text-mendelu-gray-dark">Source ID</span><span class="text-mendelu-black font-mono text-xs">{{ selectedSource.source_id }}</span></div>
              <div v-if="selectedSource.location_name" class="flex justify-between"><span class="text-mendelu-gray-dark">Region</span><span class="text-mendelu-black">{{ selectedSource.location_name }}</span></div>
              <div v-if="selectedSource.impact_sector" class="flex justify-between"><span class="text-mendelu-gray-dark">Sectors</span><span class="text-mendelu-black">{{ selectedSource.impact_sector }}</span></div>
              <div v-if="selectedSource.spatial_coverage" class="flex justify-between"><span class="text-mendelu-gray-dark">Coverage</span><span class="text-mendelu-black">{{ selectedSource.spatial_coverage }}</span></div>
              <div v-if="selectedSource.format" class="flex justify-between"><span class="text-mendelu-gray-dark">Format</span><span class="text-mendelu-black">{{ selectedSource.format }}</span></div>
              <div v-if="selectedSource.url" class="flex justify-between"><span class="text-mendelu-gray-dark">URL</span><a :href="selectedSource.url" target="_blank" class="text-mendelu-green text-xs break-all">{{ selectedSource.url }}</a></div>
              <div v-if="selectedSource.catalog_source" class="flex justify-between"><span class="text-mendelu-gray-dark">Origin</span><span class="text-mendelu-black">{{ selectedSource.catalog_source }}</span></div>
              <div v-if="selectedSource.last_processed" class="flex justify-between"><span class="text-mendelu-gray-dark">Last Processed</span><span class="text-mendelu-black">{{ formatDate(selectedSource.last_processed) }}</span></div>
            </div>
          </div>

          <!-- Schedule -->
          <div>
            <h4 class="text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-2">Schedule</h4>
            <div v-if="selectedSource.schedule" class="bg-mendelu-gray-light p-3 rounded-lg text-xs space-y-1">
              <div class="flex justify-between"><span class="text-mendelu-gray-dark">Cron</span><code class="text-mendelu-black font-mono">{{ selectedSource.schedule.cron_expression }}</code></div>
              <div class="flex justify-between"><span class="text-mendelu-gray-dark">Enabled</span><span :class="selectedSource.schedule.is_enabled ? 'text-mendelu-success' : 'text-mendelu-gray-dark'">{{ selectedSource.schedule.is_enabled ? 'Yes' : 'No' }}</span></div>
            </div>
            <p v-else class="text-xs text-mendelu-gray-dark">No schedule configured</p>
          </div>

          <!-- Error -->
          <div v-if="selectedSource.error_message" class="p-3 border-l-2 border-mendelu-alert bg-mendelu-alert/5 rounded text-sm text-mendelu-alert">
            {{ selectedSource.error_message }}
          </div>

          <!-- Processing History -->
          <div v-if="selectedSource.source_id" class="pt-4 border-t border-mendelu-gray-semi">
            <ProcessingHistory :source-id="selectedSource.source_id" />
          </div>

          <!-- Actions -->
          <div class="pt-4 border-t border-mendelu-gray-semi">
            <div class="flex gap-2">
              <button
                v-if="selectedSource.processing_status !== 'processing'"
                @click="reprocessSource(selectedSource)"
                :disabled="selectedSource._reprocessing"
                class="btn-primary flex-1 text-xs disabled:opacity-50">
                {{ selectedSource._reprocessing ? 'Triggering...' : 'Reprocess' }}
              </button>
              <button
                v-else
                @click="cancelSourceRun(selectedSource)"
                :disabled="selectedSource._cancelling"
                class="btn-ghost flex-1 text-xs text-mendelu-alert hover:bg-mendelu-alert/10 border border-mendelu-alert/20 disabled:opacity-50">
                {{ selectedSource._cancelling ? 'Cancelling...' : 'Cancel run' }}
              </button>
              <button @click="openEditModal(selectedSource)" class="btn-secondary flex-1 text-xs">Edit</button>
            </div>
            <div class="mt-3 space-y-2">
              <button v-if="selectedSource.embedding_count > 0" @click="deleteSourceEmbeddings(selectedSource)"
                      class="btn-ghost w-full text-xs text-mendelu-alert hover:bg-mendelu-alert/10 border border-mendelu-alert/20">
                Delete Embeddings
              </button>
            </div>
          </div>
        </div>
      </template>
    </Modal>

    <!-- Edit Modal -->
    <Modal
      :open="!!editingSource"
      title="Edit Source"
      subtitle="Changes to metadata update all chunks in Qdrant instantly"
      max-width="lg"
      @close="editingSource = null"
    >
      <template v-if="editingSource">
        <form @submit.prevent="saveEdit" class="space-y-4">
          <div>
            <label for="src-edit-id" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Source ID</label>
            <input id="src-edit-id" :value="editForm.source_id" type="text" class="input-field" disabled />
          </div>
          <div>
            <label for="src-edit-url" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">URL</label>
            <input id="src-edit-url" v-model="editForm.url" type="text" class="input-field" />
          </div>
          <div>
            <label for="src-edit-description" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Description</label>
            <textarea id="src-edit-description" v-model="editForm.description" rows="2" class="input-field resize-none"></textarea>
          </div>

          <div class="border-t border-mendelu-gray-semi pt-4">
            <p class="text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-3">Metadata (improves RAG retrieval)</p>
          </div>

          <div>
            <label for="src-edit-hazard" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Hazard Type</label>
            <input id="src-edit-hazard" v-model="editForm.hazard_type" type="text" class="input-field" placeholder="e.g., Drought, Extreme heat, River flood" />
          </div>
          <div>
            <label for="src-edit-region" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Region</label>
            <input id="src-edit-region" v-model="editForm.location_name" type="text" class="input-field" placeholder="e.g., Global, Europe, Mediterranean" />
          </div>
          <div>
            <label for="src-edit-sector" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Impact Sector</label>
            <input id="src-edit-sector" v-model="editForm.impact_sector" type="text" class="input-field" placeholder="e.g., Agriculture, Health, Energy" />
          </div>
          <div>
            <label for="src-edit-keywords" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Keywords (comma separated)</label>
            <input id="src-edit-keywords" v-model="editForm.keywords" type="text" class="input-field" placeholder="e.g., temperature, reanalysis, ERA5" />
          </div>

          <div class="border-t border-mendelu-gray-semi pt-4">
            <CronPicker v-model="editForm.schedule_cron" label="Schedule" />
          </div>

          <div class="flex gap-3 pt-2">
            <button type="submit" :disabled="saving" class="btn-primary flex-1 disabled:opacity-50">{{ saving ? 'Saving...' : 'Save' }}</button>
            <button type="button" @click="editingSource = null" class="btn-secondary flex-1">Cancel</button>
          </div>
        </form>
      </template>
    </Modal>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import PageHeader from '../components/PageHeader.vue'
import ProcessingHistory from '../components/ProcessingHistory.vue'
import CronPicker from '../components/CronPicker.vue'
import Modal from '../components/Modal.vue'
import { apiFetch } from '../api'
import { useToast } from '../composables/useToast'
import { useConfirm } from '../composables/useConfirm'

const toast = useToast()
const { confirm } = useConfirm()

const sources = ref<any[]>([])
const loading = ref(true)
const selectedSource = ref(null)
const editingSource = ref(null)
const saving = ref(false)
const editForm = ref<any>({ source_id: '', url: '', description: '', hazard_type: '', location_name: '', impact_sector: '', keywords: '', schedule_cron: '' })

const filters = ref<any>({ search: '', status: '', hazard: '', region: '', origin: '' })
// Debounce the free-text search so filteredSources doesn't recompute on every
// keystroke (which re-runs through every source + re-builds hazardTypes/regions).
const searchInput = ref('')
let searchDebounce: ReturnType<typeof setTimeout> | null = null
watch(searchInput, (val) => {
  if (searchDebounce) clearTimeout(searchDebounce)
  searchDebounce = setTimeout(() => {
    filters.value.search = val
  }, 250)
})

const sortField = ref('embedding_count')
const sortAsc = ref(false)

let statusPollInterval: ReturnType<typeof setInterval> | null = null

const withDataCount = computed(() => sources.value.filter(s => s.embedding_count > 10).length)
const metadataOnlyCount = computed(() => sources.value.filter(s => s.embedding_count > 0 && s.embedding_count <= 10).length)
const scheduledCount = computed(() => sources.value.filter(s => s.schedule).length)
// hazard_type can now be a comma-joined list (datasets covering multiple
// hazards in the catalog). Split for dropdown options and match with
// substring instead of equality.
const hazardTypes = computed(() => {
  const all = new Set<string>()
  for (const s of sources.value) {
    if (!s.hazard_type) continue
    for (const h of String(s.hazard_type).split(',').map(x => x.trim()).filter(Boolean)) {
      all.add(h)
    }
  }
  return [...all].sort()
})
const regions = computed(() => {
  const all = new Set<string>()
  for (const s of sources.value) {
    const raw = s.location_name || s.region_country
    if (!raw) continue
    for (const r of String(raw).split(',').map(x => x.trim()).filter(Boolean)) {
      all.add(r)
    }
  }
  return [...all].sort()
})

const filteredSources = computed(() => {
  let result = [...sources.value]
  if (filters.value.search) {
    const q = filters.value.search.toLowerCase()
    result = result.filter(s =>
      (s.dataset_name || '').toLowerCase().includes(q) ||
      (s.source_id || '').toLowerCase().includes(q) ||
      (s.hazard_type || '').toLowerCase().includes(q)
    )
  }
  if (filters.value.status) {
    if (filters.value.status === 'completed') result = result.filter(s => s.embedding_count > 10)
    else if (filters.value.status === 'metadata_only') result = result.filter(s => s.embedding_count > 0 && s.embedding_count <= 10)
    else result = result.filter(s => s.processing_status === filters.value.status)
  }
  if (filters.value.hazard) result = result.filter(s => String(s.hazard_type || '').includes(filters.value.hazard))
  if (filters.value.region) result = result.filter(s => String(s.location_name || s.region_country || '').includes(filters.value.region))
  if (filters.value.origin === 'catalog') result = result.filter(s => s.from_catalog)
  else if (filters.value.origin === 'user') result = result.filter(s => !s.from_catalog)

  result.sort((a, b) => {
    if (sortField.value === 'embedding_count') {
      return sortAsc.value ? a.embedding_count - b.embedding_count : b.embedding_count - a.embedding_count
    }
    const av = (a[sortField.value] || '').toString().toLowerCase()
    const bv = (b[sortField.value] || '').toString().toLowerCase()
    return sortAsc.value ? av.localeCompare(bv) : bv.localeCompare(av)
  })
  return result
})

function toggleSort(field) {
  if (sortField.value === field) sortAsc.value = !sortAsc.value
  else { sortField.value = field; sortAsc.value = field !== 'embedding_count' }
}
function sortIcon(field) { return sortField.value !== field ? '' : sortAsc.value ? '\u25B2' : '\u25BC' }

function statusBadge(status) {
  return {
    completed: 'badge-success', metadata_only: 'badge-warning', processing: 'badge-info',
    failed: 'badge-danger', error: 'badge-danger', pending: 'badge-neutral',
  }[status] || 'badge-neutral'
}
function statusLabel(status) {
  return {
    completed: 'Data', metadata_only: 'Metadata', processing: 'Processing',
    failed: 'Failed', error: 'Error', pending: 'Pending',
  }[status] || 'Pending'
}
function formatDate(d) { return d ? new Date(d).toLocaleString() : '--' }

async function loadSources() {
  loading.value = true
  try {
    const resp = await apiFetch('/sources/')
    if (!resp.ok) throw new Error(resp.statusText)
    sources.value = await resp.json()
  } catch (e: any) {
    console.error('Failed to load sources:', e)
    toast.error(`Could not load sources: ${e?.message || 'network error'}`)
  } finally {
    loading.value = false
  }
}

async function reprocessSource(source) {
  source._reprocessing = true
  try {
    const resp = await apiFetch(`/sources/${source.source_id}/trigger`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }
    })
    if (resp.ok) {
      source.processing_status = 'processing'
    } else {
      const err = await resp.json().catch(() => ({}))
      toast.error(err.detail || 'Failed to trigger reprocessing')
    }
  } catch (e: any) {
    console.error('Reprocess failed:', e)
    toast.error(`Reprocess failed: ${e?.message || 'network error'}`)
  } finally {
    source._reprocessing = false
  }
}

async function cancelSourceRun(source) {
  source._cancelling = true
  try {
    const resp = await apiFetch(`/sources/${source.source_id}/cancel`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
    })
    const data = await resp.json().catch(() => ({}))
    if (resp.ok) {
      source.processing_status = 'pending'
      if (data.cancelled > 0) {
        toast.success(`Cancelled ${data.cancelled} run${data.cancelled === 1 ? '' : 's'} for ${source.source_id}`)
      } else {
        toast.info?.(`No active run to cancel — status reset to pending.`) ?? toast.success('Status reset')
      }
    } else {
      toast.error(data.detail || `Cancel failed (HTTP ${resp.status})`)
    }
  } catch (e: any) {
    console.error('Cancel failed:', e)
    toast.error(`Cancel failed: ${e?.message || 'network error'}`)
  } finally {
    source._cancelling = false
  }
}

async function deleteSourceEmbeddings(source) {
  const ok = await confirm({
    title: 'Delete embeddings?',
    message: `Delete all embeddings for "${source.dataset_name || source.source_id}"?\nThis cannot be undone.`,
    confirmText: 'Delete',
    danger: true,
  })
  if (!ok) return
  try {
    const resp = await apiFetch(`/sources/${source.source_id}/embeddings?confirm=true`, { method: 'DELETE' })
    if (resp.ok) {
      await apiFetch('/qdrant/cache/clear', { method: 'POST' })
      selectedSource.value = null
      await loadSources()
    }
  } catch (e: any) {
    console.error('Error deleting embeddings:', e)
    toast.error(`Delete failed: ${e?.message || 'network error'}`)
  }
}

function openEditModal(source) {
  editForm.value = {
    source_id: source.source_id,
    url: source.url || '',
    description: source.description || '',
    hazard_type: source.hazard_type || '',
    location_name: source.location_name || '',
    impact_sector: source.impact_sector || '',
    keywords: Array.isArray(source.keywords) ? source.keywords.join(', ') : (source.keywords || ''),
    schedule_cron: source.schedule?.cron_expression || '',
  }
  editingSource.value = source
}

async function saveEdit() {
  saving.value = true
  try {
    // Update metadata (both Qdrant payloads + PostgreSQL)
    const metadata = {
      url: editForm.value.url,
      description: editForm.value.description,
      hazard_type: editForm.value.hazard_type || null,
      location_name: editForm.value.location_name || null,
      impact_sector: editForm.value.impact_sector || null,
      keywords: editForm.value.keywords || null,
    }
    const resp = await apiFetch(`/sources/${editForm.value.source_id}/metadata`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(metadata)
    })
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(err.detail || `HTTP ${resp.status}`)
    }
    const result = await resp.json()

    // Update or remove schedule based on cron field
    const hadSchedule = Boolean(selectedSource.value?.schedule)
    const newCron = (editForm.value.schedule_cron || '').trim()
    if (newCron) {
      const schedResp = await apiFetch(`/sources/${editForm.value.source_id}/schedule`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cron_expression: newCron, is_enabled: true })
      })
      if (!schedResp.ok) {
        const err = await schedResp.json().catch(() => ({}))
        throw new Error(err.detail || `Schedule update failed (HTTP ${schedResp.status})`)
      }
    } else if (hadSchedule) {
      await apiFetch(`/sources/${editForm.value.source_id}/schedule`, { method: 'DELETE' })
    }

    editingSource.value = null
    selectedSource.value = null
    await loadSources()
  } catch (e) {
    toast.error(`Error: ${e.message}`)
  } finally {
    saving.value = false
  }
}

function startPolling() {
  statusPollInterval = setInterval(() => {
    if (sources.value.some(s => s.processing_status === 'processing')) loadSources()
  }, 5000)
}

onMounted(() => { loadSources(); startPolling() })
onUnmounted(() => { if (statusPollInterval) clearInterval(statusPollInterval) })
</script>
