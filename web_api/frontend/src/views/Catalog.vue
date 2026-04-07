<template>
  <div class="space-y-5">
    <PageHeader title="Dataset Catalog" :subtitle="`${catalog.length || 246} climate data sources from D1.1.xlsx`">
      <template #actions>
        <button @click="classifyCatalog" :disabled="loading" class="btn-ghost disabled:opacity-50">
          Classify
        </button>
        <div class="flex rounded-lg border border-mendelu-gray-semi overflow-hidden">
          <button
            v-for="p in [0, 1, 2, 3]"
            :key="p"
            @click="triggerProcessing([p])"
            :disabled="processing"
            class="px-3 py-1.5 text-xs font-medium transition-all duration-150 disabled:opacity-50 border-r border-mendelu-gray-semi last:border-r-0"
            :class="processing ? 'bg-mendelu-gray-light text-mendelu-gray-dark' : 'bg-white hover:bg-mendelu-gray-light text-mendelu-black'"
          >
            {{ processing ? '...' : `Phase ${p}` }}
          </button>
        </div>
        <button @click="refreshCatalog" :disabled="loading" class="btn-secondary disabled:opacity-50">
          Refresh
        </button>
      </template>
    </PageHeader>

    <!-- Phase Distribution -->
    <div v-if="phaseStats" class="grid grid-cols-5 gap-3">
      <div v-for="(count, phase) in phaseStats.phases" :key="phase" class="stat-card !p-4">
        <div class="flex items-center justify-between mb-1">
          <span class="text-[11px] font-medium uppercase tracking-wider text-mendelu-gray-dark">Phase {{ phase }}</span>
          <span class="text-lg font-semibold text-mendelu-black tabular-nums">{{ count }}</span>
        </div>
        <p class="text-[11px] text-mendelu-gray-dark">{{ phaseLabel(Number(phase)) }}</p>
      </div>
    </div>

    <!-- Thread Crashed Banner -->
    <div v-if="progress && progress.thread_crashed" class="card !p-4 border-mendelu-alert/40">
      <div class="flex items-center justify-between">
        <div>
          <span class="text-sm font-medium text-mendelu-alert">Batch processing crashed</span>
          <p v-if="progress.thread_error" class="text-xs text-mendelu-alert/70 mt-1 font-mono max-w-xl truncate">{{ progress.thread_error }}</p>
        </div>
        <button @click="autoRestart" :disabled="restarting" class="btn-danger !py-1.5 disabled:opacity-50">
          {{ restarting ? 'Restarting...' : 'Restart' }}
        </button>
      </div>
    </div>

    <!-- Thread Alive Banner -->
    <div v-if="progress && progress.thread_alive" class="card !p-4 border-mendelu-green/30">
      <div class="flex items-center gap-2">
        <span class="relative flex h-2.5 w-2.5">
          <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-mendelu-green opacity-75"></span>
          <span class="relative inline-flex rounded-full h-2.5 w-2.5 bg-mendelu-green"></span>
        </span>
        <span class="text-sm text-mendelu-green font-medium">Processing in progress</span>
        <span v-if="progress.current_source" class="text-xs text-mendelu-gray-dark ml-auto font-mono">{{ progress.current_source }}</span>
      </div>
    </div>

    <!-- Progress Bar -->
    <div v-if="progress && progress.total > 0" class="card !p-4">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs text-mendelu-gray-dark">Processing Progress</span>
        <span class="text-xs text-mendelu-black tabular-nums">{{ progress.processed + (progress.metadata_only || 0) + progress.failed }} / {{ progress.total }}</span>
      </div>
      <div class="w-full bg-mendelu-gray-semi rounded-full h-2">
        <div class="flex h-2 rounded-full overflow-hidden">
          <div class="bg-mendelu-success transition-all duration-500" :style="{ width: `${pctOf(progress.processed)}%` }"></div>
          <div v-if="progress.metadata_only" class="bg-mendelu-green/40 transition-all duration-500" :style="{ width: `${pctOf(progress.metadata_only)}%` }"></div>
          <div v-if="progress.failed" class="bg-mendelu-alert transition-all duration-500" :style="{ width: `${pctOf(progress.failed)}%` }"></div>
        </div>
      </div>
      <div class="flex gap-4 mt-1.5 text-[11px] text-mendelu-gray-dark">
        <span class="text-mendelu-success">{{ progress.processed }} processed</span>
        <span v-if="progress.metadata_only" class="text-mendelu-green/70">{{ progress.metadata_only }} metadata only</span>
        <span v-if="progress.failed" class="text-mendelu-alert">{{ progress.failed }} failed</span>
        <span>{{ progress.pending || 0 }} pending</span>
      </div>
    </div>

    <!-- Filters -->
    <div class="flex gap-3 flex-wrap">
      <input v-model="filters.search" type="text" placeholder="Search datasets..." class="input-field !w-56" />
      <select v-model="filters.phase" class="input-field !w-auto">
        <option value="">All Phases</option>
        <option v-for="p in [0,1,2,3,4]" :key="p" :value="p">Phase {{ p }}</option>
      </select>
      <select v-model="filters.status" class="input-field !w-auto">
        <option value="">All Statuses</option>
        <option value="pending">Pending</option>
        <option value="processing">Processing</option>
        <option value="completed">Completed</option>
        <option value="metadata_only">Metadata Only</option>
        <option value="failed">Failed</option>
      </select>
      <select v-model="filters.access" class="input-field !w-auto">
        <option value="">All Access</option>
        <option v-for="a in accessTypes" :key="a" :value="a">{{ a }}</option>
      </select>
    </div>

    <!-- Table -->
    <div class="card !p-0 overflow-hidden">
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead class="sticky top-0 bg-white z-10">
            <tr class="border-b border-mendelu-gray-semi">
              <th class="table-header cursor-pointer hover:text-mendelu-black transition-colors duration-150" @click="toggleSort('dataset_name')">
                Dataset {{ sortIcon('dataset_name') }}
              </th>
              <th class="table-header">Hazard</th>
              <th class="table-header">Type</th>
              <th class="table-header">Region</th>
              <th class="table-header">Phase</th>
              <th class="table-header">Status</th>
              <th class="table-header">Access</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="entry in filteredEntries"
              :key="entry.row_index"
              class="border-b border-mendelu-gray-semi/50 hover:bg-mendelu-gray-light cursor-pointer transition-all duration-150"
              @click="selectedEntry = entry"
            >
              <td class="px-4 py-2.5 text-mendelu-black text-sm font-medium">{{ entry.dataset_name }}</td>
              <td class="px-4 py-2.5 text-mendelu-gray-dark text-xs">{{ entry.hazard || '--' }}</td>
              <td class="px-4 py-2.5 text-mendelu-gray-dark text-xs">{{ entry.data_type || '--' }}</td>
              <td class="px-4 py-2.5 text-mendelu-gray-dark text-xs">{{ entry.region_country || entry.spatial_coverage || '--' }}</td>
              <td class="px-4 py-2.5">
                <span class="badge-neutral">{{ entry.phase }}</span>
              </td>
              <td class="px-4 py-2.5">
                <span :class="statusBadgeClass(entry.processing_status)">
                  {{ entry.processing_status }}
                </span>
              </td>
              <td class="px-4 py-2.5 text-mendelu-gray-dark text-xs">{{ entry.access || '--' }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div class="text-center text-mendelu-gray-dark text-xs py-3 border-t border-mendelu-gray-semi/50">
        {{ filteredEntries.length }} of {{ catalog.length }} entries
      </div>
    </div>

    <!-- Detail Modal -->
    <div v-if="selectedEntry" class="fixed inset-0 bg-black/40 flex items-center justify-center z-50" @click.self="selectedEntry = null">
      <div class="bg-white border border-mendelu-gray-semi rounded-xl p-5 max-w-xl w-full mx-4 max-h-[80vh] overflow-y-auto shadow-lg">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-lg font-semibold text-mendelu-black">{{ selectedEntry.dataset_name }}</h3>
          <button @click="selectedEntry = null" class="btn-ghost !px-2 !py-1">&times;</button>
        </div>
        <div class="grid grid-cols-2 gap-3 text-sm">
          <div v-for="(value, key) in selectedEntryFields" :key="key">
            <span class="text-[11px] text-mendelu-gray-dark uppercase tracking-wider block">{{ formatFieldName(key) }}</span>
            <span class="text-mendelu-black">{{ value || '--' }}</span>
          </div>
        </div>
        <div v-if="selectedEntry.link" class="mt-4 pt-3 border-t border-mendelu-gray-semi">
          <a :href="selectedEntry.link" target="_blank" class="text-mendelu-green hover:text-mendelu-green-hover text-sm font-medium transition-colors duration-150">
            Open data link &rarr;
          </a>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import PageHeader from '../components/PageHeader.vue'

const catalog = ref([])
const phaseStats = ref(null)
const progress = ref(null)
const loading = ref(false)
const processing = ref(false)
const restarting = ref(false)
const selectedEntry = ref(null)

const filters = ref({ search: '', phase: '', access: '', dataType: '', status: '' })
const sortField = ref('dataset_name')
const sortAsc = ref(true)

const accessTypes = computed(() => [...new Set(catalog.value.map(e => e.access).filter(Boolean))])

function pctOf(value) {
  const total = progress.value?.total || 1
  return Math.min(100, Math.round((value / total) * 100))
}

const filteredEntries = computed(() => {
  let result = [...catalog.value]
  if (filters.value.search) {
    const q = filters.value.search.toLowerCase()
    result = result.filter(e => (e.dataset_name || '').toLowerCase().includes(q))
  }
  if (filters.value.phase !== '') result = result.filter(e => e.phase === Number(filters.value.phase))
  if (filters.value.access) result = result.filter(e => e.access === filters.value.access)
  if (filters.value.status) result = result.filter(e => e.processing_status === filters.value.status)
  result.sort((a, b) => {
    const aVal = (a[sortField.value] || '').toString().toLowerCase()
    const bVal = (b[sortField.value] || '').toString().toLowerCase()
    return sortAsc.value ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal)
  })
  return result
})

const selectedEntryFields = computed(() => {
  if (!selectedEntry.value) return {}
  const e = selectedEntry.value
  return {
    hazard: e.hazard, data_type: e.data_type, spatial_coverage: e.spatial_coverage,
    region_country: e.region_country, spatial_resolution: e.spatial_resolution,
    temporal_coverage: e.temporal_coverage, temporal_resolution: e.temporal_resolution,
    bias_corrected: e.bias_corrected, impact_sector: e.impact_sector,
    phase: e.phase, processing_status: e.processing_status, notes: e.notes,
  }
})

function toggleSort(field) {
  if (sortField.value === field) { sortAsc.value = !sortAsc.value } else { sortField.value = field; sortAsc.value = true }
}
function sortIcon(field) { return sortField.value !== field ? '' : sortAsc.value ? '\u25B2' : '\u25BC' }

function phaseLabel(phase) {
  return { 0: 'Metadata only', 1: 'Direct download', 2: 'Registration', 3: 'API portal', 4: 'Manual' }[phase] || ''
}
function statusBadgeClass(status) {
  return { pending: 'badge-neutral', processing: 'badge-info', completed: 'badge-success', metadata_only: 'badge-warning', failed: 'badge-danger' }[status] || 'badge-neutral'
}
function formatFieldName(key) { return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) }

async function refreshCatalog() {
  loading.value = true
  try {
    const [catResp, progResp] = await Promise.all([fetch('/catalog'), fetch('/catalog/progress')])
    if (catResp.ok) catalog.value = await catResp.json()
    if (progResp.ok) progress.value = await progResp.json()
  } catch (e) { console.error('Failed to load catalog:', e) }
  finally { loading.value = false }
}

async function classifyCatalog() {
  loading.value = true
  try {
    const resp = await fetch('/catalog/classify', { method: 'POST' })
    if (resp.ok) phaseStats.value = await resp.json()
  } catch (e) { console.error('Failed to classify:', e) }
  finally { loading.value = false }
}

async function triggerProcessing(phases) {
  processing.value = true
  try {
    const resp = await fetch('/catalog/process', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ phases }),
    })
    if (resp.ok) pollProgress()
  } catch (e) { console.error('Failed to trigger processing:', e) }
  finally { processing.value = false }
}

async function autoRestart() {
  restarting.value = true
  try {
    const resp = await fetch('/catalog/auto-restart', { method: 'POST' })
    if (resp.ok) pollProgress()
    else { const data = await resp.json(); console.error('Auto-restart failed:', data.detail || data) }
  } catch (e) { console.error('Auto-restart error:', e) }
  finally { restarting.value = false }
}

let progressInterval = null
function pollProgress() {
  if (progressInterval) return
  progressInterval = setInterval(async () => {
    try {
      const resp = await fetch('/catalog/progress')
      if (resp.ok) {
        progress.value = await resp.json()
        if (!progress.value.thread_alive && progress.value.pending === 0) {
          clearInterval(progressInterval); progressInterval = null; refreshCatalog()
        }
      }
    } catch (e) { clearInterval(progressInterval); progressInterval = null }
  }, 3000)
}

onMounted(() => { refreshCatalog(); classifyCatalog().catch(() => {}) })
</script>
