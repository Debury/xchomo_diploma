<template>
  <div class="space-y-5">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-xl font-semibold text-white">Dataset Catalog</h1>
        <p class="text-sm text-gray-500">233 climate data sources from D1.1.xlsx</p>
      </div>
      <div class="flex gap-2">
        <button
          @click="classifyCatalog"
          :disabled="loading"
          class="px-3 py-1.5 text-sm bg-dark-hover text-gray-300 rounded-md hover:bg-gray-600 transition-colors disabled:opacity-50"
        >
          Classify
        </button>
        <button
          @click="triggerProcessing([0])"
          :disabled="processing"
          class="px-3 py-1.5 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:opacity-50"
        >
          {{ processing ? 'Processing...' : 'Phase 0' }}
        </button>
        <button
          @click="triggerProcessing([1])"
          :disabled="processing"
          class="px-3 py-1.5 text-sm bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors disabled:opacity-50"
        >
          {{ processing ? 'Processing...' : 'Phase 1' }}
        </button>
        <button
          @click="refreshCatalog"
          :disabled="loading"
          class="px-3 py-1.5 text-sm bg-dark-hover text-gray-300 rounded-md hover:bg-gray-600 transition-colors disabled:opacity-50"
        >
          Refresh
        </button>
      </div>
    </div>

    <!-- Phase Distribution -->
    <div v-if="phaseStats" class="grid grid-cols-5 gap-3">
      <div v-for="(count, phase) in phaseStats.phases" :key="phase" class="card !p-4">
        <div class="flex items-center justify-between mb-1">
          <span class="text-[11px] font-medium uppercase tracking-wider" :class="phaseColor(Number(phase))">Phase {{ phase }}</span>
          <span class="text-lg font-semibold text-white tabular-nums">{{ count }}</span>
        </div>
        <p class="text-[11px] text-gray-600">{{ phaseLabel(Number(phase)) }}</p>
      </div>
    </div>

    <!-- Thread Crashed Banner -->
    <div v-if="progress && progress.thread_crashed" class="card !p-4 border border-red-500/40 bg-red-500/10">
      <div class="flex items-center justify-between">
        <div>
          <span class="text-sm font-medium text-red-400">Batch processing crashed</span>
          <p v-if="progress.thread_error" class="text-xs text-red-300/70 mt-1 font-mono max-w-xl truncate">{{ progress.thread_error }}</p>
        </div>
        <button
          @click="autoRestart"
          :disabled="restarting"
          class="px-3 py-1.5 text-sm bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors disabled:opacity-50 shrink-0"
        >
          {{ restarting ? 'Restarting...' : 'Restart' }}
        </button>
      </div>
    </div>

    <!-- Thread Alive Indicator -->
    <div v-if="progress && progress.thread_alive" class="card !p-4 border border-blue-500/30 bg-blue-500/5">
      <div class="flex items-center gap-2">
        <span class="relative flex h-2.5 w-2.5">
          <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
          <span class="relative inline-flex rounded-full h-2.5 w-2.5 bg-blue-500"></span>
        </span>
        <span class="text-sm text-blue-300">Processing in progress</span>
        <span v-if="progress.current_source" class="text-xs text-blue-400/70 ml-auto font-mono">{{ progress.current_source }}</span>
      </div>
    </div>

    <!-- Progress Bar -->
    <div v-if="progress && progress.total > 0" class="card !p-4">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs text-gray-400">Processing Progress</span>
        <span class="text-xs text-white tabular-nums">{{ progress.processed + progress.failed }} / {{ progress.total }}</span>
      </div>
      <div class="w-full bg-gray-700/50 rounded-full h-2">
        <div class="flex h-2 rounded-full overflow-hidden">
          <div
            class="bg-green-500 transition-all duration-500"
            :style="{ width: `${(progress.processed / progress.total) * 100}%` }"
          ></div>
          <div
            v-if="progress.failed"
            class="bg-red-500 transition-all duration-500"
            :style="{ width: `${(progress.failed / progress.total) * 100}%` }"
          ></div>
        </div>
      </div>
      <div class="flex gap-4 mt-1.5 text-[11px] text-gray-500">
        <span class="text-green-400">{{ progress.processed }} done</span>
        <span v-if="progress.failed" class="text-red-400">{{ progress.failed }} failed</span>
        <span>{{ progress.pending }} pending</span>
        <span v-if="progress.current_source" class="text-blue-400 ml-auto">{{ progress.current_source }}</span>
      </div>
    </div>

    <!-- Filters -->
    <div class="flex gap-3 flex-wrap">
      <input
        v-model="filters.search"
        type="text"
        placeholder="Search datasets..."
        class="bg-dark-card border border-dark-border rounded-md px-3 py-1.5 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-gray-500 w-56"
      />
      <select v-model="filters.phase" class="bg-dark-card border border-dark-border rounded-md px-3 py-1.5 text-sm text-white focus:outline-none focus:border-gray-500">
        <option value="">All Phases</option>
        <option v-for="p in [0,1,2,3,4]" :key="p" :value="p">Phase {{ p }}</option>
      </select>
      <select v-model="filters.status" class="bg-dark-card border border-dark-border rounded-md px-3 py-1.5 text-sm text-white focus:outline-none focus:border-gray-500">
        <option value="">All Statuses</option>
        <option value="pending">Pending</option>
        <option value="processing">Processing</option>
        <option value="completed">Completed</option>
        <option value="metadata_only">Metadata Only</option>
        <option value="failed">Failed</option>
      </select>
      <select v-model="filters.access" class="bg-dark-card border border-dark-border rounded-md px-3 py-1.5 text-sm text-white focus:outline-none focus:border-gray-500">
        <option value="">All Access</option>
        <option v-for="a in accessTypes" :key="a" :value="a">{{ a }}</option>
      </select>
    </div>

    <!-- Table -->
    <div class="card !p-0 overflow-hidden">
      <table class="w-full text-sm">
        <thead>
          <tr class="border-b border-dark-border text-left text-xs text-gray-500 uppercase tracking-wider">
            <th class="px-4 py-2.5 cursor-pointer hover:text-gray-300" @click="toggleSort('dataset_name')">
              Dataset {{ sortIcon('dataset_name') }}
            </th>
            <th class="px-4 py-2.5">Hazard</th>
            <th class="px-4 py-2.5">Type</th>
            <th class="px-4 py-2.5">Region</th>
            <th class="px-4 py-2.5">Phase</th>
            <th class="px-4 py-2.5">Status</th>
            <th class="px-4 py-2.5">Access</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="entry in filteredEntries"
            :key="entry.row_index"
            class="border-b border-dark-border/30 hover:bg-dark-hover/50 cursor-pointer transition-colors"
            @click="selectedEntry = entry"
          >
            <td class="px-4 py-2 text-white text-sm">{{ entry.dataset_name }}</td>
            <td class="px-4 py-2 text-gray-400 text-xs">{{ entry.hazard || '--' }}</td>
            <td class="px-4 py-2 text-gray-400 text-xs">{{ entry.data_type || '--' }}</td>
            <td class="px-4 py-2 text-gray-400 text-xs">{{ entry.region_country || entry.spatial_coverage || '--' }}</td>
            <td class="px-4 py-2">
              <span class="px-1.5 py-0.5 rounded text-[11px] font-medium" :class="phaseBadgeClass(entry.phase)">
                {{ entry.phase }}
              </span>
            </td>
            <td class="px-4 py-2">
              <span class="px-1.5 py-0.5 rounded text-[11px] font-medium" :class="statusBadgeClass(entry.processing_status)">
                {{ entry.processing_status }}
              </span>
            </td>
            <td class="px-4 py-2 text-gray-500 text-xs">{{ entry.access || '--' }}</td>
          </tr>
        </tbody>
      </table>
      <div class="text-center text-gray-600 text-xs py-3 border-t border-dark-border/30">
        {{ filteredEntries.length }} of {{ catalog.length }} entries
      </div>
    </div>

    <!-- Detail Modal -->
    <div v-if="selectedEntry" class="fixed inset-0 bg-black/60 flex items-center justify-center z-50" @click.self="selectedEntry = null">
      <div class="bg-dark-card border border-dark-border rounded-lg p-5 max-w-xl w-full mx-4 max-h-[80vh] overflow-y-auto">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-lg font-semibold text-white">{{ selectedEntry.dataset_name }}</h3>
          <button @click="selectedEntry = null" class="text-gray-500 hover:text-white text-lg leading-none">&times;</button>
        </div>
        <div class="grid grid-cols-2 gap-3 text-sm">
          <div v-for="(value, key) in selectedEntryFields" :key="key">
            <span class="text-[11px] text-gray-500 uppercase tracking-wider block">{{ formatFieldName(key) }}</span>
            <span class="text-gray-300">{{ value || '--' }}</span>
          </div>
        </div>
        <div v-if="selectedEntry.link" class="mt-4 pt-3 border-t border-dark-border">
          <a :href="selectedEntry.link" target="_blank" class="text-blue-400 hover:text-blue-300 text-sm">
            Open data link &rarr;
          </a>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'

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

function phaseColor(phase) {
  return { 0: 'text-blue-400', 1: 'text-green-400', 2: 'text-yellow-400', 3: 'text-orange-400', 4: 'text-red-400' }[phase] || 'text-gray-400'
}
function phaseLabel(phase) {
  return { 0: 'Metadata only', 1: 'Direct download', 2: 'Registration', 3: 'API portal', 4: 'Manual' }[phase] || ''
}
function phaseBadgeClass(phase) {
  return { 0: 'bg-blue-500/15 text-blue-400', 1: 'bg-green-500/15 text-green-400', 2: 'bg-yellow-500/15 text-yellow-400', 3: 'bg-orange-500/15 text-orange-400', 4: 'bg-red-500/15 text-red-400' }[phase] || 'bg-gray-500/15 text-gray-400'
}
function statusBadgeClass(status) {
  return { pending: 'bg-gray-500/15 text-gray-400', processing: 'bg-blue-500/15 text-blue-400', completed: 'bg-green-500/15 text-green-400', metadata_only: 'bg-amber-500/15 text-amber-400', failed: 'bg-red-500/15 text-red-400' }[status] || 'bg-gray-500/15 text-gray-400'
}
function formatFieldName(key) { return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) }

async function refreshCatalog() {
  loading.value = true
  try {
    const [catResp, progResp] = await Promise.all([fetch('/catalog'), fetch('/catalog/progress')])
    if (catResp.ok) catalog.value = await catResp.json()
    if (progResp.ok) progress.value = await progResp.json()
  } catch (e) {
    console.error('Failed to load catalog:', e)
  } finally {
    loading.value = false
  }
}

async function classifyCatalog() {
  loading.value = true
  try {
    const resp = await fetch('/catalog/classify', { method: 'POST' })
    if (resp.ok) phaseStats.value = await resp.json()
  } catch (e) {
    console.error('Failed to classify:', e)
  } finally {
    loading.value = false
  }
}

async function triggerProcessing(phases) {
  processing.value = true
  try {
    const resp = await fetch('/catalog/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ phases }),
    })
    if (resp.ok) pollProgress()
  } catch (e) {
    console.error('Failed to trigger processing:', e)
  } finally {
    processing.value = false
  }
}

async function autoRestart() {
  restarting.value = true
  try {
    const resp = await fetch('/catalog/auto-restart', { method: 'POST' })
    if (resp.ok) {
      pollProgress()
    } else {
      const data = await resp.json()
      console.error('Auto-restart failed:', data.detail || data)
    }
  } catch (e) {
    console.error('Auto-restart error:', e)
  } finally {
    restarting.value = false
  }
}

let progressInterval = null
function pollProgress() {
  if (progressInterval) return
  progressInterval = setInterval(async () => {
    try {
      const resp = await fetch('/catalog/progress')
      if (resp.ok) {
        progress.value = await resp.json()
        // Stop polling when thread is done and nothing pending
        if (!progress.value.thread_alive && progress.value.pending === 0) {
          clearInterval(progressInterval)
          progressInterval = null
          refreshCatalog()
        }
      }
    } catch (e) {
      clearInterval(progressInterval)
      progressInterval = null
    }
  }, 3000)
}

onMounted(() => {
  refreshCatalog()
  classifyCatalog()
})
</script>
