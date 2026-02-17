<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-2xl font-bold text-white">Dataset Catalog</h1>
        <p class="text-gray-400">234 climate data sources from D1.1.xlsx</p>
      </div>
      <div class="flex gap-3">
        <button
          @click="classifyCatalog"
          :disabled="loading"
          class="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors disabled:opacity-50"
        >
          Classify
        </button>
        <button
          @click="triggerProcessing([0])"
          :disabled="processing"
          class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
        >
          {{ processing ? 'Processing...' : 'Run Phase 0' }}
        </button>
        <button
          @click="triggerProcessing([1])"
          :disabled="processing"
          class="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50"
        >
          {{ processing ? 'Processing...' : 'Run Phase 1' }}
        </button>
        <button
          @click="refreshCatalog"
          :disabled="loading"
          class="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors disabled:opacity-50"
        >
          Refresh
        </button>
      </div>
    </div>

    <!-- Phase Distribution -->
    <div v-if="phaseStats" class="grid grid-cols-5 gap-4">
      <div v-for="(count, phase) in phaseStats.phases" :key="phase" class="card">
        <div class="flex items-center justify-between mb-2">
          <span class="text-xs font-medium" :class="phaseColor(Number(phase))">Phase {{ phase }}</span>
          <span class="text-lg font-bold text-white">{{ count }}</span>
        </div>
        <p class="text-xs text-gray-500">{{ phaseLabel(Number(phase)) }}</p>
      </div>
    </div>

    <!-- Progress Bar -->
    <div v-if="progress && progress.total > 0" class="card">
      <div class="flex items-center justify-between mb-2">
        <span class="text-sm text-gray-400">Processing Progress</span>
        <span class="text-sm text-white">{{ progress.processed }} / {{ progress.total }}</span>
      </div>
      <div class="w-full bg-gray-700 rounded-full h-3">
        <div
          class="bg-blue-500 h-3 rounded-full transition-all duration-500"
          :style="{ width: `${(progress.processed / progress.total) * 100}%` }"
        ></div>
      </div>
      <div class="flex gap-4 mt-2 text-xs text-gray-500">
        <span class="text-green-400">{{ progress.processed }} completed</span>
        <span class="text-red-400">{{ progress.failed }} failed</span>
        <span class="text-gray-400">{{ progress.pending }} pending</span>
      </div>
    </div>

    <!-- Filters -->
    <div class="card">
      <div class="grid grid-cols-1 md:grid-cols-5 gap-4">
        <input
          v-model="filters.search"
          type="text"
          placeholder="Search dataset name..."
          class="bg-dark-hover border border-dark-border rounded-lg px-4 py-2 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
        />
        <select v-model="filters.phase" class="bg-dark-hover border border-dark-border rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500">
          <option value="">All Phases</option>
          <option v-for="p in [0,1,2,3,4]" :key="p" :value="p">Phase {{ p }}</option>
        </select>
        <select v-model="filters.access" class="bg-dark-hover border border-dark-border rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500">
          <option value="">All Access Types</option>
          <option v-for="a in accessTypes" :key="a" :value="a">{{ a }}</option>
        </select>
        <select v-model="filters.dataType" class="bg-dark-hover border border-dark-border rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500">
          <option value="">All Data Types</option>
          <option v-for="t in dataTypes" :key="t" :value="t">{{ t }}</option>
        </select>
        <select v-model="filters.status" class="bg-dark-hover border border-dark-border rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500">
          <option value="">All Statuses</option>
          <option value="pending">Pending</option>
          <option value="processing">Processing</option>
          <option value="completed">Completed</option>
          <option value="metadata_only">Metadata Only</option>
          <option value="failed">Failed</option>
        </select>
      </div>
    </div>

    <!-- Table -->
    <div class="card overflow-x-auto">
      <table class="w-full text-sm">
        <thead>
          <tr class="border-b border-dark-border text-left text-gray-400">
            <th class="px-4 py-3 cursor-pointer hover:text-white" @click="toggleSort('dataset_name')">
              Dataset {{ sortIcon('dataset_name') }}
            </th>
            <th class="px-4 py-3">Hazard</th>
            <th class="px-4 py-3">Type</th>
            <th class="px-4 py-3">Region</th>
            <th class="px-4 py-3">Phase</th>
            <th class="px-4 py-3">Status</th>
            <th class="px-4 py-3">Access</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="entry in filteredEntries"
            :key="entry.row_index"
            class="border-b border-dark-border/50 hover:bg-dark-hover cursor-pointer transition-colors"
            @click="selectedEntry = entry"
          >
            <td class="px-4 py-3 text-white font-medium">{{ entry.dataset_name }}</td>
            <td class="px-4 py-3 text-gray-300">{{ entry.hazard || '—' }}</td>
            <td class="px-4 py-3 text-gray-300">{{ entry.data_type || '—' }}</td>
            <td class="px-4 py-3 text-gray-300">{{ entry.region_country || entry.spatial_coverage || '—' }}</td>
            <td class="px-4 py-3">
              <span class="px-2 py-1 rounded-full text-xs font-medium" :class="phaseBadgeClass(entry.phase)">
                {{ entry.phase }}
              </span>
            </td>
            <td class="px-4 py-3">
              <span class="px-2 py-1 rounded-full text-xs font-medium" :class="statusBadgeClass(entry.processing_status)">
                {{ entry.processing_status }}
              </span>
            </td>
            <td class="px-4 py-3 text-gray-300 text-xs">{{ entry.access || '—' }}</td>
          </tr>
        </tbody>
      </table>
      <div class="text-center text-gray-500 text-sm py-4">
        Showing {{ filteredEntries.length }} of {{ catalog.length }} entries
      </div>
    </div>

    <!-- Detail Modal -->
    <div v-if="selectedEntry" class="fixed inset-0 bg-black/60 flex items-center justify-center z-50" @click.self="selectedEntry = null">
      <div class="bg-dark-card border border-dark-border rounded-xl p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-xl font-bold text-white">{{ selectedEntry.dataset_name }}</h3>
          <button @click="selectedEntry = null" class="text-gray-400 hover:text-white text-2xl">&times;</button>
        </div>
        <div class="grid grid-cols-2 gap-4 text-sm">
          <div v-for="(value, key) in selectedEntryFields" :key="key">
            <span class="text-gray-500 block">{{ formatFieldName(key) }}</span>
            <span class="text-white">{{ value || '—' }}</span>
          </div>
        </div>
        <div v-if="selectedEntry.link" class="mt-4">
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
const selectedEntry = ref(null)

const filters = ref({
  search: '',
  phase: '',
  access: '',
  dataType: '',
  status: '',
})
const sortField = ref('dataset_name')
const sortAsc = ref(true)

const accessTypes = computed(() => [...new Set(catalog.value.map(e => e.access).filter(Boolean))])
const dataTypes = computed(() => [...new Set(catalog.value.map(e => e.data_type).filter(Boolean))])

const filteredEntries = computed(() => {
  let result = [...catalog.value]

  if (filters.value.search) {
    const q = filters.value.search.toLowerCase()
    result = result.filter(e => (e.dataset_name || '').toLowerCase().includes(q))
  }
  if (filters.value.phase !== '') {
    result = result.filter(e => e.phase === Number(filters.value.phase))
  }
  if (filters.value.access) {
    result = result.filter(e => e.access === filters.value.access)
  }
  if (filters.value.dataType) {
    result = result.filter(e => e.data_type === filters.value.dataType)
  }
  if (filters.value.status) {
    result = result.filter(e => e.processing_status === filters.value.status)
  }

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
    hazard: e.hazard,
    data_type: e.data_type,
    spatial_coverage: e.spatial_coverage,
    region_country: e.region_country,
    spatial_resolution: e.spatial_resolution,
    temporal_coverage: e.temporal_coverage,
    temporal_resolution: e.temporal_resolution,
    bias_corrected: e.bias_corrected,
    impact_sector: e.impact_sector,
    phase: e.phase,
    processing_status: e.processing_status,
    notes: e.notes,
  }
})

function toggleSort(field) {
  if (sortField.value === field) {
    sortAsc.value = !sortAsc.value
  } else {
    sortField.value = field
    sortAsc.value = true
  }
}

function sortIcon(field) {
  if (sortField.value !== field) return ''
  return sortAsc.value ? '▲' : '▼'
}

function phaseColor(phase) {
  const colors = { 0: 'text-blue-400', 1: 'text-green-400', 2: 'text-yellow-400', 3: 'text-orange-400', 4: 'text-red-400' }
  return colors[phase] || 'text-gray-400'
}

function phaseLabel(phase) {
  const labels = { 0: 'Metadata only', 1: 'Direct download', 2: 'Registration', 3: 'API portal', 4: 'Manual' }
  return labels[phase] || ''
}

function phaseBadgeClass(phase) {
  const map = {
    0: 'bg-blue-500/20 text-blue-400',
    1: 'bg-green-500/20 text-green-400',
    2: 'bg-yellow-500/20 text-yellow-400',
    3: 'bg-orange-500/20 text-orange-400',
    4: 'bg-red-500/20 text-red-400',
  }
  return map[phase] || 'bg-gray-500/20 text-gray-400'
}

function statusBadgeClass(status) {
  const map = {
    pending: 'bg-gray-500/20 text-gray-400',
    processing: 'bg-blue-500/20 text-blue-400',
    completed: 'bg-green-500/20 text-green-400',
    metadata_only: 'bg-amber-500/20 text-amber-400',
    failed: 'bg-red-500/20 text-red-400',
  }
  return map[status] || 'bg-gray-500/20 text-gray-400'
}

function formatFieldName(key) {
  return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

async function refreshCatalog() {
  loading.value = true
  try {
    const [catResp, progResp] = await Promise.all([
      fetch('/catalog'),
      fetch('/catalog/progress'),
    ])
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
    if (resp.ok) {
      const data = await resp.json()
      console.log('Processing started:', data)
      // Start polling progress
      pollProgress()
    }
  } catch (e) {
    console.error('Failed to trigger processing:', e)
  } finally {
    processing.value = false
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
        if (progress.value.pending === 0) {
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
