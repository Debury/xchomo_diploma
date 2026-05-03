<template>
  <div class="space-y-5">
    <PageHeader title="Dataset Catalog" :subtitle="`${catalog.length || 233} climate data sources from D1.1.xlsx`">
      <template #actions>
        <template v-if="!progress?.thread_alive">
          <button
            @click="processCatalog([1])"
            :disabled="processing !== null"
            class="btn-primary disabled:opacity-50"
            title="Download and embed actual data files for catalog sources whose URL we can resolve. Skips sources already in Qdrant. Does NOT re-run metadata embedding."
          >
            {{ processing === 1 ? 'Starting…' : 'Download data' }}
          </button>
          <button
            @click="processCatalog([0])"
            :disabled="processing !== null"
            class="btn-secondary disabled:opacity-50"
            title="Phase 0 only — re-embed catalog metadata (the title/description/hazard summary chunks). Run this if you change the catalog Excel or want metadata refresh without touching downloaded data."
          >
            {{ processing === 0 ? 'Starting…' : 'Re-embed metadata' }}
          </button>
        </template>
        <button
          v-else
          @click="cancelProcessing"
          :disabled="cancelling"
          class="btn-danger disabled:opacity-50"
        >
          {{ cancelling ? 'Cancelling…' : 'Cancel' }}
        </button>
        <button @click="refreshCatalog" :disabled="loading" class="btn-secondary disabled:opacity-50">
          Refresh
        </button>
      </template>
    </PageHeader>

    <!-- Status summary — counts the user actually cares about -->
    <div class="grid grid-cols-3 gap-3">
      <div class="stat-card !p-4">
        <span class="text-[11px] font-medium uppercase tracking-wider text-mendelu-gray-dark">Completed</span>
        <span class="text-lg font-semibold text-mendelu-success tabular-nums block">
          {{ statusCounts.completed }} / {{ catalog.length || '—' }}
        </span>
        <p class="text-[10px] text-mendelu-gray-dark mt-0.5">data downloaded + embedded</p>
      </div>
      <div class="stat-card !p-4">
        <span class="text-[11px] font-medium uppercase tracking-wider text-mendelu-gray-dark">Needs attention</span>
        <span class="text-lg font-semibold text-amber-600 tabular-nums block">
          {{ statusCounts.metadata_only }}
        </span>
        <p class="text-[10px] text-mendelu-gray-dark mt-0.5">metadata only — see reasons below</p>
      </div>
      <div class="stat-card !p-4">
        <span class="text-[11px] font-medium uppercase tracking-wider text-mendelu-gray-dark">Total chunks</span>
        <span class="text-lg font-semibold text-mendelu-black tabular-nums block">
          {{ qdrantSummary?.totalChunks?.toLocaleString() || '—' }}
        </span>
        <p class="text-[10px] text-mendelu-gray-dark mt-0.5">across all sources in Qdrant</p>
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

    <!-- Per-batch progress bar removed: it reported in-batch internal state
         (Qdrant guard pre-skips counted as "processed") and contradicted
         the headline "Completed X / 233" stat cards above. The thread-alive
         banner up top + "current source" indicator there is enough live
         feedback. -->
    <!--
    <div v-if="progress && progress.total > 0" class="card !p-4">
      … (intentionally removed) …
    </div>
    -->
    <div v-if="progress && progress.thread_alive && progress.failed > 0" class="card !p-3 border-mendelu-alert/30">
      <div class="flex items-center gap-2 text-xs">
        <span class="w-2 h-2 rounded-full bg-mendelu-alert"></span>
        <span class="text-mendelu-alert font-medium">{{ progress.failed }} {{ progress.failed === 1 ? 'failure' : 'failures' }} in this batch</span>
        <span class="text-mendelu-gray-dark ml-2">— see "Reason / Notes" column for per-row details once the batch finishes</span>
      </div>
    </div>

    <!-- Filters — kept simple. The phase / data-type filters were thesis-
         only abstractions; users on this page just want "show me what's
         broken" or "find this dataset". -->
    <div class="flex gap-3 flex-wrap">
      <input v-model="filters.search" type="text" placeholder="Search datasets..." class="input-field !w-56" />
      <select v-model="filters.status" class="input-field !w-auto">
        <option value="">All sources</option>
        <option value="completed">✅ Completed</option>
        <option value="metadata_only">⚠️ Needs attention</option>
        <option value="pending">Not yet processed</option>
      </select>
      <select v-model="filters.access" class="input-field !w-auto">
        <option value="">All access types</option>
        <option v-for="a in accessTypes" :key="a" :value="a">{{ a }}</option>
      </select>
    </div>

    <!-- Table — fixed layout with explicit column widths so Reason can't
         push Link off-screen. Long values truncate with a tooltip; click
         the row to see the full record in the modal. -->
    <div class="card !p-0 overflow-hidden">
      <div class="overflow-x-auto">
        <table class="w-full text-sm table-fixed">
          <colgroup>
            <col style="width: 22%" />
            <col style="width: 11%" />
            <col style="width: 13%" />
            <col style="width: 8%" />
            <col style="width: 14%" />
            <col style="width: 27%" />
            <col style="width: 5%" />
          </colgroup>
          <thead class="sticky top-0 bg-white z-10">
            <tr class="border-b border-mendelu-gray-semi">
              <th class="table-header cursor-pointer hover:text-mendelu-black transition-colors duration-150" @click="toggleSort('dataset_name')">
                Dataset {{ sortIcon('dataset_name') }}
              </th>
              <th class="table-header">Hazard</th>
              <th class="table-header">Region</th>
              <th class="table-header cursor-pointer hover:text-mendelu-black transition-colors duration-150" @click="toggleSort('chunk_count')">
                Chunks {{ sortIcon('chunk_count') }}
              </th>
              <th class="table-header">Status</th>
              <th class="table-header">Reason / Notes</th>
              <th class="table-header text-right">Link</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="entry in filteredEntries"
              :key="entry.row_index"
              class="border-b border-mendelu-gray-semi/50 hover:bg-mendelu-gray-light cursor-pointer transition-all duration-150"
              @click="selectedEntry = entry"
            >
              <td class="px-4 py-2.5 text-mendelu-black text-sm font-medium truncate" :title="entry.dataset_name">{{ entry.dataset_name }}</td>
              <td class="px-4 py-2.5 text-mendelu-gray-dark text-xs truncate" :title="entry.hazard || ''">{{ entry.hazard || '--' }}</td>
              <td class="px-4 py-2.5 text-mendelu-gray-dark text-xs truncate" :title="entry.region_country || entry.spatial_coverage || ''">{{ entry.region_country || entry.spatial_coverage || '--' }}</td>
              <td class="px-4 py-2.5 tabular-nums text-xs">
                <span v-if="entry.chunk_count > 10" class="text-mendelu-success font-medium">{{ entry.chunk_count.toLocaleString() }}</span>
                <span v-else-if="entry.chunk_count > 0" class="text-amber-600">{{ entry.chunk_count }}</span>
                <span v-else class="text-mendelu-gray-dark">--</span>
              </td>
              <td class="px-4 py-2.5 whitespace-nowrap overflow-hidden">
                <span :class="statusBadgeClass(entry.processing_status)">
                  {{ statusLabel(entry.processing_status) }}
                </span>
                <span
                  v-if="entry.ingest_progress?.is_alive"
                  class="ml-1.5 text-[10px] px-1.5 py-0.5 rounded bg-mendelu-green/10 text-mendelu-green font-medium"
                  :title="`Ingest running: ${entry.ingest_progress.done_files}/${entry.ingest_progress.total_files} files`"
                >
                  ⏳ {{ entry.ingest_progress.done_files }}/{{ entry.ingest_progress.total_files }}
                </span>
                <span
                  v-else-if="entry.ingest_progress?.is_killed"
                  class="ml-1.5 text-[10px] px-1.5 py-0.5 rounded bg-amber-50 text-amber-700 font-medium"
                  :title="`Killed mid-ingest: ${entry.ingest_progress.done_files}/${entry.ingest_progress.total_files} files. Click row → Resume.`"
                >
                  ⚠ {{ entry.ingest_progress.done_files }}/{{ entry.ingest_progress.total_files }}
                </span>
              </td>
              <td class="px-4 py-2.5 text-xs text-mendelu-gray-dark truncate" :title="entry.error || ''">
                {{ entry.error || '—' }}
              </td>
              <td class="px-4 py-2.5 text-right whitespace-nowrap" @click.stop>
                <a v-if="entry.link" :href="entry.link" target="_blank"
                   class="text-mendelu-green hover:text-mendelu-green-hover text-xs transition-colors duration-150">
                  Open
                </a>
                <span v-else class="text-mendelu-gray-dark text-xs">--</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <div class="text-center text-mendelu-gray-dark text-xs py-3 border-t border-mendelu-gray-semi/50">
        {{ filteredEntries.length }} of {{ catalog.length }} entries
      </div>
    </div>

    <!-- Detail Modal -->
    <Modal
      :open="!!selectedEntry"
      :title="selectedEntry?.dataset_name || ''"
      max-width="xl"
      @close="selectedEntry = null"
    >
      <template v-if="selectedEntry">
        <!-- Manual-ingest partial progress banner. Shown when a long-
             running script (SPEI-GD multi-file ingest, etc.) was started
             but didn't finish — the user knows the script path so they
             can resume from terminal. -->
        <div v-if="selectedEntry.ingest_progress?.is_partial"
             class="mb-4 p-3 rounded-lg border border-amber-200 bg-amber-50">
          <div class="flex items-center justify-between mb-1">
            <span class="text-[11px] font-medium uppercase tracking-wider text-amber-700">⚠ Partial ingest</span>
            <span class="text-xs tabular-nums text-amber-700">
              {{ selectedEntry.ingest_progress.done_files }} / {{ selectedEntry.ingest_progress.total_files }} files
            </span>
          </div>
          <div class="w-full bg-amber-100 rounded-full h-1.5 mb-2">
            <div class="bg-amber-500 h-1.5 rounded-full transition-all duration-300"
                 :style="{ width: `${(selectedEntry.ingest_progress.done_files / selectedEntry.ingest_progress.total_files) * 100}%` }"></div>
          </div>
          <p class="text-[11px] text-amber-900">
            Last update {{ formatTime(selectedEntry.ingest_progress.updated_at) }}.
            <span v-if="selectedEntry.ingest_progress.is_alive">Script is running — heartbeat is fresh.</span>
            <span v-else>Script is killed — resume below to pick up where it stopped.</span>
          </p>
          <div class="mt-2 flex items-center gap-2">
            <button
              v-if="selectedEntry.ingest_progress.is_killed"
              @click="resumeIngest(selectedEntry)"
              :disabled="resumingRow === selectedEntry.row_index"
              class="text-xs px-3 py-1.5 rounded bg-amber-600 text-white hover:bg-amber-700 disabled:bg-amber-300 disabled:cursor-not-allowed transition-colors duration-150"
            >
              {{ resumingRow === selectedEntry.row_index ? 'Launching…' : '↻ Resume ingest' }}
            </button>
            <span v-if="resumeMessage" class="text-[11px] text-amber-900">{{ resumeMessage }}</span>
          </div>
        </div>

        <!-- Reason / error banner — first thing the user should see when
             investigating a row. -->
        <div v-if="selectedEntry.error"
             class="mb-4 p-3 rounded-lg border"
             :class="selectedEntry.processing_status === 'completed' ? 'bg-mendelu-gray-light border-mendelu-gray-semi' : 'bg-amber-50 border-amber-200'">
          <span class="text-[11px] font-medium uppercase tracking-wider"
                :class="selectedEntry.processing_status === 'completed' ? 'text-mendelu-gray-dark' : 'text-amber-700'">
            Reason
          </span>
          <p class="text-sm mt-1" :class="selectedEntry.processing_status === 'completed' ? 'text-mendelu-black' : 'text-amber-900'">
            {{ selectedEntry.error }}
          </p>
        </div>

        <!-- Qdrant Stats -->
        <div v-if="selectedEntry.chunk_count > 0" class="mb-4 p-3 bg-mendelu-success/5 border border-mendelu-success/20 rounded-lg">
          <div class="flex items-center justify-between mb-2">
            <span class="text-xs font-medium text-mendelu-success uppercase tracking-wider">In Qdrant</span>
            <span class="text-sm font-semibold text-mendelu-success tabular-nums">{{ selectedEntry.chunk_count.toLocaleString() }} chunks</span>
          </div>
          <div v-if="selectedEntry.qdrant_variables && selectedEntry.qdrant_variables.length" class="flex flex-wrap gap-1">
            <span v-for="v in selectedEntry.qdrant_variables" :key="v" class="badge-info !text-[10px]">{{ v }}</span>
          </div>
        </div>
        <div v-else class="mb-4 p-3 bg-mendelu-gray-light border border-mendelu-gray-semi rounded-lg">
          <span class="text-xs text-mendelu-gray-dark">No data chunks in Qdrant yet</span>
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

        <!-- Actions -->
        <div class="mt-4 pt-3 border-t border-mendelu-gray-semi flex gap-2">
          <button
            @click="retryRow(selectedEntry)"
            :disabled="selectedEntry.reprocessing || progress?.thread_alive"
            class="btn-primary flex-1 text-xs disabled:opacity-50"
            :title="progress?.thread_alive ? 'A batch is already running — cancel it first' : 'Reset progress and try downloading this single source again'"
          >
            {{ selectedEntry.reprocessing ? 'Retrying…' : 'Retry this source' }}
          </button>
          <button
            v-if="selectedEntry.chunk_count > 0"
            @click="deleteDatasetEmbeddings(selectedEntry)"
            class="btn-ghost text-xs text-mendelu-alert hover:bg-mendelu-alert/10"
          >
            Delete embeddings
          </button>
        </div>
      </template>
    </Modal>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import PageHeader from '../components/PageHeader.vue'
import Modal from '../components/Modal.vue'
import { apiFetch } from '../api'
import { useConfirm } from '../composables/useConfirm'
import { useToast } from '../composables/useToast'

const { confirm } = useConfirm()
const toast = useToast()

const catalog = ref<any[]>([])
const qdrantDatasets = ref<any[]>([])
const progress = ref<any>(null)
const loading = ref(false)
// processing holds the active phase (0 or 1) while a trigger request is
// in flight, or null when no request is being made. Used to label the
// active button as "Starting…" so the user knows their click registered.
const processing = ref<number | null>(null)
const cancelling = ref(false)
const restarting = ref(false)
const selectedEntry = ref<any>(null)
const resumingRow = ref<number | null>(null)
const resumeMessage = ref<string>('')

const filters = ref<any>({ search: '', access: '', status: '' })
const sortField = ref('dataset_name')
const sortAsc = ref(true)

const accessTypes = computed(() => [...new Set(catalog.value.map(e => e.access).filter(Boolean))])

const qdrantSummary = computed(() => {
  if (!qdrantDatasets.value.length) return null
  const totalDatasets = qdrantDatasets.value.length
  const totalChunks = qdrantDatasets.value.reduce((sum, d) => sum + d.chunk_count, 0)
  const withData = qdrantDatasets.value.filter(d => d.chunk_count > 10 && !d.is_metadata_only).length
  return { totalDatasets, totalChunks, withData }
})

// Roll-up of processing_status across the catalog. The hero stat card uses
// these so the user immediately sees "X out of 233 are done", not a
// per-phase breakdown that means nothing without context.
const statusCounts = computed(() => {
  const out = { completed: 0, metadata_only: 0, pending: 0 }
  for (const e of catalog.value) {
    const s = e.processing_status as keyof typeof out
    if (s in out) out[s] += 1
  }
  return out
})

function pctOf(value) {
  const total = progress.value?.total || 1
  return Math.min(100, Math.round((value / total) * 100))
}

// Build a lookup from dataset_name -> qdrant info
const qdrantLookup = computed(() => {
  const map = {}
  for (const d of qdrantDatasets.value) {
    map[d.dataset_name] = d
  }
  return map
})

// Enrich catalog entries with Qdrant data
const enrichedCatalog = computed(() => {
  return catalog.value.map(entry => {
    const qd = qdrantLookup.value[entry.dataset_name]
    return {
      ...entry,
      chunk_count: qd ? qd.chunk_count : 0,
      qdrant_variables: qd ? qd.variables.map(v => v.name) : [],
      qdrant_link: qd ? qd.link : null,
      is_metadata_only: qd ? qd.is_metadata_only : false,
      qdrant_location: qd ? qd.location_name : null,
      qdrant_hazard: qd ? qd.hazard_type : null,
    }
  })
})

const filteredEntries = computed(() => {
  let result = [...enrichedCatalog.value]
  if (filters.value.search) {
    const q = filters.value.search.toLowerCase()
    result = result.filter(e => (e.dataset_name || '').toLowerCase().includes(q))
  }
  if (filters.value.access) result = result.filter(e => e.access === filters.value.access)
  if (filters.value.status) result = result.filter(e => e.processing_status === filters.value.status)

  result.sort((a, b) => {
    if (sortField.value === 'chunk_count') {
      return sortAsc.value ? a.chunk_count - b.chunk_count : b.chunk_count - a.chunk_count
    }
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

function formatTime(iso?: string | null): string {
  if (!iso) return '—'
  try {
    const d = new Date(iso)
    if (isNaN(d.getTime())) return iso
    return d.toLocaleString()
  } catch {
    return iso || '—'
  }
}

function statusLabel(status: string) {
  return ({
    completed: 'Completed',
    metadata_only: 'Needs attention',
    pending: 'Not processed',
    processing: 'Processing',
    failed: 'Failed',
  } as Record<string, string>)[status] || status
}
function statusBadgeClass(status: string) {
  return ({
    pending: 'badge-neutral',
    processing: 'badge-info',
    completed: 'badge-success',
    metadata_only: 'badge-warning',
    failed: 'badge-danger',
  } as Record<string, string>)[status] || 'badge-neutral'
}
function formatFieldName(key) { return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) }

async function loadQdrantDatasets() {
  try {
    const resp = await apiFetch('/qdrant/datasets')
    if (resp.ok) qdrantDatasets.value = await resp.json()
  } catch (e: any) {
    console.error('Failed to load Qdrant datasets:', e)
    toast.error(`Could not load datasets: ${e?.message || 'network error'}`)
  }
}

async function refreshCatalog() {
  loading.value = true
  try {
    const [catResp, progResp] = await Promise.all([apiFetch('/catalog'), apiFetch('/catalog/progress'), loadQdrantDatasets()])
    if (catResp.ok) catalog.value = await catResp.json()
    if (progResp.ok) progress.value = await progResp.json()
  } catch (e: any) {
    console.error('Failed to load catalog:', e)
    toast.error(`Could not load catalog: ${e?.message || 'network error'}`)
  }
  finally { loading.value = false }
}

// Two distinct flows so the user controls cost:
//   • [1] "Download data": Phase 1 only — fetches and chunks the actual
//     raster/CSV files. The expensive one. Skips Phase 0 entirely so
//     repeated clicks don't burn GPU re-embedding 233 metadata texts that
//     are already in Qdrant.
//   • [0] "Re-embed metadata": Phase 0 only — refreshes the small
//     metadata-summary chunks. Cheap. Use after editing the catalog
//     Excel or when metadata-text generation rules change.
async function processCatalog(phases: number[]) {
  processing.value = phases[0]  // 0 or 1, drives button label
  try {
    const resp = await apiFetch('/catalog/process', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ phases }),
    })
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}))
      throw new Error(err.detail || `HTTP ${resp.status}`)
    }
    const label = phases.includes(1) ? 'Data download started' : 'Metadata re-embed started'
    toast.success(`${label} — leave it running and check back later`)
    pollProgress()
  } catch (e: any) {
    console.error('Failed to trigger processing:', e)
    toast.error(`Could not start processing: ${e?.message || 'network error'}`)
  }
  finally { processing.value = null }
}

async function cancelProcessing() {
  cancelling.value = true
  try {
    const resp = await apiFetch('/catalog/cancel', { method: 'POST' })
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}))
      throw new Error(err.detail || `HTTP ${resp.status}`)
    }
    const data = await resp.json()
    toast.success(`Cancel signal sent. ${data.rows_reset || 0} stuck rows reset.`)
    setTimeout(() => refreshCatalog(), 2000)
  } catch (e: any) {
    console.error('Cancel failed:', e)
    toast.error(`Cancel failed: ${e?.message || 'network error'}`)
  } finally {
    cancelling.value = false
  }
}

// Per-row retry: hits /catalog/{row_index}/retry which clears the row's
// progress entry and re-runs the batch pipeline with row_filter set to
// just this one source. Designed for the "this one Zenodo URL is dead,
// the others are fine" workflow.
async function resumeIngest(entry: any) {
  if (!entry) return
  resumingRow.value = entry.row_index
  resumeMessage.value = ''
  try {
    const resp = await apiFetch('/catalog/resume-ingest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        source_id: entry.source_id,
        dataset_name: entry.dataset_name,
      }),
    })
    const data = await resp.json().catch(() => ({}))
    if (!resp.ok) {
      throw new Error(data.detail || `HTTP ${resp.status}`)
    }
    resumeMessage.value = `Launched (pid ${data.pid}). Badge will refresh momentarily.`
    toast.success(`Resumed ingest for ${entry.dataset_name}`)
    setTimeout(refreshCatalog, 3000)
  } catch (e: any) {
    console.error('Resume failed:', e)
    resumeMessage.value = `Failed: ${e?.message || 'network error'}`
    toast.error(`Resume failed: ${e?.message || 'network error'}`)
  } finally {
    resumingRow.value = null
  }
}

async function retryRow(entry: any) {
  if (!entry || entry.reprocessing) return
  entry.reprocessing = true
  try {
    const resp = await apiFetch(`/catalog/${entry.row_index}/retry`, { method: 'POST' })
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}))
      throw new Error(err.detail || `HTTP ${resp.status}`)
    }
    toast.success(`Retrying ${entry.dataset_name || entry.source_id}…`)
    pollProgress()
  } catch (e: any) {
    console.error('Error retrying row:', e)
    toast.error(`Retry failed: ${e?.message || 'network error'}`)
  } finally {
    entry.reprocessing = false
  }
}

async function deleteDatasetEmbeddings(entry) {
  const sourceId = entry.source_id
  if (!sourceId) return
  const ok = await confirm({
    title: 'Delete embeddings?',
    message: `Delete all embeddings for "${entry.dataset_name}"?\nThis cannot be undone.`,
    confirmText: 'Delete',
    danger: true,
  })
  if (!ok) return
  try {
    const resp = await apiFetch(`/sources/${sourceId}/embeddings?confirm=true`, { method: 'DELETE' })
    if (resp.ok) {
      // Clear cache and refresh
      await apiFetch('/qdrant/cache/clear', { method: 'POST' })
      await loadQdrantDatasets()
      selectedEntry.value = null
    }
  } catch (e: any) {
    console.error('Error deleting embeddings:', e)
    toast.error(`Delete failed: ${e?.message || 'network error'}`)
  }
}

async function autoRestart() {
  restarting.value = true
  try {
    const resp = await apiFetch('/catalog/auto-restart', { method: 'POST' })
    if (resp.ok) pollProgress()
    else {
      const data = await resp.json().catch(() => ({}))
      const detail = data.detail || data.message || resp.statusText
      console.error('Auto-restart failed:', detail)
      toast.error(`Auto-restart failed: ${detail}`)
    }
  } catch (e: any) {
    console.error('Auto-restart error:', e)
    toast.error(`Auto-restart error: ${e?.message || 'network error'}`)
  }
  finally { restarting.value = false }
}

let progressInterval = null
let progressTickCount = 0
const MAX_POLL_TICKS = 400  // ~20 min of 3-second polls — hard cap so a stuck
                            // batch (thread_alive=false but pending>0) can't
                            // keep firing requests forever.

function stopProgressPolling() {
  if (progressInterval) {
    clearInterval(progressInterval)
    progressInterval = null
  }
  progressTickCount = 0
}

function pollProgress() {
  if (progressInterval) return
  progressTickCount = 0
  progressInterval = setInterval(async () => {
    progressTickCount += 1
    if (progressTickCount > MAX_POLL_TICKS) {
      stopProgressPolling()
      return
    }
    try {
      const resp = await apiFetch('/catalog/progress')
      if (resp.ok) {
        progress.value = await resp.json()
        // Stop polling when the batch is finished OR stalled (no thread, no pending).
        if (!progress.value.thread_alive && (progress.value.pending || 0) === 0) {
          stopProgressPolling()
          refreshCatalog()
        }
        // Also stop if the thread crashed — don't keep spamming.
        if (progress.value.thread_crashed) {
          stopProgressPolling()
        }
      }
    } catch (e) {
      stopProgressPolling()
    }
  }, 3000)
}

onMounted(() => {
  refreshCatalog()
  // If we land on the page mid-batch, start polling so the live banner
  // appears without a manual Refresh.
  setTimeout(() => {
    if (progress.value?.thread_alive) pollProgress()
  }, 500)
})
onUnmounted(() => stopProgressPolling())
</script>
