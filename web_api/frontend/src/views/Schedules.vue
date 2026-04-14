<template>
  <div class="space-y-6">
    <PageHeader title="Schedules" subtitle="Manage dataset and per-source update schedules">
      <template #actions>
        <button @click="refreshAll" :disabled="loading" class="btn-ghost disabled:opacity-50">Refresh</button>
        <button @click="showCreateModal = true" class="btn-primary">Create Schedule</button>
      </template>
    </PageHeader>

    <!-- Dataset Schedules -->
    <div class="card">
      <div class="flex items-center justify-between mb-4">
        <div>
          <h3 class="text-sm font-bold text-mendelu-black">Dataset Schedules</h3>
          <p class="text-xs text-mendelu-gray-dark mt-0.5">Reprocesses all sources under a dataset on a cron schedule</p>
        </div>
      </div>

      <div v-if="datasetSchedules.length === 0" class="text-center py-6 text-mendelu-gray-dark text-sm">
        No dataset schedules configured
      </div>

      <div v-else class="space-y-3">
        <div v-for="sched in datasetSchedules" :key="sched.id"
             class="bg-mendelu-gray-light rounded-lg p-4 hover:bg-mendelu-gray-semi/50 transition-all duration-150">
          <div class="flex items-center justify-between">
            <div class="flex-1">
              <div class="flex items-center gap-2 mb-1">
                <span class="text-sm font-medium text-mendelu-black">{{ sched.name }}</span>
                <span :class="sched.is_enabled ? 'badge-success' : 'badge-neutral'">
                  {{ sched.is_enabled ? 'Active' : 'Paused' }}
                </span>
              </div>
              <div class="flex gap-4 text-xs text-mendelu-gray-dark">
                <span>Dataset: <span class="text-mendelu-black font-medium">{{ sched.dataset_name }}</span></span>
                <span>Cron: <code class="text-mendelu-black font-mono bg-white px-1.5 py-0.5 rounded border border-mendelu-gray-semi">{{ sched.cron_expression }}</code></span>
                <span v-if="sched.next_run_at">Next: {{ formatTime(sched.next_run_at) }}</span>
                <span v-if="sched.last_triggered_at">Last: {{ formatTime(sched.last_triggered_at) }}</span>
              </div>
            </div>
            <button @click="deleteDatasetSchedule(sched.id, sched.name)" class="btn-ghost text-xs text-mendelu-alert hover:bg-mendelu-alert/10">Remove</button>
          </div>
        </div>
      </div>
    </div>

    <!-- Per-Source Schedules -->
    <div class="card">
      <div class="flex items-center justify-between mb-4">
        <div>
          <h3 class="text-sm font-bold text-mendelu-black">Source Schedules</h3>
          <p class="text-xs text-mendelu-gray-dark mt-0.5">Per-source cron schedules — set via the Sources page or the edit modal</p>
        </div>
      </div>

      <div v-if="sourceSchedules.length === 0" class="text-center py-6">
        <p class="text-mendelu-gray-dark text-sm">No source schedules configured</p>
        <router-link to="/sources" class="text-xs text-mendelu-green hover:text-mendelu-green-hover mt-1 inline-block">
          Configure on the Sources page
        </router-link>
      </div>

      <div v-else class="space-y-3">
        <div v-for="sched in sourceSchedules" :key="sched.source_id"
             class="bg-mendelu-gray-light rounded-lg p-4 hover:bg-mendelu-gray-semi/50 transition-all duration-150">
          <div class="flex items-center justify-between">
            <div class="flex-1">
              <div class="flex items-center gap-2">
                <span class="text-sm font-medium text-mendelu-black">{{ sched.source_id }}</span>
                <span :class="sched.is_enabled ? 'badge-success' : 'badge-neutral'">
                  {{ sched.is_enabled ? 'Active' : 'Paused' }}
                </span>
              </div>
              <div class="flex gap-3 mt-1 text-xs text-mendelu-gray-dark">
                <span>Cron: <code class="text-mendelu-black font-mono bg-white px-1.5 py-0.5 rounded border border-mendelu-gray-semi">{{ sched.cron_expression }}</code></span>
                <span v-if="sched.next_run_at">Next: {{ formatTime(sched.next_run_at) }}</span>
                <span v-if="sched.last_triggered_at">Last: {{ formatTime(sched.last_triggered_at) }}</span>
              </div>
            </div>
            <div class="flex gap-2 items-center">
              <button @click="editSourceSchedule(sched)" class="btn-ghost text-xs">Edit</button>
              <button @click="deleteSourceSchedule(sched.source_id)" class="btn-ghost text-xs text-mendelu-alert hover:bg-mendelu-alert/10">Remove</button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Schedule Reference -->
    <div class="card">
      <button @click="showReference = !showReference" class="flex items-center justify-between w-full">
        <h3 class="text-sm font-medium text-mendelu-black">Cron Reference</h3>
        <svg class="w-4 h-4 text-mendelu-gray-dark transition-transform duration-150" :class="{ 'rotate-180': showReference }" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      <div v-if="showReference" class="mt-4 pt-4 border-t border-mendelu-gray-semi text-mendelu-gray-dark font-mono text-xs space-y-1">
        <p><span class="text-mendelu-black">0 2 * * *</span> &mdash; Daily at 2:00 AM</p>
        <p><span class="text-mendelu-black">0 3 * * 0</span> &mdash; Weekly Sunday 3am</p>
        <p><span class="text-mendelu-black">0 0 * * 2</span> &mdash; Every Tuesday midnight</p>
        <p><span class="text-mendelu-black">0 0 1 * *</span> &mdash; Monthly on the 1st</p>
        <p><span class="text-mendelu-black">0 */6 * * *</span> &mdash; Every 6 hours</p>
      </div>
    </div>

    <!-- Create Dataset Schedule Modal -->
    <div v-if="showCreateModal" class="fixed inset-0 bg-black/40 flex items-center justify-center z-50" @click.self="showCreateModal = false">
      <div class="bg-white border border-mendelu-gray-semi rounded-xl p-6 max-w-md w-full mx-4 shadow-lg">
        <div class="flex items-center justify-between mb-6">
          <h2 class="text-lg font-semibold text-mendelu-black">Create Dataset Schedule</h2>
          <button @click="showCreateModal = false" class="btn-ghost !px-2 !py-1">&times;</button>
        </div>

        <form @submit.prevent="createDatasetSchedule" class="space-y-4">
          <div>
            <label class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Schedule Name</label>
            <input v-model="createForm.name" type="text" class="input-field" placeholder="e.g., Weekly NCEP update" required />
          </div>
          <div>
            <label class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Dataset</label>
            <select v-model="createForm.dataset_name" class="input-field" required>
              <option value="">Select dataset...</option>
              <option v-for="ds in availableDatasets" :key="ds" :value="ds">{{ ds }}</option>
            </select>
          </div>
          <CronPicker v-model="createForm.cron_expression" label="Schedule" />
          <div class="flex gap-3 pt-2">
            <button type="submit" :disabled="creating" class="btn-primary flex-1 disabled:opacity-50">
              {{ creating ? 'Creating...' : 'Create' }}
            </button>
            <button type="button" @click="showCreateModal = false" class="btn-secondary flex-1">Cancel</button>
          </div>
        </form>
      </div>
    </div>

    <!-- Edit Source Schedule Modal -->
    <div v-if="editingSched" class="fixed inset-0 bg-black/40 flex items-center justify-center z-50" @click.self="editingSched = null">
      <div class="bg-white border border-mendelu-gray-semi rounded-xl p-6 max-w-md w-full mx-4 shadow-lg">
        <div class="flex items-center justify-between mb-6">
          <h2 class="text-lg font-semibold text-mendelu-black">Edit: {{ editingSched.source_id }}</h2>
          <button @click="editingSched = null" class="btn-ghost !px-2 !py-1">&times;</button>
        </div>
        <form @submit.prevent="saveSourceSchedule" class="space-y-4">
          <CronPicker v-model="editSchedCron" />
          <div class="flex items-center gap-3">
            <label class="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" v-model="editSchedEnabled" class="sr-only peer">
              <div class="w-9 h-5 bg-mendelu-gray-semi rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-mendelu-green"></div>
            </label>
            <span class="text-sm text-mendelu-black">Enabled</span>
          </div>
          <div class="flex gap-3 pt-2">
            <button type="submit" :disabled="savingSched" class="btn-primary flex-1 disabled:opacity-50">
              {{ savingSched ? 'Saving...' : 'Save' }}
            </button>
            <button type="button" @click="editingSched = null" class="btn-secondary flex-1">Cancel</button>
          </div>
        </form>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import PageHeader from '../components/PageHeader.vue'
import CronPicker from '../components/CronPicker.vue'

const datasetSchedules = ref([])
const sourceSchedules = ref([])
const availableDatasets = ref([])
const loading = ref(false)
const showReference = ref(false)
const showCreateModal = ref(false)
const creating = ref(false)
const createForm = ref({ name: '', dataset_name: '', cron_expression: '' })

const editingSched = ref(null)
const editSchedCron = ref('')
const editSchedEnabled = ref(true)
const savingSched = ref(false)

function formatTime(ts) {
  if (!ts) return '---'
  try {
    const date = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts)
    return date.toLocaleString()
  } catch { return String(ts) }
}

async function refreshDatasetSchedules() {
  try {
    const resp = await fetch('/schedules/datasets')
    if (resp.ok) datasetSchedules.value = await resp.json()
  } catch (e) { console.error('Failed to load dataset schedules:', e) }
}

async function refreshSourceSchedules() {
  try {
    const resp = await fetch('/sources/')
    if (!resp.ok) return
    const sources = await resp.json()
    sourceSchedules.value = sources
      .filter(s => s.schedule)
      .map(s => ({ source_id: s.source_id, ...s.schedule }))
  } catch (e) { console.error('Failed to load source schedules:', e) }
}

async function loadAvailableDatasets() {
  try {
    const resp = await fetch('/qdrant/datasets')
    if (resp.ok) {
      const data = await resp.json()
      availableDatasets.value = data.map(d => d.dataset_name).sort()
    }
  } catch (e) { console.error('Failed to load datasets:', e) }
}

async function refreshAll() {
  loading.value = true
  try {
    await Promise.all([refreshDatasetSchedules(), refreshSourceSchedules(), loadAvailableDatasets()])
  } finally {
    loading.value = false
  }
}

async function createDatasetSchedule() {
  creating.value = true
  try {
    const resp = await fetch('/schedules/datasets', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(createForm.value),
    })
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: 'Error' }))
      throw new Error(err.detail || `HTTP ${resp.status}`)
    }
    showCreateModal.value = false
    createForm.value = { name: '', dataset_name: '', cron_expression: '' }
    await refreshDatasetSchedules()
  } catch (e) {
    alert(`Error: ${e.message}`)
  } finally {
    creating.value = false
  }
}

async function deleteDatasetSchedule(id, name) {
  if (!confirm(`Delete schedule "${name}"?`)) return
  try {
    await fetch(`/schedules/datasets/${id}`, { method: 'DELETE' })
    await refreshDatasetSchedules()
  } catch (e) { console.error('Error:', e) }
}

function editSourceSchedule(sched) {
  editingSched.value = sched
  editSchedCron.value = sched.cron_expression || ''
  editSchedEnabled.value = sched.is_enabled !== false
}

async function saveSourceSchedule() {
  savingSched.value = true
  try {
    const resp = await fetch(`/sources/${editingSched.value.source_id}/schedule`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cron_expression: editSchedCron.value, is_enabled: editSchedEnabled.value }),
    })
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    editingSched.value = null
    await refreshSourceSchedules()
  } catch (e) { alert(`Error: ${e.message}`) }
  finally { savingSched.value = false }
}

async function deleteSourceSchedule(sourceId) {
  if (!confirm(`Remove schedule for "${sourceId}"?`)) return
  try {
    await fetch(`/sources/${sourceId}/schedule`, { method: 'DELETE' })
    await refreshSourceSchedules()
  } catch (e) { console.error('Error:', e) }
}

onMounted(refreshAll)
</script>
