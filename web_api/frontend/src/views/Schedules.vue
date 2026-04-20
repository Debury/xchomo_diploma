<template>
  <div class="space-y-6">
    <PageHeader title="Schedules" subtitle="Manage per-source update schedules">
      <template #actions>
        <button @click="refreshAll" :disabled="loading" class="btn-ghost disabled:opacity-50">Refresh</button>
        <button @click="openCreateModal" class="btn-primary">Create Schedule</button>
      </template>
    </PageHeader>

    <!-- Per-Source Schedules -->
    <div class="card">
      <div class="flex items-center justify-between mb-4">
        <div>
          <h3 class="text-sm font-bold text-mendelu-black">Source Schedules</h3>
          <p class="text-xs text-mendelu-gray-dark mt-0.5">Each schedule fires the ETL job for a single source on a cron expression.</p>
        </div>
      </div>

      <div v-if="loading && sourceSchedules.length === 0" class="text-center py-6 text-mendelu-gray-dark text-sm">
        Loading schedules…
      </div>

      <div v-else-if="errorMessage" class="p-3 border-l-2 border-mendelu-alert bg-mendelu-alert/5 rounded text-sm text-mendelu-alert">
        {{ errorMessage }}
      </div>

      <div v-else-if="sourceSchedules.length === 0" class="text-center py-6">
        <p class="text-mendelu-gray-dark text-sm">No schedules configured yet.</p>
        <button @click="openCreateModal" class="text-xs text-mendelu-green hover:text-mendelu-green-hover mt-1 inline-block">
          Create your first schedule
        </button>
      </div>

      <div v-else class="space-y-3">
        <div v-for="sched in sourceSchedules" :key="sched.source_id"
             class="bg-mendelu-gray-light rounded-lg p-4 hover:bg-mendelu-gray-semi/50 transition-all duration-150">
          <div class="flex items-center justify-between">
            <div class="flex-1 min-w-0">
              <div class="flex items-center gap-2">
                <span class="text-sm font-medium text-mendelu-black truncate">{{ sched.dataset_name || sched.source_id }}</span>
                <span :class="sched.is_enabled ? 'badge-success' : 'badge-neutral'">
                  {{ sched.is_enabled ? 'Active' : 'Paused' }}
                </span>
              </div>
              <div class="flex flex-wrap gap-3 mt-1 text-xs text-mendelu-gray-dark">
                <span v-if="sched.dataset_name">Source: <span class="text-mendelu-black font-mono">{{ sched.source_id }}</span></span>
                <span>Cron: <code class="text-mendelu-black font-mono bg-white px-1.5 py-0.5 rounded border border-mendelu-gray-semi">{{ sched.cron_expression }}</code></span>
                <span v-if="sched.next_run_at">Next: {{ formatTime(sched.next_run_at) }}</span>
                <span v-if="sched.last_triggered_at">Last: {{ formatTime(sched.last_triggered_at) }}</span>
              </div>
            </div>
            <div class="flex gap-2 items-center">
              <button @click="openEditModal(sched)" class="btn-ghost text-xs">Edit</button>
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

    <!-- Create / Edit Source Schedule Modal -->
    <Modal
      :open="showModal"
      :title="editingId ? `Edit: ${editingId}` : 'Create Source Schedule'"
      max-width="md"
      :scrollable="false"
      @close="closeModal"
    >
      <template v-if="showModal">
        <form @submit.prevent="saveSchedule" class="space-y-4">
          <div v-if="!editingId">
            <label for="sched-source" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Source</label>
            <select id="sched-source" v-model="form.source_id" class="input-field" required>
              <option value="" disabled>Select a source…</option>
              <option v-for="s in availableSources" :key="s.source_id" :value="s.source_id">
                {{ s.dataset_name || s.source_id }}<span v-if="s.source_id !== (s.dataset_name || s.source_id)"> ({{ s.source_id }})</span>
              </option>
            </select>
            <p v-if="availableSources.length === 0" class="text-xs text-mendelu-gray-dark mt-1">
              No sources available. Add a source first on the Sources page.
            </p>
          </div>

          <CronPicker v-model="form.cron_expression" label="Schedule" />

          <div class="flex items-center gap-3">
            <label class="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" v-model="form.is_enabled" class="sr-only peer">
              <div class="w-9 h-5 bg-mendelu-gray-semi rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-mendelu-green"></div>
            </label>
            <span class="text-sm text-mendelu-black">Enabled</span>
          </div>

          <div v-if="formError" class="p-3 border-l-2 border-mendelu-alert bg-mendelu-alert/5 rounded text-sm text-mendelu-alert">
            {{ formError }}
          </div>

          <div class="flex gap-3 pt-2">
            <button type="submit" :disabled="saving || (!editingId && !form.source_id) || !form.cron_expression" class="btn-primary flex-1 disabled:opacity-50">
              {{ saving ? 'Saving…' : (editingId ? 'Save' : 'Create') }}
            </button>
            <button type="button" @click="closeModal" class="btn-secondary flex-1">Cancel</button>
          </div>
        </form>
      </template>
    </Modal>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import PageHeader from '../components/PageHeader.vue'
import CronPicker from '../components/CronPicker.vue'
import Modal from '../components/Modal.vue'
import { apiFetch } from '../api'
import { useToast } from '../composables/useToast'
import { useConfirm } from '../composables/useConfirm'

const toast = useToast()
const { confirm } = useConfirm()

const sources = ref<any[]>([])
const sourceSchedules = ref<any[]>([])
const loading = ref(false)
const saving = ref(false)
const errorMessage = ref('')
const showReference = ref(false)

const showModal = ref(false)
const editingId = ref(null)
const form = ref<any>({ source_id: '', cron_expression: '0 3 * * 0', is_enabled: true })
const formError = ref('')

const availableSources = computed(() => {
  const scheduled = new Set(sourceSchedules.value.map(s => s.source_id))
  return sources.value
    .filter(s => !scheduled.has(s.source_id))
    .sort((a, b) => (a.dataset_name || a.source_id).localeCompare(b.dataset_name || b.source_id))
})

function formatTime(ts) {
  if (!ts) return '---'
  try {
    const date = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts)
    if (Number.isNaN(date.getTime())) return String(ts)
    return date.toLocaleString()
  } catch { return String(ts) }
}

async function loadSources() {
  try {
    const resp = await apiFetch('/sources/')
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    const data = await resp.json()
    sources.value = Array.isArray(data) ? data : []
    sourceSchedules.value = sources.value
      .filter(s => s.schedule)
      .map(s => ({
        source_id: s.source_id,
        dataset_name: s.dataset_name,
        ...s.schedule,
      }))
  } catch (e: any) {
    console.error('Failed to load sources:', e)
    const msg = e?.message || 'network error'
    errorMessage.value = `Failed to load sources: ${msg}`
    toast.error(`Could not load schedules: ${msg}`)
  }
}

async function refreshAll() {
  loading.value = true
  errorMessage.value = ''
  try {
    await loadSources()
  } finally {
    loading.value = false
  }
}

function openCreateModal() {
  editingId.value = null
  form.value = { source_id: '', cron_expression: '0 3 * * 0', is_enabled: true }
  formError.value = ''
  showModal.value = true
}

function openEditModal(sched) {
  editingId.value = sched.source_id
  form.value = {
    source_id: sched.source_id,
    cron_expression: sched.cron_expression || '0 3 * * 0',
    is_enabled: sched.is_enabled !== false,
  }
  formError.value = ''
  showModal.value = true
}

function closeModal() {
  showModal.value = false
  editingId.value = null
  formError.value = ''
}

async function saveSchedule() {
  const sourceId = editingId.value || form.value.source_id
  if (!sourceId) {
    formError.value = 'Please select a source.'
    return
  }
  if (!form.value.cron_expression) {
    formError.value = 'Please provide a cron expression.'
    return
  }
  saving.value = true
  formError.value = ''
  try {
    const resp = await apiFetch(`/sources/${encodeURIComponent(sourceId)}/schedule`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        cron_expression: form.value.cron_expression,
        is_enabled: form.value.is_enabled,
      }),
    })
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}))
      throw new Error(err.detail || `HTTP ${resp.status}`)
    }
    closeModal()
    await loadSources()
  } catch (e) {
    formError.value = e.message
  } finally {
    saving.value = false
  }
}

async function deleteSourceSchedule(sourceId) {
  const ok = await confirm({
    title: 'Remove schedule?',
    message: `Remove schedule for "${sourceId}"?`,
    confirmText: 'Remove',
    danger: true,
  })
  if (!ok) return
  try {
    const resp = await apiFetch(`/sources/${encodeURIComponent(sourceId)}/schedule`, { method: 'DELETE' })
    if (!resp.ok && resp.status !== 404) {
      const err = await resp.json().catch(() => ({}))
      throw new Error(err.detail || `HTTP ${resp.status}`)
    }
    await loadSources()
  } catch (e) {
    toast.error(`Failed to remove schedule: ${e.message}`)
  }
}

onMounted(refreshAll)
</script>
