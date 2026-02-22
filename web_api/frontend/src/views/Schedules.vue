<template>
  <div class="space-y-6">
    <PageHeader title="Schedules" subtitle="Manage Dagster and per-source schedules">
      <template #actions>
        <button @click="refreshAll" :disabled="loading" class="btn-secondary disabled:opacity-50">Refresh</button>
      </template>
    </PageHeader>

    <!-- Per-Source Schedules -->
    <div class="card">
      <h3 class="text-sm font-medium text-mendelu-black mb-3">Source Schedules</h3>
      <p class="text-xs text-mendelu-gray-dark mb-3">Per-source cron schedules managed via the Sources page</p>

      <div v-if="sourceSchedules.length === 0" class="text-center py-6">
        <p class="text-mendelu-gray-dark text-sm">No source schedules configured</p>
        <router-link to="/sources" class="text-xs text-mendelu-green hover:text-mendelu-green-hover mt-1 inline-block transition-colors duration-150">
          Configure schedules on the Sources page
        </router-link>
      </div>

      <div v-else class="space-y-3">
        <div
          v-for="sched in sourceSchedules"
          :key="sched.source_id"
          class="bg-mendelu-gray-light rounded-lg p-4 hover:bg-mendelu-gray-semi/50 transition-all duration-150"
        >
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

    <!-- Dagster Schedules -->
    <div class="card !p-0">
      <div class="px-5 py-4 border-b border-mendelu-gray-semi">
        <h3 class="text-sm font-medium text-mendelu-black">Dagster Schedules</h3>
      </div>

      <div v-if="schedules.length === 0 && !loading" class="text-center py-8">
        <p class="text-mendelu-gray-dark text-sm">No Dagster schedules found</p>
      </div>
      <div
        v-for="schedule in schedules"
        :key="schedule.name"
        class="flex items-center justify-between px-5 py-4 border-b border-mendelu-gray-semi last:border-0 hover:bg-mendelu-gray-light/50 transition-all duration-150"
      >
        <div class="flex-1">
          <div class="flex items-center gap-3">
            <h3 class="text-mendelu-black text-sm font-medium">{{ schedule.name }}</h3>
            <span :class="schedule.status === 'RUNNING' ? 'badge-success' : 'badge-neutral'">
              {{ schedule.status }}
            </span>
          </div>
          <div class="flex gap-4 mt-1 text-xs text-mendelu-gray-dark">
            <span>Cron: <code class="text-mendelu-black font-mono bg-mendelu-gray-light px-1.5 py-0.5 rounded border border-mendelu-gray-semi">{{ schedule.cron_schedule }}</code></span>
            <span v-if="schedule.job_name">Job: {{ schedule.job_name }}</span>
            <span v-if="schedule.next_run">Next: {{ formatTime(schedule.next_run) }}</span>
          </div>
        </div>
        <div class="flex gap-2 items-center">
          <label class="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              :checked="schedule.status === 'RUNNING'"
              @change="toggleSchedule(schedule.name, schedule.status !== 'RUNNING')"
              class="sr-only peer"
            >
            <div class="w-9 h-5 bg-mendelu-gray-semi peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-mendelu-green"></div>
          </label>
        </div>
      </div>
    </div>

    <!-- Schedule Reference (collapsible) -->
    <div class="card">
      <button @click="showReference = !showReference" class="flex items-center justify-between w-full">
        <h3 class="text-sm font-medium text-mendelu-black">Schedule Reference</h3>
        <svg
          class="w-4 h-4 text-mendelu-gray-dark transition-transform duration-150"
          :class="{ 'rotate-180': showReference }"
          fill="none" stroke="currentColor" viewBox="0 0 24 24"
        >
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      <div v-if="showReference" class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm mt-4 pt-4 border-t border-mendelu-gray-semi">
        <div>
          <h4 class="text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-2">Common Cron Patterns</h4>
          <div class="space-y-1 text-mendelu-gray-dark font-mono text-xs">
            <p><span class="text-mendelu-black">0 0 * * *</span> -- Daily at midnight</p>
            <p><span class="text-mendelu-black">0 2 * * *</span> -- Daily at 2:00 AM</p>
            <p><span class="text-mendelu-black">0 3 * * 0</span> -- Weekly on Sunday 3am</p>
            <p><span class="text-mendelu-black">0 0 1 * *</span> -- Monthly on the 1st</p>
          </div>
        </div>
        <div>
          <h4 class="text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-2">Available Jobs</h4>
          <div class="space-y-1 text-mendelu-gray-dark text-xs">
            <p><span class="text-mendelu-black font-mono">single_source_etl_job</span> -- Process one source</p>
            <p><span class="text-mendelu-black font-mono">catalog_full_etl_job</span> -- Full catalog (Phase 0-3)</p>
            <p><span class="text-mendelu-black font-mono">batch_catalog_etl_job</span> -- Catalog Phase 0+1</p>
            <p><span class="text-mendelu-black font-mono">catalog_metadata_only_job</span> -- Phase 0 only</p>
          </div>
        </div>
      </div>
    </div>

    <!-- Edit Source Schedule Modal -->
    <div
      v-if="editingSched"
      class="fixed inset-0 bg-black/40 flex items-center justify-center z-50"
      @click.self="editingSched = null"
    >
      <div class="bg-white border border-mendelu-gray-semi rounded-xl p-6 max-w-md w-full mx-4 shadow-lg">
        <div class="flex items-center justify-between mb-6">
          <h2 class="text-lg font-semibold text-mendelu-black">Edit Schedule: {{ editingSched.source_id }}</h2>
          <button @click="editingSched = null" class="btn-ghost !px-2 !py-1">&times;</button>
        </div>

        <form @submit.prevent="saveSourceSchedule" class="space-y-4">
          <div>
            <label class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Cron Pattern</label>
            <input v-model="editSchedCron" type="text" class="input-field font-mono" placeholder="0 2 * * *" />
            <p class="mt-1 text-xs text-mendelu-gray-dark">Format: minute hour day month weekday</p>
          </div>
          <div class="flex items-center gap-3">
            <label class="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" v-model="editSchedEnabled" class="sr-only peer">
              <div class="w-9 h-5 bg-mendelu-gray-semi peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-mendelu-green"></div>
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

const schedules = ref([])
const sourceSchedules = ref([])
const loading = ref(false)
const showReference = ref(false)

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

async function refreshDagsterSchedules() {
  try {
    const resp = await fetch('/schedules')
    if (resp.ok) schedules.value = await resp.json()
  } catch (e) {
    console.error('Failed to load Dagster schedules:', e)
  }
}

async function refreshSourceSchedules() {
  try {
    const resp = await fetch('/sources?active_only=true')
    if (!resp.ok) return
    const sources = await resp.json()
    sourceSchedules.value = sources
      .filter(s => s.schedule)
      .map(s => ({ source_id: s.source_id, ...s.schedule }))
  } catch (e) {
    console.error('Failed to load source schedules:', e)
  }
}

async function refreshAll() {
  loading.value = true
  try {
    await Promise.all([refreshDagsterSchedules(), refreshSourceSchedules()])
  } finally {
    loading.value = false
  }
}

async function toggleSchedule(name, enable) {
  try {
    const resp = await fetch(`/schedules/${name}/toggle?enable=${enable}`, { method: 'POST' })
    if (resp.ok) refreshDagsterSchedules()
  } catch (e) {
    console.error('Failed to toggle schedule:', e)
  }
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
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: 'Error' }))
      throw new Error(err.detail || `HTTP ${resp.status}`)
    }
    editingSched.value = null
    await refreshSourceSchedules()
  } catch (e) {
    alert(`Error: ${e.message}`)
  } finally {
    savingSched.value = false
  }
}

async function deleteSourceSchedule(sourceId) {
  if (!confirm(`Remove schedule for "${sourceId}"?`)) return
  try {
    await fetch(`/sources/${sourceId}/schedule`, { method: 'DELETE' })
    await refreshSourceSchedules()
  } catch (e) {
    console.error('Error deleting schedule:', e)
  }
}

onMounted(refreshAll)
</script>
