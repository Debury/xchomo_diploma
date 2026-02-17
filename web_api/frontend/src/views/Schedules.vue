<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-xl font-semibold text-mendelu-black">Schedules</h1>
        <p class="text-sm text-mendelu-gray-dark">Manage Dagster job schedules</p>
      </div>
      <button @click="refreshSchedules" :disabled="loading" class="btn-secondary disabled:opacity-50">
        Refresh
      </button>
    </div>

    <!-- Schedules List -->
    <div class="card !p-0">
      <div v-if="schedules.length === 0 && !loading" class="text-center py-8">
        <p class="text-mendelu-gray-dark text-sm">No schedules found</p>
        <p class="text-mendelu-gray-dark text-xs mt-2">Schedules are defined in dagster_project/schedules.py</p>
      </div>
      <div v-for="schedule in schedules" :key="schedule.name" class="flex items-center justify-between px-5 py-4 border-b border-mendelu-gray-semi last:border-0">
        <div class="flex-1">
          <div class="flex items-center gap-3">
            <h3 class="text-mendelu-black text-sm font-medium">{{ schedule.name }}</h3>
            <span :class="schedule.status === 'RUNNING' ? 'badge-success' : 'badge-neutral'">
              {{ schedule.status }}
            </span>
          </div>
          <div class="flex gap-4 mt-1 text-xs text-mendelu-gray-dark">
            <span>Cron: <code class="text-mendelu-black font-mono bg-mendelu-gray-light px-1 rounded">{{ schedule.cron_schedule }}</code></span>
            <span v-if="schedule.job_name">Job: {{ schedule.job_name }}</span>
            <span v-if="schedule.next_run">Next: {{ formatTime(schedule.next_run) }}</span>
          </div>
        </div>
        <div class="flex gap-2 items-center">
          <button @click="openEditModal(schedule)" class="btn-secondary !py-1.5 !px-3 !text-xs">
            Edit
          </button>
          <!-- Toggle Switch -->
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

    <!-- Schedule Reference -->
    <div class="card">
      <h3 class="text-sm font-semibold text-mendelu-black mb-3">Schedule Reference</h3>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
        <div>
          <h4 class="text-mendelu-gray-dark mb-2 font-medium">Common Cron Patterns</h4>
          <div class="space-y-1 text-mendelu-gray-dark font-mono text-xs">
            <p><span class="text-mendelu-black">0 0 * * *</span> — Daily at midnight</p>
            <p><span class="text-mendelu-black">0 6 * * *</span> — Daily at 6:00 AM</p>
            <p><span class="text-mendelu-black">0 0 * * 1</span> — Weekly on Monday</p>
            <p><span class="text-mendelu-black">0 0 1 * *</span> — Monthly on the 1st</p>
          </div>
        </div>
        <div>
          <h4 class="text-mendelu-gray-dark mb-2 font-medium">Available Jobs</h4>
          <div class="space-y-1 text-mendelu-gray-dark text-xs">
            <p><span class="text-mendelu-black font-mono">dynamic_source_etl_job</span> — Process individual sources</p>
            <p><span class="text-mendelu-black font-mono">batch_catalog_etl_job</span> — Full catalog batch processing</p>
            <p><span class="text-mendelu-black font-mono">catalog_metadata_only_job</span> — Phase 0 metadata embedding</p>
          </div>
        </div>
      </div>
    </div>

    <!-- Edit Modal -->
    <div
      v-if="editingSchedule"
      class="fixed inset-0 bg-black/40 flex items-center justify-center z-50"
      @click.self="editingSchedule = null"
    >
      <div class="bg-white border border-mendelu-gray-semi rounded-xl p-6 max-w-md w-full mx-4 shadow-lg">
        <div class="flex items-center justify-between mb-6">
          <h2 class="text-lg font-semibold text-mendelu-black">Edit Schedule</h2>
          <button @click="editingSchedule = null" class="text-mendelu-gray-dark hover:text-mendelu-black text-sm">Close</button>
        </div>

        <form @submit.prevent="saveSchedule" class="space-y-4">
          <div>
            <label class="block text-sm font-medium text-mendelu-black mb-1">Schedule Name</label>
            <input :value="editingSchedule.name" type="text" class="input-field" disabled />
          </div>
          <div>
            <label class="block text-sm font-medium text-mendelu-black mb-1">Cron Pattern</label>
            <input v-model="editCron" type="text" class="input-field font-mono" placeholder="0 0 * * *" />
            <p class="mt-1 text-xs text-mendelu-gray-dark">Format: minute hour day month weekday</p>
          </div>
          <div class="flex items-center gap-3">
            <label class="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" v-model="editEnabled" class="sr-only peer">
              <div class="w-9 h-5 bg-mendelu-gray-semi peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-mendelu-green"></div>
            </label>
            <span class="text-sm text-mendelu-black">Enabled</span>
          </div>
          <div class="flex gap-3 pt-2">
            <button type="submit" :disabled="savingSchedule" class="btn-primary flex-1 disabled:opacity-50">
              {{ savingSchedule ? 'Saving...' : 'Save' }}
            </button>
            <button type="button" @click="editingSchedule = null" class="btn-secondary flex-1">Cancel</button>
          </div>
        </form>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const schedules = ref([])
const loading = ref(false)
const editingSchedule = ref(null)
const editCron = ref('')
const editEnabled = ref(true)
const savingSchedule = ref(false)

function formatTime(ts) {
  if (!ts) return '—'
  try {
    const date = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts)
    return date.toLocaleString()
  } catch { return String(ts) }
}

function openEditModal(schedule) {
  editingSchedule.value = schedule
  editCron.value = schedule.cron_schedule || ''
  editEnabled.value = schedule.status === 'RUNNING'
}

async function saveSchedule() {
  savingSchedule.value = true
  try {
    // Toggle enable/disable
    const currentlyRunning = editingSchedule.value.status === 'RUNNING'
    if (editEnabled.value !== currentlyRunning) {
      await fetch(`/schedules/${editingSchedule.value.name}/toggle?enable=${editEnabled.value}`, { method: 'POST' })
    }
    // Update cron if changed
    if (editCron.value !== editingSchedule.value.cron_schedule) {
      await fetch(`/schedules/${editingSchedule.value.name}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cron_schedule: editCron.value })
      })
    }
    editingSchedule.value = null
    await refreshSchedules()
  } catch (e) {
    console.error('Failed to save schedule:', e)
  } finally {
    savingSchedule.value = false
  }
}

async function refreshSchedules() {
  loading.value = true
  try {
    const resp = await fetch('/schedules')
    if (resp.ok) schedules.value = await resp.json()
  } catch (e) {
    console.error('Failed to load schedules:', e)
  } finally {
    loading.value = false
  }
}

async function toggleSchedule(name, enable) {
  try {
    const resp = await fetch(`/schedules/${name}/toggle?enable=${enable}`, { method: 'POST' })
    if (resp.ok) refreshSchedules()
  } catch (e) {
    console.error('Failed to toggle schedule:', e)
  }
}

onMounted(() => { refreshSchedules() })
</script>
