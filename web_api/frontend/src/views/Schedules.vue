<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-2xl font-bold text-white">Schedules</h1>
        <p class="text-gray-400">Manage Dagster job schedules</p>
      </div>
      <button
        @click="refreshSchedules"
        :disabled="loading"
        class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
      >
        Refresh
      </button>
    </div>

    <!-- Schedules List -->
    <div class="card">
      <div v-if="schedules.length === 0 && !loading" class="text-center py-12">
        <p class="text-gray-500 text-lg">No schedules found</p>
        <p class="text-gray-600 text-sm mt-2">Schedules are defined in dagster_project/schedules.py</p>
      </div>
      <div v-for="schedule in schedules" :key="schedule.name" class="flex items-center justify-between py-4 border-b border-dark-border/50 last:border-0">
        <div class="flex-1">
          <div class="flex items-center gap-3">
            <h3 class="text-white font-medium">{{ schedule.name }}</h3>
            <span class="px-2 py-0.5 rounded-full text-xs font-medium" :class="schedule.status === 'RUNNING' ? 'bg-green-500/20 text-green-400' : 'bg-gray-500/20 text-gray-400'">
              {{ schedule.status }}
            </span>
          </div>
          <div class="flex gap-4 mt-1 text-sm text-gray-500">
            <span>Cron: <code class="text-gray-300">{{ schedule.cron_schedule }}</code></span>
            <span v-if="schedule.job_name">Job: {{ schedule.job_name }}</span>
            <span v-if="schedule.next_run">Next: {{ formatTime(schedule.next_run) }}</span>
          </div>
        </div>
        <div class="flex gap-2">
          <button
            @click="toggleSchedule(schedule.name, schedule.status !== 'RUNNING')"
            class="px-3 py-1.5 rounded-lg text-sm transition-colors"
            :class="schedule.status === 'RUNNING' ? 'bg-red-600/20 text-red-400 hover:bg-red-600/40' : 'bg-green-600/20 text-green-400 hover:bg-green-600/40'"
          >
            {{ schedule.status === 'RUNNING' ? 'Disable' : 'Enable' }}
          </button>
        </div>
      </div>
    </div>

    <!-- Schedule Help -->
    <div class="card">
      <h3 class="text-lg font-semibold text-white mb-3">Schedule Reference</h3>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
        <div>
          <h4 class="text-gray-400 mb-2">Common Cron Patterns</h4>
          <div class="space-y-1 text-gray-500 font-mono text-xs">
            <p><span class="text-gray-300">0 0 * * *</span> — Daily at midnight</p>
            <p><span class="text-gray-300">0 6 * * *</span> — Daily at 6:00 AM</p>
            <p><span class="text-gray-300">0 0 * * 1</span> — Weekly on Monday</p>
            <p><span class="text-gray-300">0 0 1 * *</span> — Monthly on the 1st</p>
          </div>
        </div>
        <div>
          <h4 class="text-gray-400 mb-2">Available Jobs</h4>
          <div class="space-y-1 text-gray-500 text-xs">
            <p><span class="text-gray-300">dynamic_source_etl_job</span> — Process individual sources</p>
            <p><span class="text-gray-300">batch_catalog_etl_job</span> — Full catalog batch processing</p>
            <p><span class="text-gray-300">catalog_metadata_only_job</span> — Phase 0 metadata embedding</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const schedules = ref([])
const loading = ref(false)

function formatTime(ts) {
  if (!ts) return '—'
  try {
    // Dagster timestamps are usually epoch seconds
    const date = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts)
    return date.toLocaleString()
  } catch {
    return String(ts)
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
    if (resp.ok) {
      refreshSchedules()
    }
  } catch (e) {
    console.error('Failed to toggle schedule:', e)
  }
}

onMounted(() => {
  refreshSchedules()
})
</script>
