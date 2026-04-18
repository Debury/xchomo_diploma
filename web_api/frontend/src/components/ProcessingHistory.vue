<template>
  <div>
    <h4 class="text-sm font-medium text-mendelu-gray-dark mb-3">Processing History</h4>

    <div v-if="loading" class="text-xs text-mendelu-gray-dark py-4 text-center">Loading...</div>

    <div v-else-if="runs.length === 0" class="text-xs text-mendelu-gray-dark py-4 text-center">
      No processing history
    </div>

    <div v-else class="space-y-2">
      <div
        v-for="run in runs"
        :key="run.id"
        class="flex items-start gap-3 p-3 bg-mendelu-gray-light rounded-lg"
      >
        <!-- Status icon -->
        <div class="mt-0.5">
          <span
            class="w-2.5 h-2.5 rounded-full inline-block"
            :class="{
              'bg-mendelu-success': run.status === 'completed',
              'bg-mendelu-alert': run.status === 'failed',
              'bg-amber-400': run.status === 'started',
              'bg-mendelu-gray-dark': !['completed', 'failed', 'started'].includes(run.status),
            }"
          ></span>
        </div>

        <!-- Details -->
        <div class="flex-1 min-w-0">
          <div class="flex items-center gap-2 mb-0.5">
            <span class="text-xs font-medium text-mendelu-black">
              {{ run.status === 'completed' ? 'Completed' : run.status === 'failed' ? 'Failed' : 'Running' }}
            </span>
            <span v-if="run.trigger_type" class="text-[10px] px-1.5 py-0.5 bg-white rounded text-mendelu-gray-dark">
              {{ run.trigger_type }}
            </span>
            <span v-if="run.phase != null" class="text-[10px] px-1.5 py-0.5 bg-white rounded text-mendelu-gray-dark">
              Phase {{ run.phase }}
            </span>
          </div>

          <div class="flex flex-wrap gap-3 text-[11px] text-mendelu-gray-dark">
            <span v-if="run.started_at">{{ formatDate(run.started_at) }}</span>
            <span v-if="run.duration_seconds != null">{{ formatDuration(run.duration_seconds) }}</span>
            <span v-if="run.chunks_processed != null">{{ run.chunks_processed.toLocaleString() }} chunks</span>
            <span v-if="run.job_name" class="font-mono">{{ run.job_name }}</span>
          </div>

          <!-- Error -->
          <div v-if="run.error_message" class="mt-1 text-[11px] text-mendelu-alert truncate">
            {{ run.error_message }}
          </div>

          <!-- Dagster link -->
          <a
            v-if="run.dagster_run_id"
            :href="`/dagit/runs/${run.dagster_run_id}`"
            target="_blank"
            class="inline-block mt-1 text-[11px] text-mendelu-green hover:underline"
          >
            Dagster run &rarr;
          </a>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import { apiFetch } from '../api'

const props = defineProps({
  sourceId: { type: String, required: true },
})

const runs = ref<any[]>([])
const loading = ref(false)

function formatDate(dateString) {
  if (!dateString) return ''
  try { return new Date(dateString).toLocaleString() }
  catch { return dateString }
}

function formatDuration(seconds) {
  if (seconds == null) return ''
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`
  return `${(seconds / 3600).toFixed(1)}h`
}

async function loadHistory() {
  if (!props.sourceId) return
  loading.value = true
  try {
    const resp = await apiFetch(`/sources/${props.sourceId}/history`)
    if (resp.ok) {
      const data = await resp.json()
      runs.value = data.runs || []
    }
  } catch (e) {
    console.error('Failed to load history:', e)
  } finally {
    loading.value = false
  }
}

watch(() => props.sourceId, loadHistory)
onMounted(loadHistory)
</script>
