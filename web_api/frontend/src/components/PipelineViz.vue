<template>
  <div class="space-y-3">
    <div
      v-for="(phase, idx) in phases"
      :key="idx"
      class="group relative"
    >
      <!-- Phase row -->
      <div
        class="flex items-center gap-4 p-3 rounded-xl transition-all duration-300"
        :class="[
          isCurrentPhase(idx)
            ? 'bg-mendelu-green/5 ring-1 ring-mendelu-green/20'
            : 'hover:bg-mendelu-gray-light/50'
        ]"
      >
        <!-- Phase number -->
        <div
          class="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 font-bold text-sm transition-all duration-300"
          :class="phaseStatusClass(idx)"
          style="font-family: var(--font-mono);"
        >
          <svg v-if="phaseCompleted(idx)" class="w-4.5 h-4.5" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7" />
          </svg>
          <span v-else-if="isCurrentPhase(idx)" class="relative flex h-2.5 w-2.5">
            <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-mendelu-green opacity-75"></span>
            <span class="relative inline-flex rounded-full h-2.5 w-2.5 bg-mendelu-green"></span>
          </span>
          <span v-else>{{ idx }}</span>
        </div>

        <!-- Phase info -->
        <div class="flex-1 min-w-0">
          <div class="flex items-center gap-2 mb-1">
            <span class="text-sm font-semibold text-mendelu-black">{{ phase.label }}</span>
            <span v-if="isCurrentPhase(idx)" class="badge-info text-[9px]">Active</span>
          </div>

          <!-- Progress bar -->
          <div v-if="phase.total > 0" class="flex items-center gap-3">
            <div class="flex-1 h-1.5 rounded-full bg-mendelu-gray-semi/60 overflow-hidden">
              <div
                class="h-full rounded-full transition-all duration-700 ease-out"
                :class="phase.failed > 0 ? 'bg-gradient-to-r from-mendelu-green to-mendelu-alert' : 'bg-gradient-to-r from-mendelu-green to-mendelu-success'"
                :style="{ width: `${Math.min(100, (phase.completed / phase.total) * 100)}%` }"
              ></div>
            </div>
            <span class="text-[11px] font-bold text-mendelu-gray-dark tabular-nums flex-shrink-0" style="font-family: var(--font-mono);">
              {{ phase.completed }}/{{ phase.total }}
            </span>
          </div>
          <span v-else class="text-[11px] text-mendelu-gray-dark/50">No entries</span>
        </div>

        <!-- Failed count -->
        <div v-if="phase.failed > 0" class="badge-danger text-[10px] flex-shrink-0">
          {{ phase.failed }} failed
        </div>
      </div>

      <!-- Connector line -->
      <div
        v-if="idx < phases.length - 1"
        class="absolute left-[29px] -bottom-3 w-px h-3"
        :class="phaseCompleted(idx) ? 'bg-mendelu-green/40' : 'bg-mendelu-gray-semi/60'"
      ></div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps({
  progress: { type: Object, default: null },
})

const phaseLabels = ['Metadata', 'Direct Download', 'Registration', 'API Portals', 'Manual']

const phases = computed(() => {
  return phaseLabels.map((label, idx) => {
    const info = props.progress?.phases?.[String(idx)]
    return {
      label,
      completed: info?.completed || 0,
      failed: info?.failed || 0,
      total: info?.total || 0,
    }
  })
})

function isCurrentPhase(idx) {
  return props.progress?.thread_alive && props.progress?.current_phase === idx
}

function phaseCompleted(idx) {
  const p = phases.value[idx]
  return p.total > 0 && p.completed >= p.total && p.failed === 0
}

function phaseStatusClass(idx) {
  if (phaseCompleted(idx)) return 'bg-mendelu-success/15 text-mendelu-success'
  if (isCurrentPhase(idx)) return 'bg-mendelu-green/15 text-mendelu-green'
  if (phases.value[idx].failed > 0) return 'bg-mendelu-alert/10 text-mendelu-alert'
  return 'bg-mendelu-gray-light text-mendelu-gray-dark'
}
</script>
