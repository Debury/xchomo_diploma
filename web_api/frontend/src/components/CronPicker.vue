<template>
  <div class="space-y-3">
    <label class="block text-sm font-medium text-mendelu-black">{{ label || 'Schedule' }}</label>

    <!-- Presets -->
    <div class="flex flex-wrap gap-2">
      <button
        v-for="preset in presets"
        :key="preset.label"
        type="button"
        @click="selectPreset(preset)"
        :class="[
          'px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors',
          modelValue === preset.cron
            ? 'bg-mendelu-green/10 text-mendelu-green border-mendelu-green/30'
            : 'bg-mendelu-gray-light text-mendelu-gray-dark border-transparent hover:border-mendelu-green/20'
        ]"
      >
        {{ preset.label }}
      </button>
    </div>

    <!-- Custom input -->
    <div>
      <input
        :value="modelValue"
        @input="$emit('update:modelValue', $event.target.value)"
        type="text"
        class="input-field font-mono text-sm"
        placeholder="0 2 * * *"
      />
      <p class="mt-1 text-xs text-mendelu-gray-dark">Format: minute hour day month weekday</p>
    </div>

    <!-- Preview next runs -->
    <div v-if="modelValue && nextRuns.length" class="bg-mendelu-gray-light rounded-lg p-3">
      <p class="text-xs font-medium text-mendelu-gray-dark mb-1.5">Next runs:</p>
      <div class="space-y-0.5">
        <p v-for="(run, i) in nextRuns" :key="i" class="text-xs text-mendelu-black">
          {{ run }}
        </p>
      </div>
    </div>
    <p v-else-if="modelValue && cronError" class="text-xs text-mendelu-alert">{{ cronError }}</p>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  modelValue: { type: String, default: '' },
  label: { type: String, default: 'Schedule' },
})

const emit = defineEmits(['update:modelValue'])

const presets = [
  { label: 'Daily 2am', cron: '0 2 * * *' },
  { label: 'Weekly Sunday 3am', cron: '0 3 * * 0' },
  { label: 'Monthly 1st', cron: '0 0 1 * *' },
  { label: 'Every 12h', cron: '0 */12 * * *' },
]

function selectPreset(preset) {
  emit('update:modelValue', preset.cron)
}

const cronError = computed(() => {
  if (!props.modelValue) return ''
  const parts = props.modelValue.trim().split(/\s+/)
  if (parts.length !== 5) return 'Cron expression must have 5 fields'
  return ''
})

const nextRuns = computed(() => {
  if (!props.modelValue || cronError.value) return []
  // Simple preview: show approximate next 3 run times
  // Real cron parsing would need croniter, so we show a simplified version
  try {
    const parts = props.modelValue.trim().split(/\s+/)
    if (parts.length !== 5) return []

    const [min, hour, dom, mon, dow] = parts
    const now = new Date()
    const runs = []

    for (let dayOffset = 0; dayOffset < 60 && runs.length < 3; dayOffset++) {
      const candidate = new Date(now)
      candidate.setDate(candidate.getDate() + dayOffset)
      candidate.setHours(hour === '*' ? 0 : parseInt(hour), min === '*' ? 0 : parseInt(min), 0, 0)

      if (candidate <= now) continue

      // Check day-of-week
      if (dow !== '*') {
        const dowList = dow.split(',').map(Number)
        if (!dowList.includes(candidate.getDay())) continue
      }

      // Check day-of-month
      if (dom !== '*') {
        const domList = dom.split(',').map(Number)
        if (!domList.includes(candidate.getDate())) continue
      }

      // Check month
      if (mon !== '*') {
        const monList = mon.split(',').map(Number)
        if (!monList.includes(candidate.getMonth() + 1)) continue
      }

      runs.push(candidate.toLocaleString())
    }

    return runs
  } catch {
    return []
  }
})
</script>
