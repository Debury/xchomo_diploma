<template>
  <div class="space-y-3">
    <label v-if="label" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider">{{ label }}</label>

    <!-- Quick presets -->
    <div class="grid grid-cols-2 gap-2">
      <button
        v-for="preset in presets"
        :key="preset.cron"
        type="button"
        @click="selectPreset(preset)"
        class="px-3 py-2 rounded-lg text-xs font-medium border transition-all duration-150 text-left"
        :class="modelValue === preset.cron
          ? 'bg-mendelu-green/10 text-mendelu-green border-mendelu-green/30'
          : 'bg-mendelu-gray-light text-mendelu-black border-transparent hover:border-mendelu-green/20'"
      >
        <span class="block">{{ preset.label }}</span>
        <span class="text-[10px] text-mendelu-gray-dark font-normal">{{ preset.desc }}</span>
      </button>
    </div>

    <!-- Custom builder -->
    <details class="text-xs">
      <summary class="cursor-pointer text-mendelu-gray-dark hover:text-mendelu-black">Custom schedule</summary>
      <div class="mt-3 space-y-3">
        <div class="grid grid-cols-3 gap-2">
          <div>
            <label class="block text-[10px] text-mendelu-gray-dark mb-1">Frequency</label>
            <select v-model="custom.frequency" @change="buildCron" class="input-field !text-xs">
              <option value="hourly">Every N hours</option>
              <option value="daily">Daily</option>
              <option value="weekly">Weekly</option>
              <option value="monthly">Monthly</option>
            </select>
          </div>
          <div v-if="custom.frequency === 'hourly'">
            <label class="block text-[10px] text-mendelu-gray-dark mb-1">Every</label>
            <select v-model="custom.hours" @change="buildCron" class="input-field !text-xs">
              <option v-for="h in [1,2,3,4,6,8,12]" :key="h" :value="h">{{ h }} hour{{ h > 1 ? 's' : '' }}</option>
            </select>
          </div>
          <div v-if="custom.frequency === 'weekly'">
            <label class="block text-[10px] text-mendelu-gray-dark mb-1">Day</label>
            <select v-model="custom.dayOfWeek" @change="buildCron" class="input-field !text-xs">
              <option v-for="(name, i) in dayNames" :key="i" :value="i">{{ name }}</option>
            </select>
          </div>
          <div v-if="custom.frequency === 'monthly'">
            <label class="block text-[10px] text-mendelu-gray-dark mb-1">Day of month</label>
            <select v-model="custom.dayOfMonth" @change="buildCron" class="input-field !text-xs">
              <option v-for="d in 28" :key="d" :value="d">{{ d }}{{ ordinal(d) }}</option>
            </select>
          </div>
          <div v-if="custom.frequency !== 'hourly'">
            <label class="block text-[10px] text-mendelu-gray-dark mb-1">At</label>
            <select v-model="custom.hour" @change="buildCron" class="input-field !text-xs">
              <option v-for="h in 24" :key="h-1" :value="h-1">{{ String(h-1).padStart(2, '0') }}:00</option>
            </select>
          </div>
        </div>
        <div>
          <input :value="modelValue" @input="$emit('update:modelValue', $event.target.value)" type="text" class="input-field font-mono !text-xs" placeholder="0 2 * * *" />
        </div>
      </div>
    </details>

    <!-- Preview -->
    <div v-if="modelValue && nextRuns.length" class="bg-mendelu-gray-light rounded-lg p-2.5">
      <p class="text-[10px] font-medium text-mendelu-gray-dark mb-1">Next runs:</p>
      <div class="flex flex-wrap gap-x-4 gap-y-0.5">
        <span v-for="(run, i) in nextRuns" :key="i" class="text-[11px] text-mendelu-black">{{ run }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  modelValue: { type: String, default: '' },
  label: { type: String, default: '' },
})
const emit = defineEmits(['update:modelValue'])

const dayNames = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

const presets = [
  { label: 'Every 6 hours', desc: 'High frequency updates', cron: '0 */6 * * *' },
  { label: 'Daily at 2am', desc: 'Overnight processing', cron: '0 2 * * *' },
  { label: 'Every Tuesday', desc: 'Weekly on Tuesday', cron: '0 2 * * 2' },
  { label: 'Every Sunday', desc: 'Weekly on Sunday', cron: '0 3 * * 0' },
  { label: 'Twice a week', desc: 'Monday & Thursday', cron: '0 2 * * 1,4' },
  { label: 'Monthly', desc: '1st of each month', cron: '0 0 1 * *' },
]

const custom = ref({ frequency: 'daily', hour: 2, dayOfWeek: 2, dayOfMonth: 1, hours: 6 })

function selectPreset(preset) {
  emit('update:modelValue', preset.cron)
}

function buildCron() {
  const f = custom.value
  let cron = ''
  if (f.frequency === 'hourly') cron = `0 */${f.hours} * * *`
  else if (f.frequency === 'daily') cron = `0 ${f.hour} * * *`
  else if (f.frequency === 'weekly') cron = `0 ${f.hour} * * ${f.dayOfWeek}`
  else if (f.frequency === 'monthly') cron = `0 ${f.hour} ${f.dayOfMonth} * *`
  emit('update:modelValue', cron)
}

function ordinal(n) {
  if (n > 3 && n < 21) return 'th'
  switch (n % 10) { case 1: return 'st'; case 2: return 'nd'; case 3: return 'rd'; default: return 'th' }
}

const nextRuns = computed(() => {
  if (!props.modelValue) return []
  try {
    const parts = props.modelValue.trim().split(/\s+/)
    if (parts.length !== 5) return []
    const [min, hour, dom, mon, dow] = parts
    const now = new Date()
    const runs = []
    for (let dayOffset = 0; dayOffset < 60 && runs.length < 3; dayOffset++) {
      const candidate = new Date(now)
      candidate.setDate(candidate.getDate() + dayOffset)

      const hours = hour.includes('/') ? [0] : hour === '*' ? [0] : hour.split(',').map(Number)
      for (const h of hours) {
        candidate.setHours(h, min === '*' ? 0 : parseInt(min), 0, 0)
        if (candidate <= now) continue
        if (dow !== '*') { const dl = dow.split(',').map(Number); if (!dl.includes(candidate.getDay())) continue }
        if (dom !== '*') { const dl = dom.split(',').map(Number); if (!dl.includes(candidate.getDate())) continue }
        if (mon !== '*') { const ml = mon.split(',').map(Number); if (!ml.includes(candidate.getMonth() + 1)) continue }
        runs.push(candidate.toLocaleString())
        if (runs.length >= 3) break
      }
    }
    return runs
  } catch { return [] }
})
</script>
