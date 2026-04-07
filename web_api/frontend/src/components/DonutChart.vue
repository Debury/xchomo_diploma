<template>
  <div class="flex flex-col items-center">
    <div class="relative">
      <svg :width="size" :height="size" :viewBox="`0 0 ${size} ${size}`">
        <!-- Background track -->
        <circle
          :cx="size / 2"
          :cy="size / 2"
          :r="radius"
          fill="none"
          stroke="currentColor"
          class="text-mendelu-gray-semi/40"
          :stroke-width="strokeWidth"
        />
        <!-- Segments -->
        <circle
          v-for="(seg, i) in computedSegments"
          :key="i"
          :cx="size / 2"
          :cy="size / 2"
          :r="radius"
          fill="none"
          :stroke="seg.color"
          :stroke-width="strokeWidth - 1"
          :stroke-dasharray="`${seg.dash} ${circumference - seg.dash}`"
          :stroke-dashoffset="seg.offset"
          stroke-linecap="round"
          class="transition-all duration-1000 ease-out"
          :style="{ opacity: mounted ? 1 : 0 }"
        />
        <!-- Center content -->
        <text
          :x="size / 2"
          :y="size / 2 - 6"
          text-anchor="middle"
          dominant-baseline="central"
          class="fill-mendelu-black font-bold"
          style="font-family: var(--font-mono); font-size: 22px; letter-spacing: -0.03em;"
        >{{ total.toLocaleString() }}</text>
        <text
          :x="size / 2"
          :y="size / 2 + 14"
          text-anchor="middle"
          dominant-baseline="central"
          class="fill-mendelu-gray-dark"
          style="font-family: var(--font-mono); font-size: 9px; text-transform: uppercase; letter-spacing: 0.1em;"
        >{{ label }}</text>
      </svg>
    </div>

    <!-- Legend -->
    <div v-if="segments.length" class="flex flex-wrap justify-center gap-x-4 gap-y-1.5 mt-4">
      <div v-for="seg in segments" :key="seg.label" class="flex items-center gap-2 group cursor-default">
        <span class="w-2.5 h-2.5 rounded-sm flex-shrink-0 transition-transform duration-200 group-hover:scale-125" :style="{ backgroundColor: seg.color }"></span>
        <span class="text-xs text-mendelu-gray-dark">{{ seg.label }}</span>
        <span class="text-xs font-bold text-mendelu-black data-value">{{ seg.value }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, onMounted } from 'vue'

const props = defineProps({
  segments: { type: Array, default: () => [] },
  size: { type: Number, default: 160 },
  label: { type: String, default: 'total' },
})

const strokeWidth = 16
const radius = 54
const circumference = 2 * Math.PI * radius
const mounted = ref(false)

const total = computed(() => props.segments.reduce((sum, s) => sum + s.value, 0))

const computedSegments = computed(() => {
  if (total.value === 0) return []
  let accumulated = 0
  const gap = 3
  return props.segments.map(seg => {
    const dash = Math.max(0, (seg.value / total.value) * circumference - gap)
    const offset = -accumulated + circumference / 4
    accumulated += dash + gap
    return { ...seg, dash: mounted.value ? dash : 0, offset }
  })
})

onMounted(() => {
  requestAnimationFrame(() => { mounted.value = true })
})
</script>
