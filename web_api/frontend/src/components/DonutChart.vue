<template>
  <div class="flex flex-col items-center">
    <svg :width="size" :height="size" :viewBox="`0 0 ${size} ${size}`">
      <circle
        v-for="(seg, i) in computedSegments"
        :key="i"
        :cx="size / 2"
        :cy="size / 2"
        :r="radius"
        fill="none"
        :stroke="seg.color"
        :stroke-width="strokeWidth"
        :stroke-dasharray="`${seg.dash} ${circumference - seg.dash}`"
        :stroke-dashoffset="seg.offset"
        stroke-linecap="round"
        class="transition-all duration-700 ease-out"
        :style="{ opacity: mounted ? 1 : 0 }"
      />
      <!-- Center text -->
      <text
        :x="size / 2"
        :y="size / 2 - 4"
        text-anchor="middle"
        dominant-baseline="central"
        class="fill-mendelu-black text-lg font-semibold"
        style="font-size: 18px"
      >{{ total }}</text>
      <text
        :x="size / 2"
        :y="size / 2 + 14"
        text-anchor="middle"
        dominant-baseline="central"
        class="fill-mendelu-gray-dark"
        style="font-size: 10px"
      >{{ label }}</text>
    </svg>
    <!-- Legend -->
    <div v-if="segments.length" class="flex flex-wrap justify-center gap-x-3 gap-y-1 mt-3">
      <div v-for="seg in segments" :key="seg.label" class="flex items-center gap-1.5">
        <span class="w-2 h-2 rounded-full flex-shrink-0" :style="{ backgroundColor: seg.color }"></span>
        <span class="text-xs text-mendelu-gray-dark">{{ seg.label }}</span>
        <span class="text-xs font-medium text-mendelu-black">{{ seg.value }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, onMounted } from 'vue'

const props = defineProps({
  segments: { type: Array, default: () => [] },
  size: { type: Number, default: 140 },
  label: { type: String, default: 'total' },
})

const strokeWidth = 14
const radius = 50
const circumference = 2 * Math.PI * radius
const mounted = ref(false)

const total = computed(() => props.segments.reduce((sum, s) => sum + s.value, 0))

const computedSegments = computed(() => {
  if (total.value === 0) return []
  let accumulated = 0
  return props.segments.map(seg => {
    const dash = (seg.value / total.value) * circumference
    const offset = -accumulated + circumference / 4
    accumulated += dash
    return { ...seg, dash: mounted.value ? dash : 0, offset }
  })
})

onMounted(() => {
  requestAnimationFrame(() => { mounted.value = true })
})
</script>
