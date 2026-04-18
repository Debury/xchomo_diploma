<template>
  <div class="flex flex-col items-center gap-1.5">
    <svg :width="size" :height="size" :viewBox="`0 0 ${size} ${size}`" class="transform -rotate-90">
      <!-- Background ring -->
      <circle
        :cx="size / 2"
        :cy="size / 2"
        :r="radius"
        fill="none"
        stroke="#eff3f4"
        :stroke-width="strokeWidth"
      />
      <!-- Progress ring -->
      <circle
        :cx="size / 2"
        :cy="size / 2"
        :r="radius"
        fill="none"
        :stroke="color"
        :stroke-width="strokeWidth"
        :stroke-dasharray="circumference"
        :stroke-dashoffset="dashOffset"
        stroke-linecap="round"
        class="transition-all duration-700 ease-out"
      />
    </svg>
    <!-- Center overlay -->
    <div class="absolute flex items-center justify-center" :style="{ width: size + 'px', height: size + 'px' }">
      <span class="text-xs font-semibold tabular-nums" :style="{ color }">{{ value }}%</span>
    </div>
    <span v-if="label" class="text-xs text-mendelu-gray-dark">{{ label }}</span>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, onMounted } from 'vue'

const props = defineProps({
  value: { type: Number, default: 0 },
  size: { type: Number, default: 56 },
  color: { type: String, default: '#82c55b' },
  label: { type: String, default: '' },
})

const strokeWidth = 5
const radius = computed(() => (props.size - strokeWidth) / 2)
const circumference = computed(() => 2 * Math.PI * radius.value)
const mounted = ref(false)

const dashOffset = computed(() => {
  if (!mounted.value) return circumference.value
  return circumference.value * (1 - props.value / 100)
})

onMounted(() => {
  requestAnimationFrame(() => { mounted.value = true })
})
</script>
