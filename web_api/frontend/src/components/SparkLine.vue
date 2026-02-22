<template>
  <svg :width="width" :height="height" :viewBox="`0 0 ${width} ${height}`" class="overflow-visible">
    <defs>
      <linearGradient :id="gradientId" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" :stop-color="color" stop-opacity="0.15" />
        <stop offset="100%" :stop-color="color" stop-opacity="0" />
      </linearGradient>
    </defs>
    <!-- Fill area -->
    <polygon :points="areaPoints" :fill="`url(#${gradientId})`" />
    <!-- Line -->
    <polyline :points="linePoints" fill="none" :stroke="color" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" />
  </svg>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  data: { type: Array, default: () => [] },
  width: { type: Number, default: 80 },
  height: { type: Number, default: 24 },
  color: { type: String, default: '#79be15' },
})

const gradientId = computed(() => `spark-${Math.random().toString(36).slice(2, 8)}`)

const linePoints = computed(() => {
  if (props.data.length < 2) return ''
  const max = Math.max(...props.data) || 1
  const min = Math.min(...props.data)
  const range = max - min || 1
  const padding = 2
  const usableHeight = props.height - padding * 2

  return props.data.map((v, i) => {
    const x = (i / (props.data.length - 1)) * props.width
    const y = padding + usableHeight - ((v - min) / range) * usableHeight
    return `${x},${y}`
  }).join(' ')
})

const areaPoints = computed(() => {
  if (!linePoints.value) return ''
  return `0,${props.height} ${linePoints.value} ${props.width},${props.height}`
})
</script>
