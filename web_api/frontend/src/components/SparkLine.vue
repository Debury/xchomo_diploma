<template>
  <svg :width="width" :height="height" :viewBox="`0 0 ${width} ${height}`" class="overflow-visible">
    <defs>
      <linearGradient :id="gradientId" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" :stop-color="color" stop-opacity="0.2" />
        <stop offset="100%" :stop-color="color" stop-opacity="0" />
      </linearGradient>
    </defs>
    <!-- Fill area -->
    <polygon :points="areaPoints" :fill="`url(#${gradientId})`" />
    <!-- Line -->
    <polyline
      :points="linePoints"
      fill="none"
      :stroke="color"
      stroke-width="1.5"
      stroke-linecap="round"
      stroke-linejoin="round"
    />
    <!-- End dot -->
    <circle
      v-if="lastPoint"
      :cx="lastPoint.x"
      :cy="lastPoint.y"
      r="2"
      :fill="color"
    />
  </svg>
</template>

<script setup lang="ts">
import { computed, type PropType } from 'vue'

const props = defineProps({
  data: { type: Array as PropType<number[]>, default: () => [] },
  width: { type: Number, default: 80 },
  height: { type: Number, default: 24 },
  color: { type: String, default: '#79be15' },
})

const gradientId = computed(() => `spark-${Math.random().toString(36).slice(2, 8)}`)

const points = computed(() => {
  if (props.data.length < 2) return []
  const max = Math.max(...props.data) || 1
  const min = Math.min(...props.data)
  const range = max - min || 1
  const padding = 3

  return props.data.map((v, i) => ({
    x: (i / (props.data.length - 1)) * props.width,
    y: padding + (props.height - padding * 2) - ((v - min) / range) * (props.height - padding * 2),
  }))
})

const linePoints = computed(() => points.value.map(p => `${p.x},${p.y}`).join(' '))

const areaPoints = computed(() => {
  if (!linePoints.value) return ''
  return `0,${props.height} ${linePoints.value} ${props.width},${props.height}`
})

const lastPoint = computed(() => {
  if (points.value.length < 2) return null
  return points.value[points.value.length - 1]
})
</script>
