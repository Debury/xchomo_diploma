<template>
  <div class="stat-card">
    <h3 class="text-xs text-mendelu-gray-dark uppercase tracking-wider font-medium mb-1">{{ label }}</h3>
    <template v-if="loading">
      <div class="skeleton h-8 w-20 mt-1"></div>
    </template>
    <template v-else>
      <div class="flex items-end gap-2">
        <p class="text-2xl font-semibold text-mendelu-black tabular-nums">
          {{ isNumeric ? animatedDisplay : value }}
        </p>
        <SparkLine v-if="sparkData.length >= 2" :data="sparkData" class="mb-1" />
      </div>
      <p v-if="delta" class="text-xs mt-0.5" :class="deltaPositive ? 'text-mendelu-success' : 'text-mendelu-alert'">
        {{ delta }}
      </p>
    </template>
  </div>
</template>

<script setup>
import { computed, toRef } from 'vue'
import { useCountUp } from '../composables/useCountUp.js'
import SparkLine from './SparkLine.vue'

const props = defineProps({
  label: { type: String, required: true },
  value: { type: [String, Number], default: '0' },
  target: { type: Number, default: null },
  delta: { type: String, default: '' },
  loading: { type: Boolean, default: false },
  sparkData: { type: Array, default: () => [] },
})

const isNumeric = computed(() => props.target !== null)
const targetRef = toRef(props, 'target')
const { displayValue } = useCountUp(targetRef)

const animatedDisplay = computed(() => {
  return displayValue.value.toLocaleString()
})

const deltaPositive = computed(() => {
  if (!props.delta) return false
  return !props.delta.startsWith('-')
})
</script>
