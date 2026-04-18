<template>
  <div class="stat-card group">
    <!-- Icon + Label row -->
    <div class="flex items-center gap-2.5 mb-3">
      <div
        class="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 transition-all duration-300 group-hover:scale-105"
        :class="iconBgClass"
      >
        <slot name="icon">
          <svg class="w-4.5 h-4.5" :class="iconColorClass" fill="none" stroke="currentColor" stroke-width="1.8" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
          </svg>
        </slot>
      </div>
      <div class="min-w-0">
        <h3 class="text-[11px] font-semibold uppercase tracking-wider text-mendelu-gray-dark truncate" style="font-family: var(--font-mono);">{{ label }}</h3>
        <p v-if="description" class="text-[10px] text-mendelu-gray-dark/60 truncate mt-0.5 leading-none">{{ description }}</p>
      </div>
    </div>

    <!-- Value -->
    <template v-if="loading">
      <div class="skeleton h-8 w-24 rounded-lg"></div>
    </template>
    <template v-else>
      <div class="flex items-end gap-3">
        <p class="text-3xl font-bold text-mendelu-black tracking-tight leading-none data-value">
          {{ isNumeric ? animatedDisplay : value }}
        </p>
        <SparkLine v-if="sparkData.length >= 2" :data="sparkData" class="mb-0.5 opacity-60 group-hover:opacity-100 transition-opacity" />
      </div>
      <div v-if="delta || suffix" class="flex items-center gap-2 mt-2">
        <span v-if="delta" class="text-xs font-medium flex items-center gap-1" :class="deltaPositive ? 'text-mendelu-success' : 'text-mendelu-alert'">
          <svg v-if="deltaPositive" class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M5 15l7-7 7 7" /></svg>
          <svg v-else class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M19 9l-7 7-7-7" /></svg>
          {{ delta }}
        </span>
        <span v-if="suffix" class="text-[10px] text-mendelu-gray-dark/60 font-medium">{{ suffix }}</span>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, toRef, type PropType } from 'vue'
import { useCountUp } from '../composables/useCountUp'
import SparkLine from './SparkLine.vue'

const props = defineProps({
  label: { type: String, required: true },
  value: { type: [String, Number], default: '0' },
  target: { type: Number, default: null },
  delta: { type: String, default: '' },
  description: { type: String, default: '' },
  suffix: { type: String, default: '' },
  loading: { type: Boolean, default: false },
  sparkData: { type: Array as PropType<number[]>, default: () => [] },
  variant: { type: String, default: 'default' },
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

const iconBgClass = computed(() => {
  const map = {
    green: 'bg-mendelu-green/10',
    success: 'bg-mendelu-success/10',
    alert: 'bg-mendelu-alert/10',
    amber: 'bg-amber-500/10',
    default: 'bg-mendelu-green/8',
  }
  return map[props.variant] || map.default
})

const iconColorClass = computed(() => {
  const map = {
    green: 'text-mendelu-green',
    success: 'text-mendelu-success',
    alert: 'text-mendelu-alert',
    amber: 'text-amber-500',
    default: 'text-mendelu-green',
  }
  return map[props.variant] || map.default
})
</script>
