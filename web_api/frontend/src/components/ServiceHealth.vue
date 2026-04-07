<template>
  <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
    <div
      v-for="svc in services"
      :key="svc.name"
      class="relative group p-4 rounded-xl transition-all duration-300 cursor-default"
      :class="svc.online
        ? 'bg-mendelu-success/5 hover:bg-mendelu-success/8'
        : 'bg-mendelu-alert/5 hover:bg-mendelu-alert/8'"
    >
      <!-- Glow dot -->
      <div class="flex items-center gap-3 mb-2.5">
        <div class="relative">
          <span
            class="block w-3 h-3 rounded-full transition-colors duration-500"
            :class="svc.online ? 'bg-mendelu-success' : 'bg-mendelu-alert'"
          ></span>
          <span
            v-if="svc.online"
            class="absolute inset-0 w-3 h-3 rounded-full bg-mendelu-success pulse-online"
          ></span>
        </div>
        <span class="text-[10px] font-bold uppercase tracking-widest" :class="svc.online ? 'text-mendelu-success' : 'text-mendelu-alert'" style="font-family: var(--font-mono);">
          {{ svc.online ? 'Online' : 'Offline' }}
        </span>
      </div>

      <h4 class="text-sm font-bold text-mendelu-black">{{ svc.name }}</h4>
      <p class="text-[10px] text-mendelu-gray-dark/60 mt-0.5">{{ svc.description }}</p>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  health: { type: Object, required: true },
})

const services = computed(() => [
  {
    name: 'API',
    description: 'FastAPI backend',
    online: true,
  },
  {
    name: 'Qdrant',
    description: 'Vector database',
    online: props.health.qdrant,
  },
  {
    name: 'Dagster',
    description: 'ETL orchestrator',
    online: props.health.dagster,
  },
  {
    name: 'LLM',
    description: 'Language model',
    online: props.health.llmOnline,
  },
])
</script>
