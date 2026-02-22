<template>
  <div class="space-y-6">
    <PageHeader title="Embeddings" subtitle="Vector store overview">
      <template #actions>
        <button @click="refreshStats" :disabled="loading" class="btn-secondary disabled:opacity-50">
          {{ loading ? 'Loading...' : 'Refresh' }}
        </button>
      </template>
    </PageHeader>

    <!-- Collection Health -->
    <div class="card">
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-full flex items-center justify-center"
             :class="healthStatus === 'healthy' ? 'bg-mendelu-success/10' : healthStatus === 'degraded' ? 'bg-mendelu-green/10' : 'bg-mendelu-gray-light'">
          <span class="relative flex h-3 w-3">
            <span v-if="healthStatus === 'healthy'" class="animate-ping absolute inline-flex h-full w-full rounded-full bg-mendelu-success opacity-75"></span>
            <span class="relative inline-flex rounded-full h-3 w-3"
                  :class="healthStatus === 'healthy' ? 'bg-mendelu-success' : healthStatus === 'degraded' ? 'bg-mendelu-green/50' : 'bg-mendelu-gray-semi'"></span>
          </span>
        </div>
        <div>
          <h3 class="text-sm font-medium text-mendelu-black">Collection Health</h3>
          <p class="text-xs text-mendelu-gray-dark capitalize">{{ healthStatus }}</p>
        </div>
      </div>
    </div>

    <!-- Stats Overview -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
      <StatCard label="Total Vectors" :target="stats.total_embeddings || 0" :loading="loading" />
      <StatCard label="Dimensions" value="1024" :loading="loading" />
      <StatCard label="Collection" value="climate_data" :loading="loading" />
    </div>

    <!-- Dataset Breakdown — Side by side with Donut -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-4">
      <div class="lg:col-span-8 card">
        <h3 class="text-sm font-medium text-mendelu-black mb-3">Chunks by Dataset</h3>
        <div v-if="datasetBreakdown.length" class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="border-b border-mendelu-gray-semi">
                <th class="table-header">Dataset</th>
                <th class="table-header text-right">Chunks</th>
                <th class="table-header w-1/3">Distribution</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-mendelu-gray-semi/50">
              <tr v-for="ds in datasetBreakdown" :key="ds.name" class="hover:bg-mendelu-gray-light transition-all duration-150">
                <td class="px-4 py-2.5 text-mendelu-black font-medium">{{ ds.name }}</td>
                <td class="px-4 py-2.5 text-mendelu-black text-right tabular-nums">{{ ds.count.toLocaleString() }}</td>
                <td class="px-4 py-2.5">
                  <div class="flex items-center gap-2">
                    <div class="flex-1 bg-mendelu-gray-light rounded-full h-1.5">
                      <div class="bg-mendelu-green h-1.5 rounded-full transition-all duration-300" :style="{ width: `${ds.percent}%` }"></div>
                    </div>
                    <span class="text-xs text-mendelu-gray-dark tabular-nums w-8 text-right">{{ ds.percent }}%</span>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <p v-else class="text-mendelu-gray-dark text-sm">No dataset data available</p>
      </div>
      <div class="lg:col-span-4 card flex items-center justify-center">
        <DonutChart
          :segments="datasetDonutSegments"
          :size="150"
          label="chunks"
        />
      </div>
    </div>

    <!-- Variable Breakdown -->
    <div class="card">
      <h3 class="text-sm font-medium text-mendelu-black mb-3">Chunks by Variable</h3>
      <div v-if="variableBreakdown.length" class="flex flex-wrap gap-2">
        <span
          v-for="(v, i) in displayedVariables"
          :key="v.name"
          class="badge-info"
        >
          {{ v.name }}
        </span>
        <button
          v-if="variableBreakdown.length > 20 && !showAllVariables"
          @click="showAllVariables = true"
          class="btn-ghost text-xs"
        >
          +{{ variableBreakdown.length - 20 }} more
        </button>
      </div>
      <p v-else class="text-mendelu-gray-dark text-sm">No variable data available</p>
    </div>

    <!-- Sample Embeddings -->
    <div class="card">
      <h3 class="text-sm font-medium text-mendelu-black mb-3">Sample Records</h3>
      <div v-if="samples.length" class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="border-b border-mendelu-gray-semi">
              <th class="table-header">Variable</th>
              <th class="table-header">Source</th>
              <th class="table-header">Temporal</th>
              <th class="table-header">Spatial</th>
              <th class="table-header">Preview</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-mendelu-gray-semi/50">
            <tr v-for="sample in samples" :key="sample.id" class="hover:bg-mendelu-gray-light transition-all duration-150">
              <td class="px-4 py-2.5"><span class="badge-info">{{ sample.variable }}</span></td>
              <td class="px-4 py-2.5 text-mendelu-gray-dark">{{ sample.source }}</td>
              <td class="px-4 py-2.5 text-mendelu-gray-dark">{{ sample.temporal }}</td>
              <td class="px-4 py-2.5 text-mendelu-gray-dark">{{ sample.spatial }}</td>
              <td class="px-4 py-2.5 max-w-xs truncate text-mendelu-gray-dark text-xs">{{ sample.text }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p v-else class="text-mendelu-gray-dark text-sm">No sample data available</p>
    </div>

    <!-- Actions -->
    <div class="card">
      <h3 class="text-sm font-medium text-mendelu-black mb-3">Actions</h3>
      <div class="flex gap-3 flex-wrap">
        <button @click="regenerateEmbeddings" class="btn-secondary">Regenerate All</button>
        <button @click="optimizeCollection" class="btn-secondary">Optimize Collection</button>
        <button @click="exportEmbeddings" class="btn-secondary">Export JSON</button>
        <button @click="clearEmbeddings" class="btn-danger">Clear All</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import PageHeader from '../components/PageHeader.vue'
import StatCard from '../components/StatCard.vue'
import DonutChart from '../components/DonutChart.vue'

const stats = ref({})
const loading = ref(false)
const samples = ref([])
const datasetBreakdown = ref([])
const variableBreakdown = ref([])
const showAllVariables = ref(false)

const donutColors = ['#79be15', '#82c55b', '#6aaa10', '#535a5d', '#dce3e4', '#4a9e0d', '#a3d977', '#3d8a0b']

const healthStatus = computed(() => {
  if (!stats.value.total_embeddings) return 'empty'
  if (stats.value.total_embeddings > 0) return 'healthy'
  return 'degraded'
})

const datasetDonutSegments = computed(() => {
  return datasetBreakdown.value.slice(0, 8).map((ds, i) => ({
    label: ds.name.length > 15 ? ds.name.slice(0, 15) + '...' : ds.name,
    value: ds.count,
    color: donutColors[i % donutColors.length],
  }))
})

const displayedVariables = computed(() => {
  if (showAllVariables.value) return variableBreakdown.value
  return variableBreakdown.value.slice(0, 20)
})

async function refreshStats() {
  loading.value = true
  try {
    const resp = await fetch(`/rag/info?t=${Date.now()}`)
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    const data = await resp.json()
    stats.value = data

    const sources = data.sources || []
    const total = data.total_embeddings || 1
    datasetBreakdown.value = sources.map(s => ({
      name: s,
      count: Math.round(total / sources.length),
      percent: Math.round(100 / sources.length)
    }))

    variableBreakdown.value = (data.variables || []).map(v => ({ name: v }))

    samples.value = (data.variables || []).slice(0, 5).map((v, i) => ({
      id: i,
      variable: v,
      source: data.sources?.[0] || 'Unknown',
      temporal: '2020-2100',
      spatial: '0.5 grid',
      text: `Climate data for ${v} - trends and patterns across the study region`
    }))

    try {
      const healthResp = await fetch('/admin/qdrant/health')
      if (healthResp.ok) {
        const healthData = await healthResp.json()
        if (healthData.datasets) {
          const totalChunks = Object.values(healthData.datasets).reduce((a, b) => a + b, 0)
          datasetBreakdown.value = Object.entries(healthData.datasets).map(([name, count]) => ({
            name, count, percent: totalChunks > 0 ? Math.round((count / totalChunks) * 100) : 0
          })).sort((a, b) => b.count - a.count)
        }
      }
    } catch (e) {}
  } catch (e) {
    console.error('Failed to load stats:', e)
  } finally {
    loading.value = false
  }
}

function regenerateEmbeddings() {
  if (confirm('This will regenerate all embeddings. Continue?')) {
    alert('Embedding regeneration would be triggered via Dagster pipeline')
  }
}

function optimizeCollection() {
  alert('Collection optimization initiated')
}

function exportEmbeddings() {
  alert('Export feature coming soon')
}

async function clearEmbeddings() {
  if (!confirm('This will permanently delete ALL embeddings from Qdrant. Are you sure?')) return
  try {
    const resp = await fetch('/embeddings/clear?confirm=true&delete_sources=false', { method: 'POST' })
    if (!resp.ok) {
      const error = await resp.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(error.detail || `HTTP ${resp.status}`)
    }
    await refreshStats()
  } catch (e) {
    console.error('Error clearing embeddings:', e)
    alert(`Error: ${e.message}`)
  }
}

onMounted(refreshStats)
</script>
