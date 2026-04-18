<template>
  <div class="space-y-6 topo-bg">

    <!-- ===== Hero Welcome ===== -->
    <div class="animate-in">
      <div class="flex flex-col md:flex-row md:items-end md:justify-between gap-4 mb-2">
        <div>
          <p class="text-xs font-bold uppercase tracking-widest text-mendelu-green mb-2" style="font-family: var(--font-mono);">Climate Data Observatory</p>
          <h1 class="text-3xl md:text-4xl font-bold text-mendelu-black tracking-tight leading-tight">
            {{ greeting }}, <span class="text-mendelu-green">researcher</span>
          </h1>
          <p class="text-mendelu-gray-dark mt-1.5 text-sm max-w-lg">
            Your RAG pipeline is {{ systemStatus }}.
            <span v-if="stats.total_embeddings" class="text-mendelu-black font-medium">{{ stats.total_embeddings?.toLocaleString() }} vectors</span> indexed across
            <span v-if="qdrantDatasets.length" class="text-mendelu-black font-medium">{{ qdrantDatasets.length }} datasets</span>.
          </p>
        </div>
        <div class="flex items-center gap-2">
          <button
            @click="refreshAll"
            :disabled="loading"
            class="btn-secondary flex items-center gap-2 disabled:opacity-50"
          >
            <svg class="w-4 h-4 transition-transform" :class="{ 'animate-spin': loading }" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            {{ loading ? 'Refreshing...' : 'Refresh' }}
          </button>
        </div>
      </div>
    </div>

    <!-- ===== Stat Cards ===== -->
    <div class="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <StatCard
        label="Vectors"
        description="Total embeddings in Qdrant"
        :target="stats.total_embeddings || 0"
        :loading="loading"
        variant="green"
        class="animate-in stagger-1"
      >
        <template #icon>
          <svg class="w-4.5 h-4.5 text-mendelu-green" fill="none" stroke="currentColor" stroke-width="1.8" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
          </svg>
        </template>
      </StatCard>

      <StatCard
        label="Datasets"
        :description="`${metadataOnlyCount} metadata-only`"
        :target="datasetsWithData"
        :loading="loading"
        variant="success"
        class="animate-in stagger-2"
      >
        <template #icon>
          <svg class="w-4.5 h-4.5 text-mendelu-success" fill="none" stroke="currentColor" stroke-width="1.8" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
          </svg>
        </template>
      </StatCard>

      <StatCard
        label="Variables"
        description="Unique climate parameters"
        :target="stats.variables?.length || 0"
        :loading="loading"
        variant="green"
        class="animate-in stagger-3"
      >
        <template #icon>
          <svg class="w-4.5 h-4.5 text-mendelu-green" fill="none" stroke="currentColor" stroke-width="1.8" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z" />
          </svg>
        </template>
      </StatCard>

      <StatCard
        label="LLM"
        :value="health.llmOnline ? 'Connected' : 'Offline'"
        :loading="loading"
        :variant="health.llmOnline ? 'success' : 'alert'"
        :description="health.llmOnline ? 'Ready for queries' : 'Service unavailable'"
        class="animate-in stagger-4"
      >
        <template #icon>
          <svg class="w-4.5 h-4.5" :class="health.llmOnline ? 'text-mendelu-success' : 'text-mendelu-alert'" fill="none" stroke="currentColor" stroke-width="1.8" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
          </svg>
        </template>
      </StatCard>
    </div>

    <!-- ===== Main Content Grid ===== -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

      <!-- LEFT COLUMN: Top Datasets + Hazard Distribution -->
      <div class="lg:col-span-7 space-y-6">

        <!-- Top Datasets by Chunk Count -->
        <div class="card animate-in stagger-5">
          <div class="flex items-center gap-2.5 mb-5">
            <div class="w-8 h-8 rounded-lg bg-mendelu-green/10 flex items-center justify-center">
              <svg class="w-4 h-4 text-mendelu-green" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
              </svg>
            </div>
            <div>
              <h3 class="text-sm font-bold text-mendelu-black">Largest Datasets</h3>
              <p class="text-[10px] text-mendelu-gray-dark/60">Top sources by embedding count</p>
            </div>
          </div>

          <div class="space-y-3">
            <div v-for="ds in topDatasets" :key="ds.dataset_name" class="flex items-center gap-3">
              <div class="flex-1 min-w-0">
                <div class="flex items-center gap-2 mb-1">
                  <span class="text-sm font-medium text-mendelu-black truncate">{{ ds.dataset_name }}</span>
                  <span v-if="ds.hazard_type" class="badge-info !text-[9px] !py-0 flex-shrink-0">{{ ds.hazard_type }}</span>
                </div>
                <div class="w-full bg-mendelu-gray-semi/30 rounded-full h-1.5">
                  <div class="bg-mendelu-green rounded-full h-1.5 transition-all duration-700"
                       :style="{ width: `${(ds.chunk_count / maxChunks) * 100}%` }"></div>
                </div>
              </div>
              <span class="text-xs font-medium text-mendelu-gray-dark tabular-nums flex-shrink-0 w-16 text-right">
                {{ ds.chunk_count.toLocaleString() }}
              </span>
            </div>
          </div>

          <div class="mt-4 pt-3 border-t border-mendelu-gray-semi/40 flex justify-between items-center">
            <span class="text-[11px] text-mendelu-gray-dark">{{ qdrantDatasets.length }} total datasets</span>
            <router-link to="/catalog" class="text-[11px] font-medium text-mendelu-green hover:text-mendelu-green-hover transition-colors">
              View all &rarr;
            </router-link>
          </div>
        </div>

        <!-- Hazard Type Distribution -->
        <div class="card animate-in stagger-6">
          <div class="flex items-center gap-2.5 mb-5">
            <div class="w-8 h-8 rounded-lg bg-mendelu-green/10 flex items-center justify-center">
              <svg class="w-4 h-4 text-mendelu-green" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z" />
                <path stroke-linecap="round" stroke-linejoin="round" d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z" />
              </svg>
            </div>
            <div>
              <h3 class="text-sm font-bold text-mendelu-black">Hazard Coverage</h3>
              <p class="text-[10px] text-mendelu-gray-dark/60">Datasets by climate hazard type</p>
            </div>
          </div>
          <div class="flex items-center justify-center py-2">
            <DonutChart
              :segments="hazardSegments"
              :size="180"
              label="hazards"
            />
          </div>
        </div>
      </div>

      <!-- RIGHT COLUMN: Activity + Services + Quick Actions -->
      <div class="lg:col-span-5 space-y-6">

        <!-- Activity Feed -->
        <div class="card animate-in stagger-5" style="min-height: 320px;">
          <ActivityFeed />
        </div>

        <!-- Services Health -->
        <div class="card animate-in stagger-6">
          <div class="flex items-center gap-2.5 mb-4">
            <div class="w-8 h-8 rounded-lg bg-mendelu-green/10 flex items-center justify-center">
              <svg class="w-4 h-4 text-mendelu-green" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01" />
              </svg>
            </div>
            <div>
              <h3 class="text-sm font-bold text-mendelu-black">Infrastructure</h3>
              <p class="text-[10px] text-mendelu-gray-dark/60">Service health status</p>
            </div>
          </div>
          <ServiceHealth :health="health" />
        </div>

        <!-- Quick Actions -->
        <div class="animate-in stagger-7 grid grid-cols-1 gap-3">
          <router-link to="/chat" class="group relative overflow-hidden rounded-2xl p-5 transition-all duration-300 hover:scale-[1.01]" :class="isDark ? 'bg-dark-card border border-dark-border hover:border-mendelu-green/30' : 'bg-gradient-to-br from-mendelu-green to-mendelu-green-hover'">
            <div class="relative z-10 flex items-center gap-4">
              <div class="w-11 h-11 rounded-xl bg-white/20 flex items-center justify-center backdrop-blur-sm">
                <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
              </div>
              <div>
                <h4 class="text-sm font-bold text-white">Ask a Question</h4>
                <p class="text-xs text-white/70">Query climate data with AI-powered RAG</p>
              </div>
              <svg class="w-5 h-5 text-white/40 ml-auto group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
              </svg>
            </div>
          </router-link>

          <router-link to="/sources" class="stat-card group">
            <div class="flex items-center gap-4">
              <div class="w-11 h-11 rounded-xl bg-mendelu-green/10 flex items-center justify-center group-hover:bg-mendelu-green/15 transition-all duration-200">
                <svg class="w-5 h-5 text-mendelu-green" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                </svg>
              </div>
              <div>
                <h4 class="text-sm font-bold text-mendelu-black">Manage Sources</h4>
                <p class="text-xs text-mendelu-gray-dark">Add, update, or reprocess data sources</p>
              </div>
              <svg class="w-5 h-5 text-mendelu-gray-dark/30 ml-auto group-hover:translate-x-1 group-hover:text-mendelu-green transition-all" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
              </svg>
            </div>
          </router-link>
        </div>
      </div>
    </div>

    <!-- ===== Region Coverage ===== -->
    <div v-if="regionCounts.length" class="card animate-in stagger-8">
      <div class="flex items-center justify-between mb-4">
        <div class="flex items-center gap-2.5">
          <div class="w-8 h-8 rounded-lg bg-mendelu-green/10 flex items-center justify-center">
            <svg class="w-4 h-4 text-mendelu-green" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div>
            <h3 class="text-sm font-bold text-mendelu-black">Geographic Coverage</h3>
            <p class="text-[10px] text-mendelu-gray-dark/60">Datasets by region</p>
          </div>
        </div>
      </div>
      <div class="flex flex-wrap gap-2">
        <span v-for="r in regionCounts" :key="r.name"
              class="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium bg-mendelu-gray-light text-mendelu-black border border-mendelu-gray-semi/40">
          {{ r.name }}
          <span class="text-mendelu-gray-dark tabular-nums">{{ r.count }}</span>
        </span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useThemeStore } from '../stores/theme'
import StatCard from '../components/StatCard.vue'
import DonutChart from '../components/DonutChart.vue'
import ActivityFeed from '../components/ActivityFeed.vue'
import ServiceHealth from '../components/ServiceHealth.vue'
import { apiFetch } from '../api'
import { useToast } from '../composables/useToast'

const toast = useToast()

const themeStore = useThemeStore()
const isDark = computed(() => themeStore.theme === 'dark')

const stats = ref<any>({})
const health = ref<any>({ llm: 'Checking...', qdrant: false, dagster: false, llmOnline: false })
const qdrantDatasets = ref<any[]>([])
const loading = ref(false)

const hazardColors = ['#79be15', '#82c55b', '#6aaa10', '#3d7a0a', '#535a5d', '#8b95a0', '#b8c0c5', '#dce3e4', '#4a9e2f', '#2d6b16']

const greeting = computed(() => {
  const h = new Date().getHours()
  if (h < 12) return 'Good morning'
  if (h < 17) return 'Good afternoon'
  return 'Good evening'
})

const systemStatus = computed(() => {
  const online = health.value.qdrant && health.value.dagster
  if (online && health.value.llmOnline) return 'fully operational'
  if (online) return 'operational (LLM unavailable)'
  return 'experiencing issues'
})

// Single pass over qdrantDatasets produces all aggregates.
// Previously each aggregate iterated the array separately — 5x the work on mutation.
const datasetAggregates = computed(() => {
  const withData: any[] = []
  const hazardCounts: Record<string, number> = {}
  const regionCountsMap: Record<string, number> = {}
  let metadataOnly = 0

  for (const ds of qdrantDatasets.value) {
    const isMetadataOnly = ds.is_metadata_only || ds.chunk_count <= 10
    if (isMetadataOnly) metadataOnly++
    else withData.push(ds)

    const hazard = ds.hazard_type || 'Unknown'
    hazardCounts[hazard] = (hazardCounts[hazard] || 0) + 1

    const region = ds.location_name || 'Unknown'
    regionCountsMap[region] = (regionCountsMap[region] || 0) + 1
  }

  withData.sort((a, b) => (b.chunk_count || 0) - (a.chunk_count || 0))
  const topDatasets = withData.slice(0, 10)

  const hazardSegments = Object.entries(hazardCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([label, value], i) => ({
      label,
      value,
      color: hazardColors[i % hazardColors.length],
    }))

  const regionCounts = Object.entries(regionCountsMap)
    .sort((a, b) => b[1] - a[1])
    .map(([name, count]) => ({ name, count }))

  return {
    datasetsWithData: withData.length,
    metadataOnlyCount: metadataOnly,
    topDatasets,
    maxChunks: topDatasets[0]?.chunk_count || 1,
    hazardSegments,
    regionCounts,
  }
})

const datasetsWithData = computed(() => datasetAggregates.value.datasetsWithData)
const metadataOnlyCount = computed(() => datasetAggregates.value.metadataOnlyCount)
const topDatasets = computed(() => datasetAggregates.value.topDatasets)
const maxChunks = computed(() => datasetAggregates.value.maxChunks)
const hazardSegments = computed(() => datasetAggregates.value.hazardSegments)
const regionCounts = computed(() => datasetAggregates.value.regionCounts)

async function loadStats() {
  try {
    const resp = await apiFetch(`/rag/info?t=${Date.now()}`)
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    stats.value = await resp.json()
    health.value.qdrant = true
  } catch (e: any) {
    health.value.qdrant = false
    toast.error(`Failed to load RAG stats: ${e?.message ?? 'network error'}`, loadStats)
  }
}

async function checkHealth() {
  try {
    const resp = await apiFetch(`/health?t=${Date.now()}`)
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    const data = await resp.json()
    health.value.dagster = data.dagster_available
    health.value.llmOnline = data.status === 'healthy'
  } catch (e: any) {
    health.value.dagster = false
    health.value.llmOnline = false
    toast.error(`Health check failed: ${e?.message ?? 'network error'}`, checkHealth)
  }
}

async function loadQdrantDatasets() {
  try {
    const resp = await apiFetch('/qdrant/datasets')
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    qdrantDatasets.value = await resp.json()
  } catch (e: any) {
    toast.error(`Failed to load datasets: ${e?.message ?? 'network error'}`, loadQdrantDatasets)
  }
}

async function refreshAll() {
  loading.value = true
  try {
    await Promise.all([loadStats(), checkHealth(), loadQdrantDatasets()])
  } finally {
    loading.value = false
  }
}

onMounted(refreshAll)
</script>
