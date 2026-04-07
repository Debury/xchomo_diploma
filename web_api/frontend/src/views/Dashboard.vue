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
            <span v-if="stats.variables" class="text-mendelu-black font-medium">{{ stats.variables?.length }} climate variables</span>.
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
        label="Variables"
        description="Unique climate parameters"
        :target="stats.variables?.length || 0"
        :loading="loading"
        variant="success"
        class="animate-in stagger-2"
      >
        <template #icon>
          <svg class="w-4.5 h-4.5 text-mendelu-success" fill="none" stroke="currentColor" stroke-width="1.8" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z" />
          </svg>
        </template>
      </StatCard>

      <StatCard
        label="Sources"
        description="Active data sources"
        :target="stats.sources?.length || 0"
        :delta="staleSources > 0 ? `${staleSources} stale` : ''"
        :loading="loading"
        :variant="staleSources > 0 ? 'amber' : 'green'"
        class="animate-in stagger-3"
      >
        <template #icon>
          <svg class="w-4.5 h-4.5" :class="staleSources > 0 ? 'text-amber-500' : 'text-mendelu-green'" fill="none" stroke="currentColor" stroke-width="1.8" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
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

      <!-- LEFT COLUMN: Pipeline + Phase Distribution -->
      <div class="lg:col-span-7 space-y-6">

        <!-- Processing Pipeline -->
        <div class="card animate-in stagger-5">
          <div class="flex items-center justify-between mb-5">
            <div class="flex items-center gap-2.5">
              <div class="w-8 h-8 rounded-lg bg-mendelu-green/10 flex items-center justify-center">
                <svg class="w-4 h-4 text-mendelu-green" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                </svg>
              </div>
              <div>
                <h3 class="text-sm font-bold text-mendelu-black">Processing Pipeline</h3>
                <p class="text-[10px] text-mendelu-gray-dark/60">5-phase catalog ingestion</p>
              </div>
            </div>
            <div v-if="catalogProgress?.thread_alive" class="flex items-center gap-2">
              <span class="relative flex h-2 w-2">
                <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-mendelu-green opacity-75"></span>
                <span class="relative inline-flex rounded-full h-2 w-2 bg-mendelu-green"></span>
              </span>
              <span class="text-[10px] font-bold text-mendelu-green uppercase tracking-wider" style="font-family: var(--font-mono);">Processing</span>
            </div>
          </div>

          <PipelineViz :progress="catalogProgress" />

          <!-- Summary stats -->
          <div v-if="catalogProgress" class="flex gap-6 mt-5 pt-4 border-t border-mendelu-gray-semi/40">
            <div class="text-center">
              <p class="text-lg font-bold text-mendelu-black data-value">{{ catalogProgress.processed || 0 }}</p>
              <p class="text-[10px] text-mendelu-gray-dark/60 font-medium uppercase tracking-wider" style="font-family: var(--font-mono);">Processed</p>
            </div>
            <div class="text-center">
              <p class="text-lg font-bold data-value" :class="(catalogProgress.failed || 0) > 0 ? 'text-mendelu-alert' : 'text-mendelu-gray-dark'">{{ catalogProgress.failed || 0 }}</p>
              <p class="text-[10px] text-mendelu-gray-dark/60 font-medium uppercase tracking-wider" style="font-family: var(--font-mono);">Failed</p>
            </div>
            <div class="text-center">
              <p class="text-lg font-bold text-mendelu-black data-value">{{ catalogProgress.pending || 0 }}</p>
              <p class="text-[10px] text-mendelu-gray-dark/60 font-medium uppercase tracking-wider" style="font-family: var(--font-mono);">Pending</p>
            </div>
          </div>
        </div>

        <!-- Phase Distribution -->
        <div class="card animate-in stagger-6">
          <div class="flex items-center gap-2.5 mb-5">
            <div class="w-8 h-8 rounded-lg bg-mendelu-green/10 flex items-center justify-center">
              <svg class="w-4 h-4 text-mendelu-green" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z" />
                <path stroke-linecap="round" stroke-linejoin="round" d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z" />
              </svg>
            </div>
            <div>
              <h3 class="text-sm font-bold text-mendelu-black">Phase Distribution</h3>
              <p class="text-[10px] text-mendelu-gray-dark/60">Catalog entries by access type</p>
            </div>
          </div>
          <div class="flex items-center justify-center py-2">
            <DonutChart
              :segments="phaseSegments"
              :size="180"
              label="entries"
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

          <router-link to="/catalog" class="stat-card group">
            <div class="flex items-center gap-4">
              <div class="w-11 h-11 rounded-xl bg-mendelu-green/10 flex items-center justify-center group-hover:bg-mendelu-green/15 transition-all duration-200">
                <svg class="w-5 h-5 text-mendelu-green" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
              </div>
              <div>
                <h4 class="text-sm font-bold text-mendelu-black">Browse Catalog</h4>
                <p class="text-xs text-mendelu-gray-dark">{{ catalog.length || 0 }} climate datasets</p>
              </div>
              <svg class="w-5 h-5 text-mendelu-gray-dark/30 ml-auto group-hover:translate-x-1 group-hover:text-mendelu-green transition-all" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
              </svg>
            </div>
          </router-link>
        </div>
      </div>
    </div>

    <!-- ===== Data Freshness Bar ===== -->
    <div v-if="stats.sources?.length" class="card animate-in stagger-8">
      <div class="flex items-center justify-between mb-4">
        <div class="flex items-center gap-2.5">
          <div class="w-8 h-8 rounded-lg bg-mendelu-green/10 flex items-center justify-center">
            <svg class="w-4 h-4 text-mendelu-green" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div>
            <h3 class="text-sm font-bold text-mendelu-black">Data Freshness</h3>
            <p class="text-[10px] text-mendelu-gray-dark/60">Source update status (30-day threshold)</p>
          </div>
        </div>
        <div class="flex items-center gap-4 text-xs">
          <span class="flex items-center gap-1.5">
            <span class="w-2.5 h-2.5 rounded-sm bg-mendelu-success/80"></span>
            <span class="text-mendelu-gray-dark">Fresh</span>
          </span>
          <span class="flex items-center gap-1.5">
            <span class="w-2.5 h-2.5 rounded-sm bg-amber-400/80"></span>
            <span class="text-mendelu-gray-dark">Stale</span>
          </span>
        </div>
      </div>

      <!-- Freshness bar -->
      <div class="h-4 rounded-full overflow-hidden bg-mendelu-gray-semi/30 flex">
        <div
          class="h-full bg-gradient-to-r from-mendelu-success to-mendelu-green transition-all duration-700 rounded-l-full"
          :style="{ width: `${freshPercent}%` }"
        ></div>
        <div
          v-if="staleSources > 0"
          class="h-full bg-gradient-to-r from-amber-400 to-amber-500 transition-all duration-700"
          :class="{ 'rounded-r-full': staleSources > 0 }"
          :style="{ width: `${100 - freshPercent}%` }"
        ></div>
      </div>
      <div class="flex justify-between mt-2">
        <span class="text-[11px] font-medium text-mendelu-success data-value">{{ (stats.sources?.length || 0) - staleSources }} fresh</span>
        <span v-if="staleSources > 0" class="text-[11px] font-medium text-amber-500 data-value">{{ staleSources }} need update</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useThemeStore } from '../stores/theme'
import StatCard from '../components/StatCard.vue'
import DonutChart from '../components/DonutChart.vue'
import ActivityFeed from '../components/ActivityFeed.vue'
import PipelineViz from '../components/PipelineViz.vue'
import ServiceHealth from '../components/ServiceHealth.vue'

const themeStore = useThemeStore()
const isDark = computed(() => themeStore.theme === 'dark')

const stats = ref({})
const health = ref({ llm: 'Checking...', qdrant: false, dagster: false, llmOnline: false })
const catalogProgress = ref(null)
const catalog = ref([])
const loading = ref(false)
const staleSources = ref(0)

const phaseColors = ['#79be15', '#82c55b', '#6aaa10', '#535a5d', '#dce3e4']

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

const freshPercent = computed(() => {
  const total = stats.value.sources?.length || 0
  if (total === 0) return 100
  return Math.round(((total - staleSources.value) / total) * 100)
})

const phaseSegments = computed(() => {
  const progress = catalogProgress.value
  if (!progress?.phases || !Object.keys(progress.phases).length) {
    return [
      { label: 'Phase 0', value: catalog.value.length || 233, color: phaseColors[0] },
    ]
  }
  return Object.entries(progress.phases).map(([phase, info]) => ({
    label: `Phase ${phase}`,
    value: info.total || 0,
    color: phaseColors[parseInt(phase)] || '#dce3e4',
  })).filter(s => s.value > 0)
})

async function loadStats() {
  try {
    const resp = await fetch(`/rag/info?t=${Date.now()}`)
    if (resp.ok) {
      stats.value = await resp.json()
      health.value.qdrant = true
    }
  } catch (e) {
    health.value.qdrant = false
  }
}

async function checkHealth() {
  try {
    const resp = await fetch(`/health?t=${Date.now()}`)
    if (resp.ok) {
      const data = await resp.json()
      health.value.dagster = data.dagster_available
      health.value.llmOnline = data.status === 'healthy'
    }
  } catch (e) {
    health.value.dagster = false
    health.value.llmOnline = false
  }
}

async function loadCatalogProgress() {
  try {
    const resp = await fetch('/catalog/progress')
    if (resp.ok) catalogProgress.value = await resp.json()
  } catch (e) {}
}

async function loadCatalogCount() {
  try {
    const resp = await fetch('/catalog')
    if (resp.ok) catalog.value = await resp.json()
  } catch (e) {}
}

async function loadSourceFreshness() {
  try {
    const resp = await fetch('/sources?active_only=true')
    if (resp.ok) {
      const sources = await resp.json()
      const thirtyDaysAgo = Date.now() - 30 * 24 * 60 * 60 * 1000
      staleSources.value = sources.filter(s => {
        if (s.processing_status !== 'completed') return false
        if (!s.last_processed) return true
        return new Date(s.last_processed).getTime() < thirtyDaysAgo
      }).length
    }
  } catch (e) {}
}

async function refreshAll() {
  loading.value = true
  try {
    await Promise.all([loadStats(), checkHealth(), loadCatalogProgress(), loadCatalogCount(), loadSourceFreshness()])
  } finally {
    loading.value = false
  }
}

onMounted(refreshAll)
</script>
