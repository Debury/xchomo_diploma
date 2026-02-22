<template>
  <div class="space-y-6">
    <PageHeader title="Dashboard" subtitle="System overview">
      <template #actions>
        <button
          @click="refreshAll"
          :disabled="loading"
          class="btn-secondary disabled:opacity-50"
        >
          {{ loading ? 'Loading...' : 'Refresh' }}
        </button>
      </template>
    </PageHeader>

    <!-- Animated Stat Cards -->
    <div class="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <StatCard
        label="Vectors"
        :target="stats.total_embeddings || 0"
        :loading="loading"
      />
      <StatCard
        label="Variables"
        :target="stats.variables?.length || 0"
        :loading="loading"
      />
      <StatCard
        label="Sources"
        :target="stats.sources?.length || 0"
        :delta="staleSources > 0 ? `${staleSources} stale` : ''"
        :loading="loading"
      />
      <StatCard
        label="LLM"
        :value="health.llmOnline ? 'Connected' : 'Offline'"
        :loading="loading"
      />
    </div>

    <!-- Two-column: Donut + Activity Feed -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-4">
      <!-- Left: Phase Distribution + Active Processing -->
      <div class="lg:col-span-7 space-y-4">
        <!-- Phase Distribution Donut -->
        <div class="card">
          <h3 class="text-sm font-medium text-mendelu-black mb-4">Phase Distribution</h3>
          <div class="flex items-center justify-center">
            <DonutChart
              :segments="phaseSegments"
              :size="160"
              label="entries"
            />
          </div>
        </div>

        <!-- Active Processing -->
        <div v-if="catalogProgress && catalogProgress.thread_alive" class="card border-l-2 border-l-mendelu-green">
          <div class="flex items-center gap-2 mb-3">
            <span class="relative flex h-2.5 w-2.5">
              <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-mendelu-green opacity-75"></span>
              <span class="relative inline-flex rounded-full h-2.5 w-2.5 bg-mendelu-green"></span>
            </span>
            <h3 class="text-sm font-medium text-mendelu-black">Processing Active</h3>
            <span v-if="catalogProgress.current_phase != null" class="badge-info ml-auto">Phase {{ catalogProgress.current_phase }}</span>
          </div>
          <div v-if="catalogProgress.phases && Object.keys(catalogProgress.phases).length" class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-3">
            <div v-for="(info, phase) in catalogProgress.phases" :key="phase" class="bg-mendelu-gray-light rounded-lg p-3">
              <div class="flex items-center justify-between mb-1">
                <span class="text-xs text-mendelu-black font-medium">Phase {{ phase }}</span>
                <span class="text-[10px] text-mendelu-gray-dark">{{ phaseLabel(phase) }}</span>
              </div>
              <div class="flex gap-2 text-xs mb-1.5">
                <span class="text-mendelu-success">{{ info.completed }}/{{ info.total }}</span>
                <span v-if="info.failed > 0" class="text-mendelu-alert">{{ info.failed }} failed</span>
              </div>
              <div class="w-full bg-mendelu-gray-semi rounded-full h-1">
                <div
                  class="bg-mendelu-green h-1 rounded-full transition-all duration-300"
                  :style="{ width: info.total > 0 ? `${Math.min(100, (info.completed / info.total) * 100)}%` : '0%' }"
                ></div>
              </div>
            </div>
          </div>
          <div class="flex gap-4 text-xs text-mendelu-gray-dark">
            <span>{{ catalogProgress.processed }} processed</span>
            <span v-if="catalogProgress.failed" class="text-mendelu-alert">{{ catalogProgress.failed }} failed</span>
            <span>{{ catalogProgress.pending }} pending</span>
          </div>
        </div>
      </div>

      <!-- Right: Activity Feed -->
      <div class="lg:col-span-5">
        <div class="card h-full">
          <ActivityFeed />
        </div>
      </div>
    </div>

    <!-- Services Health -->
    <div class="card">
      <h3 class="text-sm font-medium text-mendelu-black mb-4">Services</h3>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div class="flex items-center gap-3 p-3 rounded-lg bg-mendelu-gray-light/60">
          <span class="relative flex h-2.5 w-2.5 flex-shrink-0">
            <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-mendelu-success opacity-75"></span>
            <span class="relative inline-flex rounded-full h-2.5 w-2.5 bg-mendelu-success"></span>
          </span>
          <div>
            <span class="text-sm font-medium text-mendelu-black">API</span>
            <p class="text-[10px] text-mendelu-gray-dark">Online</p>
          </div>
        </div>
        <div class="flex items-center gap-3 p-3 rounded-lg bg-mendelu-gray-light/60">
          <span class="w-2.5 h-2.5 rounded-full flex-shrink-0" :class="health.qdrant ? 'bg-mendelu-success' : 'bg-mendelu-alert'"></span>
          <div>
            <span class="text-sm font-medium text-mendelu-black">Qdrant</span>
            <p class="text-[10px] text-mendelu-gray-dark">{{ health.qdrant ? 'Online' : 'Offline' }}</p>
          </div>
        </div>
        <div class="flex items-center gap-3 p-3 rounded-lg bg-mendelu-gray-light/60">
          <span class="w-2.5 h-2.5 rounded-full flex-shrink-0" :class="health.dagster ? 'bg-mendelu-success' : 'bg-mendelu-alert'"></span>
          <div>
            <span class="text-sm font-medium text-mendelu-black">Dagster</span>
            <p class="text-[10px] text-mendelu-gray-dark">{{ health.dagster ? 'Online' : 'Offline' }}</p>
          </div>
        </div>
        <div class="flex items-center gap-3 p-3 rounded-lg bg-mendelu-gray-light/60">
          <span class="w-2.5 h-2.5 rounded-full flex-shrink-0" :class="health.llmOnline ? 'bg-mendelu-success' : 'bg-mendelu-gray-semi'"></span>
          <div>
            <span class="text-sm font-medium text-mendelu-black">LLM</span>
            <p class="text-[10px] text-mendelu-gray-dark">{{ health.llmOnline ? 'Connected' : 'Unavailable' }}</p>
          </div>
        </div>
      </div>
    </div>

    <!-- Quick Actions -->
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
      <router-link to="/chat" class="stat-card group">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 rounded-lg bg-mendelu-green/10 flex items-center justify-center group-hover:bg-mendelu-green/20 transition-all duration-150">
            <svg class="w-5 h-5 text-mendelu-green" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
          </div>
          <div>
            <h4 class="text-sm font-medium text-mendelu-black">Ask a Question</h4>
            <p class="text-xs text-mendelu-gray-dark">Query climate data with AI</p>
          </div>
        </div>
      </router-link>
      <router-link to="/catalog" class="stat-card group">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 rounded-lg bg-mendelu-green/10 flex items-center justify-center group-hover:bg-mendelu-green/20 transition-all duration-150">
            <svg class="w-5 h-5 text-mendelu-green" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
            </svg>
          </div>
          <div>
            <h4 class="text-sm font-medium text-mendelu-black">View Catalog</h4>
            <p class="text-xs text-mendelu-gray-dark">Browse {{ catalog.length || 246 }} sources</p>
          </div>
        </div>
      </router-link>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import PageHeader from '../components/PageHeader.vue'
import StatCard from '../components/StatCard.vue'
import DonutChart from '../components/DonutChart.vue'
import ActivityFeed from '../components/ActivityFeed.vue'

const stats = ref({})
const health = ref({ llm: 'Checking...', qdrant: false, dagster: false, llmOnline: false })
const catalogProgress = ref(null)
const catalog = ref([])
const loading = ref(false)
const staleSources = ref(0)

const phaseColors = ['#79be15', '#82c55b', '#6aaa10', '#535a5d', '#dce3e4']

const phaseSegments = computed(() => {
  const progress = catalogProgress.value
  if (!progress?.phases || !Object.keys(progress.phases).length) {
    // Default distribution when no live progress
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

function phaseLabel(phase) {
  const labels = { '0': 'Metadata', '1': 'Direct download', '2': 'Registration', '3': 'API portals', '4': 'Manual' }
  return labels[String(phase)] || ''
}

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
      health.value.llmOnline = data.dagster_available
    }
  } catch (e) {
    health.value.dagster = false
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
