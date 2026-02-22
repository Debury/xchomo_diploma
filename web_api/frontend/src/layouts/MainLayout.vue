<template>
  <div class="min-h-screen flex">
    <!-- Mobile overlay -->
    <div
      v-if="sidebarOpen"
      class="fixed inset-0 bg-black/40 z-40 md:hidden"
      @click="sidebarOpen = false"
    ></div>

    <!-- Sidebar -->
    <aside
      class="w-60 bg-mendelu-black flex flex-col fixed inset-y-0 left-0 z-50 transform transition-transform duration-200 md:relative md:translate-x-0"
      :class="sidebarOpen ? 'translate-x-0' : '-translate-x-full'"
    >
      <!-- Logo -->
      <div class="px-5 py-5 border-b border-white/10">
        <div class="flex items-center gap-3">
          <div class="w-9 h-9 rounded-full bg-mendelu-green flex items-center justify-center">
            <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
          </div>
          <div>
            <h1 class="text-sm font-semibold text-white tracking-tight">MENDELU</h1>
            <p class="text-[11px] text-gray-400">Climate RAG</p>
          </div>
        </div>
      </div>

      <!-- Navigation -->
      <nav class="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto">
        <router-link
          v-for="item in navItems"
          :key="item.path"
          :to="item.path"
          class="flex items-center gap-3 px-3 py-2 rounded-lg text-[13px] transition-all duration-150"
          :class="isActive(item.path)
            ? 'bg-mendelu-green/15 text-mendelu-green'
            : 'text-gray-400 hover:bg-white/10 hover:text-white'"
          @click="sidebarOpen = false"
        >
          <component :is="item.icon" class="w-4 h-4 flex-shrink-0" />
          <span>{{ item.label }}</span>
        </router-link>
      </nav>

      <!-- User Section -->
      <div class="px-4 py-3 border-t border-white/10">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-2 min-w-0">
            <div class="w-7 h-7 rounded-full bg-mendelu-green/20 flex items-center justify-center text-xs text-mendelu-green font-medium flex-shrink-0">
              {{ authStore.user?.username?.[0]?.toUpperCase() || 'U' }}
            </div>
            <span class="text-xs text-gray-400 truncate">{{ authStore.user?.username }}</span>
          </div>
          <button @click="handleLogout" class="text-gray-500 hover:text-white transition-all duration-150 p-1 rounded hover:bg-white/10">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"></path>
            </svg>
          </button>
        </div>
      </div>
    </aside>

    <!-- Main Content -->
    <main class="flex-1 flex flex-col overflow-hidden md:ml-0">
      <!-- Header -->
      <header class="h-12 bg-white border-b border-mendelu-gray-semi flex items-center justify-between px-4 md:px-6 flex-shrink-0">
        <div class="flex items-center gap-3">
          <!-- Mobile hamburger -->
          <button
            @click="sidebarOpen = !sidebarOpen"
            class="md:hidden p-1 rounded-lg text-mendelu-gray-dark hover:bg-mendelu-gray-light transition-all duration-150"
          >
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <h2 class="text-sm font-medium text-mendelu-black">{{ $route.name }}</h2>
        </div>
        <div class="flex items-center gap-4">
          <div class="flex items-center gap-1.5 text-xs text-mendelu-gray-dark">
            <div
              class="w-1.5 h-1.5 rounded-full transition-colors duration-300"
              :class="apiHealthy ? 'bg-mendelu-success' : 'bg-mendelu-alert'"
            ></div>
            <span>{{ apiHealthy ? 'Online' : 'Offline' }}</span>
          </div>
          <a href="/docs" target="_blank" class="text-mendelu-gray-dark hover:text-mendelu-green text-xs transition-all duration-150">
            API Docs
          </a>
        </div>
      </header>

      <!-- Page Content -->
      <div class="flex-1 overflow-auto p-4 md:p-6 bg-mendelu-gray-light">
        <router-view />
      </div>
    </main>
  </div>
</template>

<script setup>
import { h, ref, onMounted, onUnmounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useAuthStore } from '../stores/auth'

const router = useRouter()
const route = useRoute()
const authStore = useAuthStore()

const sidebarOpen = ref(false)
const apiHealthy = ref(true)
let healthInterval = null

function isActive(path) {
  if (path === '/') return route.path === '/' || route.path === ''
  return route.path.startsWith(path)
}

async function checkHealth() {
  try {
    const resp = await fetch(`/health?t=${Date.now()}`)
    apiHealthy.value = resp.ok
  } catch {
    apiHealthy.value = false
  }
}

onMounted(() => {
  checkHealth()
  healthInterval = setInterval(checkHealth, 30000)
})

onUnmounted(() => {
  if (healthInterval) clearInterval(healthInterval)
})

// SVG icon components
const IconDashboard = (_, { attrs }) => h('svg', { ...attrs, fill: 'none', stroke: 'currentColor', viewBox: '0 0 24 24' }, [
  h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '1.5', d: 'M4 5a1 1 0 011-1h4a1 1 0 011 1v5a1 1 0 01-1 1H5a1 1 0 01-1-1V5zm10 0a1 1 0 011-1h4a1 1 0 011 1v2a1 1 0 01-1 1h-4a1 1 0 01-1-1V5zM4 15a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H5a1 1 0 01-1-1v-4zm10-1a1 1 0 011-1h4a1 1 0 011 1v5a1 1 0 01-1 1h-4a1 1 0 01-1-1v-5z' })
])
const IconChat = (_, { attrs }) => h('svg', { ...attrs, fill: 'none', stroke: 'currentColor', viewBox: '0 0 24 24' }, [
  h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '1.5', d: 'M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z' })
])
const IconCatalog = (_, { attrs }) => h('svg', { ...attrs, fill: 'none', stroke: 'currentColor', viewBox: '0 0 24 24' }, [
  h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '1.5', d: 'M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2' })
])
const IconSources = (_, { attrs }) => h('svg', { ...attrs, fill: 'none', stroke: 'currentColor', viewBox: '0 0 24 24' }, [
  h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '1.5', d: 'M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4' })
])
const IconEmbeddings = (_, { attrs }) => h('svg', { ...attrs, fill: 'none', stroke: 'currentColor', viewBox: '0 0 24 24' }, [
  h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '1.5', d: 'M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01' })
])
const IconETL = (_, { attrs }) => h('svg', { ...attrs, fill: 'none', stroke: 'currentColor', viewBox: '0 0 24 24' }, [
  h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '1.5', d: 'M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z' })
])
const IconSchedules = (_, { attrs }) => h('svg', { ...attrs, fill: 'none', stroke: 'currentColor', viewBox: '0 0 24 24' }, [
  h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '1.5', d: 'M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z' })
])
const IconSettings = (_, { attrs }) => h('svg', { ...attrs, fill: 'none', stroke: 'currentColor', viewBox: '0 0 24 24' }, [
  h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '1.5', d: 'M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z' }),
  h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '1.5', d: 'M15 12a3 3 0 11-6 0 3 3 0 016 0z' })
])

const navItems = [
  { path: '/', label: 'Dashboard', icon: IconDashboard },
  { path: '/chat', label: 'Chat', icon: IconChat },
  { path: '/catalog', label: 'Catalog', icon: IconCatalog },
  { path: '/sources', label: 'Sources', icon: IconSources },
  { path: '/embeddings', label: 'Embeddings', icon: IconEmbeddings },
  { path: '/etl', label: 'ETL Monitor', icon: IconETL },
  { path: '/schedules', label: 'Schedules', icon: IconSchedules },
  { path: '/settings', label: 'Settings', icon: IconSettings },
]

function handleLogout() {
  authStore.logout()
  router.push('/login')
}
</script>
