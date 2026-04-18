<template>
  <div class="min-h-screen flex flex-col mesh-gradient">
    <!-- Top Navigation -->
    <header
      class="h-14 border-b flex items-center px-4 md:px-6 flex-shrink-0 z-50 transition-colors duration-150"
      :class="isDark ? 'bg-dark-bg border-dark-border' : 'bg-white border-mendelu-gray-semi'"
    >
      <!-- Logo -->
      <router-link to="/" class="flex items-center gap-2.5 mr-8 flex-shrink-0 group">
        <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-mendelu-green to-mendelu-success flex items-center justify-center shadow-sm transition-transform duration-200 group-hover:scale-105">
          <svg class="w-4.5 h-4.5 text-white" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
        <div class="hidden sm:flex flex-col">
          <span class="text-sm font-bold tracking-tight leading-none">ClimateRAG</span>
          <span class="text-[9px] font-medium tracking-widest uppercase leading-none mt-0.5" :class="isDark ? 'text-gray-500' : 'text-mendelu-gray-dark/60'">Observatory</span>
        </div>
      </router-link>

      <!-- Desktop Navigation -->
      <nav class="hidden md:flex items-center gap-0.5 flex-1">
        <router-link
          v-for="item in navItems"
          :key="item.path"
          :to="item.path"
          class="px-3 py-1.5 rounded-lg text-[13px] font-medium transition-all duration-200 relative"
          :class="isActive(item.path)
            ? (isDark ? 'text-white' : 'text-mendelu-green')
            : (isDark ? 'text-gray-400 hover:text-gray-200' : 'text-mendelu-gray-dark hover:text-mendelu-black')"
        >
          <span class="relative z-10">{{ item.label }}</span>
          <span
            v-if="isActive(item.path)"
            class="absolute inset-0 rounded-lg transition-all duration-200"
            :class="isDark ? 'bg-white/8' : 'bg-mendelu-green/8'"
          ></span>
        </router-link>
      </nav>

      <!-- Right section -->
      <div class="flex items-center gap-3 ml-auto">
        <!-- System status indicator -->
        <div class="flex items-center gap-2 text-xs" :class="isDark ? 'text-gray-400' : 'text-mendelu-gray-dark'">
          <div class="relative">
            <div
              class="w-2 h-2 rounded-full transition-colors duration-500"
              :class="apiHealthy ? 'bg-mendelu-success' : 'bg-mendelu-alert'"
            ></div>
            <div
              v-if="apiHealthy"
              class="absolute inset-0 w-2 h-2 rounded-full bg-mendelu-success pulse-online"
            ></div>
          </div>
          <span class="hidden sm:inline font-medium font-mono text-[11px]">{{ apiHealthy ? 'ONLINE' : 'OFFLINE' }}</span>
        </div>

        <!-- Theme toggle -->
        <button
          @click="themeStore.toggle()"
          class="p-1.5 rounded-lg transition-all duration-200"
          :class="isDark ? 'text-gray-400 hover:bg-white/8 hover:text-gray-200' : 'text-mendelu-gray-dark hover:bg-mendelu-gray-light'"
          :title="isDark ? 'Switch to light mode' : 'Switch to dark mode'"
        >
          <svg v-if="isDark" class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
          </svg>
          <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
          </svg>
        </button>

        <div class="w-px h-5" :class="isDark ? 'bg-dark-border/50' : 'bg-mendelu-gray-semi'"></div>

        <a href="/docs" target="_blank" class="text-[11px] font-medium tracking-wide uppercase transition-colors" :class="isDark ? 'text-gray-500 hover:text-gray-300' : 'text-mendelu-gray-dark/60 hover:text-mendelu-green'">Docs</a>

        <div class="flex items-center gap-2">
          <div class="w-7 h-7 rounded-lg bg-gradient-to-br from-mendelu-green/20 to-mendelu-success/20 flex items-center justify-center text-[10px] text-mendelu-green font-bold tracking-tight">
            {{ authStore.user?.username?.[0]?.toUpperCase() || 'U' }}
          </div>
          <button @click="handleLogout" class="text-[11px] font-medium tracking-wide uppercase transition-colors" :class="isDark ? 'text-gray-500 hover:text-gray-300' : 'text-mendelu-gray-dark/60 hover:text-mendelu-black'">
            Logout
          </button>
        </div>
      </div>

      <!-- Mobile hamburger -->
      <button
        @click="mobileOpen = !mobileOpen"
        class="md:hidden ml-3 p-1.5 rounded-lg transition-colors"
        :class="isDark ? 'text-gray-400 hover:bg-white/10' : 'text-mendelu-gray-dark hover:bg-mendelu-gray-light'"
      >
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path v-if="!mobileOpen" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
          <path v-else stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </header>

    <!-- Mobile dropdown menu -->
    <Transition name="dropdown">
      <div
        v-if="mobileOpen"
        class="md:hidden border-b px-4 py-2 space-y-0.5 z-40"
        :class="isDark ? 'bg-dark-card border-dark-border' : 'bg-white border-mendelu-gray-semi'"
      >
        <router-link
          v-for="item in navItems"
          :key="item.path"
          :to="item.path"
          class="block px-3 py-2 rounded-lg text-sm font-medium transition-colors"
          :class="isActive(item.path)
            ? (isDark ? 'bg-white/10 text-white' : 'bg-mendelu-green/10 text-mendelu-green')
            : (isDark ? 'text-gray-400 hover:bg-white/5' : 'text-mendelu-gray-dark hover:bg-mendelu-gray-light')"
          @click="mobileOpen = false"
        >
          {{ item.label }}
        </router-link>
      </div>
    </Transition>

    <!-- Page Content -->
    <main class="flex-1 overflow-auto p-4 md:p-6 lg:p-8 relative">
      <div class="max-w-7xl mx-auto">
        <ErrorBoundary>
          <router-view />
        </ErrorBoundary>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useAuthStore } from '../stores/auth'
import { useThemeStore } from '../stores/theme'
import ErrorBoundary from '../components/ErrorBoundary.vue'

const router = useRouter()
const route = useRoute()
const authStore = useAuthStore()
const themeStore = useThemeStore()

const isDark = computed(() => themeStore.theme === 'dark')
const mobileOpen = ref(false)
const apiHealthy = ref(true)
let healthInterval: ReturnType<typeof setInterval> | null = null

function stopHealthPolling(): void {
  if (healthInterval) {
    clearInterval(healthInterval)
    healthInterval = null
  }
}

function startHealthPolling(): void {
  stopHealthPolling()
  checkHealth()
  healthInterval = setInterval(checkHealth, 30000)
}

function isActive(path: string): boolean {
  if (path === '/') return route.path === '/' || route.path === ''
  return route.path.startsWith(path)
}

async function checkHealth(): Promise<void> {
  if (typeof document !== 'undefined' && document.hidden) return  // idle tab skip
  try {
    const resp = await fetch(`/health?t=${Date.now()}`)
    apiHealthy.value = resp.ok
  } catch {
    apiHealthy.value = false
  }
}

function onVisibilityChange(): void {
  if (document.hidden) {
    stopHealthPolling()
  } else {
    startHealthPolling()
  }
}

function prefetchRoutes(): void {
  const chunks = [
    () => import('../views/Chat.vue'),
    () => import('../views/Sources.vue'),
    () => import('../views/Catalog.vue'),
    () => import('../views/ETLMonitor.vue'),
    () => import('../views/Schedules.vue'),
    () => import('../views/Settings.vue'),
  ]
  const schedule = (fn: () => void) => {
    const ric = (window as any).requestIdleCallback
    if (typeof ric === 'function') ric(fn, { timeout: 2000 })
    else setTimeout(fn, 200)
  }
  for (const load of chunks) schedule(() => { load().catch(() => { /* ignore prefetch errors */ }) })
}

onMounted(() => {
  startHealthPolling()
  document.addEventListener('visibilitychange', onVisibilityChange)
  prefetchRoutes()
})

onUnmounted(() => {
  stopHealthPolling()
  document.removeEventListener('visibilitychange', onVisibilityChange)
})

const navItems = [
  { path: '/', label: 'Dashboard' },
  { path: '/chat', label: 'Chat' },
  { path: '/sources', label: 'Sources' },
  { path: '/catalog', label: 'Catalog' },
  { path: '/etl', label: 'ETL Monitor' },
  { path: '/schedules', label: 'Schedules' },
  { path: '/settings', label: 'Settings' },
]

function handleLogout(): void {
  stopHealthPolling()
  document.removeEventListener('visibilitychange', onVisibilityChange)
  authStore.logout()
  router.push('/login')
}
</script>

<style scoped>
.dropdown-enter-active {
  transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}
.dropdown-leave-active {
  transition: all 0.15s ease-in;
}
.dropdown-enter-from,
.dropdown-leave-to {
  opacity: 0;
  transform: translateY(-4px);
}
</style>
