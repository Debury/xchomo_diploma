<template>
  <div class="min-h-screen flex flex-col">
    <!-- Top Navigation -->
    <header class="h-14 border-b flex items-center px-4 md:px-6 flex-shrink-0 z-50 transition-colors"
      :class="isDark ? 'bg-dark-card border-dark-border' : 'bg-white border-mendelu-gray-semi'"
    >
      <!-- Logo -->
      <router-link to="/" class="flex items-center gap-2.5 mr-8 flex-shrink-0">
        <div class="w-7 h-7 rounded-md bg-mendelu-green flex items-center justify-center">
          <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
        <span class="text-sm font-semibold tracking-tight hidden sm:block">ClimateRAG</span>
      </router-link>

      <!-- Desktop Navigation -->
      <nav class="hidden md:flex items-center gap-1 flex-1">
        <router-link
          v-for="item in navItems"
          :key="item.path"
          :to="item.path"
          class="px-3 py-1.5 rounded-md text-[13px] font-medium transition-all duration-150"
          :class="isActive(item.path)
            ? (isDark ? 'bg-white/10 text-white' : 'bg-mendelu-green/10 text-mendelu-green')
            : (isDark ? 'text-gray-400 hover:bg-white/5 hover:text-gray-200' : 'text-mendelu-gray-dark hover:bg-mendelu-gray-light hover:text-mendelu-black')"
        >
          {{ item.label }}
        </router-link>
      </nav>

      <!-- Right section -->
      <div class="flex items-center gap-3 ml-auto">
        <!-- Health dot -->
        <div class="flex items-center gap-1.5 text-xs" :class="isDark ? 'text-gray-400' : 'text-mendelu-gray-dark'">
          <div
            class="w-1.5 h-1.5 rounded-full transition-colors duration-300"
            :class="apiHealthy ? 'bg-mendelu-success' : 'bg-mendelu-alert'"
          ></div>
          <span class="hidden sm:inline">{{ apiHealthy ? 'Online' : 'Offline' }}</span>
        </div>

        <!-- Theme toggle -->
        <button
          @click="themeStore.toggle()"
          class="p-1.5 rounded-md transition-colors"
          :class="isDark ? 'text-gray-400 hover:bg-white/10 hover:text-gray-200' : 'text-mendelu-gray-dark hover:bg-mendelu-gray-light'"
          :title="isDark ? 'Switch to light mode' : 'Switch to dark mode'"
        >
          <!-- Sun icon (show in dark mode) -->
          <svg v-if="isDark" class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
          </svg>
          <!-- Moon icon (show in light mode) -->
          <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
          </svg>
        </button>

        <div class="w-px h-5" :class="isDark ? 'bg-dark-border' : 'bg-mendelu-gray-semi'"></div>

        <a href="/docs" target="_blank" class="text-xs transition-colors" :class="isDark ? 'text-gray-500 hover:text-gray-300' : 'text-mendelu-gray-dark hover:text-mendelu-green'">Docs</a>

        <div class="flex items-center gap-2">
          <div class="w-6 h-6 rounded-md bg-mendelu-green/20 flex items-center justify-center text-[10px] text-mendelu-green font-semibold">
            {{ authStore.user?.username?.[0]?.toUpperCase() || 'U' }}
          </div>
          <button @click="handleLogout" class="text-xs transition-colors" :class="isDark ? 'text-gray-500 hover:text-gray-300' : 'text-mendelu-gray-dark hover:text-mendelu-black'">
            Logout
          </button>
        </div>
      </div>

      <!-- Mobile hamburger -->
      <button
        @click="mobileOpen = !mobileOpen"
        class="md:hidden ml-3 p-1.5 rounded-md transition-colors"
        :class="isDark ? 'text-gray-400 hover:bg-white/10' : 'text-mendelu-gray-dark hover:bg-mendelu-gray-light'"
      >
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path v-if="!mobileOpen" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
          <path v-else stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </header>

    <!-- Mobile dropdown menu -->
    <div
      v-if="mobileOpen"
      class="md:hidden border-b px-4 py-2 space-y-0.5 z-40"
      :class="isDark ? 'bg-dark-card border-dark-border' : 'bg-white border-mendelu-gray-semi'"
    >
      <router-link
        v-for="item in navItems"
        :key="item.path"
        :to="item.path"
        class="block px-3 py-2 rounded-md text-sm transition-colors"
        :class="isActive(item.path)
          ? (isDark ? 'bg-white/10 text-white' : 'bg-mendelu-green/10 text-mendelu-green')
          : (isDark ? 'text-gray-400 hover:bg-white/5' : 'text-mendelu-gray-dark hover:bg-mendelu-gray-light')"
        @click="mobileOpen = false"
      >
        {{ item.label }}
      </router-link>
    </div>

    <!-- Page Content -->
    <main class="flex-1 overflow-auto p-4 md:p-6">
      <router-view />
    </main>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useAuthStore } from '../stores/auth'
import { useThemeStore } from '../stores/theme'

const router = useRouter()
const route = useRoute()
const authStore = useAuthStore()
const themeStore = useThemeStore()

const isDark = computed(() => themeStore.theme === 'dark')
const mobileOpen = ref(false)
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

const navItems = [
  { path: '/', label: 'Dashboard' },
  { path: '/chat', label: 'Chat' },
  { path: '/catalog', label: 'Catalog' },
  { path: '/sources', label: 'Sources' },
  { path: '/embeddings', label: 'Embeddings' },
  { path: '/etl', label: 'ETL Monitor' },
  { path: '/schedules', label: 'Schedules' },
  { path: '/settings', label: 'Settings' },
]

function handleLogout() {
  authStore.logout()
  router.push('/login')
}
</script>
