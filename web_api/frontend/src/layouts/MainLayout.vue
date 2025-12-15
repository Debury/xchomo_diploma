<template>
  <div class="min-h-screen flex">
    <!-- Sidebar -->
    <aside class="w-64 bg-dark-card border-r border-dark-border flex flex-col">
      <!-- Logo -->
      <div class="p-6 border-b border-dark-border">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-xl">
            🌍
          </div>
          <div>
            <h1 class="text-lg font-bold text-white">Climate RAG</h1>
            <p class="text-xs text-gray-500">Data Pipeline</p>
          </div>
        </div>
      </div>

      <!-- Navigation -->
      <nav class="flex-1 p-4 space-y-1">
        <router-link 
          v-for="item in navItems" 
          :key="item.path"
          :to="item.path"
          class="flex items-center gap-3 px-4 py-3 rounded-lg text-gray-400 hover:bg-dark-hover hover:text-white transition-colors"
          :class="{ 'bg-dark-hover text-white': $route.path === item.path }"
        >
          <span class="text-xl">{{ item.icon }}</span>
          <span>{{ item.label }}</span>
        </router-link>
      </nav>

      <!-- User Section -->
      <div class="p-4 border-t border-dark-border">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <div class="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-sm">
              {{ authStore.user?.username?.[0]?.toUpperCase() || 'U' }}
            </div>
            <span class="text-sm text-gray-300">{{ authStore.user?.username }}</span>
          </div>
          <button @click="handleLogout" class="text-gray-500 hover:text-white transition-colors">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"></path>
            </svg>
          </button>
        </div>
      </div>
    </aside>

    <!-- Main Content -->
    <main class="flex-1 flex flex-col overflow-hidden">
      <!-- Header -->
      <header class="h-16 bg-dark-card border-b border-dark-border flex items-center justify-between px-6">
        <h2 class="text-xl font-semibold text-white">{{ $route.name }}</h2>
        <div class="flex items-center gap-4">
          <div class="flex items-center gap-2 text-sm text-gray-400">
            <div class="w-2 h-2 rounded-full bg-green-500"></div>
            <span>Online</span>
          </div>
          <a href="/docs" target="_blank" class="text-gray-400 hover:text-white text-sm transition-colors">
            API Docs
          </a>
        </div>
      </header>

      <!-- Page Content -->
      <div class="flex-1 overflow-auto p-6 bg-dark-bg">
        <router-view />
      </div>
    </main>
  </div>
</template>

<script setup>
import { useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth'

const router = useRouter()
const authStore = useAuthStore()

const navItems = [
  { path: '/', label: 'Dashboard', icon: '📊' },
  { path: '/chat', label: 'Chat', icon: '💬' },
  { path: '/sources', label: 'Sources', icon: '📁' },
  { path: '/embeddings', label: 'Embeddings', icon: '🔢' }
]

function handleLogout() {
  authStore.logout()
  router.push('/login')
}
</script>
