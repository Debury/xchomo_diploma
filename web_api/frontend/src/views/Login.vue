<template>
  <div class="min-h-screen flex items-center justify-center bg-mendelu-gray-light">
    <div class="w-full max-w-md">
      <div class="card shadow-lg">
        <!-- Logo -->
        <div class="text-center mb-8">
          <div class="w-16 h-16 rounded-full bg-mendelu-green flex items-center justify-center mx-auto mb-4">
            <svg class="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
          </div>
          <h1 class="page-title">MENDELU Climate RAG</h1>
          <p class="page-subtitle">Sign in to continue</p>
        </div>

        <!-- Form -->
        <form @submit.prevent="handleLogin" class="space-y-4">
          <div>
            <label class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-2">Username</label>
            <input
              v-model="username"
              type="text"
              required
              autocomplete="username"
              class="input-field"
              placeholder="Enter username"
            >
          </div>

          <div>
            <label class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-2">Password</label>
            <input
              v-model="password"
              type="password"
              required
              autocomplete="current-password"
              class="input-field"
              placeholder="Enter password"
            >
          </div>

          <div v-if="error" class="p-3 border-l-2 border-mendelu-alert bg-mendelu-alert/5 rounded-lg text-mendelu-alert text-sm">
            {{ error }}
          </div>

          <button
            type="submit"
            :disabled="loading"
            class="w-full btn-primary py-3 flex items-center justify-center gap-2"
          >
            <svg v-if="loading" class="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <span>{{ loading ? 'Signing in...' : 'Sign In' }}</span>
          </button>
        </form>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth'

const router = useRouter()
const authStore = useAuthStore()

const username = ref('')
const password = ref('')
const loading = ref(false)
const error = ref('')

async function handleLogin() {
  error.value = ''
  loading.value = true

  try {
    await authStore.login(username.value, password.value)
    try {
      await router.replace('/')
    } catch (navErr) {
      // Fallback if client-side navigation is blocked by a guard mismatch
      window.location.assign('/app/')
    }
  } catch (err) {
    error.value = err.message || 'Login failed'
  } finally {
    loading.value = false
  }
}
</script>
