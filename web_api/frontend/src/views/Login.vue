<template>
  <div class="min-h-screen flex items-center justify-center bg-dark-bg">
    <div class="w-full max-w-md">
      <div class="card">
        <!-- Logo -->
        <div class="text-center mb-8">
          <div class="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-3xl mx-auto mb-4">
            🌍
          </div>
          <h1 class="text-2xl font-bold text-white">Climate RAG</h1>
          <p class="text-gray-500 mt-1">Sign in to continue</p>
        </div>

        <!-- Form -->
        <form @submit.prevent="handleLogin" class="space-y-4">
          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">Username</label>
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
            <label class="block text-sm font-medium text-gray-300 mb-2">Password</label>
            <input 
              v-model="password"
              type="password" 
              required
              autocomplete="current-password"
              class="input-field"
              placeholder="Enter password"
            >
          </div>

          <div v-if="error" class="p-3 bg-red-500/10 border border-red-500/50 rounded-lg text-red-400 text-sm">
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

<script setup>
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
    router.push('/')
  } catch (err) {
    error.value = err.message
  } finally {
    loading.value = false
  }
}
</script>
