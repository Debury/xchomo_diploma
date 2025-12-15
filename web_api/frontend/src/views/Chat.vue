<template>
  <div class="h-full flex flex-col">
    <!-- Chat Header -->
    <div class="flex items-center justify-between mb-6">
      <div>
        <h1 class="text-2xl font-bold text-white">Climate Data Chat</h1>
        <p class="text-gray-400">Ask questions about your climate datasets</p>
      </div>
      <button 
        @click="clearChat"
        class="px-4 py-2 bg-dark-hover rounded-lg text-gray-400 hover:text-white transition-colors"
      >
        Clear Chat
      </button>
    </div>

    <!-- Chat Messages -->
    <div class="flex-1 overflow-y-auto space-y-4 mb-6 pr-2" ref="messagesContainer">
      <div 
        v-for="(msg, idx) in messages" 
        :key="idx"
        class="flex"
        :class="msg.role === 'user' ? 'justify-end' : 'justify-start'"
      >
        <div 
          class="max-w-3xl px-4 py-3 rounded-lg"
          :class="msg.role === 'user' 
            ? 'bg-blue-600 text-white' 
            : 'bg-dark-card border border-dark-border text-gray-200'"
        >
          <div class="whitespace-pre-wrap">{{ msg.content }}</div>
          <div v-if="msg.meta" class="mt-2 pt-2 border-t border-gray-600 text-xs text-gray-400">
            <span v-if="msg.meta.llm_time_ms">⏱️ {{ msg.meta.llm_time_ms.toFixed(0) }}ms LLM</span>
            <span v-if="msg.meta.search_time_ms" class="ml-3">🔍 {{ msg.meta.search_time_ms.toFixed(0) }}ms search</span>
            <span v-if="msg.meta.provider" class="ml-3">🤖 {{ msg.meta.provider }}</span>
          </div>
        </div>
      </div>

      <!-- Loading indicator -->
      <div v-if="loading" class="flex justify-start">
        <div class="bg-dark-card border border-dark-border px-4 py-3 rounded-lg">
          <div class="flex items-center space-x-2">
            <div class="animate-pulse flex space-x-1">
              <div class="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
              <div class="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
              <div class="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
            </div>
            <span class="text-gray-400">Thinking...</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Input Area -->
    <div class="bg-dark-card border border-dark-border rounded-lg p-4">
      <div class="flex items-end space-x-4">
        <div class="flex-1">
          <textarea
            v-model="input"
            @keydown.enter.exact.prevent="sendMessage"
            rows="3"
            class="w-full bg-dark-hover border border-dark-border rounded-lg px-4 py-3 text-white placeholder-gray-500 resize-none focus:outline-none focus:border-blue-500"
            placeholder="Ask about your climate data... (Enter to send)"
            :disabled="loading"
          ></textarea>
        </div>
        <button
          @click="sendMessage"
          :disabled="loading || !input.trim()"
          class="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {{ loading ? 'Sending...' : 'Send' }}
        </button>
      </div>
      
      <!-- Quick Questions -->
      <div class="mt-3 flex flex-wrap gap-2">
        <button
          v-for="q in quickQuestions"
          :key="q"
          @click="input = q; sendMessage()"
          class="text-xs px-3 py-1.5 bg-dark-hover text-gray-400 rounded-full hover:text-white transition-colors"
        >
          {{ q }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue'

const input = ref('')
const messages = ref([])
const loading = ref(false)
const messagesContainer = ref(null)

const quickQuestions = [
  'What variables are available?',
  'Show me temperature trends',
  'What is the spatial coverage?',
  'List data sources'
]

async function sendMessage() {
  const question = input.value.trim()
  if (!question || loading.value) return
  
  // Add user message
  messages.value.push({ role: 'user', content: question })
  input.value = ''
  loading.value = true
  
  await scrollToBottom()
  
  try {
    const resp = await fetch('/rag/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, limit: 5 })
    })
    
    const data = await resp.json()
    
    if (data.error) {
      messages.value.push({ 
        role: 'assistant', 
        content: `Error: ${data.error}` 
      })
    } else {
      messages.value.push({ 
        role: 'assistant', 
        content: data.answer,
        meta: {
          llm_time_ms: data.llm_time_ms,
          search_time_ms: data.search_time_ms,
          provider: data.provider
        }
      })
    }
  } catch (e) {
    messages.value.push({ 
      role: 'assistant', 
      content: `Connection error: ${e.message}` 
    })
  } finally {
    loading.value = false
    await scrollToBottom()
  }
}

function clearChat() {
  messages.value = []
}

async function scrollToBottom() {
  await nextTick()
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}
</script>
