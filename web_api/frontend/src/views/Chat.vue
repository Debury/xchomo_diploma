<template>
  <div class="h-full flex flex-col">
    <PageHeader title="Climate Data Chat" subtitle="Ask questions about your climate datasets">
      <template #actions>
        <button @click="clearChat" class="btn-ghost">Clear Chat</button>
      </template>
    </PageHeader>

    <!-- Filter Bar -->
    <div class="card !p-3 mb-4">
      <div class="flex items-center gap-3 flex-wrap">
        <span class="text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider">Filters</span>
        <select v-model="filterSource" class="input-field !w-auto !py-1.5 text-xs">
          <option value="">All Sources</option>
          <option v-for="s in availableSources" :key="s" :value="s">{{ s }}</option>
        </select>
        <select v-model="filterVariable" class="input-field !w-auto !py-1.5 text-xs">
          <option value="">All Variables</option>
          <option v-for="v in availableVariables" :key="v" :value="v">{{ v }}</option>
        </select>
        <span v-if="filterSource || filterVariable" class="badge-info">Filtered</span>
      </div>
    </div>

    <!-- Chat Messages -->
    <div class="flex-1 overflow-y-auto space-y-4 mb-4 pr-2" ref="messagesContainer">
      <div
        v-for="(msg, idx) in messages"
        :key="idx"
        class="flex"
        :class="msg.role === 'user' ? 'justify-end' : 'justify-start'"
      >
        <div
          class="max-w-3xl px-4 py-3 rounded-xl transition-all duration-150"
          :class="msg.role === 'user'
            ? 'bg-mendelu-green/10 text-mendelu-black border border-mendelu-green/20'
            : 'bg-white border-l-2 border-l-mendelu-green border border-mendelu-gray-semi text-mendelu-black shadow-sm'"
        >
          <div class="whitespace-pre-wrap text-sm">{{ msg.content }}</div>

          <!-- Spatial badge -->
          <div v-if="msg.spatial" class="mt-2">
            <span class="badge-info">{{ msg.spatial }}</span>
          </div>

          <!-- Timing metadata -->
          <div v-if="msg.meta" class="mt-2 pt-2 border-t border-mendelu-gray-semi text-xs text-mendelu-gray-dark">
            <span v-if="msg.meta.llm_time_ms">{{ msg.meta.llm_time_ms.toFixed(0) }}ms LLM</span>
            <span v-if="msg.meta.search_time_ms" class="ml-3">{{ msg.meta.search_time_ms.toFixed(0) }}ms search</span>
            <span v-if="msg.meta.provider" class="ml-3">{{ msg.meta.provider }}</span>
          </div>

          <!-- Chunk Details -->
          <div v-if="msg.chunks && msg.chunks.length" class="mt-3">
            <button
              @click="msg.showChunks = !msg.showChunks"
              class="btn-ghost !px-2 !py-1 text-xs flex items-center gap-1 text-mendelu-green"
            >
              <svg class="w-3 h-3 transition-transform duration-150" :class="{ 'rotate-90': msg.showChunks }" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
              </svg>
              {{ msg.chunks.length }} retrieved chunks
            </button>
            <div v-if="msg.showChunks" class="mt-2 space-y-2">
              <div
                v-for="(chunk, ci) in msg.chunks"
                :key="ci"
                class="p-3 rounded-lg text-xs bg-mendelu-gray-light"
              >
                <div class="flex items-center gap-2 mb-1 flex-wrap">
                  <span class="font-medium text-mendelu-black">
                    Score: {{ (chunk.score * 100).toFixed(1) }}%
                  </span>
                  <span v-if="chunk.dataset" class="badge-info">{{ chunk.dataset }}</span>
                  <span v-if="chunk.variable" class="badge-success">{{ chunk.variable }}</span>
                </div>
                <div v-if="chunk.coordinates || chunk.time_range" class="flex gap-3 mb-1 text-mendelu-gray-dark">
                  <span v-if="chunk.coordinates">{{ chunk.coordinates }}</span>
                  <span v-if="chunk.time_range">{{ chunk.time_range }}</span>
                </div>
                <p class="text-mendelu-gray-dark line-clamp-3">{{ chunk.text }}</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Loading indicator -->
      <div v-if="loading" class="flex justify-start">
        <div class="bg-white border border-mendelu-gray-semi px-4 py-3 rounded-xl shadow-sm min-w-[280px]">
          <div class="flex items-center space-x-2 mb-1.5">
            <div class="flex space-x-1">
              <div class="w-2 h-2 bg-mendelu-green rounded-full animate-bounce"></div>
              <div class="w-2 h-2 bg-mendelu-green rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
              <div class="w-2 h-2 bg-mendelu-green rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
            </div>
            <span class="text-mendelu-black text-sm font-medium">{{ loadingStage }}</span>
            <span class="text-mendelu-gray-dark text-xs tabular-nums ml-auto">{{ loadingElapsed }}s</span>
          </div>
          <p class="text-[11px] text-mendelu-gray-dark/70 leading-tight">
            Semantic search + LLM synthesis usually takes 15–25 s. Hang on.
          </p>
        </div>
      </div>
    </div>

    <!-- Input Area -->
    <div class="card !p-4">
      <div class="relative">
        <textarea
          v-model="input"
          @keydown.enter.exact.prevent="sendMessage"
          rows="3"
          class="input-field resize-none pr-16"
          placeholder="Ask about your climate data... (Enter to send)"
          :disabled="loading"
        ></textarea>
        <button
          @click="sendMessage"
          :disabled="loading || !input.trim()"
          class="absolute right-2 bottom-2 btn-primary !py-1.5 !px-3 disabled:opacity-50"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
          </svg>
        </button>
      </div>

      <!-- Quick Questions -->
      <div v-if="messages.length === 0" class="mt-3 flex flex-wrap gap-2">
        <button
          v-for="q in quickQuestions"
          :key="q"
          @click="input = q; sendMessage()"
          class="btn-ghost text-xs text-mendelu-green hover:bg-mendelu-green/10"
        >
          {{ q }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick, onMounted, onUnmounted } from 'vue'
import PageHeader from '../components/PageHeader.vue'
import { apiFetch } from '../api'

const input = ref('')
const messages = ref<any[]>([])
const loading = ref(false)
const loadingStage = ref('Searching Qdrant…')
const loadingElapsed = ref(0)
const messagesContainer = ref(null)
const filterSource = ref('')
const filterVariable = ref('')
const availableSources = ref<any[]>([])
const availableVariables = ref<any[]>([])
let loadingTimer = null

const quickQuestions = [
  'What variables are available?',
  'Show me temperature trends',
  'Temperature in Czech Republic',
  'Drought indices for Central Europe'
]

onMounted(async () => {
  try {
    const resp = await apiFetch('/rag/info')
    if (resp.ok) {
      const data = await resp.json()
      availableSources.value = data.sources || []
      availableVariables.value = data.variables || []
    }
  } catch (e) {}
})

onUnmounted(() => {
  if (loadingTimer) {
    clearInterval(loadingTimer)
    loadingTimer = null
  }
})

function startLoadingTimer() {
  loadingElapsed.value = 0
  loadingStage.value = 'Searching Qdrant…'
  if (loadingTimer) clearInterval(loadingTimer)
  loadingTimer = setInterval(() => {
    loadingElapsed.value += 1
    if (loadingElapsed.value === 3) loadingStage.value = 'Reranking results…'
    if (loadingElapsed.value === 6) loadingStage.value = 'Asking the LLM…'
    if (loadingElapsed.value === 25) loadingStage.value = 'Still working… long query'
  }, 1000)
}

function stopLoadingTimer() {
  if (loadingTimer) {
    clearInterval(loadingTimer)
    loadingTimer = null
  }
}

async function sendMessage() {
  const question = input.value.trim()
  if (!question || loading.value) return

  messages.value.push({ role: 'user', content: question })
  input.value = ''
  loading.value = true
  startLoadingTimer()
  await scrollToBottom()

  try {
    const body: Record<string, any> = { question, limit: 5 }
    if (filterSource.value) body.source_filter = filterSource.value
    if (filterVariable.value) body.variable_filter = filterVariable.value

    const resp = await apiFetch('/rag/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    })

    const data = await resp.json().catch(() => null)
    if (!resp.ok) {
      const errMsg = (data && (data.detail || data.error)) || `HTTP ${resp.status}`
      messages.value.push({ role: 'assistant', content: `Error: ${errMsg}` })
    } else if (!data) {
      messages.value.push({ role: 'assistant', content: 'Error: malformed server response' })
    } else if (data.error) {
      messages.value.push({ role: 'assistant', content: `Error: ${data.error}` })
    } else {
      const chunks = (data.contexts || data.results || []).map(c => ({
        score: c.score || 0,
        dataset: c.dataset_name || c.source || '',
        variable: c.variable || '',
        coordinates: c.lat !== undefined ? `${c.lat?.toFixed(1)}\u00b0N, ${c.lon?.toFixed(1)}\u00b0E` : '',
        time_range: c.time_range || '',
        text: c.text || c.content || ''
      }))

      const spatial = data.spatial_filter
        ? `Filtered: ${data.spatial_filter.description || `${data.spatial_filter.lat_min}-${data.spatial_filter.lat_max}\u00b0N`}`
        : null

      messages.value.push({
        role: 'assistant',
        content: data.answer,
        spatial,
        chunks,
        showChunks: false,
        meta: {
          llm_time_ms: data.llm_time_ms,
          search_time_ms: data.search_time_ms,
          provider: data.provider
        }
      })
    }
  } catch (e) {
    messages.value.push({ role: 'assistant', content: `Connection error: ${e.message}` })
  } finally {
    loading.value = false
    stopLoadingTimer()
    await scrollToBottom()
  }
}

function clearChat() { messages.value = [] }

async function scrollToBottom() {
  await nextTick()
  if (messagesContainer.value) messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
}
</script>
