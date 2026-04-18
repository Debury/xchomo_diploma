<template>
  <div class="h-full flex flex-col">
    <PageHeader title="Climate Data Chat" subtitle="Ask questions about your climate datasets">
      <template #actions>
        <button @click="clearChat" class="btn-ghost">Clear Chat</button>
      </template>
    </PageHeader>

    <!-- Filter Bar — compact, inline, search-driven -->
    <div class="mb-4 flex items-center gap-2 flex-wrap text-xs">
      <span class="font-medium text-mendelu-gray-dark uppercase tracking-wider">Filters</span>
      <input
        v-model="filterSearch"
        type="search"
        placeholder="Search sources &amp; variables…"
        class="input-field !w-48 !py-1 text-xs"
      />
      <select v-model="filterSource" class="input-field !w-40 !py-1 text-xs">
        <option value="">All sources ({{ filteredSources.length }})</option>
        <option v-for="s in filteredSources" :key="s" :value="s">{{ s }}</option>
      </select>
      <select v-model="filterVariable" class="input-field !w-40 !py-1 text-xs">
        <option value="">All variables ({{ filteredVariables.length }})</option>
        <option v-for="v in filteredVariables" :key="v" :value="v">{{ v }}</option>
      </select>
      <button
        v-if="filterSource || filterVariable || filterSearch"
        @click="clearFilters"
        class="text-mendelu-green hover:underline ml-1"
      >Clear</button>
    </div>

    <!-- Chat Messages -->
    <div class="flex-1 overflow-y-auto space-y-4 mb-4 pr-2" ref="messagesContainer">
      <div
        v-for="(msg, idx) in messages"
        :key="idx"
        class="flex items-start gap-3"
        :class="msg.role === 'user' ? 'justify-end' : 'justify-start'"
      >
        <!-- Assistant avatar (left of bubble) -->
        <div
          v-if="msg.role === 'assistant'"
          class="flex-shrink-0 w-8 h-8 rounded-full bg-mendelu-green/15 border border-mendelu-green/30 flex items-center justify-center"
          aria-hidden="true"
          title="ClimateRAG"
        >
          <svg class="w-4 h-4 text-mendelu-green" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
        </div>

        <div
          class="max-w-3xl px-4 py-3 rounded-xl transition-all duration-150"
          :class="msg.role === 'user'
            ? 'bg-mendelu-green/10 text-mendelu-black border border-mendelu-green/20'
            : 'bg-white border-l-2 border-l-mendelu-green border border-mendelu-gray-semi text-mendelu-black shadow-sm'"
        >
          <div
            v-if="msg.role === 'assistant'"
            class="chat-markdown text-sm"
            v-html="renderMarkdown(msg.content)"
          ></div>
          <div v-else class="whitespace-pre-wrap text-sm">{{ msg.content }}</div>

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

        <!-- User avatar (right of bubble) -->
        <div
          v-if="msg.role === 'user'"
          class="flex-shrink-0 w-8 h-8 rounded-full bg-mendelu-gray-light border border-mendelu-gray-semi flex items-center justify-center"
          aria-hidden="true"
          title="You"
        >
          <svg class="w-4 h-4 text-mendelu-gray-dark" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
          </svg>
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
    <div class="card !p-3">
      <div class="flex items-end gap-2">
        <textarea
          v-model="input"
          @keydown.enter.exact.prevent="sendMessage"
          rows="3"
          class="input-field resize-none flex-1"
          placeholder="Ask about your climate data... (Enter to send)"
          :disabled="loading"
        ></textarea>
        <button
          @click="sendMessage"
          :disabled="loading || !input.trim()"
          class="btn-primary flex items-center justify-center !w-10 !h-10 !p-0 flex-shrink-0 disabled:opacity-50"
          aria-label="Send message"
          title="Send (Enter)"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
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
import { ref, computed, nextTick, onMounted, onUnmounted } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import PageHeader from '../components/PageHeader.vue'
import { apiFetch } from '../api'
import { useToast } from '../composables/useToast'

// Render assistant answers as Markdown. `marked` parses to HTML, DOMPurify
// strips any script/event-handler content before we v-html it — the LLM
// output is not trusted input, so sanitization is mandatory.
marked.setOptions({ breaks: true, gfm: true })
function renderMarkdown(text: string): string {
  if (!text) return ''
  try {
    const html = marked.parse(text, { async: false }) as string
    return DOMPurify.sanitize(html)
  } catch {
    return text
  }
}

const toast = useToast()

const input = ref('')
const messages = ref<any[]>([])
const loading = ref(false)
const loadingStage = ref('Searching Qdrant…')
const loadingElapsed = ref(0)
const messagesContainer = ref(null)
const filterSource = ref('')
const filterVariable = ref('')
const filterSearch = ref('')
const availableSources = ref<any[]>([])
const availableVariables = ref<any[]>([])
let loadingTimer = null

const filteredSources = computed(() => {
  const q = filterSearch.value.toLowerCase().trim()
  if (!q) return availableSources.value
  return availableSources.value.filter((s: any) => String(s).toLowerCase().includes(q))
})
const filteredVariables = computed(() => {
  const q = filterSearch.value.toLowerCase().trim()
  if (!q) return availableVariables.value
  return availableVariables.value.filter((v: any) => String(v).toLowerCase().includes(q))
})
function clearFilters() {
  filterSearch.value = ''
  filterSource.value = ''
  filterVariable.value = ''
}

const quickQuestions = [
  'What variables are available?',
  'Show me temperature trends',
  'Temperature in Czech Republic',
  'Drought indices for Central Europe'
]

async function loadRagInfo() {
  try {
    const resp = await apiFetch('/rag/info')
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    const data = await resp.json()
    availableSources.value = data.sources || []
    availableVariables.value = data.variables || []
  } catch (e: any) {
    toast.error(`Could not load source/variable filters: ${e?.message ?? 'network error'}`, loadRagInfo)
  }
}

onMounted(loadRagInfo)

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

<style scoped>
/* Inline typography for markdown-rendered assistant answers.
   Tailwind typography plugin isn't installed, so we do the small bits here. */
.chat-markdown :deep(p) { margin: 0 0 0.5rem 0; }
.chat-markdown :deep(p:last-child) { margin-bottom: 0; }
.chat-markdown :deep(strong) { font-weight: 600; color: inherit; }
.chat-markdown :deep(em) { font-style: italic; }
.chat-markdown :deep(ul),
.chat-markdown :deep(ol) { margin: 0.25rem 0 0.5rem 1.25rem; padding: 0; }
.chat-markdown :deep(li) { margin: 0.125rem 0; }
.chat-markdown :deep(ul) { list-style: disc; }
.chat-markdown :deep(ol) { list-style: decimal; }
.chat-markdown :deep(h1),
.chat-markdown :deep(h2),
.chat-markdown :deep(h3) { font-weight: 600; margin: 0.5rem 0 0.25rem 0; }
.chat-markdown :deep(h1) { font-size: 1.05rem; }
.chat-markdown :deep(h2) { font-size: 1rem; }
.chat-markdown :deep(h3) { font-size: 0.95rem; }
.chat-markdown :deep(code) {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.85em;
  background: rgba(0, 0, 0, 0.05);
  padding: 0.05rem 0.3rem;
  border-radius: 0.25rem;
}
.chat-markdown :deep(pre) {
  background: rgba(0, 0, 0, 0.05);
  padding: 0.5rem 0.75rem;
  border-radius: 0.375rem;
  overflow-x: auto;
  margin: 0.5rem 0;
}
.chat-markdown :deep(pre code) { background: transparent; padding: 0; }
.chat-markdown :deep(a) { color: #2e6f40; text-decoration: underline; }
.chat-markdown :deep(blockquote) {
  border-left: 3px solid rgba(46, 111, 64, 0.4);
  padding-left: 0.75rem;
  margin: 0.5rem 0;
  color: #4a5568;
}
.chat-markdown :deep(table) {
  border-collapse: collapse;
  margin: 0.5rem 0;
  font-size: 0.85em;
}
.chat-markdown :deep(th),
.chat-markdown :deep(td) {
  border: 1px solid rgba(0, 0, 0, 0.1);
  padding: 0.25rem 0.5rem;
  text-align: left;
}
.chat-markdown :deep(th) { background: rgba(0, 0, 0, 0.04); font-weight: 600; }
</style>
