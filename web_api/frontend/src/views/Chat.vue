<template>
  <div class="h-full flex flex-col">
    <PageHeader title="Climate Data Chat" subtitle="Ask questions about your climate datasets">
      <template #actions>
        <button @click="clearChat" class="btn-ghost">Clear Chat</button>
      </template>
    </PageHeader>

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
            <div class="flex items-center gap-2 flex-wrap">
              <button
                @click="msg.showChunks = !msg.showChunks"
                class="btn-ghost !px-2 !py-1 text-xs flex items-center gap-1 text-mendelu-green"
              >
                <svg class="w-3 h-3 transition-transform duration-150" :class="{ 'rotate-90': msg.showChunks }" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
                </svg>
                {{ msg.chunks.length }} retrieved chunks
              </button>
              <span class="text-mendelu-gray-semi">·</span>
              <button
                @click="exportMessageData(msg, 'csv')"
                :disabled="!!msg.exporting"
                class="px-2 py-1 text-xs rounded-md font-medium bg-mendelu-green text-white hover:bg-mendelu-green/90 transition-colors duration-150 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
                title="Downloads every chunk in the collection matching the cited datasets/variables, filtered to the year range mentioned in your question."
              >
                <svg class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5 5-5M12 15V3" />
                </svg>
                {{ msg.exporting === 'csv' ? 'Exporting…' : 'Export full dataset (CSV)' }}
              </button>
              <button
                @click="exportMessageData(msg, 'json')"
                :disabled="!!msg.exporting"
                class="px-2 py-1 text-xs rounded-md border border-mendelu-gray-semi text-mendelu-black hover:bg-mendelu-gray-light transition-colors duration-150 disabled:opacity-50"
                title="Same as CSV but as a JSON file"
              >{{ msg.exporting === 'json' ? 'Exporting…' : 'JSON' }}</button>
            </div>
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
            AgenticRAG pipeline (multi-step) usually takes 30–60 s. Hang on.
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
          placeholder="Ask about your documents… (Enter to send)"
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
import { ref, nextTick, onUnmounted } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import PageHeader from '../components/PageHeader.vue'
import { apiFetch } from '../api'
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

const input = ref('')
const messages = ref<any[]>([])
const loading = ref(false)
const loadingStage = ref('Searching Qdrant…')
const loadingElapsed = ref(0)
const messagesContainer = ref(null)
let loadingTimer = null

const quickQuestions = [
  'What variables are available?',
  'Show me temperature trends',
  'Temperature in Czech Republic',
  'Drought indices for Central Europe'
]


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
    const resp = await apiFetch('/rag/docs/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    })
    const data = await resp.json().catch(() => null)
    console.log('[AgenticRAG] response:', data)
    if (!resp.ok) {
      const errMsg = (data && (data.detail || data.error)) || `HTTP ${resp.status}`
      messages.value.push({ role: 'assistant', content: `Error: ${errMsg}` })
    } else if (!data) {
      messages.value.push({ role: 'assistant', content: 'Error: malformed server response' })
    } else {
      // Support multiple RAG backends:
      // 1. Climate RAG: data.chunks with {source_id, variable, similarity, text, metadata}
      // 2. AgenticRAG (Ollama/PDF): data.sources with {source, file_type, header_path, score, rerank_score, text}
      const rawChunks = data.chunks || data.sources || data.contexts || data.results || []
      const chunks = rawChunks.map((c: any) => {
        const meta = c.metadata || {}
        const lat = c.lat ?? meta.lat
        const lon = c.lon ?? meta.lon
        return {
          score: c.similarity ?? c.score ?? c.rerank_score ?? 0,
          dataset: c.dataset_name || meta.dataset_name || c.source_id || c.source || '',
          variable: c.variable || meta.variable || c.header_path || '',
          coordinates: (lat !== undefined && lon !== undefined)
            ? `${Number(lat).toFixed(1)}\u00b0N, ${Number(lon).toFixed(1)}\u00b0E`
            : '',
          time_range: c.time_range || meta.time_range || '',
          text: c.text || c.content || '',
          // Keep the full payload so the per-message export can dump
          // exactly what the chat answer cited, not just the rendered subset above.
          source_id: c.source_id || meta.source_id,
          metadata: meta,
        }
      })

      const spatial = data.spatial_filter
        ? `Filtered: ${data.spatial_filter.description || `${data.spatial_filter.lat_min}-${data.spatial_filter.lat_max}\u00b0N`}`
        : null

      messages.value.push({
        role: 'assistant',
        content: data.answer,
        spatial,
        chunks,
        showChunks: chunks.length > 0,
        // Per-message bulk-export state. The original question + filters
        // are captured here so the Export button works even after later
        // chats overwrite the input box or the user changes filters.
        question,
        sourceFilter: filterSource.value || null,
        variableFilter: filterVariable.value || null,
        exporting: '',
        meta: {
          llm_time_ms: data.llm_time_ms,
          search_time_ms: data.search_time_ms,
          provider: data.provider || 'AgenticRAG (Ollama)'
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

// Find the user question that produced this assistant message — used
// only to slug the export filename.
function questionForMessage(msg: any): string {
  if (msg?.question) return msg.question
  const idx = messages.value.indexOf(msg)
  for (let i = idx - 1; i >= 0; i--) {
    if (messages.value[i].role === 'user') return messages.value[i].content || ''
  }
  return ''
}

// Stable column order for the CSV. Anything else from the metadata is
// appended alphabetically so dataset-specific fields aren't dropped.
const EXPORT_PRIORITY_COLS = [
  'rank', 'score', 'source_id', 'dataset_name', 'variable', 'long_name',
  'standard_name', 'units', 'unit',
  'time_start', 'time_end', 'temporal_frequency',
  'lat_min', 'lat_max', 'lon_min', 'lon_max',
  'stats_mean', 'stats_min', 'stats_max', 'stats_std',
  'region_country', 'spatial_coverage', 'hazard_type', 'impact_sector',
  'text',
]

function flattenChunkForExport(rank: number, chunk: any): Record<string, any> {
  const meta = chunk?.metadata || {}
  const flat: Record<string, any> = {
    rank,
    score: chunk?.score ?? null,
    source_id: meta.source_id ?? chunk?.source_id ?? null,
    variable: meta.variable ?? chunk?.variable ?? null,
    text: chunk?.text ?? '',
  }
  for (const [k, v] of Object.entries(meta)) {
    if (k in flat) continue
    flat[k] = (v && typeof v === 'object') ? JSON.stringify(v) : v
  }
  return flat
}

function csvEscape(v: any): string {
  if (v === null || v === undefined) return ''
  const s = String(v)
  // Quote if it contains comma, quote, newline, or carriage return.
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}

function rowsToCsv(rows: Record<string, any>[]): string {
  if (rows.length === 0) return '﻿rank\n'
  const seen = new Set<string>()
  const cols: string[] = []
  for (const c of EXPORT_PRIORITY_COLS) {
    if (rows.some(r => c in r) && !seen.has(c)) { cols.push(c); seen.add(c) }
  }
  const extras = new Set<string>()
  for (const r of rows) for (const k of Object.keys(r)) if (!seen.has(k)) extras.add(k)
  cols.push(...[...extras].sort())
  // BOM so Excel auto-detects UTF-8.
  const lines: string[] = ['﻿' + cols.join(',')]
  for (const r of rows) {
    lines.push(cols.map(c => csvEscape(r[c])).join(','))
  }
  return lines.join('\n') + '\n'
}

// Pull unique (dataset_name, variable) pairs out of the cited chunks.
// These tell the export endpoint "I want every chunk in the collection
// that's like one of these" — combined with the year filter, that's the
// "give me all 2024 data on this variable" experience a scientist expects.
function citedPairsFromMessage(msg: any): Array<{ dataset_name?: string; variable?: string; source_id?: string }> {
  const seen = new Set<string>()
  const out: Array<any> = []
  for (const c of (msg?.chunks || [])) {
    const meta = c.metadata || {}
    const dataset_name = (meta.dataset_name || c.dataset || '').trim()
    const source_id = (meta.source_id || c.source_id || '').trim()
    const variable = (meta.variable || c.variable || '').trim()
    if (!dataset_name && !source_id) continue
    if (!variable) continue
    const key = `${dataset_name}::${source_id}::${variable}`
    if (seen.has(key)) continue
    seen.add(key)
    const pair: any = { variable }
    if (dataset_name) pair.dataset_name = dataset_name
    else pair.source_id = source_id
    out.push(pair)
  }
  return out
}

// Hand the question + cited pairs to the backend export. The endpoint
// expands them to "every chunk in Qdrant matching one of these
// (dataset, variable) pairs", auto-detects the year window from the
// question, and streams a CSV of every overlapping data point.
async function exportMessageData(msg: any, fmt: 'csv' | 'json') {
  if (msg.exporting) return
  const cited = citedPairsFromMessage(msg)
  if (cited.length === 0) {
    toast.error('No (dataset, variable) info on the cited chunks — try re-sending the question.')
    return
  }
  const question = questionForMessage(msg)
  const slug = question.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '').slice(0, 60) || 'export'

  msg.exporting = fmt
  try {
    const resp = await apiFetch('/rag/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        cited_pairs: cited,
        fmt,
      }),
    })
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}))
      throw new Error(err.detail || `HTTP ${resp.status}`)
    }
    if (fmt === 'csv') {
      const yMin = resp.headers.get('X-Export-Year-Min')
      const yMax = resp.headers.get('X-Export-Year-Max')
      const auto = resp.headers.get('X-Export-Year-Auto') === '1'
      const blob = await resp.blob()
      triggerDownload(blob, `climate_export_${slug}.csv`)
      // Backend doesn't pre-count when streaming, so derive count from rows
      // the user actually got — minus the header line.
      const rowCount = (await blob.text().catch(() => '')).split('\n').length - 2
      const yearNote = (yMin && yMax)
        ? ` covering ${yMin}–${yMax}${auto ? ' (auto-detected)' : ''}`
        : ''
      toast.success(`Exported ~${Math.max(0, rowCount)} rows${yearNote}`)
    } else {
      const data = await resp.json()
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      triggerDownload(blob, `climate_export_${slug}.json`)
      const yearNote = (data.year_min && data.year_max)
        ? ` covering ${data.year_min}–${data.year_max}${data.year_auto_detected ? ' (auto-detected)' : ''}`
        : ''
      const truncNote = data.truncated ? ' (truncated to 50k rows — use CSV for full export)' : ''
      toast.success(`Exported ${data.count ?? 0} chunks${yearNote}${truncNote}`)
    }
  } catch (e: any) {
    console.error('Export failed:', e)
    toast.error(`Export failed: ${e?.message || 'unknown error'}`)
  } finally {
    msg.exporting = ''
  }
}

function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

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
