<template>
  <div class="space-y-6">
    <PageHeader title="Settings" subtitle="System configuration and status">
      <template #actions>
        <button @click="refreshSettings" :disabled="loading" class="btn-secondary disabled:opacity-50">Refresh</button>
      </template>
    </PageHeader>

    <!-- LLM Configuration -->
    <div class="card">
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-sm font-medium text-mendelu-black">LLM Configuration</h3>
        <span v-if="saveSuccess" class="text-xs text-mendelu-success font-medium">Saved!</span>
        <button v-if="hasChanges" @click="saveSettings" :disabled="saving" class="btn-primary !py-1.5 !text-xs disabled:opacity-50">
          {{ saving ? 'Saving...' : 'Save Changes' }}
        </button>
      </div>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label for="settings-openrouter-key" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">OpenRouter API Key</label>
          <div class="relative">
            <input
              id="settings-openrouter-key"
              :type="revealedKeys.openrouter_api_key ? 'text' : 'password'"
              v-model="credentialEdits.openrouter_api_key"
              :placeholder="credentials.openrouter_api_key?.configured ? credentials.openrouter_api_key?.masked : 'sk-or-v1-...'"
              class="input-field pr-14"
            />
            <button
              v-if="credentials.openrouter_api_key?.configured || credentialEdits.openrouter_api_key"
              type="button"
              @click="toggleReveal('openrouter_api_key')"
              class="absolute right-2 top-1/2 -translate-y-1/2 btn-ghost !px-2 !py-0.5 text-xs"
            >
              {{ revealedKeys.openrouter_api_key ? 'Hide' : 'Show' }}
            </button>
          </div>
          <div class="flex items-center gap-1.5 mt-1">
            <span class="w-1.5 h-1.5 rounded-full" :class="credentials.openrouter_api_key?.configured ? 'bg-mendelu-success' : 'bg-mendelu-gray-semi'"></span>
            <span class="text-[10px] text-mendelu-gray-dark">{{ credentials.openrouter_api_key?.configured ? 'Configured' : 'Not configured' }}</span>
          </div>
        </div>
        <div>
          <label for="settings-llm-model" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">LLM Model</label>
          <div class="flex gap-2">
            <input id="settings-llm-model" type="text" v-model="editableSettings.model" placeholder="anthropic/claude-sonnet-4.6" class="input-field flex-1" />
            <button @click="testConnection" :disabled="testingConnection" class="btn-secondary !py-1.5 !text-xs whitespace-nowrap disabled:opacity-50">
              {{ testingConnection ? 'Testing...' : connectionStatus === 'ok' ? '✓ OK' : connectionStatus === 'fail' ? '✗ Fail' : 'Test' }}
            </button>
          </div>
          <div class="flex flex-wrap gap-1 mt-2">
            <button v-for="m in quickModels" :key="m.id" @click="editableSettings.model = m.id"
              class="text-[10px] px-2 py-0.5 rounded-full transition-colors duration-150"
              :class="editableSettings.model === m.id ? 'bg-mendelu-green text-white' : 'bg-mendelu-gray-light text-mendelu-gray-dark hover:bg-mendelu-gray-semi'">
              {{ m.label }}
            </button>
          </div>
        </div>
        <div>
          <label for="settings-temperature" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Temperature: {{ editableSettings.temperature }}</label>
          <input id="settings-temperature" type="range" v-model.number="editableSettings.temperature" min="0" max="1" step="0.1" class="w-full h-2 bg-mendelu-gray-semi rounded-lg appearance-none cursor-pointer accent-mendelu-green" />
          <div class="flex justify-between text-[10px] text-mendelu-gray-dark mt-1">
            <span>Precise (0)</span>
            <span>Creative (1)</span>
          </div>
        </div>
        <div>
          <label for="settings-top-k" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Top-K Results</label>
          <input id="settings-top-k" type="number" v-model.number="editableSettings.top_k" min="1" max="50" class="input-field" />
        </div>
        <div>
          <label for="settings-reranker" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Cross-encoder reranker</label>
          <div class="flex items-start gap-2 mt-2">
            <input id="settings-reranker" type="checkbox" v-model="editableSettings.use_reranker" class="w-4 h-4 accent-mendelu-green mt-0.5" />
            <div class="text-sm">
              <div class="text-mendelu-black">Re-rank retrieved chunks with <span class="font-mono text-xs">BAAI/bge-reranker-v2-m3</span></div>
              <p class="text-xs text-mendelu-gray-dark mt-1 leading-snug">
                <span class="font-medium text-mendelu-alert">Off by default — it is slower.</span>
                Turning it on adds roughly 2–4&nbsp;seconds per query (cross-encoder
                inference runs on top of the vector search). On our golden-query
                evaluation (see <span class="font-mono">docs/rag_eval_v2_*.md</span>) the no-reranker
                pipeline already matches or beats the reranked one on Faithfulness
                (95–100%) and Answer Correctness (78–85%), so the trade-off rarely pays off.
                Enable if you notice the top chunks look noisy on a specific query.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Portal Adapters & Credentials -->
    <div class="card">
      <div class="flex items-center justify-between mb-4">
        <div>
          <h3 class="text-sm font-medium text-mendelu-black">Portal Adapters & Credentials</h3>
          <p class="text-[10px] text-mendelu-gray-dark mt-0.5">Phase 3 data portal configurations</p>
        </div>
        <div class="flex items-center gap-2">
          <span class="text-[10px] px-1.5 py-0.5 rounded-full bg-mendelu-gray-light text-mendelu-gray-dark" style="font-family: var(--font-mono);">
            {{ configuredAdapterCount }}/{{ adapterGroups.length }} configured
          </span>
          <button
            @click="showAdapterForm = !showAdapterForm"
            class="btn-ghost !py-1.5 !text-xs"
            :class="showAdapterForm ? 'text-mendelu-gray-dark' : 'text-mendelu-green'"
          >
            {{ showAdapterForm ? 'Cancel' : '+ Add adapter' }}
          </button>
          <button v-if="hasCredentialChanges" @click="saveCredentials" :disabled="savingCredentials" class="btn-primary !py-1.5 !text-xs disabled:opacity-50">
            {{ savingCredentials ? 'Saving...' : 'Save Credentials' }}
          </button>
        </div>
      </div>

      <!-- Add-adapter form. Collapsed by default. When expanded, lets the
           operator register a new portal config without a code deploy — useful
           for a portal whose Python adapter isn't yet shipped but whose data
           you still want to collect credentials for. -->
      <div v-if="showAdapterForm" class="mb-4 border border-mendelu-green/30 bg-mendelu-green/[0.03] rounded-xl p-4 space-y-3">
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label class="block text-[10px] font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Name *</label>
            <input v-model="adapterDraft.name" type="text" placeholder="e.g. CEDA CMIP6" class="input-field !py-1.5 text-xs" />
          </div>
          <div>
            <label class="block text-[10px] font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Datasets (hint)</label>
            <input v-model="adapterDraft.datasets" type="text" placeholder="e.g. CMIP6, CORDEX" class="input-field !py-1.5 text-xs" />
          </div>
        </div>
        <div>
          <label class="block text-[10px] font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Description</label>
          <input v-model="adapterDraft.description" type="text" placeholder="One-liner shown next to the adapter name" class="input-field !py-1.5 text-xs" />
        </div>

        <div>
          <div class="flex items-center justify-between mb-2">
            <span class="text-[10px] font-medium text-mendelu-gray-dark uppercase tracking-wider">Credential fields</span>
            <button @click="addDraftField" type="button" class="btn-ghost !px-2 !py-0.5 text-xs text-mendelu-green">+ Field</button>
          </div>
          <div v-for="(f, idx) in adapterDraft.fields" :key="idx" class="grid grid-cols-12 gap-2 mb-2 items-start">
            <input v-model="f.key" type="text" placeholder="key (e.g. ceda_token)" class="input-field !py-1.5 text-xs col-span-3" />
            <input v-model="f.label" type="text" placeholder="Display label" class="input-field !py-1.5 text-xs col-span-4" />
            <input v-model="f.hint" type="text" placeholder="Hint (optional)" class="input-field !py-1.5 text-xs col-span-4" />
            <button
              v-if="adapterDraft.fields.length > 1"
              @click="removeDraftField(idx)"
              type="button"
              class="btn-ghost !px-2 !py-1 text-xs text-mendelu-alert col-span-1"
              title="Remove field"
            >×</button>
          </div>
          <p class="text-[10px] text-mendelu-gray-dark/70">
            Leave this list empty for a public (no-auth) adapter. Keys are normalised to lowercase/underscore.
          </p>
        </div>

        <div class="flex justify-end gap-2 pt-2">
          <button @click="resetAdapterDraft" type="button" class="btn-ghost text-xs">Cancel</button>
          <button
            @click="submitAdapter"
            :disabled="adapterSubmitting || !adapterDraft.name.trim()"
            class="btn-primary !py-1.5 !text-xs disabled:opacity-50"
          >{{ adapterSubmitting ? 'Saving…' : 'Add adapter' }}</button>
        </div>
      </div>

      <div class="space-y-2">
        <div v-for="adapter in adapterGroups" :key="adapter.id" class="border rounded-xl overflow-hidden transition-colors duration-150" :class="adapter.expanded ? 'border-mendelu-green/30 bg-mendelu-green/[0.02]' : 'border-mendelu-gray-semi/60'">
          <!-- Adapter header (clickable) -->
          <button
            @click="adapter.expanded = !adapter.expanded"
            class="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-mendelu-gray-light/50 transition-colors duration-150"
          >
            <!-- Status dot -->
            <span class="w-2.5 h-2.5 rounded-full flex-shrink-0" :class="adapterConfigured(adapter) ? 'bg-mendelu-success' : adapter.public ? 'bg-blue-400' : 'bg-mendelu-gray-semi'"></span>

            <!-- Name + description -->
            <div class="flex-1 min-w-0">
              <div class="flex items-center gap-2">
                <span class="text-sm font-medium text-mendelu-black">{{ adapter.name }}</span>
                <span v-if="adapter.public" class="text-[9px] px-1.5 py-0.5 rounded bg-blue-50 text-blue-500 font-medium uppercase tracking-wider" style="font-family: var(--font-mono);">Public</span>
                <span v-else-if="adapterConfigured(adapter)" class="text-[9px] px-1.5 py-0.5 rounded bg-mendelu-success/10 text-mendelu-success font-medium uppercase tracking-wider" style="font-family: var(--font-mono);">Ready</span>
                <span v-else class="text-[9px] px-1.5 py-0.5 rounded bg-amber-50 text-amber-500 font-medium uppercase tracking-wider" style="font-family: var(--font-mono);">Needs Setup</span>
              </div>
              <p class="text-[10px] text-mendelu-gray-dark mt-0.5">{{ adapter.description }}</p>
            </div>

            <!-- Datasets badge -->
            <span v-if="adapter.datasets" class="text-[10px] text-mendelu-gray-dark hidden sm:block">{{ adapter.datasets }}</span>

            <!-- Custom badge -->
            <span v-if="!adapter.builtin" class="text-[9px] px-1.5 py-0.5 rounded bg-mendelu-green/10 text-mendelu-green font-medium uppercase tracking-wider" style="font-family: var(--font-mono);">Custom</span>

            <!-- Chevron -->
            <svg class="w-4 h-4 text-mendelu-gray-dark transition-transform duration-150 flex-shrink-0" :class="{ 'rotate-180': adapter.expanded }" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          <!-- Expanded content -->
          <div v-if="adapter.expanded" class="px-4 pb-4 pt-1 border-t border-mendelu-gray-semi/40">
            <!-- Public adapters - no credentials needed -->
            <div v-if="adapter.public" class="flex items-center gap-2 py-2">
              <svg class="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span class="text-xs text-mendelu-gray-dark">No credentials required — publicly accessible data.</span>
            </div>

            <!-- Credential fields -->
            <div v-else class="space-y-3 pt-2">
              <div v-for="field in adapter.fields" :key="field.key">
                <div class="flex items-center gap-2 mb-1">
                  <span class="w-1.5 h-1.5 rounded-full" :class="credentials[field.key]?.configured ? 'bg-mendelu-success' : 'bg-mendelu-gray-semi'"></span>
                  <label :for="`settings-cred-${field.key}`" class="text-xs font-medium text-mendelu-gray-dark">{{ field.label }}</label>
                  <span v-if="credentials[field.key]?.configured" class="text-[9px] text-mendelu-success font-medium ml-auto" style="font-family: var(--font-mono);">configured</span>
                </div>
                <div class="relative">
                  <input
                    :id="`settings-cred-${field.key}`"
                    :type="revealedKeys[field.key] ? 'text' : 'password'"
                    v-model="credentialEdits[field.key]"
                    :placeholder="credentials[field.key]?.configured ? credentials[field.key]?.masked : 'Not configured'"
                    class="input-field pr-14"
                  />
                  <button
                    v-if="credentials[field.key]?.configured || credentialEdits[field.key]"
                    type="button"
                    @click="toggleReveal(field.key)"
                    class="absolute right-2 top-1/2 -translate-y-1/2 btn-ghost !px-2 !py-0.5 text-xs"
                  >
                    {{ revealedKeys[field.key] ? 'Hide' : 'Show' }}
                  </button>
                </div>
                <p v-if="field.hint" class="text-[10px] text-mendelu-gray-dark/60 mt-1">{{ field.hint }}</p>
              </div>
            </div>

            <!-- Custom-adapter controls: only shown on non-builtin entries.
                 The built-in adapters are immutable from the UI. -->
            <div v-if="!adapter.builtin" class="flex justify-end pt-3 mt-2 border-t border-mendelu-gray-semi/30">
              <button
                type="button"
                @click="removeAdapter(adapter)"
                class="btn-ghost !py-1 !px-2 text-xs text-mendelu-alert hover:underline"
              >Remove adapter</button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- LLM Provider -->
    <div class="card">
      <h3 class="text-sm font-medium text-mendelu-black mb-3">LLM Provider</h3>
      <div class="bg-mendelu-gray-light rounded-lg p-4">
        <div class="flex items-center justify-between mb-2">
          <span class="text-mendelu-black font-medium text-sm">OpenRouter</span>
          <span class="w-2.5 h-2.5 rounded-full" :class="settings?.llm?.providers?.openrouter ? 'bg-mendelu-success' : 'bg-mendelu-gray-semi'"></span>
        </div>
        <div class="grid grid-cols-2 gap-4 mt-3">
          <div>
            <span class="text-xs text-mendelu-gray-dark uppercase tracking-wider block mb-0.5">Main Model</span>
            <span class="text-mendelu-black font-mono text-sm">{{ settings?.llm?.model || '—' }}</span>
          </div>
          <div>
            <span class="text-xs text-mendelu-gray-dark uppercase tracking-wider block mb-0.5">Fast Model (Query Expansion)</span>
            <span class="text-mendelu-black font-mono text-sm">{{ settings?.llm?.fast_model || '—' }}</span>
          </div>
        </div>
        <p class="text-xs text-mendelu-gray-dark mt-2">{{ settings?.llm?.providers?.openrouter ? 'API key configured' : 'No API key set — add OPENROUTER_API_KEY to .env' }}</p>
      </div>
    </div>

    <!-- Embedding Model -->
    <div class="card">
      <div class="flex items-center justify-between mb-3">
        <h3 class="text-sm font-medium text-mendelu-black">Embedding Model</h3>
        <button v-if="hasEmbedChanges" @click="saveEmbedSettings" :disabled="savingEmbed" class="btn-primary !py-1.5 !text-xs disabled:opacity-50">
          {{ savingEmbed ? 'Saving…' : 'Save batch sizes' }}
        </button>
      </div>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div>
          <span class="text-xs text-mendelu-gray-dark uppercase tracking-wider block mb-0.5">Model</span>
          <span class="text-mendelu-black font-mono text-sm">{{ settings?.embedding_model?.name || 'BAAI/bge-large-en-v1.5' }}</span>
        </div>
        <div>
          <span class="text-xs text-mendelu-gray-dark uppercase tracking-wider block mb-0.5">Dimensions</span>
          <span class="text-mendelu-black font-mono text-sm">{{ settings?.embedding_model?.dimensions || 1024 }}</span>
        </div>
        <div>
          <span class="text-xs text-mendelu-gray-dark uppercase tracking-wider block mb-0.5">Distance Metric</span>
          <span class="text-mendelu-black font-mono text-sm">{{ settings?.embedding_model?.distance || 'COSINE' }}</span>
        </div>
        <div>
          <span class="text-xs text-mendelu-gray-dark uppercase tracking-wider block mb-0.5">Device</span>
          <span class="text-mendelu-black font-mono text-sm">{{ settings?.embedding_model?.device || 'auto' }}</span>
        </div>
      </div>

      <!-- Batch size — critical knob for GPU memory.
           BAAI-large at FP32 + 512 tokens needs ~16 MB / sample of VRAM.
           A batch that doesn't fit in VRAM falls back to unified memory
           (PCIe spill) which is 5-10× slower — that's the symptom of
           "GPU at 100% but no progress for minutes". Recommended sizes
           below are derived from VRAM tiers. -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4 pt-4 border-t border-mendelu-gray-semi/40">
        <div>
          <label for="settings-embed-doc-batch" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">
            Document batch size
            <span class="text-[10px] text-mendelu-gray-dark/70 normal-case ml-1">(catalog batch + ETL)</span>
          </label>
          <input
            id="settings-embed-doc-batch"
            type="number"
            v-model.number="editableSettings.embedding_batch_size"
            min="1" max="2048" step="1"
            class="input-field"
          />
          <div class="flex flex-wrap gap-1 mt-2">
            <button v-for="p in batchPresets" :key="p.size"
              @click="editableSettings.embedding_batch_size = p.size"
              class="text-[10px] px-2 py-0.5 rounded-full transition-colors duration-150"
              :class="editableSettings.embedding_batch_size === p.size ? 'bg-mendelu-green text-white' : 'bg-mendelu-gray-light text-mendelu-gray-dark hover:bg-mendelu-gray-semi'">
              {{ p.size }} <span class="opacity-70">({{ p.label }})</span>
            </button>
          </div>
        </div>
        <div>
          <label for="settings-embed-query-batch" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">
            Query batch size
            <span class="text-[10px] text-mendelu-gray-dark/70 normal-case ml-1">(RAG chat)</span>
          </label>
          <input
            id="settings-embed-query-batch"
            type="number"
            v-model.number="editableSettings.embedding_query_batch_size"
            min="1" max="512" step="1"
            class="input-field"
          />
          <p class="text-[10px] text-mendelu-gray-dark/70 mt-1 leading-snug">
            Queries are usually 1–10 texts → small batch is fine. Bump only
            if you have a heavily-multi-user deployment.
          </p>
        </div>
      </div>
      <p class="text-[11px] text-mendelu-alert mt-3 leading-snug">
        <span class="font-medium">⚠ If embedding takes minutes per batch with GPU at 100 % but no Qdrant chunks landing,</span>
        the batch is too big and PyTorch is spilling tensors over PCIe to system RAM.
        Drop the doc batch size to the next preset down.
      </p>
    </div>

    <!-- Qdrant -->
    <div class="card">
      <h3 class="text-sm font-medium text-mendelu-black mb-3">Vector Database (Qdrant)</h3>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div>
          <span class="text-xs text-mendelu-gray-dark uppercase tracking-wider block mb-0.5">Host</span>
          <span class="text-mendelu-black font-mono text-sm">{{ settings?.qdrant?.host || 'localhost' }}</span>
        </div>
        <div>
          <span class="text-xs text-mendelu-gray-dark uppercase tracking-wider block mb-0.5">Port</span>
          <span class="text-mendelu-black font-mono text-sm">{{ settings?.qdrant?.port || 6333 }}</span>
        </div>
        <div>
          <span class="text-xs text-mendelu-gray-dark uppercase tracking-wider block mb-0.5">Embeddings</span>
          <span class="text-mendelu-black font-mono text-sm tabular-nums">{{ embeddingStats?.total_embeddings?.toLocaleString() || '---' }}</span>
        </div>
        <div>
          <span class="text-xs text-mendelu-gray-dark uppercase tracking-wider block mb-0.5">Collection</span>
          <span class="text-mendelu-black font-mono text-sm">{{ embeddingStats?.collection_name || 'climate_data' }}</span>
        </div>
      </div>
    </div>

    <!-- Vector Database Backups -->
    <div class="card">
      <div class="flex items-center justify-between mb-4">
        <div>
          <h3 class="text-sm font-medium text-mendelu-black">Vector Database Backups</h3>
          <p class="text-[10px] text-mendelu-gray-dark mt-0.5">Export or restore Qdrant collection snapshots</p>
        </div>
        <button @click="refreshSnapshots" :disabled="snapshotsLoading" class="btn-ghost !py-1 !text-xs disabled:opacity-50">
          {{ snapshotsLoading ? 'Loading…' : 'Refresh' }}
        </button>
      </div>

      <div class="flex flex-wrap gap-2 mb-4">
        <button
          @click="exportSnapshot"
          :disabled="exportingSnapshot || importingSnapshot"
          class="btn-primary !py-1.5 !text-xs disabled:opacity-50"
        >
          {{ exportingSnapshot ? exportProgress.stage : 'Export Snapshot' }}
        </button>
        <button
          @click="triggerImportPicker"
          :disabled="exportingSnapshot || importingSnapshot"
          class="btn-secondary !py-1.5 !text-xs disabled:opacity-50"
          title="Restore from a .snapshot file. WILL OVERWRITE current points."
        >
          {{ importingSnapshot ? importProgress.stage : 'Import Snapshot' }}
        </button>
        <input
          ref="importFileInput"
          type="file"
          accept=".snapshot,application/octet-stream"
          class="hidden"
          @change="onSnapshotFilePicked"
        />
      </div>

      <!-- Export progress -->
      <div v-if="exportingSnapshot" class="mb-4 px-3 py-2.5 rounded-lg border border-mendelu-green/30 bg-mendelu-green/[0.04]">
        <div class="flex items-center justify-between text-[11px] mb-1.5">
          <span class="text-mendelu-black font-medium">{{ exportProgress.stage }}</span>
          <span class="text-mendelu-gray-dark tabular-nums">{{ exportProgressLabel }}</span>
        </div>
        <div class="w-full bg-mendelu-gray-semi/60 rounded-full h-1.5 overflow-hidden">
          <!-- Determinate (download with known length): width = % done.
               Indeterminate (creating, or download w/o Content-Length): an
               animated bar slides across to show "still working". -->
          <div
            v-if="exportProgress.indeterminate"
            class="h-1.5 rounded-full bg-mendelu-green animate-snapshot-pulse"
            style="width: 35%;"
          ></div>
          <div
            v-else
            class="h-1.5 rounded-full bg-mendelu-green transition-[width] duration-150 ease-linear"
            :style="{ width: `${exportProgress.percent}%` }"
          ></div>
        </div>
        <!-- Concurrency warning. Heavy disk I/O during snapshot creation
             slows ETL and other Qdrant operations to a crawl. Better to
             keep the system idle for the few minutes it takes. -->
        <p class="text-[11px] text-mendelu-gray-dark mt-2 leading-snug">
          <span class="font-medium text-mendelu-alert">⚠ Don't trigger ETL jobs, catalog processing, or other heavy operations until this finishes.</span>
          Snapshot creation reads the full collection from disk; running concurrent
          writes will slow it down and may produce a less consistent backup.
          You can safely leave the page open or close the tab — the create
          continues on the server, and reopening Settings will resume showing this progress.
        </p>
      </div>

      <!-- Import progress (upload bytes) -->
      <div v-if="importingSnapshot" class="mb-4 px-3 py-2.5 rounded-lg border border-mendelu-alert/30 bg-mendelu-alert/[0.04]">
        <div class="flex items-center justify-between text-[11px] mb-1.5">
          <span class="text-mendelu-black font-medium">{{ importProgress.stage }}</span>
          <span class="text-mendelu-gray-dark tabular-nums">{{ importProgressLabel }}</span>
        </div>
        <div class="w-full bg-mendelu-gray-semi/60 rounded-full h-1.5 overflow-hidden">
          <div
            v-if="importProgress.indeterminate"
            class="h-1.5 rounded-full bg-mendelu-alert animate-snapshot-pulse"
            style="width: 35%;"
          ></div>
          <div
            v-else
            class="h-1.5 rounded-full bg-mendelu-alert transition-[width] duration-150 ease-linear"
            :style="{ width: `${importProgress.percent}%` }"
          ></div>
        </div>
      </div>

      <p class="text-[11px] text-mendelu-alert mb-3 leading-snug">
        <span class="font-medium">Warning:</span> Importing a snapshot will replace all current points
        in the active collection. Export first if you want a recovery point.
      </p>

      <div v-if="snapshots.length > 0" class="border-t border-mendelu-gray-semi/40 pt-3">
        <div class="text-[10px] font-medium text-mendelu-gray-dark uppercase tracking-wider mb-2">
          Existing snapshots ({{ snapshots.length }})
        </div>
        <div class="space-y-1.5">
          <div
            v-for="snap in sortedSnapshots"
            :key="snap.name"
            class="flex items-center gap-3 px-3 py-2 rounded-lg bg-mendelu-gray-light/50 text-xs"
          >
            <div class="flex-1 min-w-0">
              <div class="font-mono text-mendelu-black truncate" :title="snap.name">{{ snap.name }}</div>
              <div class="text-[10px] text-mendelu-gray-dark mt-0.5 tabular-nums">
                {{ snapshotTimestamp(snap) }}
              </div>
            </div>
            <span class="text-mendelu-gray-dark tabular-nums whitespace-nowrap">{{ formatBytes(snap.size) }}</span>
            <button
              @click="downloadExistingSnapshot(snap.name)"
              class="btn-ghost !px-2 !py-0.5 text-[11px] text-mendelu-green"
            >Download</button>
            <button
              @click="deleteSnapshot(snap.name)"
              class="btn-ghost !px-2 !py-0.5 text-[11px] text-mendelu-alert"
            >Delete</button>
          </div>
        </div>
      </div>
      <div v-else-if="!snapshotsLoading" class="text-[11px] text-mendelu-gray-dark/70 italic">
        No snapshots stored on the server yet.
      </div>
    </div>

    <!-- ETL temp file cleanup — leftover /tmp/tmp*.nc partials from
         crashed/killed batches accumulate to several GB. Show what's
         there + offer a one-click cleanup. -->
    <div class="card">
      <div class="flex items-center justify-between mb-3">
        <div>
          <h3 class="text-sm font-medium text-mendelu-black">ETL temp files</h3>
          <p class="text-[10px] text-mendelu-gray-dark mt-0.5">
            Partial downloads left behind by killed batches. Safe to delete when nothing is running.
          </p>
        </div>
        <div class="flex gap-2">
          <button @click="refreshTmp" :disabled="tmpLoading" class="btn-ghost !py-1 !text-xs disabled:opacity-50">
            {{ tmpLoading ? 'Loading…' : 'Refresh' }}
          </button>
          <button
            @click="cleanTmp"
            :disabled="tmpLoading || cleaningTmp || !tmp.files?.length"
            class="btn-primary !py-1 !text-xs disabled:opacity-50"
            title="Delete all /tmp/tmp*. Refused while a catalog batch is running."
          >
            {{ cleaningTmp ? 'Cleaning…' : `Clean ${tmp.count || 0} files` }}
          </button>
        </div>
      </div>

      <div v-if="tmp.files?.length" class="space-y-1">
        <div class="flex items-center justify-between text-xs text-mendelu-gray-dark mb-1">
          <span>{{ tmp.count }} files, total <span class="font-medium tabular-nums text-mendelu-black">{{ formatBytes(tmp.total_bytes) }}</span></span>
        </div>
        <div class="max-h-40 overflow-y-auto border border-mendelu-gray-semi/40 rounded-lg">
          <div v-for="f in tmp.files" :key="f.path" class="flex items-center gap-3 px-3 py-1.5 text-xs odd:bg-mendelu-gray-light/40">
            <span class="font-mono text-mendelu-gray-dark truncate flex-1" :title="f.path">{{ f.path }}</span>
            <span class="text-mendelu-gray-dark tabular-nums whitespace-nowrap">{{ formatBytes(f.size_bytes) }}</span>
          </div>
        </div>
      </div>
      <div v-else-if="!tmpLoading" class="text-[11px] text-mendelu-gray-dark/70 italic">
        No leftover temp files.
      </div>
    </div>

    <!-- System Resources -->
    <div class="card">
      <h3 class="text-sm font-medium text-mendelu-black mb-3">System Resources</h3>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <span class="text-xs text-mendelu-gray-dark uppercase tracking-wider block mb-2">Disk Usage</span>
          <div v-if="settings?.disk" class="space-y-1">
            <div class="w-full bg-mendelu-gray-semi rounded-full h-2.5">
              <div
                class="bg-mendelu-green h-2.5 rounded-full transition-all duration-300"
                :style="{ width: `${diskPercent}%` }"
              ></div>
            </div>
            <span class="text-xs text-mendelu-gray-dark tabular-nums">
              {{ settings.disk.used_gb }} GB / {{ settings.disk.total_gb }} GB
              ({{ settings.disk.free_gb }} GB free)
            </span>
          </div>
          <span v-else class="text-mendelu-gray-dark text-sm">---</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, reactive, onMounted, onBeforeUnmount } from 'vue'
import PageHeader from '../components/PageHeader.vue'
import { apiFetch } from '../api'
import { useToast } from '../composables/useToast'

const toast = useToast()

const settings = ref(null)
const embeddingStats = ref(null)
const loading = ref(false)
const saving = ref(false)
const savingCredentials = ref(false)
const saveSuccess = ref(false)

const testingConnection = ref(false)
const connectionStatus = ref('')  // '', 'ok', 'fail'

const quickModels = [
  { id: 'anthropic/claude-sonnet-4.6', label: 'Sonnet 4.6' },
  { id: 'anthropic/claude-opus-4-6', label: 'Opus 4.6' },
  { id: 'anthropic/claude-haiku-4-5', label: 'Haiku 4.5' },
  { id: 'x-ai/grok-4.1-fast', label: 'Grok 4.1' },
  { id: 'openai/gpt-4o-mini', label: 'GPT-4o Mini' },
]

interface EditableSettings {
  model: string
  temperature: number
  top_k: number
  use_reranker: boolean
  batch_size?: number
  embedding_batch_size: number
  embedding_query_batch_size: number
}

const editableSettings = reactive<EditableSettings>({
  model: '',
  temperature: 0.1,
  top_k: 10,
  use_reranker: false,
  embedding_batch_size: 64,
  embedding_query_batch_size: 32,
})

// Recommended embedding batch sizes by VRAM tier. BAAI-large at FP32 +
// 512 tokens needs ~16 MB/sample of working VRAM, so 64 ≈ 1 GB ≈ safe
// for 4 GB cards once the 1.3 GB model is loaded. Bigger cards can push
// proportionally higher.
const batchPresets = [
  { size: 32, label: '2 GB GPU' },
  { size: 64, label: '4 GB (RTX 3050)' },
  { size: 128, label: '8 GB (RTX 3070)' },
  { size: 256, label: '12 GB (RTX 3080 Ti)' },
  { size: 512, label: '16+ GB (4090, A100)' },
  { size: 1024, label: '24+ GB (workstation)' },
]
const savingEmbed = ref(false)

const hasEmbedChanges = computed(() => {
  return editableSettings.embedding_batch_size !== originalSettings.value.embedding_batch_size
    || editableSettings.embedding_query_batch_size !== originalSettings.value.embedding_query_batch_size
})

async function saveEmbedSettings() {
  savingEmbed.value = true
  try {
    const resp = await apiFetch('/settings/system', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        embedding_batch_size: editableSettings.embedding_batch_size,
        embedding_query_batch_size: editableSettings.embedding_query_batch_size,
      })
    })
    if (resp.ok) {
      originalSettings.value.embedding_batch_size = editableSettings.embedding_batch_size
      originalSettings.value.embedding_query_batch_size = editableSettings.embedding_query_batch_size
      toast.success('Batch sizes saved — next embedding call will use them')
    } else {
      const err = await resp.json().catch(() => ({}))
      throw new Error(err.detail || `HTTP ${resp.status}`)
    }
  } catch (e: any) {
    toast.error(`Save failed: ${e?.message || 'network error'}`)
  } finally {
    savingEmbed.value = false
  }
}

async function testConnection() {
  testingConnection.value = true
  connectionStatus.value = ''
  // Use a 30s abort controller so the test button can't hang indefinitely
  // if the LLM provider is down. 30s covers our observed p99 (~22s end-to-end).
  const controller = new AbortController()
  const abortTimer = setTimeout(() => controller.abort(), 30_000)
  try {
    const resp = await apiFetch('/rag/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: 'test connection', top_k: 1, use_llm: true, use_reranker: false }),
      signal: controller.signal,
    })
    connectionStatus.value = resp.ok ? 'ok' : 'fail'
  } catch {
    connectionStatus.value = 'fail'
  } finally {
    clearTimeout(abortTimer)
    testingConnection.value = false
    setTimeout(() => { connectionStatus.value = '' }, 5000)
  }
}

const originalSettings = ref<any>({})
const hasChanges = computed(() => {
  return editableSettings.model !== originalSettings.value.model ||
    editableSettings.temperature !== originalSettings.value.temperature ||
    editableSettings.top_k !== originalSettings.value.top_k ||
    editableSettings.use_reranker !== originalSettings.value.use_reranker
})

const credentials = ref<Record<string, { configured?: boolean; masked?: string }>>({})
const revealedKeys = reactive<Record<string, boolean>>({})

async function toggleReveal(key) {
  if (revealedKeys[key]) {
    // Hide — clear the fetched value, restore to empty edit
    revealedKeys[key] = false
    credentialEdits[key] = ''
    return
  }
  // If user already typed something, just toggle visibility
  if (credentialEdits[key]) {
    revealedKeys[key] = true
    return
  }
  // Fetch full value from backend
  try {
    const resp = await apiFetch(`/settings/credentials/${key}`)
    if (resp.ok) {
      const data = await resp.json()
      credentialEdits[key] = data.value
      revealedKeys[key] = true
    }
  } catch (e: any) {
    console.error('Failed to reveal credential:', e)
    toast.error(`Could not reveal credential: ${e?.message || 'network error'}`)
  }
}

const credentialEdits = reactive<Record<string, string>>({
  openrouter_api_key: '', cds_api_key: '',
  nasa_earthdata_user: '', nasa_earthdata_password: '',
  cmems_username: '', cmems_password: '',
  mistral_username: '', mistral_password: '',
  esgf_username: '', esgf_password: '',
})

// Adapters grouped by portal with their credential fields
const builtinAdapters = reactive([
  {
    id: 'cds', builtin: true,
    name: 'Copernicus CDS',
    description: 'Climate Data Store — ERA5, CERRA, seasonal forecasts',
    datasets: 'ERA5, CERRA, SEAS5',
    public: false,
    expanded: false,
    fields: [
      { key: 'cds_api_key', label: 'Personal Access Token', hint: 'CDS v2024+ format — get at cds.climate.copernicus.eu/profile' },
    ],
  },
  {
    id: 'nasa', builtin: true,
    name: 'NASA Earthdata',
    description: 'GES DISC, PO.DAAC — MERRA-2, GPM, sea level data',
    datasets: 'MERRA-2, GPM, GRACE',
    public: false,
    expanded: false,
    fields: [
      { key: 'nasa_earthdata_user', label: 'Username', hint: null },
      { key: 'nasa_earthdata_password', label: 'Password', hint: 'Register at urs.earthdata.nasa.gov' },
    ],
  },
  {
    id: 'cmems', builtin: true,
    name: 'Marine Copernicus (CMEMS)',
    description: 'Copernicus Marine — ocean temperature, sea level, ice',
    datasets: 'SST, sea level, sea ice',
    public: false,
    expanded: false,
    fields: [
      { key: 'cmems_username', label: 'Username', hint: null },
      { key: 'cmems_password', label: 'Password', hint: 'Register at data.marine.copernicus.eu' },
    ],
  },
  {
    id: 'esgf', builtin: true,
    name: 'ESGF',
    description: 'Earth System Grid Federation — CMIP6, CORDEX projections',
    datasets: 'CMIP6, CORDEX',
    public: false,
    expanded: false,
    fields: [
      { key: 'esgf_username', label: 'Username', hint: null },
      { key: 'esgf_password', label: 'Password', hint: 'Register at any ESGF node, e.g. esgf-data.dkrz.de' },
    ],
  },
  // Mistral / CINECA tile removed — meteohub.agenziaitaliameteo.it serves
  // generated download URLs without auth headers, so no portal credentials
  // are needed; users just paste the URL into a Source upload.
  {
    id: 'noaa', builtin: true,
    name: 'NOAA',
    description: 'PSL, NCEI — reanalysis, observational datasets',
    datasets: '20CRv3, NCEP/NCAR',
    public: true,
    expanded: false,
    fields: [],
  },
  {
    id: 'eidc', builtin: true,
    name: 'EIDC / CEDA',
    description: 'Hydro-JULES, UK environmental data via OpenDAP',
    datasets: 'Hydro-JULES, CHESS',
    public: true,
    expanded: false,
    fields: [],
  },
])

// Custom adapters registered through POST /settings/adapters. Fetched on
// mount and re-fetched whenever the user adds/removes one.
const customAdapters = ref<any[]>([])

// Unified list rendered by the template. Built-in first, then custom.
const adapterGroups = computed(() => [...builtinAdapters, ...customAdapters.value])

function adapterConfigured(adapter) {
  if (adapter.public) return true
  return adapter.fields.every(f => credentials.value[f.key]?.configured)
}

const configuredAdapterCount = computed(() => {
  return adapterGroups.value.filter(a => adapterConfigured(a)).length
})

// "Add adapter" form state.
const showAdapterForm = ref(false)
const adapterDraft = ref({
  name: '',
  description: '',
  datasets: '',
  fields: [{ key: '', label: '', hint: '' }],
})
const adapterSubmitting = ref(false)

function addDraftField() {
  adapterDraft.value.fields.push({ key: '', label: '', hint: '' })
}
function removeDraftField(idx: number) {
  if (adapterDraft.value.fields.length > 1) adapterDraft.value.fields.splice(idx, 1)
}
function resetAdapterDraft() {
  adapterDraft.value = { name: '', description: '', datasets: '', fields: [{ key: '', label: '', hint: '' }] }
  showAdapterForm.value = false
}

async function loadCustomAdapters() {
  try {
    const resp = await apiFetch('/settings/adapters')
    if (!resp.ok) return
    const data = await resp.json()
    // Keep the same shape as builtinAdapters so the template needs no branching.
    customAdapters.value = (data.adapters || []).map((a: any) => ({
      ...a,
      builtin: false,
      expanded: false,
    }))
  } catch (e: any) {
    console.error('Failed to load custom adapters:', e)
    toast.error(`Could not load custom adapters: ${e?.message || 'network error'}`)
  }
}

async function submitAdapter() {
  const draft = adapterDraft.value
  if (!draft.name.trim()) {
    toast.error('Adapter name is required')
    return
  }
  const validFields = draft.fields.filter(f => f.key.trim() && f.label.trim())
  // An adapter with zero fields would be a public one — if the user actually
  // wanted public, they can still submit an empty field list.
  const payload = {
    name: draft.name.trim(),
    description: draft.description.trim(),
    datasets: draft.datasets.trim(),
    fields: validFields.map(f => ({ key: f.key.trim(), label: f.label.trim(), hint: f.hint?.trim() || null })),
  }
  adapterSubmitting.value = true
  try {
    const resp = await apiFetch('/settings/adapters', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    const body = await resp.json().catch(() => ({}))
    if (!resp.ok) {
      const msg = Array.isArray(body.detail)
        ? body.detail.map((d: any) => d.msg || JSON.stringify(d)).join('; ')
        : (body.detail || `HTTP ${resp.status}`)
      throw new Error(msg)
    }
    toast.success(`Adapter "${body.name}" added`)
    resetAdapterDraft()
    await Promise.all([loadCustomAdapters(), refreshSettings()])
  } catch (e: any) {
    toast.error(`Could not add adapter: ${e?.message || e}`)
  } finally {
    adapterSubmitting.value = false
  }
}

async function removeAdapter(adapter: any) {
  if (!adapter?.id || adapter.builtin) return
  if (!window.confirm(`Remove adapter "${adapter.name}"? Stored credentials for its fields stay behind and can be cleared separately.`)) return
  try {
    const resp = await apiFetch(`/settings/adapters/${adapter.id}`, { method: 'DELETE' })
    if (!resp.ok && resp.status !== 204) {
      const err = await resp.json().catch(() => ({}))
      throw new Error(err.detail || `HTTP ${resp.status}`)
    }
    toast.success(`Removed "${adapter.name}"`)
    await Promise.all([loadCustomAdapters(), refreshSettings()])
  } catch (e: any) {
    toast.error(`Delete failed: ${e?.message || e}`)
  }
}

const hasCredentialChanges = computed(() => {
  return Object.values(credentialEdits).some(v => v !== '')
})

const diskPercent = computed(() => {
  if (!settings.value?.disk) return 0
  return Math.round((settings.value.disk.used_gb / settings.value.disk.total_gb) * 100)
})

async function refreshSettings() {
  loading.value = true
  try {
    const [sysResp, embResp, credResp] = await Promise.all([
      apiFetch('/settings/system'), apiFetch('/embeddings/stats'), apiFetch('/settings/credentials'),
    ])
    if (sysResp.ok) {
      settings.value = await sysResp.json()
      if (settings.value.llm) {
        editableSettings.model = settings.value.llm.model || 'anthropic/claude-sonnet-4.6'
        editableSettings.temperature = settings.value.llm.temperature ?? 0.1
        editableSettings.top_k = settings.value.llm.top_k ?? 10
        editableSettings.use_reranker = settings.value.llm.use_reranker ?? false
      }
      if (settings.value.embedding_model) {
        editableSettings.embedding_batch_size = settings.value.embedding_model.doc_batch_size ?? 64
        editableSettings.embedding_query_batch_size = settings.value.embedding_model.query_batch_size ?? 32
      }
      originalSettings.value = { ...editableSettings }
    }
    if (embResp.ok) embeddingStats.value = await embResp.json()
    if (credResp.ok) {
      credentials.value = await credResp.json()
      Object.keys(credentialEdits).forEach(k => credentialEdits[k] = '')
    }
  } catch (e: any) {
    console.error('Failed to load settings:', e)
    toast.error(`Could not load settings: ${e?.message || 'network error'}`)
  } finally {
    loading.value = false
  }
}

async function saveSettings() {
  saving.value = true
  try {
    const resp = await apiFetch('/settings/system', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: editableSettings.model, temperature: editableSettings.temperature,
        top_k: editableSettings.top_k, batch_size: editableSettings.batch_size,
        use_reranker: editableSettings.use_reranker,
      })
    })
    if (resp.ok) {
      originalSettings.value = { ...editableSettings }
      saveSuccess.value = true
      setTimeout(() => { saveSuccess.value = false }, 2000)
    } else {
      const err = await resp.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(err.detail || `HTTP ${resp.status}`)
    }
  } catch (e: any) {
    console.error('Failed to save settings:', e)
    toast.error(`Save failed: ${e?.message || 'network error'}`)
  } finally {
    saving.value = false
  }
}

async function saveCredentials() {
  savingCredentials.value = true
  try {
    const payload = {}
    Object.entries(credentialEdits).forEach(([k, v]) => { if (v !== '') payload[k] = v })
    if (Object.keys(payload).length === 0) return
    const resp = await apiFetch('/settings/credentials', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    if (resp.ok) {
      await refreshSettings()
    } else {
      const err = await resp.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(err.detail || `HTTP ${resp.status}`)
    }
  } catch (e: any) {
    console.error('Failed to save credentials:', e)
    toast.error(`Save failed: ${e?.message || 'network error'}`)
  } finally {
    savingCredentials.value = false
  }
}

// --- Qdrant snapshot export/import ---
//
// Snapshots can be multi-GB so the download is streamed via the backend
// (which proxies to Qdrant's REST API). We trigger the browser save
// dialog by turning the response into a Blob + object URL — fetch is
// required (not a plain <a href>) because the endpoint is JWT-protected.

interface QdrantSnapshot {
  name: string
  size: number | null
  creation_time: string | null
  checksum: string | null
}

const snapshots = ref<QdrantSnapshot[]>([])
const snapshotsLoading = ref(false)
const exportingSnapshot = ref(false)
const importingSnapshot = ref(false)
const importFileInput = ref<HTMLInputElement | null>(null)

interface SnapshotProgress {
  stage: string
  percent: number          // 0-100, only used when indeterminate=false
  bytes: number            // bytes done
  total: number            // 0 if unknown
  elapsedMs: number        // ms since start
  indeterminate: boolean   // true when we don't know the total
}

function makeProgress(stage = 'Working…'): SnapshotProgress {
  return { stage, percent: 0, bytes: 0, total: 0, elapsedMs: 0, indeterminate: true }
}

const exportProgress = reactive<SnapshotProgress>(makeProgress('Creating…'))
const importProgress = reactive<SnapshotProgress>(makeProgress('Uploading…'))

function formatDuration(ms: number): string {
  const s = Math.floor(ms / 1000)
  if (s < 60) return `${s}s`
  return `${Math.floor(s / 60)}m ${s % 60}s`
}

const exportProgressLabel = computed(() => {
  if (exportProgress.indeterminate) {
    const t = formatDuration(exportProgress.elapsedMs)
    return exportProgress.bytes > 0
      ? `${formatBytes(exportProgress.bytes)} • ${t}`
      : t
  }
  return `${formatBytes(exportProgress.bytes)} / ${formatBytes(exportProgress.total)} • ${formatDuration(exportProgress.elapsedMs)}`
})

const importProgressLabel = computed(() => {
  if (importProgress.indeterminate) return formatDuration(importProgress.elapsedMs)
  return `${formatBytes(importProgress.bytes)} / ${formatBytes(importProgress.total)} • ${formatDuration(importProgress.elapsedMs)}`
})

function resetProgress(p: SnapshotProgress, stage: string) {
  p.stage = stage
  p.percent = 0
  p.bytes = 0
  p.total = 0
  p.elapsedMs = 0
  p.indeterminate = true
}

function formatBytes(n: number | null | undefined): string {
  if (n == null) return '—'
  if (n < 1024) return `${n} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let v = n / 1024
  let i = 0
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++ }
  return `${v.toFixed(v >= 100 ? 0 : 1)} ${units[i]}`
}

// Qdrant names snapshots like "<collection>-<id>-<YYYY-MM-DD-HH-MM-SS>.snapshot".
// The creation_time field on the API response is often null in 1.17, so we
// fall back to parsing the embedded timestamp.
const SNAPSHOT_TS_RE = /(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})/

function parseSnapshotDate(snap: QdrantSnapshot): Date | null {
  if (snap.creation_time) {
    const d = new Date(snap.creation_time)
    if (!isNaN(d.getTime())) return d
  }
  const m = snap.name?.match(SNAPSHOT_TS_RE)
  if (m) {
    // Qdrant emits these in UTC.
    const [, y, mo, da, h, mi, s] = m
    const d = new Date(Date.UTC(+y, +mo - 1, +da, +h, +mi, +s))
    if (!isNaN(d.getTime())) return d
  }
  return null
}

function snapshotTimestamp(snap: QdrantSnapshot): string {
  const d = parseSnapshotDate(snap)
  if (!d) return 'unknown date'
  // Local-time short form, e.g. "2026-05-03 14:22:31"
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}

const sortedSnapshots = computed(() => {
  // Newest first. parseSnapshotDate returns null for un-dated entries, which
  // we sink to the bottom of the list.
  return [...snapshots.value].sort((a, b) => {
    const da = parseSnapshotDate(a)?.getTime() ?? -Infinity
    const db = parseSnapshotDate(b)?.getTime() ?? -Infinity
    return db - da
  })
})

// --- /tmp ETL leftovers ---
const tmp = ref<any>({ files: [], total_bytes: 0, count: 0 })
const tmpLoading = ref(false)
const cleaningTmp = ref(false)

async function refreshTmp() {
  tmpLoading.value = true
  try {
    const resp = await apiFetch('/admin/tmp/list')
    if (resp.ok) tmp.value = await resp.json()
  } catch (e: any) {
    toast.error(`Could not list /tmp: ${e?.message || 'network error'}`)
  } finally {
    tmpLoading.value = false
  }
}

async function cleanTmp() {
  if (!window.confirm(`Delete ${tmp.value.count || 0} temp files (${formatBytes(tmp.value.total_bytes || 0)})?`)) return
  cleaningTmp.value = true
  try {
    const resp = await apiFetch('/admin/tmp/clean', { method: 'POST' })
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}))
      throw new Error(err.detail || `HTTP ${resp.status}`)
    }
    const data = await resp.json()
    toast.success(`Deleted ${data.deleted} files, freed ${formatBytes(data.freed_bytes)}`)
    await refreshTmp()
  } catch (e: any) {
    toast.error(`Clean failed: ${e?.message || 'network error'}`)
  } finally {
    cleaningTmp.value = false
  }
}

async function refreshSnapshots() {
  snapshotsLoading.value = true
  try {
    const resp = await apiFetch('/admin/qdrant/snapshot/list')
    if (resp.ok) {
      const data = await resp.json()
      snapshots.value = data.snapshots || []
    } else {
      const err = await resp.json().catch(() => ({}))
      throw new Error(err.detail || `HTTP ${resp.status}`)
    }
  } catch (e: any) {
    console.error('Failed to list snapshots:', e)
    toast.error(`Could not list snapshots: ${e?.message || 'network error'}`)
  } finally {
    snapshotsLoading.value = false
  }
}

// Stream a snapshot download into memory chunk-by-chunk so we can update
// the progress bar as bytes arrive. JWT auth means a plain <a href> won't
// work — we need apiFetch + blob anyway.
async function downloadSnapshotByName(name: string, p: SnapshotProgress, startedAt: number) {
  p.stage = 'Downloading…'
  p.indeterminate = false
  p.percent = 0
  p.bytes = 0
  p.total = 0
  const resp = await apiFetch(`/admin/qdrant/snapshot/download/${encodeURIComponent(name)}`)
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}))
    throw new Error(err.detail || `HTTP ${resp.status}`)
  }
  // Content-Length might be missing if a proxy strips it on streaming
  // responses — fall back to indeterminate in that case.
  const lenHdr = resp.headers.get('Content-Length')
  const total = lenHdr ? parseInt(lenHdr, 10) : 0
  if (!total || isNaN(total)) p.indeterminate = true
  else p.total = total

  const reader = resp.body?.getReader()
  if (!reader) {
    // No streaming support — fall back to a simple blob.
    const blob = await resp.blob()
    p.bytes = blob.size
    p.total = blob.size
    p.percent = 100
    p.elapsedMs = Date.now() - startedAt
    triggerBrowserDownload(blob, name)
    return
  }

  const chunks: BlobPart[] = []
  let received = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    if (value) {
      chunks.push(value as BlobPart)
      received += value.byteLength
      p.bytes = received
      if (total) p.percent = Math.min(100, (received / total) * 100)
      p.elapsedMs = Date.now() - startedAt
    }
  }
  if (!total) {
    // Promote to determinate now that we know the final size.
    p.indeterminate = false
    p.total = received
    p.percent = 100
  }
  triggerBrowserDownload(new Blob(chunks, { type: 'application/octet-stream' }), name)
}

function triggerBrowserDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

// Active-create state lives on the backend so the progress bar survives a
// page reload. We POST to start (returns immediately), then poll
// /snapshot/active until status flips to completed/failed. The bytes-level
// download progress can only resume in the same tab that clicked Export,
// because the actual HTTP download is browser-local — for a reloaded tab
// we just refresh the snapshot list and let the user click Download.

let activePollHandle: number | null = null
let activeTickerHandle: number | null = null
// Set when this tab kicked off the create. Lets us auto-trigger the
// download once the create completes, but only here — a different tab
// that polls the same active state should not start a 2nd download.
let exportInitiatedHere = false

function stopActivePolling() {
  if (activePollHandle !== null) { window.clearInterval(activePollHandle); activePollHandle = null }
  if (activeTickerHandle !== null) { window.clearInterval(activeTickerHandle); activeTickerHandle = null }
}

async function dismissActiveServerState() {
  try { await apiFetch('/admin/qdrant/snapshot/active/dismiss', { method: 'POST' }) } catch {}
}

async function pollActiveOnce(): Promise<void> {
  let resp: Response
  try {
    resp = await apiFetch('/admin/qdrant/snapshot/active')
  } catch {
    return  // transient network blip — keep polling
  }
  if (!resp.ok) return
  const data = await resp.json().catch(() => null)
  if (!data?.active) {
    // Nothing in flight on the server; clean up our local state.
    stopActivePolling()
    exportingSnapshot.value = false
    exportInitiatedHere = false
    return
  }
  if (data.status === 'running') {
    // Keep showing progress; the ticker updates elapsed time independently.
    if (data.started_at && exportProgress.elapsedMs === 0) {
      // Page just loaded mid-run — re-anchor elapsed from the server time.
      const t = Date.parse(data.started_at)
      if (!isNaN(t)) exportProgress.elapsedMs = Date.now() - t
    }
    return
  }
  // Terminal: completed | failed.
  stopActivePolling()
  if (data.status === 'completed') {
    const snap = data.snapshot
    toast.success(`Snapshot created: ${snap?.name || 'new snapshot'}`)
    await refreshSnapshots()
    if (exportInitiatedHere && snap?.name) {
      // Same tab that kicked it off — trigger the download automatically.
      try {
        const startedAt = Date.now()
        if (snap.size) exportProgress.total = snap.size
        await downloadSnapshotByName(snap.name, exportProgress, startedAt)
      } catch (e: any) {
        toast.error(`Download failed: ${e?.message || 'network error'}`)
      }
    }
  } else {
    toast.error(`Snapshot create failed: ${data.error || 'unknown error'}`)
  }
  exportInitiatedHere = false
  exportingSnapshot.value = false
  await dismissActiveServerState()
}

function startActivePolling(serverStartedAt?: string | null) {
  stopActivePolling()
  exportingSnapshot.value = true
  // Anchor elapsed time to the server's started_at if available, otherwise now.
  const anchor = serverStartedAt ? Date.parse(serverStartedAt) : Date.now()
  const safeAnchor = isNaN(anchor) ? Date.now() : anchor
  activeTickerHandle = window.setInterval(() => {
    exportProgress.elapsedMs = Date.now() - safeAnchor
  }, 500)
  activePollHandle = window.setInterval(() => { pollActiveOnce() }, 2000)
}

async function exportSnapshot() {
  resetProgress(exportProgress, 'Creating snapshot…')
  exportingSnapshot.value = true
  exportInitiatedHere = true
  try {
    const resp = await apiFetch('/admin/qdrant/snapshot/create', { method: 'POST' })
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}))
      throw new Error(err.detail || `HTTP ${resp.status}`)
    }
    const data = await resp.json()
    // Either status=started or status=already_running — either way we poll.
    startActivePolling(data.started_at)
  } catch (e: any) {
    console.error('Snapshot export failed:', e)
    toast.error(`Export failed: ${e?.message || 'network error'}`)
    exportingSnapshot.value = false
    exportInitiatedHere = false
  }
}

// On mount, ask the backend whether a create is already running (or has
// just finished) so a reloaded tab restores its progress UI instead of
// looking like nothing's happening.
async function restoreActiveCreateState() {
  try {
    const resp = await apiFetch('/admin/qdrant/snapshot/active')
    if (!resp.ok) return
    const data = await resp.json()
    if (!data?.active) return
    if (data.status === 'running') {
      resetProgress(exportProgress, 'Creating snapshot…')
      // exportInitiatedHere stays false — this tab didn't start it, so
      // we won't auto-download on completion.
      startActivePolling(data.started_at)
    } else {
      // Stale completed/failed state — quietly clear it so the UI is clean.
      await dismissActiveServerState()
    }
  } catch (e) {
    console.warn('Could not restore snapshot state:', e)
  }
}

async function downloadExistingSnapshot(name: string) {
  exportingSnapshot.value = true
  resetProgress(exportProgress, 'Downloading…')
  const startedAt = Date.now()
  try {
    await downloadSnapshotByName(name, exportProgress, startedAt)
  } catch (e: any) {
    console.error('Snapshot download failed:', e)
    toast.error(`Download failed: ${e?.message || 'network error'}`)
  } finally {
    exportingSnapshot.value = false
  }
}

async function deleteSnapshot(name: string) {
  if (!window.confirm(`Delete snapshot "${name}"? This cannot be undone.`)) return
  try {
    const resp = await apiFetch(`/admin/qdrant/snapshot/${encodeURIComponent(name)}`, { method: 'DELETE' })
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}))
      throw new Error(err.detail || `HTTP ${resp.status}`)
    }
    toast.success(`Deleted ${name}`)
    await refreshSnapshots()
  } catch (e: any) {
    console.error('Snapshot delete failed:', e)
    toast.error(`Delete failed: ${e?.message || 'network error'}`)
  }
}

function triggerImportPicker() {
  importFileInput.value?.click()
}

async function onSnapshotFilePicked(ev: Event) {
  const input = ev.target as HTMLInputElement
  const file = input.files?.[0]
  // Reset so picking the same file twice still triggers @change.
  input.value = ''
  if (!file) return

  // Two-step confirmation — this destroys the live collection.
  const confirmMsg =
    `Restore collection from "${file.name}" (${formatBytes(file.size)})?\n\n` +
    `This WILL REPLACE all current points in the active Qdrant collection. ` +
    `Make sure you have an export of the current state first.`
  if (!window.confirm(confirmMsg)) return
  if (!window.confirm('Last chance — proceed with restore?')) return

  importingSnapshot.value = true
  resetProgress(importProgress, 'Uploading…')
  importProgress.indeterminate = false
  importProgress.total = file.size
  const startedAt = Date.now()
  try {
    await uploadSnapshotWithProgress(file, importProgress, startedAt)
    toast.success(`Restored snapshot from ${file.name}`)
    await Promise.all([refreshSettings(), refreshSnapshots()])
  } catch (e: any) {
    console.error('Snapshot restore failed:', e)
    toast.error(`Restore failed: ${e?.message || 'network error'}`)
  } finally {
    importingSnapshot.value = false
  }
}

// fetch() doesn't expose upload progress events, so the upload uses XHR.
// After bytes finish uploading the server still has to verify+recover the
// snapshot — we flip back to indeterminate during that "Restoring…" phase.
function uploadSnapshotWithProgress(file: File, p: SnapshotProgress, startedAt: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const form = new FormData()
    form.append('snapshot', file)
    const xhr = new XMLHttpRequest()
    xhr.open('POST', '/admin/qdrant/snapshot/restore')
    const token = localStorage.getItem('auth_token')
    if (token && token !== 'undefined' && token !== 'null') {
      xhr.setRequestHeader('Authorization', `Bearer ${token}`)
    }
    xhr.upload.addEventListener('progress', (ev) => {
      if (ev.lengthComputable) {
        p.bytes = ev.loaded
        p.total = ev.total
        p.percent = Math.min(100, (ev.loaded / ev.total) * 100)
      }
      p.elapsedMs = Date.now() - startedAt
    })
    xhr.upload.addEventListener('load', () => {
      // Bytes done. Server still needs to load + recover the snapshot.
      p.stage = 'Restoring on server…'
      p.indeterminate = true
    })
    const tickerId = window.setInterval(() => {
      p.elapsedMs = Date.now() - startedAt
    }, 500)
    xhr.addEventListener('loadend', () => window.clearInterval(tickerId))
    xhr.addEventListener('error', () => reject(new Error('Network error during upload')))
    xhr.addEventListener('abort', () => reject(new Error('Upload aborted')))
    xhr.addEventListener('load', () => {
      if (xhr.status === 401) {
        // Mirror apiFetch's 401 handling.
        localStorage.removeItem('auth_token')
        localStorage.removeItem('auth_user')
        window.location.assign('/app/login')
        return reject(new Error('Session expired'))
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve()
      } else {
        let detail = `HTTP ${xhr.status}`
        try {
          const body = JSON.parse(xhr.responseText)
          if (body?.detail) detail = body.detail
        } catch {}
        reject(new Error(detail))
      }
    })
    xhr.send(form)
  })
}

onMounted(() => {
  refreshSettings()
  loadCustomAdapters()
  refreshSnapshots()
  refreshTmp()
  restoreActiveCreateState()
})

onBeforeUnmount(() => {
  stopActivePolling()
})
</script>
