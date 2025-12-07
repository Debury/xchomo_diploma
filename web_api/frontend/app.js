// DOM Elements
const healthPanel = document.getElementById('healthStatus');
const sourceForm = document.getElementById('createSourceForm');
const sourceFormMessage = document.getElementById('sourceFormMessage');
const sourcesTable = document.getElementById('sourcesTable');
const ragForm = document.getElementById('ragForm');
const ragAnswer = document.getElementById('ragAnswer');
const ragChunks = document.getElementById('ragChunks');
const ragStatus = document.getElementById('ragStatus');
const embeddingStatsSummary = document.getElementById('embeddingStatsSummary');
const embeddingStatsRaw = document.getElementById('embeddingStatsRaw');
const embeddingCount = document.getElementById('embeddingCount');
const sourcesCount = document.getElementById('sourcesCount');
const variablesCount = document.getElementById('variablesCount');

const refreshHealthBtn = document.getElementById('refreshHealth');
const refreshSourcesBtn = document.getElementById('refreshSources');
const refreshEmbeddingsBtn = document.getElementById('refreshEmbeddings');
const clearEmbeddingsBtn = document.getElementById('clearEmbeddings');

const headers = {
    'Content-Type': 'application/json',
};

const fmtDate = (value) => (value ? new Date(value).toLocaleString() : '—');

// Toggle chunks visibility (for when there are many chunks)
function toggleChunks(chunksId) {
    const container = document.getElementById(chunksId);
    const toggle = document.getElementById(`${chunksId}-toggle`);
    if (container && toggle) {
        if (container.classList.contains('hidden')) {
            container.classList.remove('hidden');
            toggle.textContent = '▲ Hide';
        } else {
            container.classList.add('hidden');
            toggle.textContent = '▼ Show';
        }
    }
}

// Health Check
async function checkHealth() {
    healthPanel.innerHTML = '<div class="flex items-center gap-2"><div class="spinner"></div> Checking...</div>';
    try {
        const res = await fetch('/health');
        const data = await res.json();
        const statusIcon = data.status === 'healthy' ? '✅' : '❌';
        const statusColor = data.status === 'healthy' ? 'text-green-400' : 'text-red-400';
        const dagsterIcon = data.dagster_available ? '✅' : '❌';
        const dagsterColor = data.dagster_available ? 'text-green-400' : 'text-red-400';
        
        healthPanel.innerHTML = `
            <div class="space-y-2">
                <div class="flex items-center gap-2">
                    <span>${statusIcon}</span>
                    <span class="text-sm text-gray-300">API status:</span>
                    <span class="text-sm font-semibold ${statusColor}">${data.status}</span>
                </div>
                <div class="flex items-center gap-2">
                    <span>${dagsterIcon}</span>
                    <span class="text-sm text-gray-300">Dagster:</span>
                    <span class="text-sm font-semibold ${dagsterColor}">${data.dagster_available ? 'reachable' : 'unreachable'}</span>
                </div>
                <div class="text-xs text-gray-500 mt-2">Checked at ${fmtDate(data.timestamp)}</div>
            </div>
        `;
    } catch (err) {
        healthPanel.innerHTML = `<div class="text-red-400 text-sm">❌ Health check failed: ${err.message}</div>`;
    }
}

// Parse CSV string
function parseCSV(value) {
    return value
        .split(',')
        .map((v) => v.trim())
        .filter(Boolean);
}

// Create Source Form
sourceForm?.addEventListener('submit', async (event) => {
    event.preventDefault();
    sourceFormMessage.textContent = 'Saving…';
    sourceFormMessage.className = 'text-sm text-gray-400';
    
    const formData = new FormData(sourceForm);
    const payload = {
        source_id: formData.get('source_id')?.trim(),
        url: formData.get('url')?.trim(),
        format: formData.get('format')?.trim() || null,
        variables: parseCSV(formData.get('variables') || ''),
        tags: parseCSV(formData.get('tags') || ''),
        description: formData.get('description')?.trim() || null,
        is_active: formData.get('is_active') === 'on',
    };

    if (!payload.variables.length) delete payload.variables;
    if (!payload.tags.length) delete payload.tags;

    try {
        const res = await fetch('/sources', {
            method: 'POST',
            headers,
            body: JSON.stringify(payload),
        });
        if (!res.ok) {
            const detail = await res.json().catch(() => ({}));
            throw new Error(detail.detail || res.statusText);
        }
        sourceForm.reset();
        sourceFormMessage.textContent = '✅ Source created successfully!';
        sourceFormMessage.className = 'text-sm text-green-400';
        await loadSources();
        setTimeout(() => {
            sourceFormMessage.textContent = '';
        }, 3000);
    } catch (err) {
        sourceFormMessage.textContent = `❌ Error: ${err.message}`;
        sourceFormMessage.className = 'text-sm text-red-400';
    }
});

// Load Sources
async function loadSources() {
    sourcesTable.innerHTML = `
        <tr>
            <td colspan="5" class="px-4 py-8 text-center">
                <div class="flex items-center justify-center gap-2">
                    <div class="spinner"></div>
                    <span class="text-gray-400">Loading sources...</span>
                </div>
            </td>
        </tr>
    `;
    
    try {
        const res = await fetch('/sources?active_only=false');
        const data = await res.json();
        
        if (!Array.isArray(data) || !data.length) {
            sourcesTable.innerHTML = `
                <tr>
                    <td colspan="5" class="px-4 py-8 text-center text-gray-400">
                        No sources configured yet. Create one above!
                    </td>
                </tr>
            `;
            sourcesCount.textContent = '0';
            return;
        }
        
        sourcesCount.textContent = data.length.toString();
        sourcesTable.innerHTML = '';
        
        data.forEach((source) => {
            const isActive = source.is_active !== false;
            const statusClass = isActive ? 'completed' : 'failed';
            const statusText = isActive ? 'Active' : 'Inactive';
            const processingStatus = source.processing_status || 'pending';
            
            const row = document.createElement('tr');
            row.className = 'hover:bg-gray-700 transition-colors';
            row.innerHTML = `
                <td class="px-4 py-3">
                    <div class="font-semibold text-gray-100">${source.source_id}</div>
                    ${source.description ? `<div class="text-xs text-gray-400 mt-1">${source.description}</div>` : ''}
                </td>
                <td class="px-4 py-3">
                    <span class="px-2 py-1 bg-gray-700 text-gray-300 rounded text-xs">${source.format || 'auto'}</span>
                </td>
                <td class="px-4 py-3">
                    <span class="status-badge ${statusClass}">${statusText}</span>
                    <div class="text-xs text-gray-400 mt-1">${processingStatus}</div>
                </td>
                <td class="px-4 py-3 text-sm text-gray-400">${fmtDate(source.updated_at || source.created_at)}</td>
                <td class="px-4 py-3">
                    <div class="flex flex-wrap gap-2">
                        <button 
                            data-trigger="${source.source_id}" 
                            ${!isActive ? 'disabled' : ''}
                            class="px-3 py-1 text-xs bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded transition-colors"
                            ${!isActive ? 'title="Activate source first"' : ''}
                        >
                            Trigger ETL
                        </button>
                        <button 
                            data-edit="${source.source_id}"
                            class="px-3 py-1 text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 rounded transition-colors"
                        >
                            Edit
                        </button>
                        <button 
                            data-delete="${source.source_id}"
                            class="px-3 py-1 text-xs bg-red-600 hover:bg-red-700 text-white rounded transition-colors"
                        >
                            Delete
                        </button>
                    </div>
                </td>
            `;
            sourcesTable.appendChild(row);
        });
    } catch (err) {
        sourcesTable.innerHTML = `
            <tr>
                <td colspan="5" class="px-4 py-8 text-center text-red-400">
                    Failed to load sources: ${err.message}
                </td>
            </tr>
        `;
    }
}

// Sources Table Actions
sourcesTable?.addEventListener('click', async (event) => {
    const target = event.target;
    if (!(target instanceof HTMLButtonElement)) return;
    
    const sourceId = target.dataset.trigger || target.dataset.edit || target.dataset.delete;
    if (!sourceId) return;
    
    if (target.dataset.trigger) {
        target.textContent = 'Triggering…';
        target.disabled = true;
        try {
            const res = await fetch(`/sources/${sourceId}/trigger`, {
                method: 'POST',
                headers,
            });
            if (!res.ok) {
                const detail = await res.json().catch(() => ({}));
                throw new Error(detail.detail || res.statusText);
            }
            target.textContent = '✅ Queued';
            target.classList.remove('bg-blue-600', 'hover:bg-blue-700');
            target.classList.add('bg-green-600');
        } catch (err) {
            target.textContent = 'Retry';
            target.disabled = false;
            alert(`Failed to trigger ETL: ${err.message}`);
        }
    } else if (target.dataset.edit) {
        // Simple toggle active status
        try {
            const res = await fetch(`/sources?active_only=false`, { headers });
            const sources = await res.json();
            const source = sources.find(s => s.source_id === sourceId);
            if (!source) {
                alert(`Source ${sourceId} not found`);
                return;
            }
            
            const newActive = !source.is_active;
            if (!confirm(`${newActive ? 'Activate' : 'Deactivate'} source ${sourceId}?`)) {
                return;
            }
            
            const updateRes = await fetch(`/sources/${sourceId}`, {
                method: 'PUT',
                headers,
                body: JSON.stringify({ is_active: newActive })
            });
            
            if (!updateRes.ok) {
                const detail = await updateRes.json().catch(() => ({}));
                throw new Error(detail.detail || updateRes.statusText);
            }
            
            await loadSources();
        } catch (err) {
            alert(`Failed to edit source: ${err.message}`);
        }
    } else if (target.dataset.delete) {
        if (!confirm(`Delete source ${sourceId}? This will also remove its embeddings.`)) {
            return;
        }
        const original = target.textContent;
        target.textContent = 'Deleting…';
        target.disabled = true;
        try {
            const res = await fetch(`/sources/${sourceId}?hard_delete=true`, {
                method: 'DELETE',
                headers,
            });
            if (!res.ok) {
                const detail = await res.json().catch(() => ({}));
                throw new Error(detail.detail || res.statusText);
            }
            await loadSources();
        } catch (err) {
            alert(`Failed to delete source: ${err.message}`);
            target.textContent = original;
            target.disabled = false;
        }
    }
});

// Embedding Stats
function renderEmbeddingStats(data) {
    if (!data || Object.keys(data).length === 0) {
        embeddingStatsSummary.innerHTML = `
            <div class="text-center text-gray-400 py-8">
                <p>No embeddings stored yet.</p>
                <p class="text-sm mt-2">Create a source and trigger ETL to generate embeddings.</p>
            </div>
        `;
        embeddingStatsRaw.textContent = '{}';
        embeddingCount.textContent = '0';
        variablesCount.textContent = '0';
        return;
    }

    const total = data.total_embeddings ?? 0;
    const collection = data.collection_name || '—';
    const variables = data.variables || [];
    const sources = data.sources || [];
    const dateRange = data?.date_range
        ? `${data.date_range.earliest || '—'} → ${data.date_range.latest || '—'}`
        : '—';

    embeddingCount.textContent = total.toLocaleString();
    variablesCount.textContent = variables.length.toString();

    embeddingStatsSummary.innerHTML = `
        <div class="grid grid-cols-1 gap-4">
            <div class="bg-gray-700 rounded-lg p-4">
                <div class="text-sm text-gray-400 mb-1">Collection</div>
                <div class="text-lg font-semibold text-gray-100">${collection}</div>
            </div>
            <div class="bg-gray-700 rounded-lg p-4">
                <div class="text-sm text-gray-400 mb-1">Sources</div>
                <div class="text-lg font-semibold text-gray-100">${sources.length}</div>
                <div class="text-xs text-gray-500 mt-1">${sources.slice(0, 3).join(', ')}${sources.length > 3 ? ` +${sources.length - 3} more` : ''}</div>
            </div>
            <div class="bg-gray-700 rounded-lg p-4">
                <div class="text-sm text-gray-400 mb-1">Variables</div>
                <div class="text-lg font-semibold text-gray-100">${variables.length}</div>
                <div class="text-xs text-gray-500 mt-1">${variables.slice(0, 3).join(', ')}${variables.length > 3 ? ` +${variables.length - 3} more` : ''}</div>
            </div>
            <div class="bg-gray-700 rounded-lg p-4">
                <div class="text-sm text-gray-400 mb-1">Date Range</div>
                <div class="text-lg font-semibold text-gray-100">${dateRange}</div>
            </div>
        </div>
    `;

    embeddingStatsRaw.textContent = JSON.stringify(data, null, 2);
}

async function loadEmbeddingStats() {
    if (embeddingStatsSummary) {
        embeddingStatsSummary.innerHTML = `
            <div class="text-center text-gray-400 py-8">
                <div class="inline-block spinner"></div>
                <p class="mt-2">Loading...</p>
            </div>
        `;
    }
    try {
        const res = await fetch('/embeddings/stats');
        const data = await res.json();
        renderEmbeddingStats(data);
    } catch (err) {
        if (embeddingStatsSummary) {
            embeddingStatsSummary.innerHTML = `
                <div class="text-center text-red-400 py-8">
                    <p>Failed to load stats: ${err.message}</p>
                </div>
            `;
        }
        if (embeddingStatsRaw) {
            embeddingStatsRaw.textContent = '';
        }
    }
}

// RAG Form
ragForm?.addEventListener('submit', async (event) => {
    event.preventDefault();
    ragStatus.innerHTML = '<span class="flex items-center gap-2"><div class="spinner"></div> Thinking…</span>';
    ragStatus.className = 'text-sm text-gray-400';
    ragAnswer.innerHTML = '';
    ragChunks.innerHTML = '';
    
    const formData = new FormData(ragForm);
    const payload = {
        question: formData.get('question')?.trim(),
        top_k: 10,  // Increased to capture more relevant chunks (e.g., both TMAX and TMIN for temperature range queries)
        use_llm: true,
    };

    try {
        const res = await fetch('/rag/chat', {
            method: 'POST',
            headers,
            body: JSON.stringify(payload),
        });
        if (!res.ok) {
            const detail = await res.json().catch(() => ({}));
            throw new Error(detail.detail || res.statusText);
        }
        const data = await res.json();
        
        // Display answer
        if (data.answer) {
            ragAnswer.innerHTML = `
                <div class="rag-answer-card">
                    <div class="rag-answer-header">
                        <strong class="text-lg text-gray-100">Answer</strong>
                        <span class="px-3 py-1 bg-blue-600 bg-opacity-20 text-blue-400 rounded-full text-xs font-semibold">
                            ${data.llm_used ? '🤖 LLM Generated' : '📄 Context Only'}
                        </span>
                    </div>
                    <div class="rag-answer-text">${data.answer.replace(/\n/g, '<br>')}</div>
                </div>
            `;
        } else {
            ragAnswer.innerHTML = '<div class="text-gray-400">No answer generated.</div>';
        }
        
        // Display chunks
        if (data.chunks && data.chunks.length > 0) {
            // Create collapsible header for chunks (useful when there are many chunks)
            const chunksHeader = document.createElement('div');
            chunksHeader.className = 'mb-4';
            const isCollapsible = data.chunks.length > 5; // Collapsible if more than 5 chunks
            const chunksId = `chunks-${Date.now()}`;
            
            if (isCollapsible) {
                chunksHeader.innerHTML = `
                    <div class="flex items-center justify-between cursor-pointer hover:bg-gray-700 p-2 rounded" id="${chunksId}-header">
                        <h3 class="text-lg font-semibold text-gray-100">Retrieved Context (${data.chunks.length} chunks)</h3>
                        <span id="${chunksId}-toggle" class="text-gray-400 text-sm">▼ Show</span>
                    </div>
                `;
                // Add event listener instead of inline onclick
                const headerDiv = document.getElementById(`${chunksId}-header`);
                if (headerDiv) {
                    headerDiv.addEventListener('click', () => toggleChunks(chunksId));
                }
            } else {
                chunksHeader.innerHTML = `<h3 class="text-lg font-semibold text-gray-100">Retrieved Context (${data.chunks.length} chunks)</h3>`;
            }
            ragChunks.appendChild(chunksHeader);
            
            // Create container for chunks
            const chunksContainer = document.createElement('div');
            chunksContainer.id = chunksId;
            chunksContainer.className = isCollapsible ? 'hidden' : '';
            
            data.chunks.forEach((chunk, idx) => {
                const card = document.createElement('div');
                card.className = 'chunk-card';
                
                const meta = chunk.metadata || {};
                // Support both old schema (lat_min) and new schema (latitude_min)
                const latMin = meta.latitude_min ?? meta.lat_min;
                const latMax = meta.latitude_max ?? meta.lat_max;
                const lonMin = meta.longitude_min ?? meta.lon_min;
                const lonMax = meta.longitude_max ?? meta.lon_max;
                
                const timeInfo = meta.time_start || meta.time || '';
                const spatialInfo = latMin !== undefined && latMax !== undefined
                    ? `Lat: ${latMin.toFixed(2)}° to ${latMax.toFixed(2)}°${lonMin !== undefined && lonMax !== undefined ? `, Lon: ${lonMin.toFixed(2)}° to ${lonMax.toFixed(2)}°` : ''}`
                    : '';
                // Support both old schema (stat_mean) and new schema (stats_mean)
                const statMean = meta.stats_mean ?? meta.stat_mean;
                const unit = meta.unit || '';
                const statsInfo = statMean !== undefined
                    ? `Mean: ${statMean.toFixed(2)}${unit ? ' ' + unit : ''}`
                    : '';
                
                card.innerHTML = `
                    <div class="chunk-header">
                        <div class="flex items-center gap-2">
                            <strong class="text-gray-100">${chunk.source_id || 'Unknown Source'}</strong>
                            <span class="px-2 py-0.5 bg-gray-700 text-gray-400 rounded text-xs">#${idx + 1}</span>
                        </div>
                        <span class="similarity-badge">${(chunk.similarity * 100).toFixed(1)}% match</span>
                    </div>
                    <div class="chunk-meta">
                        ${chunk.variable ? `<span class="meta-tag">Variable: ${chunk.variable}</span>` : ''}
                        ${timeInfo ? `<span class="meta-tag">Time: ${timeInfo}</span>` : ''}
                        ${spatialInfo ? `<span class="meta-tag">${spatialInfo}</span>` : ''}
                        ${statsInfo ? `<span class="meta-tag">${statsInfo}</span>` : ''}
                    </div>
                    <div class="text-sm text-gray-300 leading-relaxed">${chunk.text || 'No preview available.'}</div>
                `;
                chunksContainer.appendChild(card);
            });
            
            ragChunks.appendChild(chunksContainer);
        } else {
            ragChunks.innerHTML = '<div class="text-gray-400 text-center py-4">No context chunks retrieved.</div>';
        }
        
        ragStatus.textContent = data.references && data.references.length 
            ? `References: ${data.references.join(', ')}` 
            : 'No references available.';
        ragStatus.className = 'text-sm text-gray-400';
    } catch (err) {
        ragStatus.innerHTML = `<span class="text-red-400">❌ Error: ${err.message}</span>`;
    }
});

// Clear Embeddings
clearEmbeddingsBtn?.addEventListener('click', async () => {
    if (!confirm('This will delete all embeddings from the vector database. Continue?')) {
        return;
    }
    const original = clearEmbeddingsBtn.textContent;
    clearEmbeddingsBtn.textContent = 'Clearing…';
    clearEmbeddingsBtn.disabled = true;
    try {
        const res = await fetch('/embeddings/clear?confirm=true', {
            method: 'POST',
            headers,
        });
        if (!res.ok) {
            const detail = await res.json().catch(() => ({}));
            throw new Error(detail.detail || res.statusText);
        }
        const data = await res.json();
        await loadEmbeddingStats();
        alert(`Cleared ${data.removed_embeddings || 0} embeddings from ${data.collection_name || 'database'}`);
    } catch (err) {
        alert(`Failed to clear embeddings: ${err.message}`);
    } finally {
        clearEmbeddingsBtn.textContent = original;
        clearEmbeddingsBtn.disabled = false;
    }
});

// Event Listeners
refreshHealthBtn?.addEventListener('click', checkHealth);
refreshSourcesBtn?.addEventListener('click', loadSources);
refreshEmbeddingsBtn?.addEventListener('click', loadEmbeddingStats);

// Initial Load
checkHealth();
loadSources();
loadEmbeddingStats();
