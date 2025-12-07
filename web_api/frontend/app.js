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

const refreshHealthBtn = document.getElementById('refreshHealth');
const refreshSourcesBtn = document.getElementById('refreshSources');
const refreshEmbeddingsBtn = document.getElementById('refreshEmbeddings');
const clearEmbeddingsBtn = document.getElementById('clearEmbeddings');

const headers = {
    'Content-Type': 'application/json',
};

const fmtDate = (value) => (value ? new Date(value).toLocaleString() : '—');

async function checkHealth() {
    healthPanel.innerHTML = '<span class="loading"></span> Checking…';
    try {
        const res = await fetch('/health');
        const data = await res.json();
        const statusIcon = data.status === 'healthy' ? '✅' : '❌';
        const dagsterIcon = data.dagster_available ? '✅' : '❌';
        healthPanel.innerHTML = `
            <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                <div>${statusIcon} API status: <strong style="color: var(--accent-light);">${data.status}</strong></div>
                <div>${dagsterIcon} Dagster reachable: <strong style="color: var(--accent-light);">${data.dagster_available ? 'yes' : 'no'}</strong></div>
                <div class="hint" style="margin-top: 0.5rem;">Checked at ${fmtDate(data.timestamp)}</div>
            </div>
        `;
    } catch (err) {
        healthPanel.innerHTML = `❌ Health check failed: <span style="color: var(--danger);">${err.message}</span>`;
    }
}

function parseCSV(value) {
    return value
        .split(',')
        .map((v) => v.trim())
        .filter(Boolean);
}

sourceForm?.addEventListener('submit', async (event) => {
    event.preventDefault();
    sourceFormMessage.textContent = 'Saving…';
    const formData = new FormData(sourceForm);

    const payload = {
        source_id: formData.get('source_id')?.trim(),
        url: formData.get('url')?.trim(),
        format: formData.get('format')?.trim() || null,
        variables: parseCSV(formData.get('variables') || ''),
        tags: parseCSV(formData.get('tags') || ''),
        description: formData.get('description')?.trim() || null,
        is_active: true,  // Always create as active (checkbox value: formData.get('is_active') === 'on')
    };
    
    console.log('Creating source with is_active:', payload.is_active);

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
        sourceFormMessage.textContent = 'Source created – trigger it below.';
        await loadSources();
    } catch (err) {
        sourceFormMessage.textContent = `Error: ${err.message}`;
    }
});

async function loadSources() {
    sourcesTable.innerHTML = '<tr><td colspan="5" style="text-align: center; padding: 2rem;"><span class="loading"></span> Loading…</td></tr>';
    try {
        const res = await fetch('/sources?active_only=false');
        const data = await res.json();
        if (!Array.isArray(data) || !data.length) {
            sourcesTable.innerHTML = '<tr><td colspan="5">No sources configured yet.</td></tr>';
            return;
        }
        sourcesTable.innerHTML = '';
        data.forEach((source) => {
            const activeStatus = source.active ? 'active' : 'inactive';
            const activeClass = source.active ? 'completed' : 'failed';
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>
                    <strong>${source.source_id}</strong>
                    <div class="hint">${source.description || ''}</div>
                </td>
                <td>${source.format}</td>
                <td>
                    <span class="status-pill ${activeClass}">${activeStatus}</span>
                    <div class="hint">Processing: ${source.processing_status}</div>
                </td>
                <td>${fmtDate(source.last_processed)}</td>
                <td>
                    <div class="table-actions">
                        <button data-trigger="${source.source_id}" ${!source.active ? 'disabled title="Activate source first"' : ''}>Trigger ETL</button>
                        <button class="ghost" data-edit="${source.source_id}">Edit</button>
                        <button class="ghost danger" data-delete="${source.source_id}">Delete</button>
                    </div>
                </td>
            `;
            sourcesTable.appendChild(row);
        });
    } catch (err) {
        sourcesTable.innerHTML = `<tr><td colspan="5">Failed to load sources: ${err.message}</td></tr>`;
    }
}

sourcesTable?.addEventListener('click', async (event) => {
    const target = event.target;
    if (target instanceof HTMLButtonElement && target.dataset.trigger) {
        const sourceId = target.dataset.trigger;
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
            target.textContent = 'Queued ✅';
        } catch (err) {
            target.textContent = 'Retry';
            target.disabled = false;
            alert(`Failed to trigger ETL: ${err.message}`);
        }
    } else if (target instanceof HTMLButtonElement && target.dataset.edit) {
        const sourceId = target.dataset.edit;
        try {
            // Fetch current source data
            const res = await fetch(`/sources?active_only=false`, { headers });
            const sources = await res.json();
            const source = sources.find(s => s.source_id === sourceId);
            if (!source) {
                alert(`Source ${sourceId} not found`);
                return;
            }
            
            // Simple prompt to toggle active status
            const newActive = !source.active;
            const confirmMsg = newActive 
                ? `Activate source ${sourceId}?` 
                : `Deactivate source ${sourceId}?`;
            
            if (!confirm(confirmMsg)) {
                return;
            }
            
            // Update source
            const updateRes = await fetch(`/sources/${sourceId}`, {
                method: 'PUT',
                headers,
                body: JSON.stringify({
                    is_active: newActive
                })
            });
            
            if (!updateRes.ok) {
                const detail = await updateRes.json().catch(() => ({}));
                throw new Error(detail.detail || updateRes.statusText);
            }
            
            await loadSources();
            alert(`Source ${sourceId} ${newActive ? 'activated' : 'deactivated'}`);
        } catch (err) {
            alert(`Failed to edit source: ${err.message}`);
        }
    } else if (target instanceof HTMLButtonElement && target.dataset.delete) {
        const sourceId = target.dataset.delete;
        if (!confirm(`Delete source ${sourceId}?`)) {
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

function summarizeList(items = [], maxItems = 5) {
    if (!items.length) {
        return '—';
    }
    if (items.length <= maxItems) {
        return items.join(', ');
    }
    const head = items.slice(0, maxItems).join(', ');
    return `${head} … (+${items.length - maxItems} more)`;
}

function renderEmbeddingStats(data) {
    if (!embeddingStatsSummary || !embeddingStatsRaw) {
        return;
    }

    if (!data || Object.keys(data).length === 0) {
        embeddingStatsSummary.innerHTML = '<p class="hint">No embeddings stored yet.</p>';
        embeddingStatsRaw.textContent = '{}';
        return;
    }

    const total = data.total_embeddings ?? 0;
    const collection = data.collection_name || '—';
    const variables = summarizeList(data.variables || []);
    const sources = summarizeList(data.sources || []);
    const dateRange = data?.date_range
        ? `${data.date_range.earliest || '—'} → ${data.date_range.latest || '—'}`
        : '—';

    embeddingStatsSummary.innerHTML = `
        <div class="stat-card">
            <p class="hint">Total embeddings</p>
            <strong>${total.toLocaleString()}</strong>
        </div>
        <div class="stat-card">
            <p class="hint">Collection</p>
            <strong>${collection}</strong>
        </div>
        <div class="stat-card">
            <p class="hint">Sources</p>
            <strong>${sources}</strong>
        </div>
        <div class="stat-card">
            <p class="hint">Variables tracked</p>
            <strong>${variables}</strong>
        </div>
        <div class="stat-card">
            <p class="hint">Date range</p>
            <strong>${dateRange}</strong>
        </div>
    `;

    embeddingStatsRaw.textContent = JSON.stringify(data, null, 2);
}

async function loadEmbeddingStats() {
    if (embeddingStatsSummary) {
        embeddingStatsSummary.innerHTML = '<p class="hint"><span class="loading"></span> Loading…</p>';
    }
    try {
        const res = await fetch('/embeddings/stats');
        const data = await res.json();
        renderEmbeddingStats(data);
    } catch (err) {
        if (embeddingStatsSummary) {
            embeddingStatsSummary.innerHTML = `<p class="hint">Failed to load stats: ${err.message}</p>`;
        }
        if (embeddingStatsRaw) {
            embeddingStatsRaw.textContent = '';
        }
    }
}

ragForm?.addEventListener('submit', async (event) => {
    event.preventDefault();
    ragStatus.innerHTML = '<span class="loading"></span> Thinking…';
    ragAnswer.textContent = '';
    ragChunks.innerHTML = '';
    const formData = new FormData(ragForm);
    const payload = {
        question: formData.get('question')?.trim(),
        top_k: 3,
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
        
        // Display answer with better formatting
        if (data.answer) {
            ragAnswer.innerHTML = `
                <div class="rag-answer-content">
                    <div class="rag-answer-header">
                        <strong>Answer</strong>
                        ${data.llm_used ? '<span class="badge">LLM Generated</span>' : '<span class="badge">Context Only</span>'}
                    </div>
                    <div class="rag-answer-text">${data.answer.replace(/\n/g, '<br>')}</div>
                </div>
            `;
        } else {
            ragAnswer.textContent = 'No answer generated.';
        }
        
        // Display context chunks with enhanced metadata
        ragChunks.innerHTML = '';
        if (data.chunks && data.chunks.length > 0) {
            const chunksHeader = document.createElement('div');
            chunksHeader.className = 'chunks-header';
            chunksHeader.innerHTML = `<h3>Retrieved Context (${data.chunks.length} chunks)</h3>`;
            ragChunks.appendChild(chunksHeader);
            
            data.chunks.forEach((chunk, idx) => {
                const card = document.createElement('div');
                card.className = 'chunk-card';
                
                // Extract metadata for display
                const meta = chunk.metadata || {};
                const timeInfo = meta.time_start || meta.time || '';
                const spatialInfo = meta.lat_min !== undefined 
                    ? `Lat: ${meta.lat_min?.toFixed(2)}° to ${meta.lat_max?.toFixed(2)}°`
                    : '';
                const statsInfo = meta.stat_mean !== undefined
                    ? `Mean: ${meta.stat_mean?.toFixed(2)}${meta.unit ? ' ' + meta.unit : ''}`
                    : '';
                
                card.innerHTML = `
                    <header>
                        <div>
                            <strong>${chunk.source_id || 'Unknown Source'}</strong>
                            <span class="chunk-rank">#${idx + 1}</span>
                        </div>
                        <span class="similarity-badge">${(chunk.similarity * 100).toFixed(1)}% match</span>
                    </header>
                    <div class="chunk-meta">
                        ${chunk.variable ? `<span class="meta-tag">Variable: ${chunk.variable}</span>` : ''}
                        ${timeInfo ? `<span class="meta-tag">Time: ${timeInfo}</span>` : ''}
                        ${spatialInfo ? `<span class="meta-tag">${spatialInfo}</span>` : ''}
                        ${statsInfo ? `<span class="meta-tag">${statsInfo}</span>` : ''}
                    </div>
                    <div class="chunk-text">${chunk.text || 'No preview available.'}</div>
                `;
                ragChunks.appendChild(card);
            });
        } else {
            ragChunks.innerHTML = '<p class="hint">No context chunks retrieved.</p>';
        }
        
        ragStatus.textContent = data.references && data.references.length 
            ? `References: ${data.references.join(', ')}` 
            : 'No references available.';
    } catch (err) {
        ragStatus.textContent = `Chat error: ${err.message}`;
    }
});

refreshHealthBtn?.addEventListener('click', checkHealth);
refreshSourcesBtn?.addEventListener('click', loadSources);
refreshEmbeddingsBtn?.addEventListener('click', loadEmbeddingStats);
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
        if (embeddingStatsSummary) {
            embeddingStatsSummary.innerHTML = `
                <div class="stat-card">
                    <p class="hint">Status</p>
                    <strong>Cleared ${data.removed_embeddings} from ${data.collection_name}</strong>
                </div>`;
        }
        if (embeddingStatsRaw) {
            embeddingStatsRaw.textContent = JSON.stringify(data, null, 2);
        }
        await loadEmbeddingStats();
    } catch (err) {
        alert(`Failed to clear embeddings: ${err.message}`);
    } finally {
        clearEmbeddingsBtn.textContent = original;
        clearEmbeddingsBtn.disabled = false;
    }
});

// Initial load
checkHealth();
loadSources();
loadEmbeddingStats();
