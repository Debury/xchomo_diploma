const healthPanel = document.getElementById('healthStatus');
const sourceForm = document.getElementById('createSourceForm');
const sourceFormMessage = document.getElementById('sourceFormMessage');
const sourcesTable = document.getElementById('sourcesTable');
const ragForm = document.getElementById('ragForm');
const ragAnswer = document.getElementById('ragAnswer');
const ragChunks = document.getElementById('ragChunks');
const ragStatus = document.getElementById('ragStatus');
const embeddingStats = document.getElementById('embeddingStats');

const refreshHealthBtn = document.getElementById('refreshHealth');
const refreshSourcesBtn = document.getElementById('refreshSources');
const refreshEmbeddingsBtn = document.getElementById('refreshEmbeddings');

const headers = {
    'Content-Type': 'application/json',
};

const fmtDate = (value) => (value ? new Date(value).toLocaleString() : '—');

async function checkHealth() {
    healthPanel.textContent = 'Checking…';
    try {
        const res = await fetch('/health');
        const data = await res.json();
        healthPanel.innerHTML = `API status: <strong>${data.status}</strong><br/>Dagster reachable: <strong>${data.dagster_available ? 'yes' : 'no'}</strong><br/>Checked at ${fmtDate(data.timestamp)}`;
    } catch (err) {
        healthPanel.textContent = `Health check failed: ${err.message}`;
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
        sourceFormMessage.textContent = 'Source created – trigger it below.';
        await loadSources();
    } catch (err) {
        sourceFormMessage.textContent = `Error: ${err.message}`;
    }
});

async function loadSources() {
    sourcesTable.innerHTML = '<tr><td colspan="5">Loading…</td></tr>';
    try {
        const res = await fetch('/sources?active_only=false');
        const data = await res.json();
        if (!Array.isArray(data) || !data.length) {
            sourcesTable.innerHTML = '<tr><td colspan="5">No sources configured yet.</td></tr>';
            return;
        }
        sourcesTable.innerHTML = '';
        data.forEach((source) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>
                    <strong>${source.source_id}</strong>
                    <div class="hint">${source.description || ''}</div>
                </td>
                <td>${source.format}</td>
                <td><span class="status-pill ${source.processing_status}">${source.processing_status}</span></td>
                <td>${fmtDate(source.last_processed)}</td>
                <td>
                    <button data-trigger="${source.source_id}">Trigger ETL</button>
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
    }
});

async function loadEmbeddingStats() {
    embeddingStats.textContent = 'Loading…';
    try {
        const res = await fetch('/embeddings/stats');
        const data = await res.json();
        embeddingStats.textContent = JSON.stringify(data, null, 2);
    } catch (err) {
        embeddingStats.textContent = `Failed to load stats: ${err.message}`;
    }
}

ragForm?.addEventListener('submit', async (event) => {
    event.preventDefault();
    ragStatus.textContent = 'Thinking…';
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
        ragAnswer.textContent = data.answer;
        ragChunks.innerHTML = '';
        data.chunks.forEach((chunk) => {
            const card = document.createElement('div');
            card.className = 'chunk-card';
            card.innerHTML = `
                <header>
                    <strong>${chunk.source_id}</strong>
                    <span>${(chunk.similarity * 100).toFixed(1)}%</span>
                </header>
                <div class="hint">${chunk.variable || 'unknown variable'}</div>
                <p>${chunk.text || 'No preview available.'}</p>
            `;
            ragChunks.appendChild(card);
        });
        ragStatus.textContent = data.references.length ? `References: ${data.references.join(', ')}` : 'No references available.';
    } catch (err) {
        ragStatus.textContent = `Chat error: ${err.message}`;
    }
});

refreshHealthBtn?.addEventListener('click', checkHealth);
refreshSourcesBtn?.addEventListener('click', loadSources);
refreshEmbeddingsBtn?.addEventListener('click', loadEmbeddingStats);

// Initial load
checkHealth();
loadSources();
loadEmbeddingStats();
