"""RAG (Retrieval-Augmented Generation) endpoints."""

import asyncio
import csv
import io
import json
import logging
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from web_api.models import RAGChatRequest, RAGChatResponse, RAGChunk
from web_api.rag_endpoint import rag_query, simple_search, get_collection_info, RAGRequest, RAGResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query", response_model=RAGResponse)
async def rag_query_endpoint(request: RAGRequest):
    """Optimized RAG endpoint with timeout handling."""
    return await rag_query(request)


@router.get("/search")
async def rag_search_only(
    query: str,
    top_k: int = 5,
    source_id: Optional[str] = None,
    variable: Optional[str] = None,
):
    """Fast vector search without LLM - for testing and debugging."""
    filters = {}
    if source_id:
        filters["source_id"] = source_id
    if variable:
        filters["variable"] = variable
    return await simple_search(query, top_k, filters if filters else None)


@router.get("/info")
async def rag_info():
    """Get collection info: variables, sources, count."""
    return await get_collection_info()


# --- Bulk export of retrieval matches ---
#
# The chat answer only surfaces the top-N chunks the LLM actually saw, but
# users often want the full set of data points that the same query would
# match — for example to drop into Excel and plot a trend. This endpoint
# runs the embedding search with a large limit (no LLM, no reranker) and
# returns either JSON or a CSV download.

class RAGExportRequest(BaseModel):
    question: str
    # When None or 0 → return EVERY chunk matching the filters via Qdrant
    # scroll. When > 0 → run vector search and return the top_k by relevance.
    top_k: Optional[int] = None
    source_filter: Optional[str] = None
    variable_filter: Optional[str] = None
    # Cited-pairs is what the chat-export button actually uses: a list of
    # ``{"dataset_name": ..., "variable": ...}`` taken from the chunks that
    # the chat answer cited. The export expands this to "give me everything
    # in the collection with the same (dataset_name, variable)" — i.e. all
    # the data points the chat would have seen if it had retrieved more
    # than 5 results. Combines as OR-of-AND.
    cited_pairs: Optional[List[Dict[str, str]]] = None
    # Optional time window — keep only chunks whose time range overlaps
    # [year_min-01-01, year_max-12-31]. When omitted, we try to parse a
    # year range out of the question itself ("2024 to 2026", "2024-2026",
    # Slovak "2024 az 2026", or just a single "2024").
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    fmt: Optional[str] = None  # "csv" or "json"; if absent => JSON


_YEAR_RANGE_RE = re.compile(
    r"\b(19\d{2}|20\d{2})\s*(?:-|–|—|to|až|az|do)\s*(19\d{2}|20\d{2})\b",
    re.IGNORECASE,
)
_YEAR_SINGLE_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def _extract_years(question: str) -> tuple[Optional[int], Optional[int]]:
    """Pull a year range out of free-text. Falls back to a single year."""
    m = _YEAR_RANGE_RE.search(question)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (min(a, b), max(a, b))
    years = [int(y) for y in _YEAR_SINGLE_RE.findall(question)]
    if not years:
        return (None, None)
    if len(years) == 1:
        return (years[0], years[0])
    return (min(years), max(years))


def _chunk_overlaps_years(meta: dict, year_min: int, year_max: int) -> bool:
    """True if the chunk's [time_start, time_end] overlaps [year_min..year_max].

    Compares ISO date prefixes lexicographically — "2024-06-21" < "2026-12-31"
    works as expected since they're zero-padded. Chunks missing time fields
    pass through (we can't prove they don't match, and dropping them would
    silently hide catalog metadata entries).
    """
    ts = (meta.get("time_start") or "")[:10]
    te = (meta.get("time_end") or "")[:10]
    if not ts and not te:
        return True  # no time info — leave it in
    lo = f"{year_min:04d}-01-01"
    hi = f"{year_max:04d}-12-31"
    # Treat missing endpoints as open-ended on that side.
    chunk_lo = ts or "0000-01-01"
    chunk_hi = te or "9999-12-31"
    # Overlap check: chunk_lo <= hi AND chunk_hi >= lo.
    return chunk_lo <= hi and chunk_hi >= lo


# Stable column order for the CSV — these are the metadata fields users
# most often want to see first. Any other payload keys are appended in
# alphabetical order so dataset-specific fields aren't dropped.
_PRIORITY_COLUMNS = [
    "rank", "score", "source_id", "dataset_name", "variable", "long_name",
    "standard_name", "units", "unit",
    "time_start", "time_end", "temporal_frequency",
    "lat_min", "lat_max", "lon_min", "lon_max",
    "stats_mean", "stats_min", "stats_max", "stats_std",
    "region_country", "spatial_coverage", "hazard_type", "impact_sector",
]


def _slugify(s: str, max_len: int = 60) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s.strip().lower()).strip("_")
    return (s[:max_len] or "export").rstrip("_")


def _flatten_chunk(rank: int, chunk: dict) -> dict:
    """Pull metadata up to top level so each row has a flat schema."""
    meta = chunk.get("metadata") or {}
    flat: dict = {
        "rank": rank,
        "score": chunk.get("score"),
        "source_id": meta.get("source_id") or chunk.get("source_id"),
        "variable": meta.get("variable") or chunk.get("variable"),
    }
    for k, v in meta.items():
        if k in flat:
            continue
        # Lists/dicts get JSON-serialised so CSV stays tabular.
        if isinstance(v, (list, dict)):
            import json
            flat[k] = json.dumps(v, ensure_ascii=False)
        else:
            flat[k] = v
    return flat


def _csv_columns(rows: list[dict]) -> list[str]:
    seen: set[str] = set()
    cols: list[str] = []
    for c in _PRIORITY_COLUMNS:
        if any(c in r for r in rows) and c not in seen:
            cols.append(c)
            seen.add(c)
    extras = sorted({k for r in rows for k in r.keys()} - seen)
    cols.extend(extras)
    return cols


def _build_export_filter(req: RAGExportRequest):
    """Translate the request into a Qdrant Filter, or None for no filtering."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    must_clauses = []
    if req.source_filter:
        must_clauses.append(FieldCondition(key="source_id", match=MatchValue(value=req.source_filter)))
    if req.variable_filter:
        must_clauses.append(FieldCondition(key="variable", match=MatchValue(value=req.variable_filter)))

    should_clauses = []
    if req.cited_pairs:
        for p in req.cited_pairs:
            sub_must = []
            # Prefer dataset_name (stable across catalog + ingested data).
            # Fall back to source_id if the chunk only had source_id.
            ds = (p.get("dataset_name") or "").strip()
            sid = (p.get("source_id") or "").strip()
            var = (p.get("variable") or "").strip()
            if ds:
                sub_must.append(FieldCondition(key="dataset_name", match=MatchValue(value=ds)))
            elif sid:
                sub_must.append(FieldCondition(key="source_id", match=MatchValue(value=sid)))
            if var:
                sub_must.append(FieldCondition(key="variable", match=MatchValue(value=var)))
            if sub_must:
                should_clauses.append(Filter(must=sub_must))

    if not must_clauses and not should_clauses:
        return None
    return Filter(
        must=must_clauses or None,
        should=should_clauses or None,
    )


async def _stream_chunks(qfilter, year_min: Optional[int], year_max: Optional[int]):
    """Async generator: scroll Qdrant page-by-page, drop chunks outside the
    year window, yield the rest one at a time. Memory stays bounded to a
    single page (~512 records).
    """
    from web_api.rag_endpoint import _get_components

    _, db, _emb, _llm = _get_components()
    client = getattr(db, "client", None)
    if client is None:
        raise HTTPException(500, "Qdrant client unavailable")

    PAGE = 512
    offset = None
    while True:
        # Wrap the sync scroll call so it doesn't block the FastAPI event
        # loop while Qdrant is fetching.
        records, next_offset = await asyncio.to_thread(
            client.scroll,
            collection_name=db.collection,
            scroll_filter=qfilter,
            limit=PAGE,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for rec in records or []:
            payload = rec.payload or {}
            if year_min and year_max and not _chunk_overlaps_years(payload, year_min, year_max):
                continue
            yield {
                "score": 1.0,
                "source_id": payload.get("source_id"),
                "variable": payload.get("variable"),
                "metadata": payload,
            }
        if not next_offset:
            break
        offset = next_offset


def _csv_value(v: Any) -> str:
    """Stringify a payload value the way DictWriter would, with empty for None."""
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


@router.post("/export")
async def rag_export(request: RAGExportRequest):
    """Bulk-export retrieval matches for a query as JSON or CSV.

    Three filtering modes:
      - ``cited_pairs`` set → "expand the chat answer": every chunk in the
        collection whose ``(dataset_name, variable)`` matches one of the
        cited chunks. Combined with the auto-detected year window from the
        question, this is what a scientist asking "European mean surface
        temperature 2024" actually wants out of a chat answer that only
        showed 4 chunks.
      - ``top_k > 0`` → vector search, sorted by semantic relevance.
      - neither → scroll everything matching ``source_filter`` /
        ``variable_filter`` (use sparingly — collection-scale exports).

    The CSV path streams page-by-page so memory stays bounded; the JSON
    path collects up to 50k rows in memory.
    """
    q = (request.question or "").strip()
    if not q:
        raise HTTPException(400, "Question is required")

    # Year window — explicit params win; otherwise sniff from the question.
    year_min = request.year_min
    year_max = request.year_max
    auto_years = False
    if year_min is None and year_max is None:
        ymin, ymax = _extract_years(q)
        if ymin and ymax:
            year_min, year_max = ymin, ymax
            auto_years = True

    raw_top_k = request.top_k
    is_search_mode = bool(raw_top_k and raw_top_k > 0 and not request.cited_pairs)

    # Vector-search path is unchanged: use simple_search and post-filter.
    if is_search_mode:
        filters: dict = {}
        if request.source_filter:
            filters["source_id"] = request.source_filter
        if request.variable_filter:
            filters["variable"] = request.variable_filter
        top_k = min(int(raw_top_k), 10_000)  # type: ignore[arg-type]
        chunks = await simple_search(q, top_k=top_k, filters=filters or None)
        if year_min and year_max:
            chunks = [
                c for c in chunks
                if _chunk_overlaps_years(c.get("metadata") or {}, year_min, year_max)
            ]
        rows = [_flatten_chunk(i + 1, c) for i, c in enumerate(chunks)]
        return _build_response(request, rows, mode="search",
                               year_min=year_min, year_max=year_max,
                               auto_years=auto_years, q=q)

    # Streaming path: scroll Qdrant, apply year filter inline.
    qfilter = _build_export_filter(request)
    mode = "cited_pairs" if request.cited_pairs else "scroll_all"

    if (request.fmt or "").lower() == "csv":
        return _stream_csv_response(qfilter, year_min, year_max, auto_years, q, mode)

    # JSON path — collect into memory, capped.
    JSON_CAP = 50_000
    chunks = []
    async for c in _stream_chunks(qfilter, year_min, year_max):
        chunks.append(c)
        if len(chunks) >= JSON_CAP:
            break
    rows = [_flatten_chunk(i + 1, c) for i, c in enumerate(chunks)]
    truncated = len(rows) >= JSON_CAP
    return {
        "question": q,
        "mode": mode,
        "count": len(rows),
        "truncated": truncated,
        "year_min": year_min,
        "year_max": year_max,
        "year_auto_detected": auto_years,
        "cited_pairs": request.cited_pairs,
        "chunks": rows,
    }


def _build_response(request: RAGExportRequest, rows: list[dict], *,
                    mode: str, year_min, year_max, auto_years: bool, q: str):
    """Synchronous-friendly response builder for the search-mode path."""
    if (request.fmt or "").lower() == "csv":
        cols = _csv_columns(rows)
        filename = f"climate_export_{_slugify(q)}.csv"

        def csv_iter():
            buf = io.StringIO()
            buf.write("﻿")  # BOM
            writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            yield buf.getvalue()
            buf.seek(0); buf.truncate(0)
            for r in rows:
                writer.writerow({c: _csv_value(r.get(c)) for c in cols})
                yield buf.getvalue()
                buf.seek(0); buf.truncate(0)

        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Export-Mode": mode,
            "X-Export-Count": str(len(rows)),
        }
        if year_min and year_max:
            headers["X-Export-Year-Min"] = str(year_min)
            headers["X-Export-Year-Max"] = str(year_max)
            headers["X-Export-Year-Auto"] = "1" if auto_years else "0"
        return StreamingResponse(csv_iter(), media_type="text/csv; charset=utf-8", headers=headers)

    return {
        "question": q,
        "mode": mode,
        "count": len(rows),
        "year_min": year_min,
        "year_max": year_max,
        "year_auto_detected": auto_years,
        "chunks": rows,
    }


def _stream_csv_response(qfilter, year_min, year_max, auto_years: bool, q: str, mode: str):
    """Build a StreamingResponse that scrolls Qdrant page-by-page and emits
    CSV rows as they arrive. Column set is derived from the first ~100
    rows; later rows that introduce new keys will have those keys dropped
    (extrasaction='ignore') — acceptable since chunks within one
    (dataset, variable) share their schema.
    """
    filename = f"climate_export_{_slugify(q)}.csv"
    HEADER_SAMPLE = 100   # rows used to derive the column set
    BATCH_FLUSH = 50      # rows between yields after columns are known

    async def csv_iter():
        buf = io.StringIO()
        buf.write("﻿")
        cols: Optional[list[str]] = None
        writer: Optional[csv.DictWriter] = None
        rank = 0
        pending: list[dict] = []

        def init_writer_from_pending():
            nonlocal cols, writer
            cols = _csv_columns(pending)
            writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            for r in pending:
                writer.writerow({c: _csv_value(r.get(c)) for c in cols})

        async for chunk in _stream_chunks(qfilter, year_min, year_max):
            rank += 1
            pending.append(_flatten_chunk(rank, chunk))

            # First flush: discover columns once we have a representative sample.
            if cols is None and len(pending) >= HEADER_SAMPLE:
                init_writer_from_pending()
                yield buf.getvalue()
                buf.seek(0); buf.truncate(0)
                pending = []
                continue

            # Subsequent flushes.
            if cols is not None and len(pending) >= BATCH_FLUSH:
                assert writer is not None
                for r in pending:
                    writer.writerow({c: _csv_value(r.get(c)) for c in cols})
                yield buf.getvalue()
                buf.seek(0); buf.truncate(0)
                pending = []

        # Tail: handle (a) fewer than HEADER_SAMPLE total rows, (b) leftover
        # rows after the last batch.
        if cols is None:
            if not pending:
                # Empty result — emit a stub header so Excel doesn't choke.
                buf.write(",".join(_PRIORITY_COLUMNS[:5]) + "\n")
                yield buf.getvalue()
                return
            init_writer_from_pending()
            yield buf.getvalue()
            return

        if pending:
            assert writer is not None
            for r in pending:
                writer.writerow({c: _csv_value(r.get(c)) for c in cols})
            yield buf.getvalue()

    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "X-Export-Mode": mode,
    }
    if year_min and year_max:
        headers["X-Export-Year-Min"] = str(year_min)
        headers["X-Export-Year-Max"] = str(year_max)
        headers["X-Export-Year-Auto"] = "1" if auto_years else "0"
    return StreamingResponse(csv_iter(), media_type="text/csv; charset=utf-8", headers=headers)


@router.post("/chat", response_model=RAGChatResponse)
async def rag_chat_legacy(request: RAGChatRequest):
    """Legacy chat endpoint - delegates to optimized rag_query()."""
    if not (request.question or "").strip():
        raise HTTPException(400, "Question is required")

    # Honour the runtime default for `use_reranker` (toggled in Settings)
    # rather than hardcoding True — that was why turning it off in the UI
    # didn't actually speed up the chat.
    from web_api.routes.admin import _runtime_settings
    default_reranker = bool(_runtime_settings.get("use_reranker", False))

    try:
        rag_req = RAGRequest(
            question=request.question,
            top_k=request.top_k or 10,
            use_llm=request.use_llm if request.use_llm is not None else True,
            use_reranker=default_reranker,
            temperature=request.temperature or 0.3,
            source_id=getattr(request, "source_id", None),
            variable=getattr(request, "variable", None),
        )
        result = await rag_query(rag_req)
    except HTTPException:
        # Preserve original status codes raised by rag_query (e.g. 400 on empty query).
        raise
    except Exception as e:
        logger.error(f"RAG chat error: {e}", exc_info=True)
        raise HTTPException(500, str(e))

    chunks_model = []
    for c in result.chunks:
        meta = c.get("metadata", {})
        chunks_model.append(
            RAGChunk(
                source_id=c.get("source_id", "unknown"),
                variable=c.get("variable", "unknown"),
                similarity=c.get("score", 0.0),
                text=c.get("text", "")[:200],
                metadata=meta,
            )
        )

    return RAGChatResponse(
        question=result.question,
        answer=result.answer,
        references=result.references,
        chunks=chunks_model,
        llm_used=result.llm_used,
    )
