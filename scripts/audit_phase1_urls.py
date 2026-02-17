#!/usr/bin/env python3
"""Audit all Phase 1 catalog URLs — run inside Docker container."""
import requests
import json
from src.catalog.excel_reader import read_catalog
from src.catalog.phase_classifier import classify_all
from src.catalog.batch_orchestrator import DIRECT_DOWNLOAD_URLS

entries = read_catalog("data/Kopie souboru D1.1.xlsx")
grouped = classify_all(entries)
phase1 = grouped.get(1, [])

# Deduplicate by dataset_name
seen = set()
unique = []
for e in phase1:
    if e.dataset_name not in seen:
        seen.add(e.dataset_name)
        unique.append(e)

print(f"Phase 1: {len(phase1)} entries, {len(unique)} unique datasets")
print()

results = {"working": [], "html": [], "dead": [], "error": [], "override": []}
for e in unique:
    url = DIRECT_DOWNLOAD_URLS.get(e.dataset_name, e.link).strip()
    is_override = e.dataset_name in DIRECT_DOWNLOAD_URLS
    try:
        r = requests.head(
            url, timeout=15, allow_redirects=True,
            headers={"User-Agent": "ClimateRAG/1.0"},
        )
        ct = r.headers.get("Content-Type", "")
        cl = r.headers.get("Content-Length", "?")
        if r.status_code == 200 and "html" not in ct.lower():
            bucket = "override" if is_override else "working"
            results[bucket].append(
                {"name": e.dataset_name, "url": url, "ct": ct, "size": cl}
            )
        elif "html" in ct.lower():
            results["html"].append(
                {"name": e.dataset_name, "url": url, "status": r.status_code}
            )
        else:
            results["dead"].append(
                {"name": e.dataset_name, "url": url, "status": r.status_code}
            )
    except Exception as ex:
        results["error"].append(
            {"name": e.dataset_name, "url": url, "err": str(ex)[:120]}
        )

print("=== WORKING (override URLs) ===")
for r in results["override"]:
    print(f'  OK  {r["name"]} | {r["ct"]} | {r["size"]}')

print()
print("=== WORKING (catalog URLs) ===")
for r in results["working"]:
    print(f'  OK  {r["name"]} | {r["ct"]} | {r["size"]}')

print()
print("=== HTML (landing pages, need override) ===")
for r in results["html"]:
    print(f'  HTML {r["name"]} | {r["status"]} | {r["url"]}')

print()
print("=== DEAD (404/error) ===")
for r in results["dead"]:
    print(f'  DEAD {r["name"]} | {r["status"]} | {r["url"]}')

print()
print("=== CONNECTION ERROR ===")
for r in results["error"]:
    print(f'  ERR  {r["name"]} | {r["err"]}')

print()
total_ok = len(results["override"]) + len(results["working"])
total_bad = len(results["html"]) + len(results["dead"]) + len(results["error"])
print(
    f"Summary: {len(results['override'])} overrides OK, "
    f"{len(results['working'])} catalog OK, "
    f"{len(results['html'])} HTML, "
    f"{len(results['dead'])} dead, "
    f"{len(results['error'])} errors"
)
print(f"Total: {total_ok} downloadable, {total_bad} need attention")
