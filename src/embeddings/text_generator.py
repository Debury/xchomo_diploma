"""Simple text generation utilities for describing metadata records."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class TextGenerator:
    """Turn metadata dictionaries into descriptive narratives."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.verbosity = self.config.get("verbosity", "medium")

    def generate_batch(self, metadata_list: List[Dict[str, Any]]) -> List[str]:
        return [self.generate_text(metadata) for metadata in metadata_list]

    def generate_text(self, metadata: Dict[str, Any]) -> str:
        lines = [
            f"Dataset {metadata.get('dataset_id', 'unknown')} describes variable {metadata.get('variable')}.",
            f"Long name: {metadata.get('long_name', 'N/A')} ({metadata.get('unit', 'unitless')}).",
            f"Dimensions: {', '.join(metadata.get('dimensions', []))} with shape {metadata.get('shape')}.",
        ]

        if self.verbosity in {"medium", "high"}:
            stats = (
                f"Statistics — min {metadata.get('stat_min')}, max {metadata.get('stat_max')}, "
                f"mean {metadata.get('stat_mean')}, std {metadata.get('stat_std')}"
            )
            lines.append(stats)

        if self.verbosity == "high":
            spatial = metadata.get("spatial_extent", {})
            temporal = metadata.get("temporal_extent", {})
            if spatial:
                lines.append(f"Spatial extent: {spatial}.")
            if temporal:
                lines.append(f"Temporal coverage: {temporal}.")
            samples = metadata.get("sample_values")
            if samples:
                lines.append(f"Sample values: {samples[:5]}.")

        return "\n".join(lines)