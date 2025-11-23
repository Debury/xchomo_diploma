"""Embedding ingestion pipeline for climate datasets."""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import xarray as xr

from .database import VectorDatabase
from .generator import EmbeddingGenerator
from .metadata import MetadataExtractor
from .text_generator import TextGenerator


class EmbeddingPipeline:
    """Turn structured climate artifacts into dense embeddings stored in Qdrant."""

    def __init__(
        self,
        generator: Optional[EmbeddingGenerator] = None,
        database: Optional[VectorDatabase] = None,
        chunk_size: int = 800,
        metadata_extractor: Optional[MetadataExtractor] = None,
        text_generator: Optional[TextGenerator] = None,
    ) -> None:
        self.generator = generator or EmbeddingGenerator()
        self.database = database or VectorDatabase()
        self.chunk_size = chunk_size
        self.metadata_extractor = metadata_extractor or MetadataExtractor()
        self.text_generator = text_generator or TextGenerator()
        self.metrics: Dict[str, Any] = {
            "pipeline": {
                "total_datasets_processed": 0,
                "total_embeddings_created": 0,
                "total_pipeline_time": 0.0,
            },
            "last_run": {},
        }

    # ------------------------------------------------------------------
    def process_dataset(
        self,
        file_path: str,
        source_id: Optional[str] = None,
        variables: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        dataset = xr.open_dataset(path)
        try:
            segments = self._dataset_to_segments(
                dataset,
                source_id=source_id or path.stem,
                variables=set(variables) if variables else None,
                file_path=str(path.resolve()),
            )
        finally:
            dataset.close()

        if not segments:
            duration = time.perf_counter() - start_time
            self._update_metrics(num_embeddings=0, duration=duration)
            return {"file": str(path), "num_embeddings": 0, "variables": [], "processing_time": duration}

        documents = [segment["text"] for segment in segments]
        metadatas = [segment["metadata"] for segment in segments]
        ids = [segment["id"] for segment in segments]

        embeddings = self.generator.generate_embeddings(documents)
        self.database.add_embeddings(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

        variables_found = sorted({meta.get("variable") for meta in metadatas if meta.get("variable")})
        duration = time.perf_counter() - start_time
        self._update_metrics(num_embeddings=len(documents), duration=duration)
        self.metrics["last_run"] = {
            "file": str(path),
            "num_embeddings": len(documents),
            "variables": variables_found,
            "processing_time": duration,
        }

        return {
            "file": str(path),
            "num_embeddings": len(documents),
            "variables": variables_found,
            "processing_time": duration,
        }

    def process_directory(
        self,
        directory: str,
        pattern: str = "*.nc",
        source_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        results: List[Dict[str, Any]] = []
        total_embeddings = 0
        start_time = time.perf_counter()
        for file in sorted(dir_path.rglob(pattern)):
            source_id = f"{source_prefix or dir_path.stem}-{file.stem}"
            stats = self.process_dataset(str(file), source_id=source_id)
            total_embeddings += stats.get("num_embeddings", 0)
            results.append(stats)

        duration = time.perf_counter() - start_time
        return {
            "directory": str(dir_path),
            "num_files": len(results),
            "num_embeddings": total_embeddings,
            "files": results,
            "processing_time": duration,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Return aggregated pipeline metrics."""
        return self.metrics

    # ------------------------------------------------------------------
    def _dataset_to_segments(
        self,
        dataset: xr.Dataset,
        source_id: str,
        variables: Optional[set[str]] = None,
        file_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        segments: List[Dict[str, Any]] = []
        timestamp = dt.datetime.utcnow().isoformat()
        metadata_records = self.metadata_extractor.extract_from_dataset(
            data=dataset,
            file_path=file_path or "",
            dataset_id=source_id,
        )
        filtered_records = (
            [record for record in metadata_records if not variables or record["variable"] in variables]
        )
        texts = self.text_generator.generate_batch(filtered_records)

        for record, text in zip(filtered_records, texts):
            for chunk_index, chunk_text in enumerate(self._chunk_text(text)):
                chunk_id = f"{record['id']}-chunk-{chunk_index}"
                metadata = dict(record)
                metadata.update(
                    {
                        "source_id": source_id,
                        "chunk_index": chunk_index,
                        "file_path": file_path or metadata.get("file_path"),
                        "timestamp": metadata.get("timestamp", timestamp),
                        "text": chunk_text,
                    }
                )
                segments.append({"id": chunk_id, "text": chunk_text, "metadata": metadata})
        return segments

    def _chunk_text(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]
        chunks = []
        current = []
        current_len = 0
        for sentence in text.split("\n"):
            sentence_len = len(sentence)
            if current_len + sentence_len + 1 > self.chunk_size and current:
                chunks.append("\n".join(current))
                current = [sentence]
                current_len = sentence_len
            else:
                current.append(sentence)
                current_len += sentence_len + 1
        if current:
            chunks.append("\n".join(current))
        return chunks

    def _update_metrics(self, num_embeddings: int, duration: float) -> None:
        pipeline_metrics = self.metrics["pipeline"]
        pipeline_metrics["total_datasets_processed"] += 1
        pipeline_metrics["total_embeddings_created"] += num_embeddings
        pipeline_metrics["total_pipeline_time"] += duration
