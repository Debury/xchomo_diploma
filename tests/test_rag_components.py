"""
Test suite for RAG components with new climate_embeddings structure.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import zipfile
import json

from climate_embeddings.embeddings import TextEmbedder, get_text_embedder, list_available_models
from climate_embeddings.index import VectorIndex, SearchResult
from climate_embeddings.loaders import load_from_zip
from climate_embeddings.rag import RAGPipeline, build_index_from_embeddings


class TestTextEmbeddings:
    """Test text embedding models."""
    
    def test_list_available_models(self):
        """Test model registry."""
        models = list_available_models()
        assert "bge-large" in models
        assert "gte-large" in models
        assert "minilm" in models
    
    def test_text_embedder_initialization(self):
        """Test embedder initialization with minilm (fast for testing)."""
        embedder = TextEmbedder(model_name="minilm")
        assert embedder.embedding_dim == 384
    
    def test_single_text_encoding(self):
        """Test encoding single text."""
        embedder = get_text_embedder("minilm")
        
        text = "What is the global temperature anomaly?"
        vec = embedder(text)
        
        assert vec.shape == (384,)
        assert vec.dtype == np.float32
        # Check L2 normalization
        assert np.abs(np.linalg.norm(vec) - 1.0) < 0.01
    
    def test_batch_encoding(self):
        """Test encoding multiple texts."""
        embedder = TextEmbedder("minilm")
        
        texts = [
            "Temperature trends in Europe",
            "Precipitation anomalies",
            "Sea level rise projections",
        ]
        
        vecs = embedder.encode(texts)
        
        assert vecs.shape == (3, 384)
        assert vecs.dtype == np.float32
    
    def test_embedding_similarity(self):
        """Test that similar texts have higher cosine similarity."""
        embedder = get_text_embedder("minilm")
        
        text1 = "global warming temperature increase"
        text2 = "climate change heat rising"
        text3 = "precipitation rainfall patterns"
        
        vec1 = embedder(text1)
        vec2 = embedder(text2)
        vec3 = embedder(text3)
        
        # Similarity between temperature-related texts
        sim_12 = np.dot(vec1, vec2)
        # Similarity between temperature and precipitation
        sim_13 = np.dot(vec1, vec3)
        
        # Temperature texts should be more similar
        assert sim_12 > sim_13


class TestVectorIndex:
    """Test vector index operations."""
    
    def test_index_creation(self):
        """Test creating empty index."""
        index = VectorIndex(dim=384, metric="cosine")
        assert len(index) == 0
        assert index.dim == 384
    
    def test_add_single_vector(self):
        """Test adding single vector."""
        index = VectorIndex(dim=128)
        
        vec = np.random.randn(128).astype(np.float32)
        vec /= np.linalg.norm(vec)
        
        idx = index.add(vec, {"source": "test", "year": 2020})
        
        assert idx == 0
        assert len(index) == 1
    
    def test_add_batch(self):
        """Test adding multiple vectors."""
        index = VectorIndex(dim=128)
        
        vecs = np.random.randn(10, 128).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        
        metadata_list = [{"id": i, "source": "test"} for i in range(10)]
        indices = index.add_batch(vecs, metadata_list)
        
        assert len(indices) == 10
        assert len(index) == 10
    
    def test_search_basic(self):
        """Test basic similarity search."""
        index = VectorIndex(dim=128, metric="cosine")
        
        # Add some vectors
        for i in range(5):
            vec = np.random.randn(128).astype(np.float32)
            vec /= np.linalg.norm(vec)
            index.add(vec, {"id": i})
        
        # Search with query
        query = np.random.randn(128).astype(np.float32)
        query /= np.linalg.norm(query)
        
        results = index.search(query, k=3)
        
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
        # Scores should be descending
        assert results[0].score >= results[1].score >= results[2].score
    
    def test_search_with_filters(self):
        """Test search with metadata filters."""
        index = VectorIndex(dim=128)
        
        # Add vectors with different metadata
        for i in range(10):
            vec = np.random.randn(128).astype(np.float32)
            vec /= np.linalg.norm(vec)
            source = "gistemp" if i < 5 else "era5"
            year = 2020 + i
            index.add(vec, {"source": source, "year": year})
        
        query = np.random.randn(128).astype(np.float32)
        query /= np.linalg.norm(query)
        
        # Filter by source
        results = index.search(query, k=10, filters={"source": "gistemp"})
        assert len(results) == 5
        assert all(r.metadata["source"] == "gistemp" for r in results)
        
        # Filter by year range
        results = index.search(query, k=10, filters={"year": {"$gte": 2025}})
        assert len(results) == 5
        assert all(r.metadata["year"] >= 2025 for r in results)
    
    def test_save_and_load(self):
        """Test persistence."""
        index = VectorIndex(dim=128)
        
        # Add some data
        for i in range(3):
            vec = np.random.randn(128).astype(np.float32)
            vec /= np.linalg.norm(vec)
            index.add(vec, {"id": i, "value": f"test_{i}"})
        
        # Save
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name
        
        try:
            index.save(temp_path)
            
            # Load
            loaded_index = VectorIndex.load(temp_path)
            
            assert len(loaded_index) == 3
            assert loaded_index.dim == 128
            assert loaded_index.metadata[0]["value"] == "test_0"
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestZIPLoader:
    """Test ZIP archive loading."""
    
    def test_load_from_zip_with_csv(self):
        """Test loading CSV from ZIP."""
        # Create temporary ZIP with CSV
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create sample CSV
            csv_data = "Year,Temp\n2020,14.5\n2021,14.7\n2022,15.0\n"
            csv_path = tmpdir / "temp_data.csv"
            csv_path.write_text(csv_data)
            
            # Create ZIP
            zip_path = tmpdir / "climate.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.write(csv_path, arcname="temp_data.csv")
            
            # Load from ZIP
            results = load_from_zip(zip_path)
            
            assert len(results) == 1
            assert results[0].metadata["zip_source"] == str(zip_path)
            assert "inner_path" in results[0].metadata


class TestRAGPipeline:
    """Test end-to-end RAG pipeline."""
    
    def test_build_index_from_embeddings(self):
        """Test building index from JSONL embeddings."""
        # Create temporary embeddings file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = f.name
            
            for i in range(5):
                record = {
                    "vector": np.random.randn(384).tolist(),
                    "metadata": {
                        "chunk_id": i,
                        "source": "test",
                        "year": 2020 + i,
                    }
                }
                f.write(json.dumps(record) + "\n")
        
        try:
            index = build_index_from_embeddings(temp_path, dim=384)
            
            assert len(index) == 5
            assert index.dim == 384
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_rag_pipeline_retrieve(self):
        """Test RAG retrieval."""
        # Create mock index
        index = VectorIndex(dim=384)
        embedder = get_text_embedder("minilm")
        
        # Add some climate-related embeddings
        texts = [
            "Global temperature increased by 1.1°C since 1880",
            "Arctic sea ice declining at 13% per decade",
            "CO2 levels reached 420 ppm in 2023",
        ]
        
        for i, text in enumerate(texts):
            vec = embedder(text)
            index.add(vec, {
                "chunk_id": i,
                "source": "climate_data",
                "text_summary": text,
            })
        
        # Create mock LLM client
        from src.llm.ollama_client import OllamaClient
        llm = OllamaClient()  # Will use defaults
        
        # Create RAG pipeline
        rag = RAGPipeline(
            index=index,
            text_embedder=embedder,
            llm_client=llm,
            top_k=2,
        )
        
        # Retrieve for query
        query = "What is happening to global temperatures?"
        context = rag.retrieve(query, top_k=2)
        
        assert len(context.results) == 2
        assert context.query == query
        assert len(context.formatted_context) > 0
        # First result should be about temperature (most similar)
        assert "temperature" in context.results[0].metadata["text_summary"].lower()


@pytest.mark.skipif(
    True,  # Skip by default since it requires running Ollama
    reason="Requires Ollama server running"
)
class TestRAGWithLLM:
    """Test RAG with actual LLM (requires Ollama running)."""
    
    def test_rag_ask_with_llm(self):
        """Test full RAG pipeline with LLM answer generation."""
        index = VectorIndex(dim=384)
        embedder = get_text_embedder("minilm")
        
        # Add climate data
        climate_facts = [
            "Global mean temperature in 2023 was 1.15°C above 1850-1900 average",
            "September 2023 was the warmest September on record",
            "Ocean heat content reached record high in 2023",
        ]
        
        for i, fact in enumerate(climate_facts):
            vec = embedder(fact)
            index.add(vec, {"chunk_id": i, "text": fact, "source": "climate_report_2023"})
        
        from src.llm.ollama_client import OllamaClient
        llm = OllamaClient()
        
        rag = RAGPipeline(index=index, text_embedder=embedder, llm_client=llm)
        
        answer = rag.ask("What was the temperature in 2023?")
        
        assert len(answer) > 0
        assert "2023" in answer or "temperature" in answer.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
