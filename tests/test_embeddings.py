"""
Simplified and functional tests for Phase 3: Embedding Generation & Vector Database
These tests are designed to work with the actual implementation.
"""

import pytest
import numpy as np
import xarray as xr
from pathlib import Path
import tempfile
import shutil
import warnings

warnings.filterwarnings('ignore')

from src.embeddings.generator import EmbeddingGenerator
from src.embeddings.database import VectorDatabase
from src.embeddings.search import SemanticSearcher, semantic_search
from src.embeddings.pipeline import EmbeddingPipeline


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    try:
        shutil.rmtree(temp_path)
    except:
        pass


@pytest.fixture
def sample_netcdf(temp_dir):
    """Create sample NetCDF file"""
    import pandas as pd
    
    file_path = temp_dir / "sample.nc"
    
    ds = xr.Dataset({
        't2m': xr.DataArray(
            np.random.randn(10, 5, 5) * 10 + 15,  # Temperature-like data
            dims=['time', 'latitude', 'longitude'],
            coords={
                'time': pd.date_range('2020-01-01', periods=10),
                'latitude': np.linspace(45, 55, 5),
                'longitude': np.linspace(10, 20, 5)
            },
            attrs={'units': '°C', 'long_name': '2 meter temperature'}
        )
    })
    ds.to_netcdf(file_path)
    return file_path


# ============================================================================
# EmbeddingGenerator Tests
# ============================================================================

class TestEmbeddingGenerator:
    """Tests for EmbeddingGenerator class"""
    
    def test_init_default(self):
        """Test initialization with defaults"""
        generator = EmbeddingGenerator()
        assert generator.model_name == 'all-MiniLM-L6-v2'
        assert generator.model is not None
    
    def test_generate_single_embedding(self):
        """Test generating single embedding"""
        generator = EmbeddingGenerator()
        text = "Temperature data from climate dataset"
        embedding = generator.generate_single_embedding(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
    
    def test_generate_batch_embeddings(self):
        """Test generating batch of embeddings"""
        generator = EmbeddingGenerator()
        texts = [
            "Temperature data",
            "Precipitation data",
            "Wind speed data"
        ]
        embeddings = generator.generate_embeddings(texts)
        
        assert embeddings.shape == (3, 384)
        assert not np.isnan(embeddings).any()
    
    def test_empty_input_raises_error(self):
        """Test that empty input raises error"""
        generator = EmbeddingGenerator()
        with pytest.raises(ValueError):
            generator.generate_embeddings([])
    
    def test_embedding_dimension(self):
        """Test getting embedding dimension"""
        generator = EmbeddingGenerator()
        dim = generator.get_embedding_dimension()
        assert dim == 384


# ============================================================================
# VectorDatabase Tests
# ============================================================================

class TestVectorDatabase:
    """Tests for VectorDatabase class"""
    
    def test_init(self):
        """Test database initialization"""
        db = VectorDatabase(collection_name="test_collection", auto_connect=False)
        assert db.collection_name == "test_collection"
        assert db._backend == "memory"
    
    def test_add_and_count(self):
        """Test adding embeddings and counting"""
        db = VectorDatabase(collection_name="test_add", auto_connect=False)
        
        embeddings = np.random.randn(3, 384).astype(np.float32)
        ids = ["doc1", "doc2", "doc3"]
        documents = ["text1", "text2", "text3"]
        
        db.add_embeddings(ids=ids, embeddings=embeddings, documents=documents)
        
        count = db.count()
        assert count == 3
    
    def test_query(self):
        """Test querying embeddings"""
        db = VectorDatabase(collection_name="test_query", auto_connect=False)
        
        # Add data
        embeddings = np.random.randn(5, 384).astype(np.float32)
        ids = [f"doc{i}" for i in range(5)]
        documents = [f"text{i}" for i in range(5)]
        
        db.add_embeddings(ids=ids, embeddings=embeddings, documents=documents)
        
        # Query
        query_emb = np.random.randn(384).astype(np.float32)
        results = db.query(query_embeddings=query_emb, k=2)
        
        assert 'ids' in results
        assert len(results['ids'][0]) <= 2


# ============================================================================
# SemanticSearcher Tests
# ============================================================================

class TestSemanticSearcher:
    """Tests for SemanticSearcher class"""
    
    def test_search_basic(self):
        """Test basic semantic search"""
        # Use fresh database for this test
        searcher = SemanticSearcher(
            database=VectorDatabase(collection_name="test_search_basic")
        )
        
        # Add some test data
        texts = [
            "Temperature measurements from weather station",
            "Precipitation data from rain gauges",
            "Wind speed observations"
        ]
        embeddings = searcher.generator.generate_embeddings(texts)
        ids = [f"doc{i}" for i in range(len(texts))]
        
        searcher.database.add_embeddings(
            ids=ids,
            embeddings=embeddings,
            documents=texts
        )
        
        # Search
        results = searcher.search("temperature", k=2)
        
        assert len(results) > 0
        assert 'id' in results[0]
        assert 'similarity' in results[0]


# ============================================================================
# EmbeddingPipeline Tests
# ============================================================================

class TestEmbeddingPipeline:
    """Tests for EmbeddingPipeline class"""
    
    def test_init(self):
        """Test pipeline initialization"""
        pipeline = EmbeddingPipeline()
        assert pipeline.generator is not None
        assert pipeline.database is not None
    
    def test_process_single_dataset(self, sample_netcdf):
        """Test processing a single dataset"""
        pipeline = EmbeddingPipeline(
            database=VectorDatabase(collection_name="test_pipeline_single")
        )
        
        result = pipeline.process_dataset(str(sample_netcdf))
        
        assert 'num_embeddings' in result
        assert result['num_embeddings'] > 0
    
    def test_process_directory(self, temp_dir, sample_netcdf):
        """Test processing directory of files"""
        # Copy sample file to temp dir
        test_file = temp_dir / "test.nc"
        shutil.copy(sample_netcdf, test_file)
        
        pipeline = EmbeddingPipeline(
            database=VectorDatabase(collection_name="test_pipeline_dir")
        )
        
        result = pipeline.process_directory(str(temp_dir), pattern="*.nc")
        
        assert 'num_files' in result
        assert result['num_files'] > 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for full workflow"""
    
    def test_end_to_end_workflow(self, temp_dir, sample_netcdf):
        """Test complete workflow from data to search"""
        # Copy sample file
        test_file = temp_dir / "climate_data.nc"
        shutil.copy(sample_netcdf, test_file)
        
        # Create pipeline with unique database
        pipeline = EmbeddingPipeline(
            database=VectorDatabase(collection_name="test_integration_e2e")
        )
        
        # Process data
        pipeline.process_directory(str(temp_dir))
        
        # Create searcher with same database
        searcher = SemanticSearcher(database=pipeline.database)
        
        # Search
        results = searcher.search("temperature climate", k=5)
        
        assert len(results) > 0


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling"""
    
    def test_invalid_file_path(self):
        """Test handling invalid file path"""
        pipeline = EmbeddingPipeline()
        
        with pytest.raises(Exception):
            pipeline.process_dataset("nonexistent_file.nc")
    
    def test_empty_text_list(self):
        """Test handling empty text list"""
        generator = EmbeddingGenerator()
        
        with pytest.raises(ValueError):
            generator.generate_embeddings([])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.embeddings", "--cov-report=term"])
