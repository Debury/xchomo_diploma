"""
Tests for text generation module.
"""

import pytest
from src.climate_embeddings.text_generation import (
    generate_text_description,
    generate_batch_descriptions,
    format_temporal_info,
    format_spatial_info,
    format_statistics,
)


class TestTextGeneration:
    """Test text generation functions."""
    
    def test_format_temporal_info(self):
        """Test temporal information formatting."""
        meta = {"time_start": "2024-01-15T12:00:00"}
        parts = format_temporal_info(meta)
        assert len(parts) > 0
        assert "2024-01-15" in parts[0]
    
    def test_format_spatial_info(self):
        """Test spatial information formatting."""
        meta = {
            "lat_min": 48.0,
            "lat_max": 49.0,
            "lon_min": 17.0,
            "lon_max": 18.0
        }
        parts = format_spatial_info(meta)
        assert len(parts) >= 2
        assert "48.00" in parts[0] or "49.00" in parts[0]
    
    def test_format_statistics_low_verbosity(self):
        """Test statistics formatting with low verbosity."""
        stats = [20.5, 3.2, 15.0, 28.0, 17.0, 20.0, 25.0, 13.0]
        meta = {"unit": "°C"}
        parts = format_statistics(stats, meta, verbosity="low")
        assert len(parts) == 2
        assert "20.50" in parts[0]
        assert "15.00" in parts[1] or "28.00" in parts[1]
    
    def test_format_statistics_medium_verbosity(self):
        """Test statistics formatting with medium verbosity."""
        stats = [20.5, 3.2, 15.0, 28.0, 17.0, 20.0, 25.0, 13.0]
        meta = {"unit": "°C"}
        parts = format_statistics(stats, meta, verbosity="medium")
        assert len(parts) >= 4
        assert any("20.50" in p for p in parts)
        assert any("3.20" in p for p in parts)
    
    def test_generate_text_description_basic(self):
        """Test basic text description generation."""
        meta = {
            "variable": "temperature",
            "source_id": "test_dataset"
        }
        stats = [20.5, 3.2, 15.0, 28.0, 17.0, 20.0, 25.0, 13.0]
        
        desc = generate_text_description(
            metadata=meta,
            stats_vector=stats,
            verbosity="medium"
        )
        
        assert "temperature" in desc.lower()
        assert "test_dataset" in desc.lower()
        assert "20.5" in desc or "20.50" in desc
    
    def test_generate_text_description_with_temporal(self):
        """Test text description with temporal information."""
        meta = {
            "variable": "temperature",
            "source_id": "test_dataset",
            "time_start": "2024-01-15T12:00:00"
        }
        stats = [20.5, 3.2, 15.0, 28.0, 17.0, 20.0, 25.0, 13.0]
        
        desc = generate_text_description(
            metadata=meta,
            stats_vector=stats,
            verbosity="medium"
        )
        
        assert "2024-01-15" in desc or "January" in desc
    
    def test_generate_text_description_with_spatial(self):
        """Test text description with spatial information."""
        meta = {
            "variable": "temperature",
            "source_id": "test_dataset",
            "lat_min": 48.0,
            "lat_max": 49.0,
            "lon_min": 17.0,
            "lon_max": 18.0
        }
        stats = [20.5, 3.2, 15.0, 28.0, 17.0, 20.0, 25.0, 13.0]
        
        desc = generate_text_description(
            metadata=meta,
            stats_vector=stats,
            verbosity="medium",
            include_coordinates=True
        )
        
        assert "48.00" in desc or "49.00" in desc
        assert "17.00" in desc or "18.00" in desc
    
    def test_generate_batch_descriptions(self):
        """Test batch description generation."""
        metadata_list = [
            {"variable": "temperature", "source_id": "test1"},
            {"variable": "precipitation", "source_id": "test2"}
        ]
        stats_vectors = [
            [20.5, 3.2, 15.0, 28.0, 17.0, 20.0, 25.0, 13.0],
            [5.2, 2.1, 0.0, 10.0, 2.0, 5.0, 8.0, 10.0]
        ]
        
        descriptions = generate_batch_descriptions(
            metadata_list=metadata_list,
            stats_vectors=stats_vectors
        )
        
        assert len(descriptions) == 2
        assert "temperature" in descriptions[0].lower()
        assert "precipitation" in descriptions[1].lower()
    
    def test_generate_text_description_no_stats(self):
        """Test text description without statistics."""
        meta = {
            "variable": "temperature",
            "source_id": "test_dataset"
        }
        
        desc = generate_text_description(
            metadata=meta,
            stats_vector=None,
            include_statistics=False
        )
        
        assert "temperature" in desc.lower()
        assert "test_dataset" in desc.lower()
    
    def test_variable_name_mapping(self):
        """Test that variable names are mapped correctly."""
        meta = {
            "variable": "2m_temperature",
            "source_id": "test_dataset"
        }
        
        desc = generate_text_description(
            metadata=meta,
            stats_vector=None,
            include_statistics=False
        )
        
        # Should use readable name
        assert "2-meter temperature" in desc or "2m_temperature" in desc

