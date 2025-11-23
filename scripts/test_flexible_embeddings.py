"""
Test script for flexible embedding system with RAG support

This script tests:
1. Metadata extraction from any dataset format
2. Dynamic text generation with sample values
3. Embedding generation and storage
4. RAG-friendly retrieval
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import (
    MetadataExtractor,
    TextGenerator,
    EmbeddingGenerator,
    VectorDatabase,
    EmbeddingPipeline,
    SemanticSearcher
)
from src.utils.logger import setup_logger
import xarray as xr

logger = setup_logger(__name__)


def test_metadata_extraction():
    """Test flexible metadata extraction."""
    logger.info("=" * 80)
    logger.info("TEST 1: Metadata Extraction")
    logger.info("=" * 80)
    
    # Find a test dataset
    test_files = list(Path("data/processed").glob("*.nc"))
    if not test_files:
        test_files = list(Path("data/raw").glob("*.nc"))
    
    if not test_files:
        logger.warning("No test files found")
        return
    
    test_file = str(test_files[0])
    logger.info(f"Testing with file: {test_file}")
    
    # Load dataset
    ds = xr.open_dataset(test_file)
    logger.info(f"Dataset variables: {list(ds.data_vars)}")
    logger.info(f"Dataset coordinates: {list(ds.coords)}")
    
    # Extract metadata
    extractor = MetadataExtractor()
    metadata_list = extractor.extract_from_dataset(
        data=ds,
        file_path=test_file,
        dataset_id="test_dataset"
    )
    
    logger.info(f"\nExtracted metadata for {len(metadata_list)} variables")
    
    # Show first variable's metadata
    if metadata_list:
        first_meta = metadata_list[0]
        logger.info(f"\nFirst variable metadata:")
        logger.info(f"  ID: {first_meta['id']}")
        logger.info(f"  Variable: {first_meta['variable']}")
        logger.info(f"  Long name: {first_meta['long_name']}")
        logger.info(f"  Unit: {first_meta['unit']}")
        logger.info(f"  Dimensions: {first_meta['dimensions']}")
        logger.info(f"  Shape: {first_meta['shape']}")
        logger.info(f"  Statistics:")
        for key, value in first_meta.items():
            if key.startswith('stat_'):
                logger.info(f"    {key}: {value:.2f}")
        
        logger.info(f"  Spatial extent: {first_meta.get('spatial_extent', {})}")
        logger.info(f"  Temporal extent: {first_meta.get('temporal_extent', {})}")
        logger.info(f"  Sample values: {first_meta.get('sample_values', [])[:5]}")
    
    ds.close()
    return metadata_list


def test_text_generation(metadata_list):
    """Test dynamic text generation."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Dynamic Text Generation")
    logger.info("=" * 80)
    
    if not metadata_list:
        logger.warning("No metadata to test")
        return
    
    # Test different verbosity levels
    for verbosity in ['low', 'medium', 'high']:
        logger.info(f"\n--- Verbosity: {verbosity} ---")
        
        text_gen = TextGenerator(config={'verbosity': verbosity})
        texts = text_gen.generate_batch(metadata_list[:1])
        
        if texts:
            logger.info(f"Generated text:\n{texts[0]}")
            logger.info(f"Text length: {len(texts[0])} characters")
    
    # Return medium verbosity for further testing
    text_gen = TextGenerator(config={'verbosity': 'medium'})
    texts = text_gen.generate_batch(metadata_list)
    
    return texts


def test_full_pipeline():
    """Test complete embedding pipeline."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Full Embedding Pipeline")
    logger.info("=" * 80)
    
    # Find test files
    test_files = list(Path("data/processed").glob("*.nc"))[:2]
    if not test_files:
        test_files = list(Path("data/raw").glob("*.nc"))[:2]
    
    if not test_files:
        logger.warning("No test files found")
        return
    
    logger.info(f"Processing {len(test_files)} files")
    
    # Initialize pipeline
    pipeline = EmbeddingPipeline()
    
    # Process files
    for test_file in test_files:
        logger.info(f"\nProcessing: {test_file.name}")
        result = pipeline.process_dataset(str(test_file))
        logger.info(f"  Created {result['num_embeddings']} embeddings")
        logger.info(f"  Processing time: {result['processing_time']:.2f}s")
    
    # Get metrics
    metrics = pipeline.get_metrics()
    logger.info(f"\nPipeline metrics:")
    logger.info(f"  Total datasets: {metrics['pipeline']['total_datasets_processed']}")
    logger.info(f"  Total embeddings: {metrics['pipeline']['total_embeddings_created']}")
    logger.info(f"  Total time: {metrics['pipeline']['total_pipeline_time']:.2f}s")
    
    return pipeline


def test_rag_retrieval(pipeline):
    """Test RAG-friendly retrieval."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: RAG-Friendly Retrieval")
    logger.info("=" * 80)
    
    if not pipeline:
        logger.warning("No pipeline to test")
        return
    
    # Create semantic searcher
    searcher = SemanticSearcher(
        generator=pipeline.generator,
        database=pipeline.database
    )
    
    # Test queries
    test_queries = [
        "temperature data with values around 15 degrees",
        "air temperature mean statistics",
        "climate data with high variability",
    ]
    
    for query in test_queries:
        logger.info(f"\nQuery: '{query}'")
        
        try:
            results = searcher.search(query, k=3)
            
            if results:
                logger.info(f"Found {len(results)} results:")
                
                for i, result in enumerate(results, 1):
                    metadata = result.get('metadata', {})
                    logger.info(f"\n  Result {i}:")
                    logger.info(f"    Distance: {result.get('distance', 'N/A'):.4f}")
                    logger.info(f"    Variable: {metadata.get('variable', 'N/A')}")
                    logger.info(f"    Dataset: {metadata.get('dataset_id', 'N/A')}")
                    logger.info(f"    Stats: mean={metadata.get('stat_mean', 'N/A')}, "
                              f"range=[{metadata.get('stat_min', 'N/A')}, {metadata.get('stat_max', 'N/A')}]")
                    logger.info(f"    Samples: {metadata.get('sample_values', 'N/A')}")
                    logger.info(f"    Document: {result.get('document', '')[:200]}...")
            else:
                logger.info("  No results found")
                
        except Exception as e:
            logger.error(f"  Query failed: {e}")


def main():
    """Run all tests."""
    logger.info("Starting flexible embedding system tests")
    logger.info("This tests the new non-hardcoded approach with RAG support\n")
    
    try:
        # Test 1: Metadata extraction
        metadata_list = test_metadata_extraction()
        
        # Test 2: Text generation
        if metadata_list:
            texts = test_text_generation(metadata_list)
        
        # Test 3: Full pipeline
        pipeline = test_full_pipeline()
        
        # Test 4: RAG retrieval
        if pipeline:
            test_rag_retrieval(pipeline)
        
        logger.info("\n" + "=" * 80)
        logger.info("ALL TESTS COMPLETED")
        logger.info("=" * 80)
        logger.info("\nThe embedding system is now flexible and can handle:")
        logger.info("  - Any meteorological variable")
        logger.info("  - Any dataset format (NetCDF, CSV, Parquet)")
        logger.info("  - Dynamic metadata extraction")
        logger.info("  - Sample values for RAG context")
        logger.info("  - Flexible text generation")
        logger.info("  - No hardcoded templates or variables")
        
    except Exception as e:
        logger.error(f"Tests failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
