"""
Test the quality of generated embeddings in ChromaDB.

This script validates:
1. Semantic similarity - do similar queries return similar results?
2. Embedding distribution - are embeddings well-distributed?
3. Retrieval accuracy - can we find relevant climate data?
4. Metadata consistency - are metadata fields properly stored?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.embeddings.database import VectorDatabase
from src.embeddings.generator import EmbeddingGenerator

console = Console()


def print_header(title: str):
    """Print formatted header."""
    console.print(f"\n{'='*80}")
    console.print(f"[bold cyan]{title}[/bold cyan]")
    console.print(f"{'='*80}\n")


def test_basic_stats():
    """Test 1: Basic statistics about stored embeddings."""
    print_header("TEST 1: Basic Statistics")
    
    db = VectorDatabase()
    total = db.collection.count()
    
    console.print(f"✓ Total embeddings in ChromaDB: [bold green]{total}[/bold green]")
    
    if total == 0:
        console.print("[red]✗ No embeddings found! Run ETL job first.[/red]")
        return False
    
    # Get all embeddings to analyze
    results = db.collection.get(include=['embeddings', 'metadatas', 'documents'])
    
    # Analyze sources
    sources = {}
    variables = set()
    
    for metadata in results['metadatas']:
        source = metadata.get('source_id', 'unknown')
        var = metadata.get('variable', 'unknown')
        
        sources[source] = sources.get(source, 0) + 1
        variables.add(var)
    
    table = Table(title="Embeddings by Source")
    table.add_column("Source ID", style="cyan")
    table.add_column("Count", style="green", justify="right")
    
    for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        table.add_row(source, str(count))
    
    console.print(table)
    console.print(f"\n✓ Variables: {', '.join(sorted(variables))}")
    
    # Analyze embedding dimensions
    embeddings_list = results.get('embeddings', [])
    if embeddings_list is not None and (isinstance(embeddings_list, list) or isinstance(embeddings_list, np.ndarray)):
        if len(embeddings_list) > 0:
            dim = len(embeddings_list[0])
            console.print(f"✓ Embedding dimension: [bold]{dim}[/bold]")
    
    return True


def test_semantic_similarity():
    """Test 2: Semantic similarity - do similar queries retrieve similar content?"""
    print_header("TEST 2: Semantic Similarity")
    
    db = VectorDatabase()
    generator = EmbeddingGenerator()
    
    # Test queries about climate data
    test_queries = [
        ("temperature air climate", "Should find air temperature data"),
        ("mean average statistical", "Should find statistical summaries"),
        ("kelvin celsius units", "Should find temperature unit conversions"),
        ("spatial temporal dimensions", "Should find dimensional information"),
    ]
    
    console.print("[bold]Testing semantic search with climate-related queries...[/bold]\n")
    
    for query_text, expected in test_queries:
        console.print(f"Query: [cyan]'{query_text}'[/cyan]")
        console.print(f"Expected: {expected}")
        
        # Generate query embedding
        query_embedding = generator.generate_embeddings([query_text])[0].tolist()
        
        # Search
        results = db.query(
            query_embeddings=[query_embedding],
            k=3
        )
        
        if results['ids'] and len(results['ids'][0]) > 0:
            console.print("✓ Top results:")
            for i, (doc_id, distance, doc) in enumerate(zip(
                results['ids'][0][:3],
                results['distances'][0][:3],
                results['documents'][0][:3]
            ), 1):
                # Lower distance = more similar (for cosine: 0 = identical, 2 = opposite)
                similarity = 1 - (distance / 2)  # Convert to 0-1 scale
                console.print(f"  [{i}] Similarity: [green]{similarity:.2%}[/green]")
                console.print(f"      {doc[:100]}...")
        else:
            console.print("[red]✗ No results found[/red]")
        
        console.print()
    
    return True


def test_embedding_distribution():
    """Test 3: Embedding distribution - are vectors well-distributed?"""
    print_header("TEST 3: Embedding Distribution")
    
    db = VectorDatabase()
    results = db.collection.get(include=['embeddings'])
    
    embeddings_list = results.get('embeddings', [])
    if embeddings_list is None or len(embeddings_list) < 2:
        console.print("[yellow]⚠ Not enough embeddings to analyze distribution[/yellow]")
        return True
    
    embeddings = np.array(embeddings_list)
    
    # Calculate statistics
    console.print("[bold]Statistical Analysis:[/bold]\n")
    
    # Mean and std per dimension
    mean = embeddings.mean(axis=0)
    std = embeddings.std(axis=0)
    
    console.print(f"✓ Mean across dimensions:")
    console.print(f"  Average: {mean.mean():.4f}")
    console.print(f"  Std dev: {mean.std():.4f}")
    
    console.print(f"\n✓ Std dev across dimensions:")
    console.print(f"  Average: {std.mean():.4f}")
    console.print(f"  Min: {std.min():.4f}, Max: {std.max():.4f}")
    
    # Norms (should be ~1 for normalized embeddings)
    norms = np.linalg.norm(embeddings, axis=1)
    console.print(f"\n✓ Vector norms (should be ~1 if normalized):")
    console.print(f"  Average: {norms.mean():.4f}")
    console.print(f"  Std dev: {norms.std():.4f}")
    console.print(f"  Range: [{norms.min():.4f}, {norms.max():.4f}]")
    
    # Pairwise similarity
    if len(embeddings) >= 2:
        # Calculate cosine similarity between all pairs
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(embeddings)
        
        # Get upper triangle (excluding diagonal)
        upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        
        console.print(f"\n✓ Pairwise cosine similarity:")
        console.print(f"  Average: {upper_tri.mean():.4f}")
        console.print(f"  Std dev: {upper_tri.std():.4f}")
        console.print(f"  Range: [{upper_tri.min():.4f}, {upper_tri.max():.4f}]")
        
        # Check for duplicates (very high similarity)
        duplicates = np.sum(upper_tri > 0.99)
        if duplicates > 0:
            console.print(f"  [yellow]⚠ Warning: {duplicates} pairs with >99% similarity (possible duplicates)[/yellow]")
        else:
            console.print(f"  ✓ No duplicate embeddings detected")
    
    return True


def test_retrieval_accuracy():
    """Test 4: Retrieval accuracy - can we find the right data?"""
    print_header("TEST 4: Retrieval Accuracy")
    
    db = VectorDatabase()
    generator = EmbeddingGenerator()
    
    # Get all data first
    all_results = db.collection.get(include=['embeddings', 'metadatas', 'documents'])
    
    ids_list = all_results.get('ids', [])
    if ids_list is None or len(ids_list) < 2:
        console.print("[yellow]⚠ Not enough embeddings for retrieval test[/yellow]")
        return True
    
    console.print(f"[bold]Testing retrieval with actual stored documents...[/bold]\n")
    
    # Test: Use partial document text as query, should retrieve the same document
    test_indices = min(3, len(all_results['documents']))
    
    for i in range(test_indices):
        original_doc = all_results['documents'][i]
        original_id = all_results['ids'][i]
        
        # Use first 50 chars as query
        query_text = original_doc[:50]
        
        console.print(f"Query (from doc {i+1}): [cyan]{query_text}...[/cyan]")
        
        # Generate embedding and search
        query_embedding = generator.generate_embeddings([query_text])[0].tolist()
        results = db.query(
            query_embeddings=[query_embedding],
            k=1
        )
        
        if results['ids'] and len(results['ids'][0]) > 0:
            retrieved_id = results['ids'][0][0]
            distance = results['distances'][0][0]
            similarity = 1 - (distance / 2)
            
            if retrieved_id == original_id:
                console.print(f"  ✓ [green]CORRECT[/green] - Retrieved original document")
                console.print(f"    Similarity: {similarity:.2%}")
            else:
                console.print(f"  ✗ [yellow]WRONG[/yellow] - Retrieved different document")
                console.print(f"    Expected: {original_id}")
                console.print(f"    Got: {retrieved_id}")
                console.print(f"    Similarity: {similarity:.2%}")
        else:
            console.print(f"  ✗ [red]ERROR[/red] - No results returned")
        
        console.print()
    
    return True


def test_metadata_quality():
    """Test 5: Metadata quality and consistency."""
    print_header("TEST 5: Metadata Quality")
    
    db = VectorDatabase()
    results = db.collection.get(include=['metadatas', 'documents'])
    
    metadatas_list = results.get('metadatas', [])
    if metadatas_list is None or len(metadatas_list) == 0:
        console.print("[red]✗ No metadata found[/red]")
        return False
    
    console.print("[bold]Checking metadata fields...[/bold]\n")
    
    # Expected fields
    expected_fields = ['source_id', 'variable', 'text', 'timestamp']
    
    field_coverage = {field: 0 for field in expected_fields}
    total = len(results['metadatas'])
    
    for metadata in results['metadatas']:
        for field in expected_fields:
            if field in metadata and metadata[field]:
                field_coverage[field] += 1
    
    table = Table(title="Metadata Field Coverage")
    table.add_column("Field", style="cyan")
    table.add_column("Coverage", style="green")
    table.add_column("Status", style="yellow")
    
    for field, count in field_coverage.items():
        percentage = (count / total) * 100 if total > 0 else 0
        status = "✓" if percentage == 100 else "⚠"
        table.add_row(field, f"{count}/{total} ({percentage:.0f}%)", status)
    
    console.print(table)
    
    # Check document-metadata consistency
    console.print(f"\n[bold]Document-Metadata Consistency:[/bold]")
    
    consistent = 0
    for metadata, document in zip(results['metadatas'], results['documents']):
        # Check if 'text' in metadata matches document
        if 'text' in metadata and metadata['text'] == document:
            consistent += 1
    
    consistency_rate = (consistent / total) * 100 if total > 0 else 0
    console.print(f"  Documents matching metadata text: {consistent}/{total} ({consistency_rate:.0f}%)")
    
    if consistency_rate == 100:
        console.print(f"  ✓ [green]Perfect consistency[/green]")
    elif consistency_rate > 80:
        console.print(f"  ⚠ [yellow]Good consistency[/yellow]")
    else:
        console.print(f"  ✗ [red]Poor consistency - investigate![/red]")
    
    return True


def generate_quality_report():
    """Generate overall quality score and recommendations."""
    print_header("OVERALL QUALITY ASSESSMENT")
    
    db = VectorDatabase()
    total = db.collection.count()
    
    if total == 0:
        console.print(Panel(
            "[red]No embeddings found in database.[/red]\n"
            "Please run the dynamic_source_etl_job to generate embeddings.",
            title="Quality Assessment Failed",
            border_style="red"
        ))
        return
    
    results = db.collection.get(include=['embeddings', 'metadatas'])
    
    # Calculate quality metrics
    embeddings_list = results.get('embeddings', [])
    quality_metrics = {
        "Total Embeddings": total,
        "Embedding Dimension": len(embeddings_list[0]) if embeddings_list is not None and len(embeddings_list) > 0 else 0,
        "Unique Sources": len(set(m.get('source_id', '') for m in results['metadatas'])),
        "Metadata Completeness": sum(
            1 for m in results['metadatas'] 
            if all(k in m for k in ['source_id', 'variable', 'text'])
        ) / total * 100,
    }
    
    # Calculate distribution quality
    if embeddings_list is not None and len(embeddings_list) >= 2:
        embeddings = np.array(embeddings_list)
        norms = np.linalg.norm(embeddings, axis=1)
        norm_std = norms.std()
        
        # Good if normalized (std close to 0)
        normalization_quality = 100 if norm_std < 0.1 else max(0, 100 - norm_std * 100)
        quality_metrics["Normalization Quality"] = normalization_quality
    
    # Display metrics
    table = Table(title="Quality Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    
    for metric, value in quality_metrics.items():
        if isinstance(value, float):
            table.add_row(metric, f"{value:.1f}%")
        else:
            table.add_row(metric, str(value))
    
    console.print(table)
    
    # Overall assessment
    console.print("\n[bold]Recommendations:[/bold]\n")
    
    if total < 10:
        console.print("  ⚠ [yellow]Low embedding count - consider processing more sources[/yellow]")
    else:
        console.print("  ✓ [green]Good embedding coverage[/green]")
    
    if quality_metrics["Metadata Completeness"] < 100:
        console.print("  ⚠ [yellow]Some metadata fields missing - check ETL pipeline[/yellow]")
    else:
        console.print("  ✓ [green]Complete metadata[/green]")
    
    if "Normalization Quality" in quality_metrics and quality_metrics["Normalization Quality"] > 90:
        console.print("  ✓ [green]Well-normalized embeddings[/green]")
    
    # Final verdict
    console.print()
    if total >= 5 and quality_metrics["Metadata Completeness"] >= 80:
        console.print(Panel(
            "[bold green]Embeddings are of good quality and ready for RAG usage![/bold green]",
            title="Quality Check Passed",
            border_style="green"
        ))
    else:
        console.print(Panel(
            "[bold yellow]Embeddings need improvement. Review recommendations above.[/bold yellow]",
            title="Quality Check - Needs Attention",
            border_style="yellow"
        ))


def main():
    """Run all quality tests."""
    console.print(Panel(
        "[bold cyan]ChromaDB Embedding Quality Test Suite[/bold cyan]\n"
        "Testing semantic quality, distribution, and retrieval accuracy",
        title="Embedding Quality Assessment",
        border_style="cyan"
    ))
    
    try:
        # Run tests
        test_basic_stats()
        test_semantic_similarity()
        test_embedding_distribution()
        test_retrieval_accuracy()
        test_metadata_quality()
        
        # Generate report
        generate_quality_report()
        
    except Exception as e:
        console.print(f"\n[red]Error during testing: {e}[/red]")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
