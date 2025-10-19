"""
Test script for complete source-to-ETL flow

Tests:
1. API health check
2. Create new source via API
3. List sources
4. Trigger ETL job for source
5. Monitor job status
6. Check embeddings
7. Update source
8. Delete source
"""

import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Configuration
API_BASE_URL = "http://localhost:8000"
DAGSTER_URL = "http://localhost:3000"

# Test source data
TEST_SOURCE = {
    "source_id": "xarray_air_temp",
    "url": "https://github.com/pydata/xarray-data/raw/master/air_temperature.nc",
    "description": "Sample air temperature data from xarray tutorial (format auto-detected)",
    "is_active": True,
    "tags": ["xarray", "air_temp", "tutorial", "test"]
}


def print_section(title: str):
    """Print section header"""
    console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
    console.print(f"[bold cyan]{title}[/bold cyan]")
    console.print(f"[bold cyan]{'='*80}[/bold cyan]\n")


def print_success(message: str):
    """Print success message"""
    console.print(f"[bold green]✓[/bold green] {message}")


def print_error(message: str):
    """Print error message"""
    console.print(f"[bold red]✗[/bold red] {message}")


def print_info(message: str):
    """Print info message"""
    console.print(f"[bold blue]ℹ[/bold blue] {message}")


def print_json(data: dict, title: str = "Response"):
    """Pretty print JSON data"""
    console.print(Panel(
        json.dumps(data, indent=2),
        title=title,
        border_style="blue"
    ))


async def test_api_health():
    """Test 1: Check API health"""
    print_section("TEST 1: API Health Check")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/health")
            
            if response.status_code == 200:
                data = response.json()
                print_success("API is healthy")
                print_json(data, "Health Status")
                return True
            else:
                print_error(f"API returned status {response.status_code}")
                return False
    except Exception as e:
        print_error(f"Cannot connect to API: {e}")
        return False


async def test_create_source():
    """Test 2: Create new source"""
    print_section("TEST 2: Create Source")
    
    try:
        print_info(f"Creating source: {TEST_SOURCE['source_id']}")
        
        async with httpx.AsyncClient() as client:
            # First, try to delete if exists
            try:
                await client.delete(f"{API_BASE_URL}/sources/{TEST_SOURCE['source_id']}?hard_delete=true")
                print_info("Deleted existing source")
            except:
                pass
            
            # Create new source
            response = await client.post(
                f"{API_BASE_URL}/sources",
                json=TEST_SOURCE,
                timeout=10.0
            )
            
            if response.status_code == 201 or response.status_code == 200:
                data = response.json()
                print_success("Source created successfully")
                print_json(data, "Created Source")
                return True
            else:
                print_error(f"Failed to create source: {response.status_code}")
                print_error(response.text)
                return False
    except Exception as e:
        print_error(f"Error creating source: {e}")
        return False


async def test_list_sources():
    """Test 3: List all sources"""
    print_section("TEST 3: List Sources")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/sources")
            
            if response.status_code == 200:
                sources = response.json()
                print_success(f"Found {len(sources)} source(s)")
                
                table = Table(title="Climate Data Sources")
                table.add_column("ID", style="cyan")
                table.add_column("URL", style="green", max_width=40)
                table.add_column("Format", style="yellow")
                table.add_column("Active", style="magenta")
                table.add_column("Variables", style="blue")
                
                for source in sources:
                    table.add_row(
                        source.get("source_id", "N/A"),
                        source.get("url", "N/A")[:40] + "..." if len(source.get("url", "")) > 40 else source.get("url", "N/A"),
                        source.get("format", "N/A"),
                        "✓" if source.get("is_active") else "✗",
                        ", ".join(source.get("variables", [])[:3]) if source.get("variables") else "All"
                    )
                
                console.print(table)
                return True
            else:
                print_error(f"Failed to list sources: {response.status_code}")
                return False
    except Exception as e:
        print_error(f"Error listing sources: {e}")
        return False


async def test_trigger_etl():
    """Test 4: Trigger ETL job for source"""
    print_section("TEST 4: Trigger ETL Job")
    
    try:
        print_info(f"Triggering DYNAMIC SOURCE ETL (processes all active sources)")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE_URL}/sources/{TEST_SOURCE['source_id']}/trigger?job_name=dynamic_source_etl_job",
                json={},
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                print_success("ETL job triggered successfully")
                print_json(data, "Run Information")
                return data.get("run_id")
            else:
                print_error(f"Failed to trigger ETL: {response.status_code}")
                print_error(response.text)
                return None
    except Exception as e:
        print_error(f"Error triggering ETL: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_monitor_job(run_id: str, max_wait: int = 60):
    """Test 5: Monitor job status"""
    print_section("TEST 5: Monitor Job Status")
    
    if not run_id:
        print_error("No run_id provided")
        return False
    
    print_info(f"Monitoring run: {run_id}")
    print_info(f"Check Dagster UI: {DAGSTER_URL}")
    
    start_time = time.time()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Waiting for job...", total=None)
        
        try:
            async with httpx.AsyncClient() as client:
                while time.time() - start_time < max_wait:
                    response = await client.get(
                        f"{API_BASE_URL}/runs/{run_id}/status",
                        timeout=10.0
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        status = data.get("status", "UNKNOWN")
                        
                        progress.update(task, description=f"Status: {status}")
                        
                        if status in ["SUCCESS", "FAILURE", "CANCELED"]:
                            if status == "SUCCESS":
                                print_success(f"Job completed successfully")
                            else:
                                print_error(f"Job ended with status: {status}")
                            
                            print_json(data, "Final Status")
                            return status == "SUCCESS"
                    
                    await asyncio.sleep(2)
                
                print_error(f"Job monitoring timed out after {max_wait}s")
                return False
        except Exception as e:
            print_error(f"Error monitoring job: {e}")
            return False


async def test_embeddings_stats():
    """Test 6: Check embeddings statistics"""
    print_section("TEST 6: Embeddings Statistics")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/embeddings/stats")
            
            if response.status_code == 200:
                data = response.json()
                print_success("Retrieved embeddings statistics")
                print_json(data, "Embeddings Stats")
                return True
            else:
                print_error(f"Failed to get embeddings stats: {response.status_code}")
                print_info("This is OK if embeddings haven't been generated yet")
                return False
    except Exception as e:
        print_error(f"Error getting embeddings: {e}")
        return False


async def test_update_source():
    """Test 7: Update source"""
    print_section("TEST 7: Update Source")
    
    try:
        update_data = {
            "description": "UPDATED: Sample air temperature data from xarray tutorial",
            "tags": ["xarray", "air_temp", "tutorial", "test_updated"]
        }
        
        print_info(f"Updating source: {TEST_SOURCE['source_id']}")
        
        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{API_BASE_URL}/sources/{TEST_SOURCE['source_id']}",
                json=update_data,
                timeout=10.0
            )
            
            if response.status_code == 200:
                data = response.json()
                print_success("Source updated successfully")
                print_json(data, "Updated Source")
                return True
            else:
                print_error(f"Failed to update source: {response.status_code}")
                return False
    except Exception as e:
        print_error(f"Error updating source: {e}")
        return False


async def test_delete_source():
    """Test 8: Delete source (soft delete)"""
    print_section("TEST 8: Delete Source")
    
    try:
        print_info(f"Soft deleting source: {TEST_SOURCE['source_id']}")
        
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{API_BASE_URL}/sources/{TEST_SOURCE['source_id']}",
                timeout=10.0
            )
            
            if response.status_code == 204:
                print_success("Source soft deleted (deactivated)")
                return True
            else:
                print_error(f"Failed to delete source: {response.status_code}")
                return False
    except Exception as e:
        print_error(f"Error deleting source: {e}")
        return False


async def run_all_tests():
    """Run all tests"""
    console.print(Panel(
        "[bold cyan]Climate ETL Source Flow Test Suite[/bold cyan]\n"
        "Testing complete workflow from source creation to ETL execution",
        border_style="cyan"
    ))
    
    results = {
        "API Health": False,
        "Create Source": False,
        "List Sources": False,
        "Trigger ETL": False,
        "Monitor Job": False,
        "Embeddings Stats": False,
        "Update Source": False,
        "Delete Source": False
    }
    
    # Test 1: API Health
    results["API Health"] = await test_api_health()
    if not results["API Health"]:
        print_error("API is not available. Make sure to run: python -m uvicorn web_api.main:app --reload")
        return
    
    # Test 2: Create Source
    results["Create Source"] = await test_create_source()
    if not results["Create Source"]:
        print_error("Cannot proceed without creating source")
        return
    
    # Test 3: List Sources
    results["List Sources"] = await test_list_sources()
    
    # Test 4: Trigger ETL
    run_id = await test_trigger_etl()
    results["Trigger ETL"] = run_id is not None
    
    # Test 5: Monitor Job (optional - may timeout)
    if run_id:
        results["Monitor Job"] = await test_monitor_job(run_id, max_wait=30)
    
    # Test 6: Embeddings Stats
    results["Embeddings Stats"] = await test_embeddings_stats()
    
    # Test 7: Update Source
    results["Update Source"] = await test_update_source()
    
    # Test 8: Delete Source
    results["Delete Source"] = await test_delete_source()
    
    # Print summary
    print_section("TEST SUMMARY")
    
    table = Table(title="Test Results")
    table.add_column("Test", style="cyan")
    table.add_column("Status", style="bold")
    
    for test_name, passed in results.items():
        status = "[green]PASSED ✓[/green]" if passed else "[red]FAILED ✗[/red]"
        table.add_row(test_name, status)
    
    console.print(table)
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    if passed_count == total_count:
        console.print(f"\n[bold green]All {total_count} tests passed! 🎉[/bold green]")
    else:
        console.print(f"\n[bold yellow]{passed_count}/{total_count} tests passed[/bold yellow]")


if __name__ == "__main__":
    import asyncio
    
    try:
        asyncio.run(run_all_tests())
    except KeyboardInterrupt:
        print_info("\nTests interrupted by user")
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
