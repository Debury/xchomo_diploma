"""
Script to create a demo source for Slovakia summer 2023 data.
This will add the source to the database and prepare it for processing.
"""

import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"  # Change to your server URL if different

# Path to the CSV file (relative to project root)
CSV_PATH = "data/raw/slovakia_summer_2023_sample.csv"
CSV_ABSOLUTE_PATH = Path(__file__).parent / CSV_PATH

def create_demo_source():
    """Create a source for the demo Slovakia data."""
    
    # Check if file exists
    if not CSV_ABSOLUTE_PATH.exists():
        print(f"Error: CSV file not found at {CSV_ABSOLUTE_PATH}")
        print("Please make sure the file exists before running this script.")
        return
    
    print(f"Creating demo source for: {CSV_PATH}")
    print(f"File exists: {CSV_ABSOLUTE_PATH.exists()}")
    print(f"File size: {CSV_ABSOLUTE_PATH.stat().st_size} bytes")
    
    # Source configuration
    source_data = {
        "source_id": "slovakia_summer_2023",
        "url": f"file://{CSV_ABSOLUTE_PATH.absolute()}",  # Use file:// protocol for local files
        "format": "csv",
        "description": "Sample climate data for Slovakia - Summer 2023 (June-August). Includes temperature (TMAX, TMIN, TAVG), precipitation (PRCP), humidity (hurs), wind (AWND, WSF2, WSF5), and snow (SNOW) data for 3 stations: Bratislava, Kosice, and Zilina.",
        "tags": ["slovakia", "summer", "2023", "demo", "temperature", "precipitation"],
        "variables": ["TMAX", "TMIN", "TAVG", "PRCP", "SNOW", "hurs", "AWND", "WSF2", "WSF5"],
        "spatial_bbox": [48.0, 17.0, 49.5, 22.5],  # Slovakia bounding box [min_lat, min_lon, max_lat, max_lon]
        "time_range": {
            "start": "2023-06-01",
            "end": "2023-08-31"
        },
        "is_active": True,
        "embedding_model": "all-MiniLM-L6-v2"
    }
    
    try:
        # Create the source
        print(f"\nCreating source via POST {BASE_URL}/sources")
        response = requests.post(
            f"{BASE_URL}/sources",
            json=source_data,
            timeout=30
        )
        
        if response.status_code == 200 or response.status_code == 201:
            result = response.json()
            print(f"\n✅ Source created successfully!")
            print(f"Source ID: {result.get('source_id', 'N/A')}")
            print(f"Status: {result.get('processing_status', 'N/A')}")
            print(f"\nNext steps:")
            print(f"1. Go to Dagster UI (http://localhost:3000)")
            print(f"2. Run the 'dynamic_source_etl_job' with source_id='slovakia_summer_2023'")
            print(f"3. Wait for processing to complete")
            print(f"4. Test with questions like:")
            print(f"   - 'Ukáž mi štatistiky teploty pre Slovensko v lete 2023'")
            print(f"   - 'What is the average temperature in Slovakia in summer 2023?'")
            print(f"   - 'Compare temperature between Bratislava and Kosice'")
        else:
            print(f"\n❌ Error creating source:")
            print(f"Status code: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Error: Could not connect to {BASE_URL}")
        print("Make sure the API server is running!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    create_demo_source()

