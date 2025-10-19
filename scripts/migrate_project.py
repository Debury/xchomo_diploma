"""
Migration Script - Move files from old structure to new structure
Run this after verifying the new structure works correctly.
"""

import shutil
from pathlib import Path

def migrate_project():
    """Migrate files from old structure to new structure."""
    
    print("=" * 80)
    print("CLIMATE DATA ETL PIPELINE - MIGRATION SCRIPT")
    print("=" * 80)
    print()
    
    base_dir = Path(__file__).parent
    
    # Define migrations
    migrations = [
        # Data files
        ("era5-download/test_era5_data.nc", "data/raw/test_era5_data.nc"),
        ("phase2_transformation/processed/test_era5_data.nc", "data/processed/test_era5_data.nc"),
        ("phase2_transformation/processed/test_era5_data.parquet", "data/processed/test_era5_data.parquet"),
        ("phase2_transformation/processed/transformation_report.txt", "data/processed/transformation_report.txt"),
        
        # Config files (if customized)
        ("phase2_transformation/config.yaml", "config/pipeline_config.yaml.backup"),
        
        # CDS API config
        ("era5-download/.cdsapirc", ".cdsapirc"),
    ]
    
    print("📦 Migrating files to new structure...")
    print()
    
    migrated = 0
    skipped = 0
    
    for old_path, new_path in migrations:
        old_file = base_dir / old_path
        new_file = base_dir / new_path
        
        if old_file.exists():
            # Create parent directory
            new_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            try:
                shutil.copy2(old_file, new_file)
                print(f"✅ {old_path} → {new_path}")
                migrated += 1
            except Exception as e:
                print(f"❌ Error copying {old_path}: {e}")
        else:
            skipped += 1
    
    print()
    print(f"📊 Migration Summary:")
    print(f"   Migrated: {migrated}")
    print(f"   Skipped: {skipped}")
    print()
    
    print("✅ Migration complete!")
    print("   Test the new pipeline: make run-all")
    print("   Run tests: make test")
    print()
    print("=" * 80)


if __name__ == "__main__":
    migrate_project()
