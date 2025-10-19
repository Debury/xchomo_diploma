"""
Climate Data Transformation Pipeline - Main Orchestrator
Phase 2: Transform raw climate datasets into standardized formats.
"""

import logging
import sys
from pathlib import Path
from typing import Union, Optional, Dict, List
import xarray as xr

# Import local modules
from .ingestion import DataLoader
from .transformations import DataTransformer
from .export import DataExporter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log')
    ]
)

logger = logging.getLogger(__name__)


class ClimatePipeline:
    """
    Main pipeline orchestrator for climate data transformation.
    """
    
    def __init__(self, output_dir: Union[str, Path] = "processed"):
        """
        Initialize the climate data pipeline.
        
        Parameters:
        -----------
        output_dir : str or Path
            Directory for processed outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 80)
        logger.info("CLIMATE DATA TRANSFORMATION PIPELINE - PHASE 2")
        logger.info("=" * 80)
        logger.info(f"Output directory: {self.output_dir.absolute()}")
    
    def process_file(self, 
                    input_file: Union[str, Path],
                    output_name: Optional[str] = None,
                    config: Optional[Dict] = None) -> Dict:
        """
        Process a single climate data file through the full pipeline.
        
        Parameters:
        -----------
        input_file : str or Path
            Path to input data file
        output_name : str, optional
            Base name for output files (defaults to input filename)
        config : dict, optional
            Configuration for transformations
            
        Returns:
        --------
        dict
            Dictionary with output file paths and metadata
        """
        input_file = Path(input_file)
        
        if output_name is None:
            output_name = input_file.stem
        
        # Default configuration
        default_config = {
            'rename_dims': True,
            'convert_temp': True,
            'convert_precip': False,
            'aggregate_to_daily': False,
            'aggregate_to_monthly': False,
            'normalize': True,
            'normalize_method': 'zscore',
            'export_formats': ['netcdf', 'parquet']
        }
        
        # Merge with user config
        if config:
            default_config.update(config)
        
        config = default_config
        
        logger.info("\n" + "=" * 80)
        logger.info(f"PROCESSING FILE: {input_file.name}")
        logger.info("=" * 80)
        
        try:
            # Step 1: Data Ingestion
            logger.info("\n[STEP 1] DATA INGESTION")
            logger.info("-" * 80)
            data, metadata, file_type = load_data(input_file)
            logger.info(f"File type: {file_type}")
            logger.info(f"Metadata: {metadata}")
            
            # Convert to xarray Dataset if needed
            if not isinstance(data, xr.Dataset):
                logger.info("Converting data to xarray Dataset...")
                if isinstance(data, xr.DataArray):
                    data = data.to_dataset()
                else:
                    logger.error(f"Unsupported data type for transformation: {type(data)}")
                    raise TypeError(f"Cannot transform data of type {type(data)}")
            
            # Step 2: Data Transformation
            logger.info("\n[STEP 2] DATA TRANSFORMATION")
            logger.info("-" * 80)
            transformed_data, transformation_log = transform_pipeline(
                data,
                rename_dims=config['rename_dims'],
                convert_temp=config['convert_temp'],
                convert_precip=config['convert_precip'],
                aggregate_to_daily=config['aggregate_to_daily'],
                aggregate_to_monthly=config['aggregate_to_monthly'],
                normalize=config['normalize'],
                normalize_method=config['normalize_method']
            )
            
            logger.info("Transformations completed:")
            for i, trans in enumerate(transformation_log, 1):
                logger.info(f"  {i}. {trans}")
            
            # Step 3: Data Export
            logger.info("\n[STEP 3] DATA EXPORT")
            logger.info("-" * 80)
            output_files = export_data(
                transformed_data,
                output_name,
                output_dir=self.output_dir,
                formats=config['export_formats'],
                transformation_log=transformation_log
            )
            
            logger.info("\n" + "=" * 80)
            logger.info(f"PROCESSING COMPLETE: {input_file.name}")
            logger.info("=" * 80)
            logger.info("Output files:")
            for fmt, path in output_files.items():
                logger.info(f"  {fmt}: {path}")
            
            return {
                'input_file': input_file,
                'output_files': output_files,
                'metadata': metadata,
                'transformations': transformation_log,
                'final_shape': dict(transformed_data.dims),
                'variables': list(transformed_data.data_vars)
            }
            
        except Exception as e:
            logger.error(f"\n{'='*80}")
            logger.error(f"ERROR PROCESSING {input_file.name}")
            logger.error(f"{'='*80}")
            logger.error(f"Error: {str(e)}", exc_info=True)
            raise
    
    def process_directory(self, 
                         input_dir: Union[str, Path],
                         pattern: str = "*",
                         config: Optional[Dict] = None) -> List[Dict]:
        """
        Process all files in a directory matching a pattern.
        
        Parameters:
        -----------
        input_dir : str or Path
            Directory containing input files
        pattern : str
            Glob pattern for file selection
        config : dict, optional
            Configuration for transformations
            
        Returns:
        --------
        list of dict
            List of processing results for each file
        """
        input_dir = Path(input_dir)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")
        
        # Find matching files
        files = list(input_dir.glob(pattern))
        
        if not files:
            logger.warning(f"No files found matching pattern '{pattern}' in {input_dir}")
            return []
        
        logger.info(f"\nFound {len(files)} files to process")
        
        results = []
        for i, file in enumerate(files, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"FILE {i}/{len(files)}")
            logger.info(f"{'='*80}")
            
            try:
                result = self.process_file(file, config=config)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {file.name}: {str(e)}")
                results.append({
                    'input_file': file,
                    'error': str(e),
                    'status': 'failed'
                })
        
        # Summary
        successful = sum(1 for r in results if 'error' not in r)
        failed = len(results) - successful
        
        logger.info(f"\n{'='*80}")
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total files: {len(results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        
        return results


def main():
    """
    Main entry point for the pipeline.
    Example usage with command-line arguments.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Climate Data Transformation Pipeline - Phase 2"
    )
    parser.add_argument(
        'input',
        help='Input file or directory'
    )
    parser.add_argument(
        '-o', '--output',
        default='processed',
        help='Output directory (default: processed)'
    )
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Skip normalization step'
    )
    parser.add_argument(
        '--daily',
        action='store_true',
        help='Aggregate to daily resolution'
    )
    parser.add_argument(
        '--monthly',
        action='store_true',
        help='Aggregate to monthly resolution'
    )
    parser.add_argument(
        '--formats',
        nargs='+',
        default=['netcdf', 'parquet'],
        help='Export formats (default: netcdf parquet)'
    )
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        'normalize': not args.no_normalize,
        'aggregate_to_daily': args.daily,
        'aggregate_to_monthly': args.monthly,
        'export_formats': args.formats
    }
    
    # Initialize pipeline
    pipeline = ClimatePipeline(output_dir=args.output)
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        result = pipeline.process_file(input_path, config=config)
        logger.info("\nProcessing complete!")
    elif input_path.is_dir():
        results = pipeline.process_directory(input_path, config=config)
        logger.info(f"\nBatch processing complete! Processed {len(results)} files.")
    else:
        logger.error(f"Input path not found: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    # Example: Process the test ERA5 file
    test_file = Path(__file__).parent.parent / "era5-download" / "test_era5_data.nc"
    
    if test_file.exists():
        logger.info("Running example with test ERA5 data...")
        
        pipeline = ClimatePipeline(output_dir="processed")
        
        config = {
            'rename_dims': True,
            'convert_temp': True,
            'normalize': True,
            'export_formats': ['netcdf', 'parquet']
        }
        
        result = pipeline.process_file(test_file, config=config)
        
        logger.info("\n" + "="*80)
        logger.info("EXAMPLE COMPLETE")
        logger.info("="*80)
        logger.info(f"Processed file: {result['input_file']}")
        logger.info(f"Variables: {result['variables']}")
        logger.info(f"Final shape: {result['final_shape']}")
    else:
        logger.info("No test file found. Use main() for command-line interface.")
        logger.info("Example: python pipeline.py <input_file> -o processed")
