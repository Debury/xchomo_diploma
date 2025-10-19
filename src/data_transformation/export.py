"""
Data Export Module
Saves cleaned datasets to NetCDF and Parquet formats.
"""

import logging
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
from datetime import datetime
import xarray as xr
import pandas as pd
import json
import yaml

logger = logging.getLogger(__name__)

# Pipeline version for metadata
PIPELINE_VERSION = "2.0.0"


class DataExporter:
    """
    Export climate data to multiple formats (NetCDF, Parquet).
    """
    
    def __init__(self, output_dir: Union[str, Path] = "processed"):
        """
        Initialize the data exporter.
        
        Parameters:
        -----------
        output_dir : str or Path
            Directory to save processed data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir.absolute()}")
    
    def save_netcdf(self, ds: xr.Dataset, filename: str,
                   compression_level: int = 4,
                   source_file: Optional[str] = None,
                   add_metadata: bool = True) -> Path:
        """
        Save dataset to NetCDF format.
        
        Parameters:
        -----------
        ds : xr.Dataset
            Dataset to save
        filename : str
            Output filename (without extension)
        compression_level : int
            Compression level (0-9, default 4)
        source_file : str, optional
            Original source file path for metadata
        add_metadata : bool
            Add processing metadata to file
            
        Returns:
        --------
        Path
            Path to saved file
        """
        # Ensure .nc extension
        if not filename.endswith('.nc'):
            filename = f"{filename}.nc"
        
        output_path = self.output_dir / filename
        
        logger.info(f"Saving NetCDF to {output_path}...")
        
        # Add metadata attributes
        if add_metadata:
            ds.attrs['pipeline_version'] = PIPELINE_VERSION
            ds.attrs['processing_date'] = datetime.now().isoformat()
            if source_file:
                ds.attrs['source_file'] = str(source_file)
            ds.attrs['export_format'] = 'NetCDF4'
        
        # Set up encoding with compression for all variables
        encoding = {}
        for var in ds.data_vars:
            encoding[var] = {
                'zlib': True,
                'complevel': compression_level,
                'dtype': 'float32'
            }
        
        # Save to NetCDF
        ds.to_netcdf(
            output_path,
            encoding=encoding,
            format='NETCDF4'
        )
        
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Saved NetCDF: {output_path} ({file_size:.2f} MB)")
        
        return output_path
    
    def save_parquet(self, ds: xr.Dataset, filename: str,
                    compression: str = 'snappy',
                    source_file: Optional[str] = None,
                    add_metadata: bool = True) -> Path:
        """
        Save dataset to Parquet format.
        
        Parameters:
        -----------
        ds : xr.Dataset
            Dataset to save
        filename : str
            Output filename (without extension)
        compression : str
            Compression algorithm ('snappy', 'gzip', 'brotli', 'zstd')
        source_file : str, optional
            Original source file path for metadata
        add_metadata : bool
            Add processing metadata to file
            
        Returns:
        --------
        Path
            Path to saved file
        """
        # Ensure .parquet extension
        if not filename.endswith('.parquet'):
            filename = f"{filename}.parquet"
        
        output_path = self.output_dir / filename
        
        logger.info(f"Converting to DataFrame and saving Parquet to {output_path}...")
        
        # Convert xarray Dataset to pandas DataFrame
        # Flatten multi-dimensional data
        df = ds.to_dataframe().reset_index()
        
        # Add metadata as columns if requested
        if add_metadata:
            df.attrs['pipeline_version'] = PIPELINE_VERSION
            df.attrs['processing_date'] = datetime.now().isoformat()
            if source_file:
                df.attrs['source_file'] = str(source_file)
        
        # Save to Parquet
        df.to_parquet(
            output_path,
            compression=compression,
            index=False
        )
        
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Saved Parquet: {output_path} ({file_size:.2f} MB)")
        
        return output_path
    
    def save_both_formats(self, ds: xr.Dataset, filename: str) -> dict:
        """
        Save dataset to both NetCDF and Parquet formats.
        
        Parameters:
        -----------
        ds : xr.Dataset
            Dataset to save
        filename : str
            Base filename (without extension)
            
        Returns:
        --------
        dict
            Dictionary with paths to saved files
        """
        logger.info(f"Saving dataset '{filename}' in both formats...")
        
        netcdf_path = self.save_netcdf(ds, filename)
        parquet_path = self.save_parquet(ds, filename)
        
        return {
            'netcdf': netcdf_path,
            'parquet': parquet_path
        }
    
    def save_csv(self, ds: xr.Dataset, filename: str) -> Path:
        """
        Save dataset to CSV format (for simple tabular data).
        
        Parameters:
        -----------
        ds : xr.Dataset
            Dataset to save
        filename : str
            Output filename (without extension)
            
        Returns:
        --------
        Path
            Path to saved file
        """
        # Ensure .csv extension
        if not filename.endswith('.csv'):
            filename = f"{filename}.csv"
        
        output_path = self.output_dir / filename
        
        logger.info(f"Converting to DataFrame and saving CSV to {output_path}...")
        
        # Convert xarray Dataset to pandas DataFrame
        df = ds.to_dataframe().reset_index()
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Saved CSV: {output_path} ({file_size:.2f} MB)")
        
        return output_path
    
    def generate_summary_report(self, ds: xr.Dataset, 
                               transformation_log: List[str],
                               output_files: dict) -> Path:
        """
        Generate a summary report of the transformation process.
        
        Parameters:
        -----------
        ds : xr.Dataset
            Transformed dataset
        transformation_log : list of str
            Log of transformations applied
        output_files : dict
            Dictionary of output file paths
            
        Returns:
        --------
        Path
            Path to summary report file
        """
        report_path = self.output_dir / "transformation_report.txt"
        
        logger.info(f"Generating summary report: {report_path}...")
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CLIMATE DATA TRANSFORMATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Dataset information
            f.write("DATASET INFORMATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Variables: {list(ds.data_vars)}\n")
            f.write(f"Dimensions: {dict(ds.dims)}\n")
            f.write(f"Coordinates: {list(ds.coords)}\n\n")
            
            # Variable details
            f.write("VARIABLE DETAILS\n")
            f.write("-" * 80 + "\n")
            for var in ds.data_vars:
                f.write(f"\n{var}:\n")
                f.write(f"  Shape: {ds[var].shape}\n")
                f.write(f"  Dtype: {ds[var].dtype}\n")
                f.write(f"  Min: {float(ds[var].min().values):.4f}\n")
                f.write(f"  Max: {float(ds[var].max().values):.4f}\n")
                f.write(f"  Mean: {float(ds[var].mean().values):.4f}\n")
                f.write(f"  Std: {float(ds[var].std().values):.4f}\n")
                
                if ds[var].attrs:
                    f.write(f"  Attributes: {ds[var].attrs}\n")
            
            # Transformations applied
            f.write("\n\nTRANSFORMATIONS APPLIED\n")
            f.write("-" * 80 + "\n")
            for i, transformation in enumerate(transformation_log, 1):
                f.write(f"{i}. {transformation}\n")
            
            # Output files
            f.write("\n\nOUTPUT FILES\n")
            f.write("-" * 80 + "\n")
            for format_type, filepath in output_files.items():
                file_size = filepath.stat().st_size / (1024 * 1024)  # MB
                f.write(f"{format_type.upper()}: {filepath} ({file_size:.2f} MB)\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("Report generated successfully.\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Summary report saved to {report_path}")
        
        return report_path
    
    def generate_documentation_report(self, ds: xr.Dataset,
                                     transformation_log: Optional[List[str]] = None,
                                     output_files: Optional[Dict[str, Path]] = None,
                                     format: str = 'json',
                                     filename: Optional[str] = None) -> Path:
        """
        Generate structured documentation report in JSON or YAML format.
        
        Parameters:
        -----------
        ds : xr.Dataset
            Processed dataset
        transformation_log : list of str, optional
            List of transformations applied
        output_files : dict, optional
            Dictionary of output file paths
        format : str
            Output format ('json' or 'yaml')
        filename : str, optional
            Custom filename (default: 'processing_report.json/yaml')
            
        Returns:
        --------
        Path
            Path to generated documentation file
        """
        if filename is None:
            ext = 'json' if format == 'json' else 'yaml'
            filename = f'processing_report.{ext}'
        
        report_path = self.output_dir / filename
        
        logger.info(f"Generating {format.upper()} documentation report...")
        
        # Build comprehensive report structure
        report: Dict[str, Any] = {
            'metadata': {
                'pipeline_version': PIPELINE_VERSION,
                'processing_date': datetime.now().isoformat(),
                'report_format': format
            },
            'dataset_info': {
                'dimensions': {dim: int(ds.dims[dim]) for dim in ds.dims},
                'variables': list(ds.data_vars.keys()),
                'coordinates': list(ds.coords.keys()),
                'total_size_bytes': int(ds.nbytes)
            },
            'variable_statistics': {},
            'transformations': transformation_log or [],
            'output_files': {}
        }
        
        # Add variable statistics
        for var in ds.data_vars:
            var_data = ds[var]
            stats = {
                'dtype': str(var_data.dtype),
                'shape': list(var_data.shape),
                'units': var_data.attrs.get('units', 'unknown'),
                'min': float(var_data.min().values),
                'max': float(var_data.max().values),
                'mean': float(var_data.mean().values),
                'std': float(var_data.std().values)
            }
            report['variable_statistics'][var] = stats
        
        # Add output file information
        if output_files:
            for format_type, filepath in output_files.items():
                if filepath.exists():
                    file_size_mb = filepath.stat().st_size / (1024 * 1024)
                    report['output_files'][format_type] = {
                        'path': str(filepath),
                        'size_mb': round(file_size_mb, 2),
                        'exists': True
                    }
        
        # Write to file
        with open(report_path, 'w', encoding='utf-8') as f:
            if format == 'json':
                json.dump(report, f, indent=2, ensure_ascii=False)
            else:  # yaml
                yaml.dump(report, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        logger.info(f"Documentation report saved to {report_path}")
        
        return report_path


def export_data(ds: xr.Dataset,
               filename: str,
               output_dir: Union[str, Path] = "processed",
               formats: List[str] = ['netcdf', 'parquet'],
               transformation_log: Optional[List[str]] = None) -> dict:
    """
    Convenience function to export data in multiple formats.
    
    Parameters:
    -----------
    ds : xr.Dataset
        Dataset to export
    filename : str
        Base filename (without extension)
    output_dir : str or Path
        Output directory
    formats : list of str
        List of formats to export ('netcdf', 'parquet', 'csv')
    transformation_log : list of str, optional
        Log of transformations to include in report
        
    Returns:
    --------
    dict
        Dictionary with paths to saved files and report
    """
    exporter = DataExporter(output_dir)
    output_files = {}
    
    for fmt in formats:
        if fmt == 'netcdf':
            output_files['netcdf'] = exporter.save_netcdf(ds, filename)
        elif fmt == 'parquet':
            output_files['parquet'] = exporter.save_parquet(ds, filename)
        elif fmt == 'csv':
            output_files['csv'] = exporter.save_csv(ds, filename)
        else:
            logger.warning(f"Unknown format: {fmt}")
    
    # Generate summary report if transformation log is provided
    if transformation_log:
        output_files['report'] = exporter.generate_summary_report(
            ds, transformation_log, output_files
        )
    
    return output_files


if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Export module ready")
