"""
Data Ingestion Module
Dynamically loads climate data files based on extension:
- .nc (NetCDF) via xarray
- .csv/.json via pandas
- .tif (GeoTIFF) via rasterio
"""

import os
import logging
from pathlib import Path
from typing import Union, Optional
import xarray as xr
import pandas as pd
import rasterio
import numpy as np

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Dynamic data loader for multiple climate data formats.
    """
    
    SUPPORTED_EXTENSIONS = ['.nc', '.csv', '.json', '.tif', '.tiff', '.geotiff']
    
    def __init__(self):
        """Initialize the data loader."""
        self.loaded_data = None
        self.file_type = None
        self.metadata = {}
    
    def load(self, filepath: Union[str, Path]) -> Union[xr.Dataset, pd.DataFrame, dict]:
        """
        Load data file based on extension.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the data file
            
        Returns:
        --------
        Union[xr.Dataset, pd.DataFrame, dict]
            Loaded data in appropriate format
            
        Raises:
        -------
        ValueError: If file extension is not supported
        FileNotFoundError: If file does not exist
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        extension = filepath.suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )
        
        logger.info(f"Loading file: {filepath} (type: {extension})")
        
        try:
            if extension == '.nc':
                self.loaded_data = self._load_netcdf(filepath)
                self.file_type = 'netcdf'
            elif extension == '.csv':
                self.loaded_data = self._load_csv(filepath)
                self.file_type = 'csv'
            elif extension == '.json':
                self.loaded_data = self._load_json(filepath)
                self.file_type = 'json'
            elif extension in ['.tif', '.tiff', '.geotiff']:
                self.loaded_data = self._load_geotiff(filepath)
                self.file_type = 'geotiff'
            
            logger.info(f"Successfully loaded {filepath.name}")
            return self.loaded_data
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {str(e)}")
            raise
    
    def _load_netcdf(self, filepath: Path) -> xr.Dataset:
        """
        Load NetCDF file using xarray.
        
        Parameters:
        -----------
        filepath : Path
            Path to NetCDF file
            
        Returns:
        --------
        xr.Dataset
            Loaded xarray Dataset
        """
        ds = xr.open_dataset(filepath)
        
        # Store metadata
        self.metadata = {
            'variables': list(ds.data_vars),
            'dimensions': list(ds.dims),
            'coords': list(ds.coords),
            'shape': {dim: ds.dims[dim] for dim in ds.dims},
            'attrs': dict(ds.attrs)
        }
        
        logger.debug(f"NetCDF variables: {self.metadata['variables']}")
        logger.debug(f"NetCDF dimensions: {self.metadata['dimensions']}")
        
        return ds
    
    def _load_csv(self, filepath: Path) -> pd.DataFrame:
        """
        Load CSV file using pandas and convert to xarray Dataset.
        
        Parameters:
        -----------
        filepath : Path
            Path to CSV file
            
        Returns:
        --------
        xr.Dataset
            Converted xarray Dataset
        """
        # Try to infer datetime columns
        df = pd.read_csv(filepath, parse_dates=True)
        
        # Store metadata
        self.metadata = {
            'columns': list(df.columns),
            'shape': df.shape,
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        logger.debug(f"CSV shape: {self.metadata['shape']}")
        logger.debug(f"CSV columns: {self.metadata['columns']}")
        
        # Convert DataFrame to xarray Dataset
        try:
            # If there's a time column, set it as index
            if 'time' in df.columns:
                df = df.set_index('time')
            elif any(col in df.columns for col in ['date', 'datetime', 'timestamp']):
                time_col = next(col for col in ['date', 'datetime', 'timestamp'] if col in df.columns)
                df = df.set_index(time_col)
            
            ds = xr.Dataset.from_dataframe(df)
            logger.info(f"Converted CSV to xarray Dataset with {len(ds.data_vars)} variables")
            return ds
        except Exception as e:
            logger.warning(f"Could not convert to xarray Dataset: {e}")
            # Return simple dataset with index dimension
            ds = xr.Dataset.from_dataframe(df.reset_index())
            return ds
    
    def _load_json(self, filepath: Path) -> pd.DataFrame:
        """
        Load JSON file using pandas.
        
        Parameters:
        -----------
        filepath : Path
            Path to JSON file
            
        Returns:
        --------
        xr.Dataset
            Converted xarray Dataset
        """
        import json
        
        # Load JSON file
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Try to extract data field if nested structure
        if isinstance(data, dict):
            # Look for common data fields
            for key in ['data', 'records', 'observations', 'measurements']:
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
        
        # Convert to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            raise ValueError(f"Unsupported JSON structure: {type(data)}")
        
        # Store metadata
        self.metadata = {
            'columns': list(df.columns),
            'shape': df.shape,
            'dtypes': df.dtypes.to_dict()
        }
        
        logger.debug(f"JSON shape: {self.metadata['shape']}")
        logger.debug(f"JSON columns: {self.metadata['columns']}")
        
        # Convert DataFrame to xarray Dataset
        try:
            # If there's a time column, set it as index
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df = df.set_index('time')
            
            ds = xr.Dataset.from_dataframe(df)
            logger.info(f"Converted JSON to xarray Dataset with {len(ds.data_vars)} variables")
            return ds
        except Exception as e:
            logger.warning(f"Could not convert to xarray Dataset: {e}")
            # Return simple dataset
            ds = xr.Dataset.from_dataframe(df.reset_index())
            return ds
    
    def _load_geotiff(self, filepath: Path) -> xr.Dataset:
        """
        Load GeoTIFF file using rasterio and convert to xarray.
        
        Parameters:
        -----------
        filepath : Path
            Path to GeoTIFF file
            
        Returns:
        --------
        xr.Dataset
            Converted xarray Dataset with spatial coordinates
        """
        with rasterio.open(filepath) as src:
            # Read data
            data = src.read()
            
            # Get spatial information
            transform = src.transform
            crs = src.crs
            height, width = src.height, src.width
            
            # Create coordinate arrays
            x_coords = np.arange(width) * transform.a + transform.c
            y_coords = np.arange(height) * transform.e + transform.f
            
            # Handle multiple bands
            if data.shape[0] == 1:
                # Single band
                ds = xr.Dataset(
                    {
                        'band_1': (['y', 'x'], data[0])
                    },
                    coords={
                        'x': x_coords,
                        'y': y_coords
                    }
                )
            else:
                # Multiple bands
                ds = xr.Dataset(
                    {
                        f'band_{i+1}': (['y', 'x'], data[i])
                        for i in range(data.shape[0])
                    },
                    coords={
                        'x': x_coords,
                        'y': y_coords
                    }
                )
            
            # Add CRS as attribute
            ds.attrs['crs'] = str(crs)
            ds.attrs['transform'] = str(transform)
            
            # Store metadata
            self.metadata = {
                'bands': data.shape[0],
                'shape': (height, width),
                'crs': str(crs),
                'transform': transform,
                'nodata': src.nodata
            }
            
            logger.debug(f"GeoTIFF shape: {self.metadata['shape']}")
            logger.debug(f"GeoTIFF bands: {self.metadata['bands']}")
            
            return ds
    
    def get_metadata(self) -> dict:
        """
        Get metadata about the loaded data.
        
        Returns:
        --------
        dict
            Metadata dictionary
        """
        return self.metadata


def load_data(filepath: Union[str, Path]) -> tuple:
    """
    Convenience function to load data and return both data and metadata.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the data file
        
    Returns:
    --------
    tuple
        (loaded_data, metadata, file_type)
    """
    loader = DataLoader()
    data = loader.load(filepath)
    metadata = loader.get_metadata()
    file_type = loader.file_type
    
    return data, metadata, file_type


if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test with ERA5 data if available
    test_file = Path(__file__).parent.parent / "era5-download" / "test_era5_data.nc"
    if test_file.exists():
        logger.info("Testing NetCDF loading...")
        data, metadata, file_type = load_data(test_file)
        logger.info(f"Loaded {file_type} with metadata: {metadata}")
