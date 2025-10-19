"""
Data Transformation Module
Handles dimension renaming, unit conversions, temporal aggregations, and normalization.
"""

import logging
from typing import Union, List, Dict, Optional
import xarray as xr
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataTransformer:
    """
    Transform climate data: standardize dimensions, convert units, aggregate, normalize.
    """
    
    # Dimension mapping for standardization
    DIMENSION_MAPPING = {
        'valid_time': 'time',
        'time': 'time',
        'valid_date': 'time',
        'date': 'time',
        'latitude': 'latitude',
        'lat': 'latitude',
        'y': 'latitude',
        'longitude': 'longitude',
        'lon': 'longitude',
        'long': 'longitude',
        'x': 'longitude',
        'level': 'level',
        'pressure': 'level',
        'height': 'height',
        'z': 'height'
    }
    
    def __init__(self):
        """Initialize the data transformer."""
        self.transformations_applied = []
    
    def rename_dimensions(self, ds: xr.Dataset, custom_mapping: Optional[Dict] = None) -> xr.Dataset:
        """
        Rename dimensions to standard names.
        
        Parameters:
        -----------
        ds : xr.Dataset
            Input xarray Dataset
        custom_mapping : dict, optional
            Custom dimension name mapping to override defaults
            
        Returns:
        --------
        xr.Dataset
            Dataset with renamed dimensions
            
        Raises:
        -------
        ValueError: If dataset is empty or invalid
        """
        # Validation
        if not isinstance(ds, xr.Dataset):
            raise TypeError(f"Expected xr.Dataset, got {type(ds)}")
        
        if len(ds.dims) == 0:
            raise ValueError("Dataset has no dimensions")
        
        logger.info("Renaming dimensions to standard names...")
        
        # Use custom mapping if provided, otherwise use default
        mapping = custom_mapping if custom_mapping else {}
        
        # Build rename dictionary for both dimensions and coordinates
        rename_dict = {}
        
        # Check dimensions
        for dim in ds.dims:
            dim_lower = dim.lower()
            if dim_lower in self.DIMENSION_MAPPING:
                standard_name = self.DIMENSION_MAPPING[dim_lower]
                if standard_name != dim:
                    rename_dict[dim] = standard_name
            elif dim in mapping:
                rename_dict[dim] = mapping[dim]
        
        # Check coordinates
        for coord in ds.coords:
            coord_lower = coord.lower()
            if coord_lower in self.DIMENSION_MAPPING:
                standard_name = self.DIMENSION_MAPPING[coord_lower]
                if standard_name != coord:
                    rename_dict[coord] = standard_name
            elif coord in mapping:
                rename_dict[coord] = mapping[coord]
        
        if rename_dict:
            logger.info(f"Renaming: {rename_dict}")
            ds = ds.rename(rename_dict)
            self.transformations_applied.append(f"Renamed dimensions: {rename_dict}")
            
            # Validate critical dimensions exist
            if 'time' in rename_dict.values():
                assert 'time' in ds.dims or 'time' in ds.coords, "Critical dimension 'time' missing after renaming"
        else:
            logger.info("No dimension renaming needed")
        
        return ds
    
    def convert_temperature_to_celsius(self, ds: xr.Dataset, 
                                      variables: Optional[List[str]] = None) -> xr.Dataset:
        """
        Convert temperature from Kelvin to Celsius.
        
        Parameters:
        -----------
        ds : xr.Dataset
            Input xarray Dataset
        variables : list of str, optional
            List of temperature variable names. If None, auto-detect.
            
        Returns:
        --------
        xr.Dataset
            Dataset with converted temperatures
            
        Raises:
        -------
        ValueError: If variable not found or invalid temperature range
        """
        logger.info("Converting temperature from Kelvin to Celsius...")
        
        if variables is None:
            # Auto-detect temperature variables
            temp_patterns = ['t2m', 'temp', 'temperature', 't_2m', 'air_temperature']
            variables = [var for var in ds.data_vars 
                        if any(pattern in var.lower() for pattern in temp_patterns)]
        
        if not variables:
            logger.info("No temperature variables found")
            return ds
        
        # Validate all variables exist
        missing_vars = [var for var in variables if var not in ds.data_vars]
        if missing_vars:
            raise ValueError(f"Temperature variables not found in dataset: {missing_vars}")
        
        converted = []
        for var in variables:
            # Validate temperature range (should be in Kelvin: 150-350 K)
            var_min = float(ds[var].min())
            var_max = float(ds[var].max())
            
            if var_min < 0:
                raise ValueError(f"Variable '{var}' has negative values ({var_min:.2f}), likely not in Kelvin")
            if var_max > 400:
                raise ValueError(f"Variable '{var}' has unrealistic high values ({var_max:.2f} K)")
            
            # Check if data is likely in Kelvin (values > 200)
            if var_max > 200:
                ds[var] = ds[var] - 273.15
                ds[var].attrs['units'] = 'Celsius'
                converted.append(var)
                logger.info(f"Converted {var}: {var_min:.2f}K to {var_min-273.15:.2f}°C")
            else:
                logger.warning(f"Variable '{var}' max value ({var_max:.2f}) suggests already in Celsius, skipping")

        
        if converted:
            self.transformations_applied.append(f"Converted to Celsius: {converted}")
        
        return ds
    
    def convert_precipitation_to_meters(self, ds: xr.Dataset,
                                       variables: Optional[List[str]] = None) -> xr.Dataset:
        """
        Convert precipitation from millimeters to meters with validation.
        
        Parameters:
        -----------
        ds : xr.Dataset
            Input xarray Dataset
        variables : list of str, optional
            List of precipitation variable names. If None, auto-detect.
            
        Returns:
        --------
        xr.Dataset
            Dataset with converted precipitation
            
        Raises:
        -------
        ValueError: If negative precipitation values detected
        """
        logger.info("Converting precipitation from mm to meters...")
        
        if variables is None:
            # Auto-detect precipitation variables
            precip_patterns = ['precip', 'rain', 'tp', 'precipitation', 'pr']
            variables = [var for var in ds.data_vars 
                        if any(pattern in var.lower() for pattern in precip_patterns)]
        
        if not variables:
            logger.info("No precipitation variables found")
            return ds
        
        # Validate all variables exist
        missing_vars = [var for var in variables if var not in ds.data_vars]
        if missing_vars:
            raise ValueError(f"Precipitation variables not found in dataset: {missing_vars}")
        
        converted = []
        for var in variables:
            # Validate: Check for negative values (physically impossible)
            var_min = float(ds[var].min())
            if var_min < 0:
                logger.warning(f"Variable '{var}' contains negative values (min: {var_min:.2f} mm)")
                # Replace negative values with 0
                ds[var] = ds[var].where(ds[var] >= 0, 0)
                logger.info(f"Replaced {int((ds[var] < 0).sum())} negative values with 0")
            
            # Check if units are in mm (values typically < 1000)
            if 'units' in ds[var].attrs:
                if 'mm' in ds[var].attrs['units'].lower():
                    ds[var] = ds[var] / 1000.0
                    ds[var].attrs['units'] = 'meters'
                    converted.append(var)
                    logger.info(f"Converted {var} from mm to meters (min: {var_min/1000:.4f} m)")
            else:
                # No units specified, assume mm if values seem reasonable
                var_max = float(ds[var].max())
                if var_max < 1000:  # Likely in mm
                    ds[var] = ds[var] / 1000.0
                    ds[var].attrs['units'] = 'meters'
                    converted.append(var)
                    logger.info(f"Converted {var} from mm to meters (assumed mm based on range)")
        
        if converted:
            self.transformations_applied.append(f"Converted to meters: {converted}")
        
        return ds
    
    def aggregate_hourly_to_daily(self, ds: xr.Dataset, 
                                  method: str = 'mean') -> xr.Dataset:
        """
        Aggregate hourly data to daily resolution.
        
        Parameters:
        -----------
        ds : xr.Dataset
            Input xarray Dataset with time dimension
        method : str
            Aggregation method: 'mean', 'sum', 'min', 'max'
            
        Returns:
        --------
        xr.Dataset
            Dataset aggregated to daily resolution
        """
        logger.info(f"Aggregating hourly data to daily using {method}...")
        
        if 'time' not in ds.dims:
            logger.warning("No 'time' dimension found, skipping temporal aggregation")
            return ds
        
        # Resample to daily
        resampled = ds.resample(time='1D')
        
        if method == 'mean':
            ds_daily = resampled.mean()
        elif method == 'sum':
            ds_daily = resampled.sum()
        elif method == 'min':
            ds_daily = resampled.min()
        elif method == 'max':
            ds_daily = resampled.max()
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")
        
        logger.info(f"Aggregated from {len(ds.time)} time steps to {len(ds_daily.time)} days")
        self.transformations_applied.append(f"Aggregated hourly to daily ({method})")
        
        return ds_daily
    
    def aggregate_daily_to_monthly(self, ds: xr.Dataset,
                                   method: str = 'sum') -> xr.Dataset:
        """
        Aggregate daily data to monthly resolution.
        
        Parameters:
        -----------
        ds : xr.Dataset
            Input xarray Dataset with time dimension
        method : str
            Aggregation method: 'mean', 'sum', 'min', 'max'
            
        Returns:
        --------
        xr.Dataset
            Dataset aggregated to monthly resolution
        """
        logger.info(f"Aggregating daily data to monthly using {method}...")
        
        if 'time' not in ds.dims:
            logger.warning("No 'time' dimension found, skipping temporal aggregation")
            return ds
        
        # Resample to monthly
        resampled = ds.resample(time='1MS')  # Month start
        
        if method == 'mean':
            ds_monthly = resampled.mean()
        elif method == 'sum':
            ds_monthly = resampled.sum()
        elif method == 'min':
            ds_monthly = resampled.min()
        elif method == 'max':
            ds_monthly = resampled.max()
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")
        
        logger.info(f"Aggregated from {len(ds.time)} time steps to {len(ds_monthly.time)} months")
        self.transformations_applied.append(f"Aggregated daily to monthly ({method})")
        
        return ds_monthly
    
    def normalize_variables(self, ds: xr.Dataset,
                           variables: Optional[List[str]] = None,
                           method: str = 'zscore') -> xr.Dataset:
        """
        Normalize variables using Z-score scaling.
        
        Parameters:
        -----------
        ds : xr.Dataset
            Input xarray Dataset
        variables : list of str, optional
            List of variables to normalize. If None, normalize all data variables.
        method : str
            Normalization method: 'zscore' (standardization) or 'minmax'
            
        Returns:
        --------
        xr.Dataset
            Dataset with normalized variables
        """
        logger.info(f"Normalizing variables using {method} method...")
        
        if variables is None:
            variables = list(ds.data_vars)
        
        ds_normalized = ds.copy()
        
        for var in variables:
            if var not in ds.data_vars:
                logger.warning(f"Variable {var} not found in dataset, skipping")
                continue
            
            data = ds[var]
            
            if method == 'zscore':
                # Z-score normalization: (x - mean) / std
                mean = data.mean()
                std = data.std()
                
                if std > 0:
                    ds_normalized[var] = (data - mean) / std
                    ds_normalized[var].attrs['normalization'] = 'zscore'
                    ds_normalized[var].attrs['original_mean'] = float(mean.values)
                    ds_normalized[var].attrs['original_std'] = float(std.values)
                    logger.info(f"Normalized {var} (mean={mean.values:.2f}, std={std.values:.2f})")
                else:
                    logger.warning(f"Standard deviation is 0 for {var}, skipping normalization")
            
            elif method == 'minmax':
                # Min-max normalization: (x - min) / (max - min)
                min_val = data.min()
                max_val = data.max()
                
                if max_val > min_val:
                    ds_normalized[var] = (data - min_val) / (max_val - min_val)
                    ds_normalized[var].attrs['normalization'] = 'minmax'
                    ds_normalized[var].attrs['original_min'] = float(min_val.values)
                    ds_normalized[var].attrs['original_max'] = float(max_val.values)
                    logger.info(f"Normalized {var} (min={min_val.values:.2f}, max={max_val.values:.2f})")
                else:
                    logger.warning(f"Min equals max for {var}, skipping normalization")
            
            else:
                raise ValueError(
                    f"Unknown normalization method: '{method}'. "
                    f"Supported methods: 'zscore', 'minmax'"
                )
        
        self.transformations_applied.append(f"Normalized variables ({method}): {variables}")
        
        return ds_normalized
    
    def get_transformation_log(self) -> List[str]:
        """
        Get log of all transformations applied.
        
        Returns:
        --------
        list of str
            List of transformation descriptions
        """
        return self.transformations_applied


def transform_pipeline(ds: xr.Dataset,
                      rename_dims: bool = True,
                      convert_temp: bool = True,
                      convert_precip: bool = False,
                      aggregate_to_daily: bool = False,
                      aggregate_to_monthly: bool = False,
                      normalize: bool = True,
                      normalize_method: str = 'zscore') -> tuple:
    """
    Apply full transformation pipeline to dataset.
    
    Parameters:
    -----------
    ds : xr.Dataset
        Input dataset
    rename_dims : bool
        Whether to rename dimensions
    convert_temp : bool
        Whether to convert temperature to Celsius
    convert_precip : bool
        Whether to convert precipitation to meters
    aggregate_to_daily : bool
        Whether to aggregate to daily resolution
    aggregate_to_monthly : bool
        Whether to aggregate to monthly resolution
    normalize : bool
        Whether to normalize variables
    normalize_method : str
        Normalization method ('zscore' or 'minmax')
        
    Returns:
    --------
    tuple
        (transformed_dataset, transformation_log)
    """
    transformer = DataTransformer()
    
    # Apply transformations in order
    if rename_dims:
        ds = transformer.rename_dimensions(ds)
    
    if convert_temp:
        ds = transformer.convert_temperature_to_celsius(ds)
    
    if convert_precip:
        ds = transformer.convert_precipitation_to_meters(ds)
    
    if aggregate_to_daily:
        ds = transformer.aggregate_hourly_to_daily(ds)
    
    if aggregate_to_monthly:
        ds = transformer.aggregate_daily_to_monthly(ds)
    
    if normalize:
        ds = transformer.normalize_variables(ds, method=normalize_method)
    
    log = transformer.get_transformation_log()
    
    return ds, log


if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Transformation module ready")
