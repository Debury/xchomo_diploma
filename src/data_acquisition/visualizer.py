"""
Data Visualization Module
Functions for visualizing climate data from NetCDF files.
"""

import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union
import numpy as np


def visualize_netcdf(filepath: Union[str, Path], 
                     variable: Optional[str] = None,
                     time_idx: int = 0) -> None:
    """
    Visualize NetCDF file data.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to NetCDF file
    variable : str, optional
        Variable to visualize (default: first variable)
    time_idx : int
        Time index to display (default: 0)
    """
    ds = xr.open_dataset(filepath)
    
    print("Dataset structure:")
    print(ds)
    print(f"\nDimensions: {ds.dims}")
    print(f"Coordinates: {list(ds.coords.keys())}")
    print(f"Variables: {list(ds.data_vars.keys())}")
    
    # Use first variable if not specified
    if variable is None and len(ds.data_vars) > 0:
        variable = list(ds.data_vars.keys())[0]
    
    if variable and variable in ds.data_vars:
        print(f"\n{variable} statistics:")
        print(f"  Mean: {float(ds[variable].mean()):.2f}")
        print(f"  Min: {float(ds[variable].min()):.2f}")
        print(f"  Max: {float(ds[variable].max()):.2f}")
        
        # Select timestep if time dimension exists
        if 'time' in ds.dims or 'valid_time' in ds.dims:
            time_dim = 'time' if 'time' in ds.dims else 'valid_time'
            ds_subset = ds.isel({time_dim: time_idx})
        else:
            ds_subset = ds
        
        # Plot if 2D
        if len(ds_subset[variable].dims) >= 2:
            plt.figure(figsize=(10, 6))
            ds_subset[variable].plot()
            plt.title(f"{variable} - Time index {time_idx}")
            plt.tight_layout()
            plt.show()
    
    ds.close()


def create_temperature_map(filepath: Union[str, Path],
                           output_path: Optional[Union[str, Path]] = None) -> None:
    """
    Create temperature map from NetCDF file.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to NetCDF file
    output_path : str or Path, optional
        Path to save figure (if None, displays interactive plot)
    """
    ds = xr.open_dataset(filepath)
    
    # Find temperature variable
    temp_vars = [var for var in ds.data_vars if 't2m' in var.lower() or 'temp' in var.lower()]
    
    if not temp_vars:
        print("No temperature variable found")
        ds.close()
        return
    
    temp_var = temp_vars[0]
    
    # Get first timestep
    if 'time' in ds.dims or 'valid_time' in ds.dims:
        time_dim = 'time' if 'time' in ds.dims else 'valid_time'
        temp_data = ds[temp_var].isel({time_dim: 0})
    else:
        temp_data = ds[temp_var]
    
    # Create map
    fig, ax = plt.subplots(figsize=(12, 8))
    im = temp_data.plot(ax=ax, cmap='RdYlBu_r', add_colorbar=True)
    ax.set_title(f"Temperature Map: {temp_var}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    else:
        plt.show()
    
    ds.close()


if __name__ == '__main__':
    # Example usage when run as script
    import sys
    if len(sys.argv) > 1:
        visualize_netcdf(sys.argv[1])
    else:
        print("Usage: python visualizer.py <netcdf_file>")

