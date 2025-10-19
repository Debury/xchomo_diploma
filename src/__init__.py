"""
Climate Data ETL Pipeline - Source Package
"""

__version__ = "2.0.0"
__author__ = "Climate Research Team"
__description__ = "Production ETL pipeline for climate data processing"

from . import data_acquisition
from . import data_transformation
from . import utils

__all__ = ["data_acquisition", "data_transformation", "utils"]
