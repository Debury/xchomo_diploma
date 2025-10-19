"""
Dagster Resources for Climate ETL Pipeline

Resources provide shared dependencies and configuration across ops and jobs.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict
from dagster import ConfigurableResource, InitResourceContext
from pydantic import Field

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger


class ConfigLoaderResource(ConfigurableResource):
    """
    Dagster resource for loading pipeline configuration.
    
    Provides centralized access to pipeline_config.yaml throughout the DAG.
    
    Attributes:
        config_path: Path to the pipeline configuration YAML file
    
    Example:
        @op
        def my_op(config_loader: ConfigLoaderResource):
            config = config_loader.load()
            # Use config...
    """
    
    config_path: str = Field(
        default="config/pipeline_config.yaml",
        description="Path to pipeline configuration YAML file"
    )
    
    def load(self) -> Dict[str, Any]:
        """
        Load and return the pipeline configuration.
        
        Returns:
            Dictionary containing complete pipeline configuration
        """
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        loader = ConfigLoader(config_file)
        config = loader.load()
        
        return config
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Load and return a specific configuration section.
        
        Args:
            section: Configuration section name (e.g., 'data_acquisition')
        
        Returns:
            Dictionary containing the requested configuration section
        """
        config = self.load()
        if section not in config:
            raise KeyError(f"Configuration section '{section}' not found")
        return config[section]


class LoggerResource(ConfigurableResource):
    """
    Dagster resource for structured logging.
    
    Provides consistent logging across all pipeline operations with 
    file and console handlers.
    
    Attributes:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
    
    Example:
        @op
        def my_op(logger: LoggerResource):
            logger.info("Processing data...")
    """
    
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_file: str = Field(
        default="logs/dagster_pipeline.log",
        description="Path to log file"
    )
    
    def _get_logger(self) -> logging.Logger:
        """
        Get or create a configured logger instance.
        
        Returns:
            Configured logger instance
        """
        logger = setup_logger(
            name="dagster_climate_pipeline",
            log_file=self.log_file,
            level=self.log_level  # Pass as-is, setup_logger will handle string/int
        )
        return logger
    
    def info(self, message: str, **kwargs):
        """Log info level message"""
        logger = self._get_logger()
        logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning level message"""
        logger = self._get_logger()
        logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error level message"""
        logger = self._get_logger()
        logger.error(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug level message"""
        logger = self._get_logger()
        logger.debug(message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        logger = self._get_logger()
        logger.exception(message, **kwargs)


class DataPathResource(ConfigurableResource):
    """
    Dagster resource for managing data directory paths.
    
    Provides consistent access to input/output directories across the pipeline.
    
    Attributes:
        raw_data_dir: Directory for raw downloaded data
        processed_data_dir: Directory for transformed data
        embeddings_dir: Directory for generated embeddings
    """
    
    raw_data_dir: str = Field(
        default="data/raw",
        description="Directory for raw data files"
    )
    processed_data_dir: str = Field(
        default="data/processed",
        description="Directory for processed data files"
    )
    embeddings_dir: str = Field(
        default="chroma_db",
        description="Directory for vector database"
    )
    
    def get_raw_path(self) -> Path:
        """Get raw data directory path"""
        path = Path(self.raw_data_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_processed_path(self) -> Path:
        """Get processed data directory path"""
        path = Path(self.processed_data_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_embeddings_path(self) -> Path:
        """Get embeddings directory path"""
        path = Path(self.embeddings_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


class SourceLoaderResource(ConfigurableResource):
    """
    Dagster resource for loading climate data sources from database.
    
    Provides dynamic source configuration for ETL jobs. Sources can be:
    - Added/updated via Web UI or API
    - Queried at job runtime
    - Used to configure download, transform, and embedding operations
    
    Attributes:
        database_url: SQLAlchemy database URL for source store
    
    Example:
        @op
        def process_sources(source_loader: SourceLoaderResource):
            sources = source_loader.get_active_sources()
            for source in sources:
                # Process each source...
    """
    
    database_url: str = Field(
        default="sqlite:///data/climate_sources.db",
        description="Database URL for source store"
    )
    
    def _get_store(self):
        """Get or create SourceStore instance"""
        from src.sources import get_source_store
        return get_source_store(self.database_url)
    
    def get_source(self, source_id: str):
        """
        Get a single source by ID.
        
        Args:
            source_id: Unique identifier for the source
            
        Returns:
            ClimateDataSource object or None if not found
        """
        store = self._get_store()
        return store.get_source(source_id)
    
    def get_active_sources(self) -> list:
        """
        Get all active sources.
        
        Returns:
            List of ClimateDataSource objects
        """
        store = self._get_store()
        return store.get_all_sources(active_only=True)
    
    def get_pending_sources(self) -> list:
        """
        Get sources that are pending processing.
        
        Returns:
            List of ClimateDataSource objects with status='pending'
        """
        store = self._get_store()
        return store.get_pending_sources()
    
    def update_status(self, source_id: str, status: str, error_message: str = None) -> bool:
        """
        Update processing status of a source.
        
        Args:
            source_id: Source identifier
            status: New status (pending, processing, completed, failed)
            error_message: Optional error message if status is 'failed'
            
        Returns:
            True if updated successfully
        """
        store = self._get_store()
        return store.update_processing_status(source_id, status, error_message)
    
    def get_sources_by_tags(self, tags: list) -> list:
        """
        Get sources matching any of the given tags.
        
        Args:
            tags: List of tag strings to search for
            
        Returns:
            List of matching ClimateDataSource objects
        """
        store = self._get_store()
        return store.get_sources_by_tags(tags)
