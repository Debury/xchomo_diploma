"""
Dagster Resources for Climate ETL Pipeline

Resources provide shared dependencies and configuration across ops and jobs.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from dagster import ConfigurableResource
from pydantic import Field

# ==============================================================================
# CRITICAL FIX: Add Project Root to Sys Path
# This ensures 'src' can be imported even if running from a subdirectory
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Points to /app
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now these imports will work because Python knows to look in /app
try:
    from src.utils.config_loader import ConfigLoader
    from src.utils.logger import setup_logger
except ImportError as e:
    # Fallback debug print if it still fails
    print(f"Failed to import src.utils: {e}")
    print(f"Current sys.path: {sys.path}")
    raise e


class ConfigLoaderResource(ConfigurableResource):
    """
    Dagster resource for loading pipeline configuration.
    """
    
    config_path: str = Field(
        default="config/pipeline_config.yaml",
        description="Path to pipeline configuration YAML file"
    )
    
    def load(self) -> Dict[str, Any]:
        """Load and return the pipeline configuration."""
        config_file = Path(self.config_path)
        
        # Handle relative paths in Docker
        if not config_file.is_absolute():
            config_file = PROJECT_ROOT / self.config_path

        if not config_file.exists():
            return {}
        
        loader = ConfigLoader(str(config_file))
        config = loader.load()
        
        return config
    
    def get_section(self, section: str) -> Dict[str, Any]:
        config = self.load()
        if section not in config:
            return {}
        return config[section]


class LoggerResource(ConfigurableResource):
    """Dagster resource for structured logging."""
    
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(default="logs/dagster_pipeline.log", description="Path to log file")
    
    def __init__(self, **data):
        super().__init__(**data)
        # Cache logger instance to avoid repeated initialization
        self._logger: Optional[logging.Logger] = None
    
    def _get_logger(self) -> logging.Logger:
        """Get or create cached logger instance."""
        if self._logger is None:
            # Use the setup_logger from src.utils
            self._logger = setup_logger(
                name="dagster_climate_pipeline",
                log_file=self.log_file,
                level=self.log_level
            )
        return self._logger
    
    def info(self, message: str, *args, **kwargs):
        self._get_logger().info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        self._get_logger().warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        self._get_logger().error(message, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs):
        self._get_logger().debug(message, *args, **kwargs)

    def exception(self, message: str, *args, **kwargs):
        self._get_logger().exception(message, *args, **kwargs)


class DataPathResource(ConfigurableResource):
    """Dagster resource for managing data directory paths."""
    
    raw_data_dir: str = Field(default="data/raw")
    processed_data_dir: str = Field(default="data/processed")
    embeddings_dir: str = Field(default="qdrant_db")
    
    def get_raw_path(self) -> Path:
        path = Path(self.raw_data_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_processed_path(self) -> Path:
        path = Path(self.processed_data_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_embeddings_path(self) -> Path:
        path = Path(self.embeddings_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path