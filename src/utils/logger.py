"""
Logging Configuration Utility
Provides centralized logging setup for the entire pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json
import yaml


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.
    
    Parameters:
    -----------
    name : str
        Logger name
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : Path, optional
        Path to log file
    log_format : str, optional
        Custom log format string
        
    Returns:
    --------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Handle both string and int log levels
    if isinstance(level, str):
        log_level = getattr(logging, level.upper(), logging.INFO)
    else:
        log_level = level
    
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Default format
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one.
    
    Parameters:
    -----------
    name : str
        Logger name
        
    Returns:
    --------
    logging.Logger
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set up basic configuration
    if not logger.handlers:
        logger = setup_logger(name)
    
    return logger


def configure_logging_from_config(config_path: Path) -> None:
    """
    Configure logging from a YAML configuration file.
    
    Parameters:
    -----------
    config_path : Path
        Path to YAML configuration file
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'logging' in config:
        logging_config = config['logging']
        logging.config.dictConfig(logging_config)


class MetricsLogger:
    """
    Logger for tracking pipeline performance metrics.
    """
    
    def __init__(self, metrics_file: Path = Path('logs/metrics.log')):
        """
        Initialize metrics logger.
        
        Parameters:
        -----------
        metrics_file : Path
            Path to metrics log file
        """
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger('metrics')
    
    def log_metric(self, 
                   metric_name: str, 
                   value: Any, 
                   unit: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a single metric with timestamp.
        
        Parameters:
        -----------
        metric_name : str
            Name of the metric
        value : Any
            Metric value
        unit : str, optional
            Unit of measurement
        metadata : dict, optional
            Additional metadata
        """
        metric_entry = {
            'timestamp': datetime.now().isoformat(),
            'metric_name': metric_name,
            'value': value,
            'unit': unit,
            'metadata': metadata or {}
        }
        
        # Write to JSON log file
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metric_entry) + '\n')
        
        # Also log to standard logger
        metric_str = f"{metric_name}: {value}"
        if unit:
            metric_str += f" {unit}"
        self.logger.info(metric_str)
    
    def log_file_size(self, filepath: Path, label: str = "file") -> None:
        """
        Log file size metric.
        
        Parameters:
        -----------
        filepath : Path
            Path to file
        label : str
            Label for the file
        """
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            self.log_metric(
                f"{label}_size",
                round(size_mb, 2),
                unit="MB",
                metadata={'filepath': str(filepath)}
            )
    
    def log_duration(self, operation: str, duration_seconds: float) -> None:
        """
        Log operation duration metric.
        
        Parameters:
        -----------
        operation : str
            Name of the operation
        duration_seconds : float
            Duration in seconds
        """
        self.log_metric(
            f"{operation}_duration",
            round(duration_seconds, 2),
            unit="seconds"
        )
    
    def log_processing_stats(self, 
                            input_file: Path,
                            output_files: Dict[str, Path],
                            duration: float,
                            num_variables: int,
                            num_transformations: int) -> None:
        """
        Log comprehensive processing statistics.
        
        Parameters:
        -----------
        input_file : Path
            Input file path
        output_files : dict
            Dictionary of output file paths
        duration : float
            Processing duration in seconds
        num_variables : int
            Number of variables processed
        num_transformations : int
            Number of transformations applied
        """
        stats = {
            'timestamp': datetime.now().isoformat(),
            'input_file': str(input_file),
            'input_size_mb': round(input_file.stat().st_size / (1024 * 1024), 2) if input_file.exists() else 0,
            'output_files': {},
            'duration_seconds': round(duration, 2),
            'num_variables': num_variables,
            'num_transformations': num_transformations
        }
        
        for fmt, filepath in output_files.items():
            if filepath.exists():
                stats['output_files'][fmt] = {
                    'path': str(filepath),
                    'size_mb': round(filepath.stat().st_size / (1024 * 1024), 2)
                }
        
        # Write comprehensive stats
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(stats) + '\n')
        
        self.logger.info(f"Processing completed in {duration:.2f}s - {num_variables} variables, {num_transformations} transformations")


# Module-level logger
logger = get_logger(__name__)
