"""
Configuration Loader Utility
Handles loading and managing configuration from YAML files.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv


class ConfigLoader:
    """
    Configuration loader with environment variable support.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration loader.
        
        Parameters:
        -----------
        config_path : Path, optional
            Path to configuration file
        """
        self.config_path = config_path
        self.config = {}
        
        # Load environment variables
        load_dotenv()
        
        if config_path and Path(config_path).exists():
            self.load()
    
    def load(self, config_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Parameters:
        -----------
        config_path : Path, optional
            Path to configuration file
            
        Returns:
        --------
        dict
            Configuration dictionary
        """
        if config_path:
            self.config_path = config_path
        
        if not self.config_path:
            raise ValueError("No configuration path specified")
        
        config_path = Path(self.config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Replace environment variables
        self.config = self._replace_env_vars(self.config)
        
        return self.config
    
    def _replace_env_vars(self, obj: Any) -> Any:
        """
        Recursively replace environment variables in configuration.
        
        Parameters:
        -----------
        obj : any
            Configuration object (dict, list, or value)
            
        Returns:
        --------
        any
            Configuration with env vars replaced
        """
        if isinstance(obj, dict):
            return {k: self._replace_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            # Extract env var name
            env_var = obj[2:-1]
            default_value = None
            
            # Check for default value syntax: ${VAR:default}
            if ':' in env_var:
                env_var, default_value = env_var.split(':', 1)
            
            return os.getenv(env_var, default_value)
        else:
            return obj
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation path.
        
        Parameters:
        -----------
        key_path : str
            Dot-separated path to configuration key (e.g., 'data.input.directory')
        default : any, optional
            Default value if key not found
            
        Returns:
        --------
        any
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value by dot-notation path.
        
        Parameters:
        -----------
        key_path : str
            Dot-separated path to configuration key
        value : any
            Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def save(self, output_path: Optional[Path] = None) -> None:
        """
        Save configuration to YAML file.
        
        Parameters:
        -----------
        output_path : Path, optional
            Path to save configuration (defaults to original path)
        """
        if output_path is None:
            output_path = self.config_path
        
        if output_path is None:
            raise ValueError("No output path specified")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)


# Global configuration instance
_config_loader: Optional[ConfigLoader] = None


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Parameters:
    -----------
    config_path : Path
        Path to configuration file
        
    Returns:
    --------
    dict
        Configuration dictionary
    """
    global _config_loader
    _config_loader = ConfigLoader(config_path)
    return _config_loader.config


def get_config() -> ConfigLoader:
    """
    Get global configuration loader instance.
    
    Returns:
    --------
    ConfigLoader
        Global configuration loader
    """
    global _config_loader
    
    if _config_loader is None:
        # Try to load default configuration
        default_config = Path(__file__).parent.parent.parent / "config" / "pipeline_config.yaml"
        if default_config.exists():
            _config_loader = ConfigLoader(default_config)
        else:
            _config_loader = ConfigLoader()
    
    return _config_loader
