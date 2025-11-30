import yaml
from pathlib import Path
from typing import Any, Dict

class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)

    def load(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return {}
        
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f) or {}