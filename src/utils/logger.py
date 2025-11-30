import logging
import sys
from pathlib import Path

def setup_logger(name: str, log_file: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger is retrieved multiple times
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 1. Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 2. File Handler (Optional)
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger