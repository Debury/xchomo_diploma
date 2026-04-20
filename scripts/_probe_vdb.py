import os
import sys
sys.path.insert(0, "/app")
env_had = os.environ.pop("QDRANT_HOST", None)
print("env QDRANT_HOST was:", env_had)

import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("/app/config/pipeline_config.yaml").read_text())
print("cfg qdrant host:", cfg.get("vector_db", {}).get("qdrant", {}).get("host"))

from src.embeddings.database import VectorDatabase
vdb = VectorDatabase(config=cfg)
print("resolved vdb.host:", vdb.host, "port:", vdb.port)

from src.embeddings.database import VectorDatabase
vdb2 = VectorDatabase()  # no config
print("no-config vdb.host:", vdb2.host, "port:", vdb2.port)
