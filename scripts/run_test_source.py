"""
Download specified test source and run embedding pipeline on it.
"""
import requests
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` imports work when running this script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import EmbeddingPipeline, SemanticSearcher
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

URL = "https://github.com/pydata/xarray-data/raw/master/air_temperature.nc"
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

local_path = RAW_DIR / "test_xarray_air_from_url.nc"

logger.info(f"Downloading {URL} -> {local_path}")
resp = requests.get(URL, stream=True, timeout=120)
resp.raise_for_status()
with open(local_path, 'wb') as f:
    for chunk in resp.iter_content(chunk_size=8192):
        f.write(chunk)
logger.info("Download complete")

pipeline = EmbeddingPipeline()
res = pipeline.process_dataset(str(local_path))
logger.info(f"Pipeline result: {res}")

# Run a quick semantic search to verify embeddings are queryable
searcher = SemanticSearcher(generator=pipeline.generator, database=pipeline.database)
results = searcher.search("air temperature mean statistics", k=5)
logger.info(f"Search returned {len(results)} results")
for r in results[:3]:
    logger.info(f"- id={r.get('id')} distance={r.get('distance')} variable={r.get('metadata',{}).get('variable')}")
print('DONE')
