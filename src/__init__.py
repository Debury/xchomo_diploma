# src/__init__.py

# Simply expose the package, don't try to re-import specific functions here
# to prevent circular imports with Dagster resources.
from . import climate_embeddings

__all__ = ["climate_embeddings"]