# climate_embeddings/embeddings/numpy_embeddings.py
import numpy as np

def compute_raster_embedding(data: np.ndarray) -> np.ndarray:
    """
    Turns a raw raster chunk (Time, Y, X) or (Y, X) into a fixed-size vector.
    We compute statistics that describe the distribution.
    Output Dim: 8 dimensions [mean, std, min, max, p10, p50, p90, trend]
    """
    # Flatten spatial dims, keep time if relevant, but for simple stats, flatten all valid
    valid = data[np.isfinite(data)]
    
    if valid.size == 0:
        return np.zeros(8, dtype="float32")
    
    # Calculate Trend (simple linear slope approximation if data is 1D or we take mean over space)
    trend = 0.0
    # (Simple logic: difference between second half mean and first half mean)
    if valid.size > 10:
        half = valid.size // 2
        trend = np.mean(valid[half:]) - np.mean(valid[:half])

    percentiles = np.percentile(valid, [10, 50, 90])
    
    vector = np.array([
        np.mean(valid),
        np.std(valid),
        np.min(valid),
        np.max(valid),
        percentiles[0],
        percentiles[1],
        percentiles[2],
        trend
    ], dtype="float32")
    
    return vector