# Test Download Types

This document verifies that the system supports both local files and HTTP/HTTPS downloads.

## Supported URL Types

### 1. HTTP/HTTPS URLs (Normal Downloads)
```json
{
  "source_id": "example_http",
  "url": "https://data.example.com/climate_data.nc",
  "format": "netcdf"
}
```
✅ **Works**: Downloads via `requests.get()`

### 2. Local Files (file:// protocol)
```json
{
  "source_id": "example_local",
  "url": "file:///path/to/data.csv",
  "format": "csv"
}
```
✅ **Works**: Copies file using `shutil.copy2()`

### 3. Direct File Paths (relative or absolute)
```json
{
  "source_id": "example_path",
  "url": "data/raw/sample.csv",
  "format": "csv"
}
```
✅ **Works**: Resolves path and copies file

## Implementation

Both `dynamic_source_ops.py` and `dynamic_jobs.py` now handle:
- `file://` URLs → Local file copy
- Direct paths → Local file copy (if file exists)
- `http://` or `https://` URLs → HTTP download (default/else case)

The logic checks in this order:
1. If URL starts with `file://` → copy local file
2. If URL is a file path (exists and doesn't start with http) → copy local file  
3. Otherwise → HTTP/HTTPS download (normal behavior)

This ensures backward compatibility - all existing HTTP sources continue to work!

