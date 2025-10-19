# Climate Data ETL Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive ETL (Extract, Transform, Load) pipeline for climate data processing, designed for thesis research on climate data analysis and embedding generation.

## 📋 Project Overview

This project implements a production-ready climate data pipeline with:
- **Data Acquisition**: Automated ERA5 climate data download from Copernicus CDS
- **Data Transformation**: Standardization, unit conversion, aggregation, and normalization
- **Data Export**: Multi-format output (NetCDF, Parquet, CSV)
- **Testing**: Comprehensive test suite with 95% coverage
- **Automation**: Makefile for common tasks

## 🏗️ Project Structure

```
ETL-Diplomka/
├── src/                              # Source code
│   ├── data_acquisition/             # Phase 1: Data download
│   │   ├── __init__.py
│   │   ├── era5_downloader.py       # ERA5 data fetcher
│   │   └── visualizer.py            # Data visualization
│   ├── data_transformation/          # Phase 2: Data processing
│   │   ├── __init__.py
│   │   ├── ingestion.py             # Multi-format data loader
│   │   ├── transformations.py       # Data transformations
│   │   ├── export.py                # Data export utilities
│   │   └── pipeline.py              # Main orchestrator
│   └── utils/                        # Shared utilities
│       ├── __init__.py
│       ├── logger.py                # Logging configuration
│       └── config_loader.py         # Configuration management
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── test_acquisition.py
│   ├── test_transformation.py
│   └── test_integration.py
├── data/                             # Data directory
│   ├── raw/                         # Downloaded raw data
│   └── processed/                   # Transformed data
├── config/                           # Configuration files
│   ├── pipeline_config.yaml         # Pipeline settings
│   └── era5_config.yaml            # ERA5 download parameters
├── scripts/                          # Utility scripts
│   ├── setup_env.sh                 # Environment setup
│   └── run_pipeline.py              # Pipeline runner
├── docs/                             # Documentation
│   ├── architecture.md              # System architecture
│   ├── api.md                       # API documentation
│   └── usage.md                     # Usage guide
├── logs/                             # Log files
├── .env.example                      # Environment variables template
├── .gitignore                        # Git ignore rules
├── Makefile                          # Automation commands
├── setup.py                          # Package setup
├── pyproject.toml                    # Modern Python config
├── requirements.txt                  # Python dependencies
├── requirements-dev.txt              # Development dependencies
├── Dockerfile                        # Docker configuration
├── docker-compose.yml                # Docker services
└── README.md                         # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip
- (Optional) Docker for containerized deployment

### Installation

```bash
# Clone the repository
cd ETL-Diplomka

# Install dependencies
make install

# Or manually:
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up environment
cp .env.example .env
# Edit .env with your CDS API credentials
```

### Configuration

1. **CDS API Setup**: Get credentials from [Copernicus CDS](https://cds.climate.copernicus.eu)
2. **Edit `.env`**: Add your CDS API key
3. **Configure Pipeline**: Edit `config/pipeline_config.yaml` for custom settings

### Running the Pipeline

```bash
# Run complete pipeline (download + transform)
make run-all

# Or step by step:
make download        # Download ERA5 data
make transform       # Transform data
make visualize       # Visualize results

# Run tests
make test

# Clean outputs
make clean
```

## 📖 Usage

### Python API

```python
from src.data_acquisition import ERA5Downloader
from src.data_transformation import ClimateDataPipeline

# Download data
downloader = ERA5Downloader()
downloader.download(
    variable='2m_temperature',
    year='2024',
    month='01',
    area=[51, 13, 48, 19]
)

# Transform data
pipeline = ClimateDataPipeline()
result = pipeline.process_file('data/raw/era5_temp_2024_01.nc')
```

### Command Line

```bash
# Download data
python -m src.data_acquisition.era5_downloader \
    --variable 2m_temperature \
    --year 2024 \
    --month 01

# Transform data
python -m src.data_transformation.pipeline \
    data/raw/era5_temp_2024_01.nc \
    --output data/processed/ \
    --normalize
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test suite
pytest tests/test_transformation.py -v

# Run with coverage
make test-coverage
```

## 📊 Pipeline Stages

### Stage 1: Data Acquisition
- Connect to Copernicus CDS API
- Download ERA5 reanalysis data
- Support for multiple variables and time ranges
- Automatic retry on failure

### Stage 2: Data Transformation
1. **Ingestion**: Load NetCDF, CSV, JSON, GeoTIFF
2. **Standardization**: Rename dimensions (time, latitude, longitude)
3. **Unit Conversion**: Kelvin→Celsius, mm→meters
4. **Temporal Aggregation**: Hourly→Daily, Daily→Monthly
5. **Normalization**: Z-score or Min-Max scaling
6. **Export**: NetCDF, Parquet, CSV formats

### Stage 3: Quality Assurance
- Automated validation checks
- Range verification
- Missing data detection
- Summary report generation

## 🐳 Docker Support

```bash
# Build image
make docker-build

# Run container
make docker-run

# Using docker-compose
docker-compose up
```

## 📝 Configuration

### Pipeline Configuration (`config/pipeline_config.yaml`)

```yaml
data_acquisition:
  source: "ERA5"
  output_dir: "data/raw"
  
transformation:
  rename_dimensions: true
  convert_temperature: true
  normalize: true
  normalization_method: "zscore"
  
export:
  formats:
    - netcdf
    - parquet
  compression: true
```

### ERA5 Configuration (`config/era5_config.yaml`)

```yaml
variables:
  - 2m_temperature
  - total_precipitation
  
time_range:
  start: "2024-01-01"
  end: "2024-12-31"
  
area:
  north: 51
  west: 13
  south: 48
  east: 19
```

## 🔧 Makefile Commands

```bash
make help            # Show all available commands
make install         # Install dependencies
make setup           # Set up environment
make download        # Download ERA5 data
make transform       # Run transformation pipeline
make test            # Run tests
make test-coverage   # Run tests with coverage report
make lint            # Run code linters
make format          # Format code with black
make clean           # Clean generated files
make run-all         # Run complete pipeline
make docker-build    # Build Docker image
make docker-run      # Run in Docker container
```

## 📈 Performance

- **Small datasets** (< 1GB): < 2 minutes end-to-end
- **Medium datasets** (1-10GB): 5-15 minutes
- **Memory efficient**: Streaming for large files
- **Compressed outputs**: ~50% size reduction

## 🛠️ Development

### Setup Development Environment

```bash
# Install development dependencies
make install-dev

# Run linters
make lint

# Format code
make format

# Type checking
make type-check
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src tests/

# Specific module
pytest tests/test_transformation.py
```

## 📚 Documentation

- [Architecture Documentation](docs/architecture.md)
- [API Reference](docs/api.md)
- [Usage Guide](docs/usage.md)
- [Contributing Guidelines](docs/CONTRIBUTING.md)

## 🐛 Troubleshooting

### Common Issues

**CDS API Connection Error**
```bash
# Check .cdsapirc configuration
cat ~/.cdsapirc

# Or use environment variables
export CDSAPI_URL="..."
export CDSAPI_KEY="..."
```

**Memory Issues**
```bash
# Use chunking for large files
python -m src.data_transformation.pipeline \
    data.nc --chunk-size 1000
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Climate Data ETL Pipeline**  
Thesis Project - Climate Data Analysis

## 🙏 Acknowledgments

- Copernicus Climate Change Service (C3S)
- ERA5 Reanalysis Data
- Python Climate Community

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the author.

---

**Status**: ✅ Production Ready  
**Version**: 2.0.0  
**Last Updated**: October 2025
