"""
Setup configuration for Climate Data ETL Pipeline
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = (this_directory / "requirements.txt").read_text(encoding='utf-8').splitlines()
dev_requirements = (this_directory / "requirements-dev.txt").read_text(encoding='utf-8').splitlines()

setup(
    name="climate-etl-pipeline",
    version="2.0.0",
    author="Climate Research Team",
    author_email="research@climate-data.org",
    description="Production ETL pipeline for climate data processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourname/climate-etl-pipeline",
    project_urls={
        "Bug Tracker": "https://github.com/yourname/climate-etl-pipeline/issues",
        "Documentation": "https://github.com/yourname/climate-etl-pipeline/docs",
        "Source Code": "https://github.com/yourname/climate-etl-pipeline",
    },
    packages=find_packages(exclude=["tests*", "docs*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "climate-etl=src.data_transformation.pipeline:main",
            "era5-download=src.data_acquisition.era5_downloader:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.txt", "*.md"],
    },
    zip_safe=False,
)
