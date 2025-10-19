"""
Test Suite for Climate Data Transformation Pipeline
Tests dimension names, variable ranges, and shape consistency.
"""

import sys
import unittest
import tempfile
import shutil
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
import json
import yaml

# Import modules to test
from src.data_transformation.ingestion import DataLoader
from src.data_transformation.transformations import DataTransformer, transform_pipeline
from src.data_transformation.export import DataExporter


class TestDataIngestion(unittest.TestCase):
    """Test the data ingestion module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = DataLoader()
    
    def tearDown(self):
        """Clean up test files."""
        shutil.rmtree(self.temp_dir)
    
    def test_netcdf_loading(self):
        """Test loading NetCDF files."""
        # Create test NetCDF file
        test_file = Path(self.temp_dir) / "test.nc"
        
        ds = xr.Dataset({
            't2m': (['time', 'lat', 'lon'], np.random.rand(10, 5, 5) * 50 + 250)
        }, coords={
            'time': pd.date_range('2024-01-01', periods=10, freq='h'),
            'lat': np.linspace(48, 51, 5),
            'lon': np.linspace(13, 19, 5)
        })
        ds.to_netcdf(test_file)
        ds.close()  # Close the file before loading
        
        # Test loading
        data = self.loader.load(test_file)
        
        self.assertIsInstance(data, xr.Dataset)
        self.assertIn('t2m', data.data_vars)
        self.assertEqual(self.loader.file_type, 'netcdf')
        
        # Close loaded data to release file handle
        if hasattr(data, 'close'):
            data.close()
    
    def test_csv_loading(self):
        """Test loading CSV files - Updated for unified API."""
        test_file = Path(self.temp_dir) / "test.csv"
        
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'temperature': np.random.rand(10) * 30,
            'precipitation': np.random.rand(10) * 100
        })
        df.to_csv(test_file, index=False)
        
        # Test loading - Now returns xr.Dataset (unified API)
        data = self.loader.load(test_file)
        
        self.assertIsInstance(data, xr.Dataset)  # Changed from pd.DataFrame
        self.assertIn('temperature', data.data_vars)
        self.assertIn('precipitation', data.data_vars)
        self.assertEqual(self.loader.file_type, 'csv')
    
    def test_json_loading(self):
        """Test loading JSON files - Updated for unified API."""
        test_file = Path(self.temp_dir) / "test.json"
        
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10).astype(str),
            'temperature': np.random.rand(10) * 30,
        })
        df.to_json(test_file, orient='records')
        
        # Test loading - Now returns xr.Dataset (unified API)
        data = self.loader.load(test_file)
        
        self.assertIsInstance(data, xr.Dataset)  # Changed from pd.DataFrame
        self.assertIn('temperature', data.data_vars)
        self.assertEqual(self.loader.file_type, 'json')
    
    def test_unsupported_extension(self):
        """Test that unsupported extensions raise ValueError."""
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("test content")
        
        with self.assertRaises(ValueError):
            self.loader.load(test_file)
    
    def test_file_not_found(self):
        """Test that missing files raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            self.loader.load("nonexistent_file.nc")


class TestDataTransformation(unittest.TestCase):
    """Test the data transformation module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.transformer = DataTransformer()
        
        # Create test dataset
        self.test_ds = xr.Dataset({
            't2m': (['valid_time', 'latitude', 'longitude'], 
                   np.random.rand(24, 5, 5) * 50 + 250)  # Kelvin
        }, coords={
            'valid_time': pd.date_range('2024-01-01', periods=24, freq='H'),
            'latitude': np.linspace(48, 51, 5),
            'longitude': np.linspace(13, 19, 5)
        })
    
    def test_dimension_renaming(self):
        """Test that dimensions are renamed correctly."""
        ds = self.transformer.rename_dimensions(self.test_ds)
        
        # Check renamed dimensions
        self.assertIn('time', ds.dims)
        self.assertIn('latitude', ds.dims)
        self.assertIn('longitude', ds.dims)
        self.assertNotIn('valid_time', ds.dims)
    
    def test_temperature_conversion(self):
        """Test temperature conversion from Kelvin to Celsius."""
        ds = self.transformer.convert_temperature_to_celsius(self.test_ds)
        
        # Check that temperature is now in reasonable Celsius range
        self.assertLess(ds['t2m'].max(), 100)
        self.assertGreater(ds['t2m'].min(), -100)
        self.assertEqual(ds['t2m'].attrs.get('units'), 'Celsius')
    
    def test_hourly_to_daily_aggregation(self):
        """Test aggregation from hourly to daily."""
        ds_renamed = self.transformer.rename_dimensions(self.test_ds)
        ds_daily = self.transformer.aggregate_hourly_to_daily(ds_renamed, method='mean')
        
        # Check that time dimension is reduced
        self.assertLess(len(ds_daily.time), len(ds_renamed.time))
        self.assertEqual(len(ds_daily.time), 1)  # 24 hours = 1 day
    
    def test_normalization_zscore(self):
        """Test Z-score normalization."""
        ds = self.transformer.normalize_variables(self.test_ds, method='zscore')
        
        # Check that mean is close to 0 and std is close to 1
        mean = float(ds['t2m'].mean().values)
        std = float(ds['t2m'].std().values)
        
        self.assertAlmostEqual(mean, 0.0, places=5)
        self.assertAlmostEqual(std, 1.0, places=5)
        self.assertEqual(ds['t2m'].attrs.get('normalization'), 'zscore')
    
    def test_normalization_minmax(self):
        """Test min-max normalization."""
        ds = self.transformer.normalize_variables(self.test_ds, method='minmax')
        
        # Check that values are in [0, 1] range
        min_val = float(ds['t2m'].min().values)
        max_val = float(ds['t2m'].max().values)
        
        self.assertAlmostEqual(min_val, 0.0, places=5)
        self.assertAlmostEqual(max_val, 1.0, places=5)
        self.assertEqual(ds['t2m'].attrs.get('normalization'), 'minmax')
    
    def test_full_pipeline(self):
        """Test the complete transformation pipeline - Updated for new API."""
        # Use new modular API instead of transform_pipeline function
        transformer = DataTransformer()
        
        # Apply transformations step by step
        ds = transformer.rename_dimensions(self.test_ds)
        ds = transformer.convert_temperature_to_celsius(ds, variables=['t2m'])
        ds = transformer.normalize_variables(ds, method='zscore')
        
        # Check that transformations were applied
        self.assertGreater(len(transformer.transformations_applied), 0)
        self.assertIn('time', ds.dims)
        self.assertLess(abs(ds['t2m'].mean()), 1.0)  # Normalized data
        
        # Verify transformation log
        log = transformer.transformations_applied
        self.assertTrue(any('Renamed' in t for t in log))
        self.assertTrue(any('Celsius' in t for t in log))
        self.assertTrue(any('Normalized' in t for t in log))


class TestDataExport(unittest.TestCase):
    """Test the data export module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = DataExporter(output_dir=self.temp_dir)
        
        # Create test dataset
        self.test_ds = xr.Dataset({
            'temperature': (['time', 'lat', 'lon'], np.random.rand(10, 5, 5) * 30)
        }, coords={
            'time': pd.date_range('2024-01-01', periods=10),
            'lat': np.linspace(48, 51, 5),
            'lon': np.linspace(13, 19, 5)
        })
    
    def tearDown(self):
        """Clean up test files."""
        shutil.rmtree(self.temp_dir)
    
    def test_netcdf_export(self):
        """Test NetCDF export."""
        output_path = self.exporter.save_netcdf(self.test_ds, 'test')
        
        self.assertTrue(output_path.exists())
        self.assertEqual(output_path.suffix, '.nc')
        
        # Verify content
        ds_loaded = xr.open_dataset(output_path)
        self.assertIn('temperature', ds_loaded.data_vars)
        ds_loaded.close()
    
    def test_parquet_export(self):
        """Test Parquet export."""
        output_path = self.exporter.save_parquet(self.test_ds, 'test')
        
        self.assertTrue(output_path.exists())
        self.assertEqual(output_path.suffix, '.parquet')
        
        # Verify content
        df_loaded = pd.read_parquet(output_path)
        self.assertIn('temperature', df_loaded.columns)
    
    def test_csv_export(self):
        """Test CSV export."""
        output_path = self.exporter.save_csv(self.test_ds, 'test')
        
        self.assertTrue(output_path.exists())
        self.assertEqual(output_path.suffix, '.csv')
        
        # Verify content
        df_loaded = pd.read_csv(output_path)
        self.assertIn('temperature', df_loaded.columns)
    
    def test_both_formats_export(self):
        """Test exporting to both NetCDF and Parquet."""
        output_files = self.exporter.save_both_formats(self.test_ds, 'test')
        
        self.assertIn('netcdf', output_files)
        self.assertIn('parquet', output_files)
        self.assertTrue(output_files['netcdf'].exists())
        self.assertTrue(output_files['parquet'].exists())
    
    def test_summary_report_generation(self):
        """Test summary report generation."""
        transformation_log = ['Renamed dimensions', 'Converted temperature']
        output_files = {'netcdf': Path(self.temp_dir) / 'test.nc'}
        
        # Save netcdf first
        self.exporter.save_netcdf(self.test_ds, 'test')
        
        report_path = self.exporter.generate_summary_report(
            self.test_ds,
            transformation_log,
            output_files
        )
        
        self.assertTrue(report_path.exists())
        
        # Check report content
        content = report_path.read_text()
        self.assertIn('TRANSFORMATION REPORT', content)
        self.assertIn('temperature', content)


class TestShapeConsistency(unittest.TestCase):
    """Test that data shapes remain consistent through transformations."""
    
    def test_shape_preservation(self):
        """Test that spatial dimensions are preserved."""
        # Create test dataset
        original_ds = xr.Dataset({
            'temp': (['time', 'lat', 'lon'], np.random.rand(24, 10, 10) * 50 + 250)
        }, coords={
            'time': pd.date_range('2024-01-01', periods=24, freq='H'),
            'lat': np.linspace(45, 55, 10),
            'lon': np.linspace(10, 20, 10)
        })
        
        # Apply transformations
        transformer = DataTransformer()
        ds = transformer.rename_dimensions(original_ds)
        ds = transformer.convert_temperature_to_celsius(ds)
        ds = transformer.normalize_variables(ds)
        
        # Check that spatial dimensions are preserved
        self.assertEqual(len(ds.latitude), 10)
        self.assertEqual(len(ds.longitude), 10)


class TestVariableRanges(unittest.TestCase):
    """Test that variable values are in expected ranges."""
    
    def test_celsius_temperature_range(self):
        """Test that converted temperatures are in valid range."""
        ds = xr.Dataset({
            't2m': (['time', 'lat', 'lon'], np.random.rand(10, 5, 5) * 50 + 250)
        }, coords={
            'time': pd.date_range('2024-01-01', periods=10),
            'lat': np.linspace(48, 51, 5),
            'lon': np.linspace(13, 19, 5)
        })
        
        transformer = DataTransformer()
        ds = transformer.convert_temperature_to_celsius(ds)
        
        # Check reasonable temperature range (-50 to 50 Celsius)
        self.assertGreater(ds['t2m'].min(), -100)
        self.assertLess(ds['t2m'].max(), 100)
    
    def test_normalized_range(self):
        """Test that normalized values are correctly scaled."""
        ds = xr.Dataset({
            'var': (['time'], np.random.rand(100) * 100)
        }, coords={
            'time': pd.date_range('2024-01-01', periods=100)
        })
        
        transformer = DataTransformer()
        
        # Test Z-score
        ds_zscore = transformer.normalize_variables(ds.copy(), method='zscore')
        # Most values should be within [-3, 3] standard deviations
        self.assertLess(abs(ds_zscore['var'].mean()), 0.01)
        
        # Test min-max
        transformer2 = DataTransformer()
        ds_minmax = transformer2.normalize_variables(ds.copy(), method='minmax')
        self.assertAlmostEqual(float(ds_minmax['var'].min()), 0.0, places=5)
        self.assertAlmostEqual(float(ds_minmax['var'].max()), 1.0, places=5)


class TestTransformationHelper(unittest.TestCase):
    """Test transform_pipeline helper function."""
    
    def test_transform_pipeline_all_options(self):
        """Test transform_pipeline with all transformations enabled."""
        # Create sample dataset with old dimension names
        ds = xr.Dataset({
            't2m': (['valid_time', 'latitude', 'longitude'], 
                    np.random.randn(24, 5, 8) * 10 + 273.15),
            'tp': (['valid_time', 'latitude', 'longitude'],
                   np.abs(np.random.randn(24, 5, 8)) * 1000)
        }, coords={
            'valid_time': pd.date_range('2024-01-01', periods=24, freq='h'),
            'latitude': np.linspace(-90, 90, 5),
            'longitude': np.linspace(-180, 180, 8)
        })
        
        # Apply all transformations
        result_ds, log = transform_pipeline(
            ds,
            rename_dims=True,
            convert_temp=True,
            convert_precip=True,
            aggregate_to_daily=True,
            normalize=True,
            normalize_method='minmax'
        )
        
        # Verify dimension renaming
        self.assertIn('time', result_ds.dims)
        
        # Verify temperature conversion (should be in Celsius)
        if 'temperature' in result_ds.data_vars:
            temp_mean = float(result_ds['temperature'].mean())
            self.assertLess(temp_mean, 100)  # Should be in Celsius range
        
        # Verify aggregation happened (should have 1 day)
        self.assertEqual(len(result_ds.coords['time']), 1)
        
        # Verify normalization (values should be [0, 1])
        for var in result_ds.data_vars:
            self.assertGreaterEqual(float(result_ds[var].min()), 0.0)
            self.assertLessEqual(float(result_ds[var].max()), 1.0)
        
        # Verify log is generated
        self.assertIsInstance(log, list)
        self.assertGreater(len(log), 0)
    
    def test_transform_pipeline_selective(self):
        """Test transform_pipeline with selective options."""
        ds = xr.Dataset({
            't2m': (['valid_time', 'latitude', 'longitude'], 
                    np.random.randn(24, 5, 8) * 10 + 273.15)
        }, coords={
            'valid_time': pd.date_range('2024-01-01', periods=24, freq='h'),
            'latitude': np.linspace(-90, 90, 5),
            'longitude': np.linspace(-180, 180, 8)
        })
        
        # Apply only dimension renaming
        result_ds, log = transform_pipeline(
            ds,
            rename_dims=True,
            convert_temp=False,
            convert_precip=False
        )
        
        # Should have renamed dimensions
        self.assertIn('time', result_ds.dims)
        
        # Should still have 24 time steps (no aggregation)
        self.assertEqual(len(result_ds.coords['time']), 24)
    
    def test_transform_pipeline_no_ops(self):
        """Test transform_pipeline with no operations."""
        ds = xr.Dataset({
            'var': (['time'], np.random.randn(10))
        }, coords={
            'time': pd.date_range('2024-01-01', periods=10)
        })
        
        # Apply no transformations (disable default normalize)
        result_ds, log = transform_pipeline(
            ds,
            rename_dims=False,
            convert_temp=False,
            convert_precip=False,
            normalize=False
        )
        
        # Dataset should be unchanged
        self.assertEqual(len(result_ds.coords['time']), 10)
        xr.testing.assert_allclose(ds['var'], result_ds['var'])


class TestExportFormats(unittest.TestCase):
    """Test export functionality for multiple formats."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path('tests/test_data')
        self.test_dir.mkdir(exist_ok=True)
        
        # Create sample dataset
        self.ds = xr.Dataset({
            'temperature': (['time', 'latitude', 'longitude'], 
                          np.random.randn(24, 5, 8) * 10 + 273.15),
            'precipitation': (['time', 'latitude', 'longitude'],
                            np.abs(np.random.randn(24, 5, 8)) * 0.01)
        }, coords={
            'time': pd.date_range('2024-01-01', periods=24, freq='h'),
            'latitude': np.linspace(-90, 90, 5),
            'longitude': np.linspace(-180, 180, 8)
        })
        
        self.exporter = DataExporter(output_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_export_to_csv(self):
        """Test CSV export functionality."""
        csv_file = self.exporter.save_csv(self.ds, filename='test_export.csv')
        
        # Verify file exists
        self.assertTrue(csv_file.exists())
        
        # Verify can read back
        df = pd.read_csv(csv_file)
        self.assertGreater(len(df), 0)
        self.assertIn('temperature', df.columns)
        self.assertIn('precipitation', df.columns)
    
    def test_export_to_json(self):
        """Test JSON export functionality."""
        # Export to JSON (first need to export as netcdf)
        nc_file = self.exporter.save_netcdf(self.ds, filename='test_temp.nc')
        
        # DataExporter doesn't have direct JSON export, use xarray
        json_data = self.ds.to_dict()
        json_file = self.test_dir / 'test_export.json'
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        # Verify file exists
        self.assertTrue(json_file.exists())
        
        # Verify valid JSON structure
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.assertIn('data_vars', data)
        self.assertIsInstance(data, dict)
    
    def test_export_to_parquet(self):
        """Test Parquet export functionality."""
        parquet_file = self.exporter.save_parquet(self.ds, filename='test_export.parquet')
        
        # Verify file exists
        self.assertTrue(parquet_file.exists())
        
        # Verify can read back with pandas
        df = pd.read_parquet(parquet_file)
        self.assertGreater(len(df), 0)
        self.assertIn('temperature', df.columns)
        self.assertIn('precipitation', df.columns)


class TestDocumentationReports(unittest.TestCase):
    """Test documentation report generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path('tests/test_data')
        self.test_dir.mkdir(exist_ok=True)
        
        # Create sample dataset with attributes
        self.ds = xr.Dataset({
            'temperature': (['time', 'latitude', 'longitude'], 
                          np.random.randn(24, 5, 8) * 10 + 273.15,
                          {'units': 'K', 'long_name': 'Temperature'}),
            'precipitation': (['time', 'latitude', 'longitude'],
                            np.abs(np.random.randn(24, 5, 8)) * 0.01,
                            {'units': 'm', 'long_name': 'Precipitation'})
        }, coords={
            'time': pd.date_range('2024-01-01', periods=24, freq='h'),
            'latitude': np.linspace(-90, 90, 5),
            'longitude': np.linspace(-180, 180, 8)
        })
        
        self.exporter = DataExporter(output_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_generate_json_report(self):
        """Test JSON documentation report generation."""
        report_file = self.exporter.generate_documentation_report(
            self.ds, 
            format='json',
            filename='test_report.json'
        )
        
        # Verify file exists
        self.assertTrue(report_file.exists())
        
        # Verify report structure
        with open(report_file, 'r') as f:
            report = json.load(f)
        
        # Check metadata
        self.assertIn('metadata', report)
        self.assertIn('pipeline_version', report['metadata'])
        
        # Check dataset info
        self.assertIn('dataset_info', report)
        self.assertIn('dimensions', report['dataset_info'])
        self.assertIn('variables', report['dataset_info'])
        
        # Check variable statistics
        self.assertIn('variable_statistics', report)
        self.assertIn('temperature', report['variable_statistics'])
        self.assertIn('precipitation', report['variable_statistics'])
        
        # Verify statistics content
        temp_stats = report['variable_statistics']['temperature']
        self.assertIn('min', temp_stats)
        self.assertIn('max', temp_stats)
        self.assertIn('mean', temp_stats)
        self.assertIn('units', temp_stats)
    
    def test_generate_yaml_report(self):
        """Test YAML documentation report generation."""
        report_file = self.exporter.generate_documentation_report(
            self.ds, 
            format='yaml',
            filename='test_report.yaml'
        )
        
        # Verify file exists
        self.assertTrue(report_file.exists())
        
        # Verify valid YAML structure
        with open(report_file, 'r') as f:
            report = yaml.safe_load(f)
        
        self.assertIn('metadata', report)
        self.assertIn('dataset_info', report)
        self.assertIn('variable_statistics', report)


def run_tests():
    """Run all tests and generate report."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataIngestion))
    suite.addTests(loader.loadTestsFromTestCase(TestDataTransformation))
    suite.addTests(loader.loadTestsFromTestCase(TestDataExport))
    suite.addTests(loader.loadTestsFromTestCase(TestShapeConsistency))
    suite.addTests(loader.loadTestsFromTestCase(TestVariableRanges))
    suite.addTests(loader.loadTestsFromTestCase(TestTransformationHelper))
    suite.addTests(loader.loadTestsFromTestCase(TestExportFormats))
    suite.addTests(loader.loadTestsFromTestCase(TestDocumentationReports))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 80)
    
    return result


if __name__ == "__main__":
    result = run_tests()
    
    # Exit with appropriate code
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)
