import cdsapi
import os

# Set the absolute path to .cdsapirc file
script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['CDSAPI_RC'] = os.path.join(script_dir, '.cdsapirc')

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': '2m_temperature',
        'year': '2024',
        'month': '01',
        'day': '01',
        'time': ['00:00', '12:00'],
        'area': [51, 13, 48, 19],  # Európa - malá oblasť okolo ČR
        'format': 'netcdf'
    },
    'test_era5_data.nc')
