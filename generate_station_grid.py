import pandas as pd
from tqdm import tqdm

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Geod

df = pd.read_csv('/scratch/08105/ms86336/ismn_txson_2014_2022.csv')
df.head()

df['Date_Time'] = pd.to_datetime(df['Date_Time'])

# Define grid properties
lower_left_lat = df.Latitude.min()
lower_left_lon = df.Longitude.min()
upper_right_lat = df.Latitude.max()
upper_right_lon = df.Longitude.max()

# Define the resolution of the grid in degrees using an approximate conversion from 14km to degrees
geod = Geod(ellps='WGS84')
distance_km = 14  # km
_, _, dx = geod.inv(lower_left_lon, lower_left_lat, lower_left_lon + 1, lower_left_lat)  # 1 degree longitude
dx_km = dx / 1000
grid_spacing_lon = distance_km / dx_km

_, _, dy = geod.inv(lower_left_lon, lower_left_lat, lower_left_lon, lower_left_lat + 1)  # 1 degree latitude
dy_km = dy / 1000
grid_spacing_lat = distance_km / dy_km

# Create the grid
times = pd.to_datetime(df['Date_Time'].unique())
lats = np.arange(lower_left_lat, upper_right_lat, grid_spacing_lat)
lons = np.arange(lower_left_lon, upper_right_lon, grid_spacing_lon)

# Create an empty xarray dataset for the values and a count array
grid = xr.Dataset(
    {
        'Value': (('time', 'lat', 'lon'), np.full((len(times), len(lats), len(lons)), np.nan)),
        'Count': (('time', 'lat', 'lon'), np.zeros((len(times), len(lats), len(lons))))
    },
    coords={
        'time': times,
        'lat': lats,
        'lon': lons
    }
)

# Initialize with zero to facilitate summing
grid['Value'][:] = 0  # Set all values to zero initially for summing

# Fill the station data into the nearest grid point and keep track of counts
for i, row in df.iterrows():
    lat_idx = np.abs(lats - row['Latitude']).argmin()  # Find nearest latitude
    lon_idx = np.abs(lons - row['Longitude']).argmin()  # Find nearest longitude
    time_idx = np.where(times == row['Date_Time'])[0][0]
    
    # Sum the value to the existing value in the nearest grid point
    grid['Value'][time_idx, lat_idx, lon_idx] += row['Value']
    
    # Increment the count for averaging later
    grid['Count'][time_idx, lat_idx, lon_idx] += 1

# Calculate the average by dividing the summed values by the counts, but only where counts are > 0
with np.errstate(invalid='ignore'):  # Ignore warnings due to division by zero
    grid['Value'] = xr.where(grid['Count'] > 0, grid['Value'] / grid['Count'], np.nan)
    
# Drop the count variable as it's no longer needed
grid = grid.drop_vars('Count')

# The result is an xarray dataset with the required format where values are averaged
grid

