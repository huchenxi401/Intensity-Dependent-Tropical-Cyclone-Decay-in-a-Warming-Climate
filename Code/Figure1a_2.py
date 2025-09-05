import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

file_paths = {
    'WP': '../DATA/WP/TC_decay_timescale.nc',
    'EP': '../DATA/EP/TC_decay_timescale.nc',
    'NI': '../DATA/NI/TC_decay_timescale.nc',
    'SI': '../DATA/SI/TC_decay_timescale.nc',
    'SP': '../DATA/SP/TC_decay_timescale.nc',
    'NA': '../DATA/NA/TC_decay_timescale.nc'
}

REGION_BOUNDS = {
    'EP': {'lon': (240, 260), 'lat': (15, 35)},
    'NA': {'lon': (260, 290), 'lat': (10, 40)},
    'NI': {'lon': (45, 100), 'lat': (5, 30)},
    'SI': {'lon': (30, 54), 'lat': (-30, -10)},
    'SP': {'lon': (125, 153), 'lat': (-31, -10)},
    'WP': {'lon': (100, 140), 'lat': (10, 40)}
}

all_data = {}

for region, filepath in file_paths.items():
    ds = xr.open_dataset(filepath)
    all_data[region] = {
        'year': ds['year'].values,
        'V0': ds['V0'].values,
        'decay_timescale': ds['decay_timescale'].values,
        'alllon': ds['alllon'].values,
        'alllat': ds['alllat'].values
    }
    ds.close()

global_data = {
    'year': np.concatenate([data['year'] for data in all_data.values()]),
    'V0': np.concatenate([data['V0'] for data in all_data.values()]),
    'decay_timescale': np.concatenate([data['decay_timescale'] for data in all_data.values()])
}

all_data['Global'] = global_data
REGION_BOUNDS['Global'] = {'lon': (125+100, 125+100+30), 'lat': (-30, -10)} 
plt.figure(figsize=(9, 4))
proj = ccrs.PlateCarree(central_longitude=180)
ax = plt.axes(projection=proj)
ax.set_extent([30, 290, -40, 50], crs=ccrs.PlateCarree())    

ax.coastlines(resolution='50m', linewidth=0.5, color='black', zorder=2)
ax.add_feature(cfeature.BORDERS, linewidth=0.5, color='gray', alpha=0.7, zorder=2)
ax.add_feature(cfeature.LAND, color='#F5F5F5', alpha=0.8, zorder=0)
ax.add_feature(cfeature.OCEAN, color='#F0F8FF', alpha=0.8, zorder=0)

ax.set_global()

for region, data in all_data.items():
    if region == 'Global': 
        continue
    years = data['year']
    lons = data['alllon']
    lats = data['alllat']
    
    for i in range(len(years)):
        curr_lons = lons[i]
        curr_lats = lats[i]
        mask = ~np.isnan(curr_lons) & ~np.isnan(curr_lats)
        curr_lons = curr_lons[mask]
        curr_lats = curr_lats[mask]
        color = '#2166AC' if years[i] < 2008 else '#B2182B'

        if len(curr_lons) > 0:
            plt.plot(curr_lons, curr_lats, color=color, alpha=0.4, 
                    transform=ccrs.PlateCarree(), linewidth=0.4)



box_size = 11  
for region, bounds in REGION_BOUNDS.items():
    if region == 'Global': 
        continue
    lon_min, lon_max = bounds['lon']
    lat_min, lat_max = bounds['lat']
    
    box = plt.Rectangle((lon_min, lat_min),
                       lon_max - lon_min,
                       lat_max - lat_min,
                       fill=False, color='black',
                       linewidth=1.0,
                       transform=ccrs.PlateCarree())
    ax.add_patch(box)

    text_x = lon_min + (lon_max - lon_min) / 2.0
    
sp_bounds = REGION_BOUNDS['SP']
base_x = sp_bounds['lon'][0] + 100  
base_y = sp_bounds['lat'][0] + (sp_bounds['lat'][1] - sp_bounds['lat'][0])/2 - box_size -20


box = plt.Rectangle((110, -35),
                    125 - 110,
                    -15 - -35,
                    fill=False, color='black',
                    linewidth=1.0,
                    transform=ccrs.PlateCarree())
ax.add_patch(box)
    
text_x = 110 + (125 - 110) / 2.0
    

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
tc_legend_elements = [
    Line2D([0], [0], color='#2166AC', linewidth=3.0, label='1990-2007'),
    Line2D([0], [0], color='#B2182B', linewidth=3.0, label='2008-2024')
]
legend1 = ax.legend(handles=tc_legend_elements, loc='lower right', 
                    bbox_to_anchor=(0.98, 0.02), 
                    title='Frequency (times/yr)')
ax.add_artist(legend1)


plt.savefig('./Figure1a_1.png', dpi=1000, bbox_inches='tight')
