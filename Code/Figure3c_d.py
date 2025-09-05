import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os
import xarray as xr
from matplotlib.colors import BoundaryNorm
import netCDF4 as nc
from matplotlib.transforms import Affine2D
import matplotlib.ticker as ticker

coast_file = '../DATA/Precipitation/coastline_distance.nc'
ds_coast = xr.open_dataset(coast_file)

if np.max(ds_coast.lon) > 180:
    ds_coast = ds_coast.assign_coords(lon=(((ds_coast.lon + 180) % 360) - 180))
    ds_coast = ds_coast.sortby('lon')

PLOT_REGION_BOUNDS = {
    'NA': {'lon': (-100, -70), 'lat': (15, 40)},
    'EP': {'lon': (-124, -100), 'lat': (15, 35)},
    'NI': {'lon': (70, 100), 'lat': (5, 30)},
    'SI': {'lon': (30, 54), 'lat': (-30, -10)},
    'SP': {'lon': (125, 153), 'lat': (-31, -10)},
    'WP': {'lon': (100, 140), 'lat': (10, 40)}
}

distance_bins = np.arange(0, 501, 50)  # 0-50, 50-100, ..., 450-500

COLOR_RANGE = {
    'NA': {'weak': 15, 'strong': 15},
    'EP': {'weak': 15, 'strong': 15},
    'NI': {'weak': 15, 'strong': 15},
    'SI': {'weak': 15, 'strong': 15},
    'SP': {'weak': 15, 'strong': 15},
    'WP': {'weak': 15, 'strong': 15}
}

def compute_diff(rainfall_data):
    diff_data = {}
    
    for region, periods in rainfall_data.items():
        
        diff_data[region] = {
            'diff_weak': [late - early for late, early in zip(
                periods['weak_late'],
                periods['weak_early']
            )],
            'diff_strong': [late - early for late, early in zip(
                periods['strong_late'],
                periods['strong_early']
            )]
        }

    return diff_data


def load_rainfall_from_nc(nc_file):
    ds = xr.open_dataset(nc_file)

    rainfall_data = {}
    for i, region in enumerate(ds.region.values):
        region_name = region.decode('utf-8') if isinstance(region, bytes) else str(region)
        rainfall_data[region_name] = {
            'weak_late': ds.weak_late[i, :].values.tolist(),
            'weak_early': ds.weak_early[i, :].values.tolist(),
            'strong_late': ds.strong_late[i, :].values.tolist(),
            'strong_early': ds.strong_early[i, :].values.tolist()
        }
    
    return rainfall_data


def build_region_diff_layer(diff_data, region, diff_type, ds_coast):
    distance = ds_coast.distance_to_coastline.values
    land_mask = ds_coast.land_mask.values if 'land_mask' in ds_coast else (distance > 0)
    lons = ds_coast.lon.values
    lats = ds_coast.lat.values

    layer = np.zeros_like(distance)
    
    if land_mask.dtype == bool:
        valid_land = land_mask
    else:
        valid_land = land_mask > 0
    
    valid_land = valid_land & (distance > 0) & (~np.isnan(distance))

    bounds = PLOT_REGION_BOUNDS[region]
    lon_min, lon_max = bounds['lon']
    lat_min, lat_max = bounds['lat']
    
    region_mask = np.zeros_like(distance, dtype=bool)
    for i, lon_val in enumerate(lons):
        if lon_min <= lon_val <= lon_max:
            for j, lat_val in enumerate(lats):
                if lat_min <= lat_val <= lat_max:
                    if distance.shape[0] == len(lons) and distance.shape[1] == len(lats):
                        region_mask[i, j] = True
                    elif distance.shape[0] == len(lats) and distance.shape[1] == len(lons):
                        region_mask[j, i] = True
    
    diff_values = diff_data[region][diff_type]
    
    for i in range(len(distance_bins) - 1):
        if i < len(diff_values): 
            bin_min = distance_bins[i]
            bin_max = distance_bins[i+1]
            
            distance_mask = (distance >= bin_min) & (distance < bin_max)
            combined_mask = region_mask & distance_mask & valid_land

            layer[combined_mask] = diff_values[i]
    
    distance_limit = 510  # km
    distance_limit_mask = ds_coast.distance_to_coastline.values <= distance_limit
    
    final_valid_mask = region_mask & valid_land & distance_limit_mask
    
    return layer, final_valid_mask

def plot_region_difference(diff_data, region, diff_type, ds_coast):
    lons = ds_coast.lon.values
    lats = ds_coast.lat.values
    distance = ds_coast.distance_to_coastline.values
    
    layer, valid_mask = build_region_diff_layer(diff_data, region, diff_type, ds_coast)
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.colors import ListedColormap
    import matplotlib.colors as mcolors
    colors_negative = [
        '#8B4513',  
        '#A0522D',  
        '#B8651F',  
        '#CD7F32', 
        '#D2B48C',  
        '#DDD5C7', 
        '#E8E2D5'   
    ]

    colors_zero = ['white']  

    colors_positive = [
        '#E8F5E8', 
        '#D5E8D4', 
        '#B8D6B8',  
        '#7FB069', 
        '#5A8A3A',  
        '#3E6B1F', 
        '#2D5016'  
    ]
    all_colors = colors_negative + colors_zero + colors_positive
    cmap = ListedColormap(all_colors)
    if diff_type == 'diff_weak':
        max_abs_diff = COLOR_RANGE[region]['weak']
    else:
        max_abs_diff = COLOR_RANGE[region]['strong']
    
    levels = np.linspace(-max_abs_diff, max_abs_diff, 16) 
    norm = BoundaryNorm(levels, cmap.N)
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.spines['top'].set_linewidth(4.0)
    ax.spines['bottom'].set_linewidth(4.0)
    ax.spines['left'].set_linewidth(4.0)
    ax.spines['right'].set_linewidth(4.0)
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    bounds = PLOT_REGION_BOUNDS[region]
    lon_min, lon_max = bounds['lon']
    lat_min, lat_max = bounds['lat']
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.set_facecolor('lightgray')
    lon_2d, lat_2d = np.meshgrid(lons, lats)
    background = np.full_like(layer.T, np.nan)

    is_ocean = ds_coast.land_mask.values.T == 0
    background[is_ocean] = -999

    is_land = ds_coast.land_mask.values.T == 1

    bounds = PLOT_REGION_BOUNDS[region]
    lon_min, lon_max = bounds['lon']
    lat_min, lat_max = bounds['lat']

    lon_2d_for_mask, lat_2d_for_mask = np.meshgrid(lons, lats)
    in_region = ((lon_2d_for_mask >= lon_min) & (lon_2d_for_mask <= lon_max) & 
             (lat_2d_for_mask >= lat_min) & (lat_2d_for_mask <= lat_max))

    has_precip_data = np.abs(layer.T) > 1e-10

    land_no_data = is_land
    background[land_no_data] = -888

    ocean_plot = np.where(background == -999, 1, np.nan)
    ax.contourf(lon_2d, lat_2d, ocean_plot, levels=[0.5, 1.5], 
           colors=["#F0F8FF"], transform=ccrs.PlateCarree(), zorder=0)

    masked_data = np.ma.masked_array(layer.T, mask=~valid_mask.T)
   
    cf = ax.contourf(lon_2d, lat_2d, masked_data, levels=levels, 
                cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), extend='both')
    
    #cs_zero = ax.contour(lon_2d, lat_2d, masked_data, levels=[0], 
    #                colors='k', linewidths=1.0, transform=ccrs.PlateCarree())
    
    masked_distance = np.ma.masked_array(distance.T, mask=~valid_mask.T)
    key_distances = [100, 300, 500] 
    cs = ax.contour(lon_2d, lat_2d, masked_distance, levels=key_distances, colors='black',
                linewidths=1.0, alpha=0.8, 
               transform=ccrs.PlateCarree())

    ax.coastlines(resolution='50m', linewidth=0.5, color='gray', zorder=4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, color='gray', 
               alpha=0.8, zorder=4)
    ax.add_feature(cfeature.LAKES, alpha=0.8, color='#E6F3FF', 
               edgecolor='black', linewidth=0.5, zorder=3)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.set_facecolor('#FAFAFA')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                linewidth=0.5, color='white', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    gl.xformatter = LongitudeFormatter(zero_direction_label=True, number_format='.0f')
    gl.yformatter = LatitudeFormatter(number_format='.0f')
    gl.xlabel_style = {'size': 24} 
    gl.ylabel_style = {'size': 24}
    diff_titles = {
        'diff_weak': f'Precipitation Change: Weak TCs (2008-2024 minus 1990-2007) - {region}',
        'diff_strong': f'Precipitation Change: Strong TCs (2008-2024 minus 1990-2007) - {region}'
    }


    filename = f"./Figure3{diff_type}_{region}_precipitation.png"
    plt.savefig(filename, dpi=1000, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
    plt.close(fig)
    
    return filename


def main():

    nc_file = '../DATA/Precipitation/rainfall_data.nc'
   
    rainfall_data_from_nc = load_rainfall_from_nc(nc_file)
    
    diff_data = compute_diff(rainfall_data_from_nc)
    
    coast_file = '../DATA/Precipitation/coastline_distance.nc'
    ds_coast = xr.open_dataset(coast_file)
    
    if np.max(ds_coast.lon) > 180:
        ds_coast = ds_coast.assign_coords(lon=(((ds_coast.lon + 180) % 360) - 180))
        ds_coast = ds_coast.sortby('lon')

    saved_files = []
    for region in PLOT_REGION_BOUNDS.keys():
        for diff_type in ['diff_weak', 'diff_strong']:
            if region not in diff_data:
                continue
            
            filename = plot_region_difference(diff_data, region, diff_type, ds_coast)
            saved_files.append(filename)
    

if __name__ == "__main__":
    main()