import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats
import pandas as pd
from scipy.stats import bootstrap
def read_tc_data(file_path):
    with nc.Dataset(file_path, 'r') as ds:
        decay_timescale = ds.variables['decay_timescale'][:]
        V0 = ds.variables['V0'][:]
        year = ds.variables['year'][:]
        month = ds.variables['year'][:]
        lon = ds.variables['lon'][:, 0]  
        lat = abs(ds.variables['lat'][:, 0])
    return decay_timescale, V0, year, month, lon, lat


def calculate_bootstrap_ci(x_vals, y_vals, unique_years, n_bootstrap=10000, confidence=0.95):

    def trend_func(x, y):
        slope, intercept = np.polyfit(x, y, 1)
        return slope * unique_years + intercept
    
    data = (x_vals, y_vals)

    rng = np.random.default_rng(42)  
    res = bootstrap(data, trend_func, n_resamples=n_bootstrap, 
                   confidence_level=confidence, random_state=rng,
                   paired=True)
    
    return res.confidence_interval.low, res.confidence_interval.high



def combine_tc_data(file_paths):
    all_decay = []
    all_V0 = []
    all_year = []
    all_month = []
    all_lon = []
    all_lat = []
    
    for file_path in file_paths:
        decay, V0, year, month, lon, lat = read_tc_data(file_path)
        all_decay.extend(decay)
        all_V0.extend(V0)
        all_year.extend(year)
        all_month.extend(month)
        all_lon.extend(lon)
        all_lat.extend(lat)
    
    return (np.array(all_decay), np.array(all_V0), np.array(all_year),
            np.array(all_month), np.array(all_lon), np.array(all_lat))

def plot_annual_rh_trend(rh_values, years, V0s, output_file):

    plt.figure(figsize=(5, 4))
    unique_years = np.arange(1990, 2025)
    weak_data = {
        'years': [],
        'values': [],
        'running_mean': []
    }
    strong_data = {
        'years': [],
        'values': [],
        'running_mean': []
    }

    weak_mask = (V0s >= 11) & (V0s < 33)
    strong_mask = V0s >= 33
    
    weak_data['years'] = years[weak_mask]
    weak_data['values'] = rh_values[weak_mask]
    strong_data['years'] = years[strong_mask]
    strong_data['values'] = rh_values[strong_mask]

    for data in [weak_data, strong_data]:
        for year in unique_years:
            if year <= 1991:
                year_mask = (data['years'] >= 1990) & (data['years'] <= year+3)
            elif year >= 2016: 
                year_mask = (data['years'] >= year-3) & (data['years'] <= 2023)
            else: 
                year_mask = (data['years'] >= year-3) & (data['years'] <= year+3)

            values_in_window = data['values'][year_mask]
            
            if len(values_in_window) > 0:
                data['running_mean'].append(np.nanmean(values_in_window))
            else:
                data['running_mean'].append(np.nan)

    if len(weak_data['running_mean']) > 0:
        valid_mask = ~np.isnan(weak_data['running_mean'])
        x_vals = unique_years[valid_mask]
        y_vals = np.array(weak_data['running_mean'])[valid_mask]
        valid_mask = ~np.isnan(weak_data['running_mean'])
        if np.sum(valid_mask) > 1:
            slope_weak, intercept_weak, r_weak, p_weak, _ = stats.linregress(
                unique_years[valid_mask], 
                np.array(weak_data['running_mean'])[valid_mask]
            )
            line_weak = slope_weak * unique_years + intercept_weak
            ci_low, ci_high = calculate_bootstrap_ci(x_vals, y_vals, unique_years)
            plt.plot(unique_years, line_weak, '--', color='.7', linewidth=1.0)
            plt.plot(unique_years, ci_low, '--', color='.7', alpha=0.8, linewidth=1)
            plt.plot(unique_years, ci_high, '--', color='.7', alpha=0.8,  linewidth=1)
            plt.plot(unique_years, weak_data['running_mean'], '-', color='.7', 
                label=f'Weak R={r_weak:.2f}**')
            #plt.plot(unique_years, weak_data['running_mean'], '-', color='.7',
            #    label=f'Weak R={r_weak:.2f}**')

    if len(strong_data['running_mean']) > 0:
        valid_mask = ~np.isnan(weak_data['running_mean'])
        x_vals = unique_years[valid_mask]
        y_vals = np.array(strong_data['running_mean'])[valid_mask]
        
        valid_mask = ~np.isnan(strong_data['running_mean'])
        if np.sum(valid_mask) > 1:
            slope_strong, intercept_strong, r_strong, p_strong, _ = stats.linregress(
                unique_years[valid_mask], 
                np.array(strong_data['running_mean'])[valid_mask]
            )
            line_strong = slope_strong * unique_years + intercept_strong
            ci_low, ci_high = calculate_bootstrap_ci(x_vals, y_vals, unique_years)
            plt.plot(unique_years, line_strong, '--', color='black', linewidth=1.0)
            plt.plot(unique_years, ci_low, '--', color='black', alpha=0.8,  linewidth=1)
            plt.plot(unique_years, ci_high, '--', color='black', alpha=0.8,  linewidth=1)
            plt.plot(unique_years, strong_data['running_mean'], '-', color='black',
                label=f'Strong R={r_strong:.2f}')
            #plt.plot(unique_years, strong_data['running_mean'], '-', color='black',
            #    label=f'Strong R={r_strong:.2f}**')
    plt.xlabel('Year')
    plt.ylabel('Absolute Landfall Latitude')

    
    plt.legend(loc='upper left')

    
    plt.savefig(output_file, dpi=1000, bbox_inches='tight')
    plt.close()



def main():
    file_paths = [
        '../DATA/RAFT/TC_decay_timescale_raft.nc'
    ]

    print("Reading TC data...")
    decay_timescale, V0, year, month, tc_lon, tc_lat = combine_tc_data(file_paths)
    
    weak_lats = []
    strong_lats = []
    weak_years = []
    strong_years = []
    all_lats = []
    all_years = []
    all_V0s = []

    for intensity in ['weak', 'strong']:
        if intensity == 'weak':
            mask = (V0 >= 11) & (V0 < 33)
            suffix = '11m'
        else:
            mask = V0 >= 33
            suffix = '33m'
        
        print(f"\nProcessing {intensity} intensity TCs...")
        
        valid_decay = decay_timescale[mask]
        valid_years = year[mask]
        valid_lats = np.abs(tc_lat[mask])
        valid_V0s = V0[mask]
        print(f"Number of valid TCs: {len(valid_decay)}")
        
        if intensity == 'weak':
            weak_lats.extend(valid_lats)
            weak_years.extend(valid_years)
        else:
            strong_lats.extend(valid_lats)
            strong_years.extend(valid_years)
        
        all_lats.extend(valid_lats)
        all_years.extend(valid_years)
        all_V0s.extend(valid_V0s)
        



    all_lats = np.array(all_lats)
    all_years = np.array(all_years)
    all_V0s = np.array(all_V0s)
    
    plot_annual_rh_trend(all_lats, all_years, all_V0s,
                        './Figure5d.png')


if __name__ == "__main__":
    main()