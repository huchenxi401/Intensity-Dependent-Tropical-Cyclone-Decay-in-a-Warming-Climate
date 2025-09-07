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
        month = ds.variables['month'][:]
        lon = ds.variables['lon'][:, 0]  
        lat = abs(ds.variables['lat'][:, 0])  
        #lon = ds.variables['lat'][:, 0]
    return decay_timescale, V0, year, month, lon, lat

def calculate_confidence_interval(x, y, confidence=0.95):

    n = len(x)
    if n < 3:
        return None, None, None, None

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    y_pred = slope * x + intercept
    residuals = y - y_pred

    mse = np.sum(residuals**2) / (n - 2) 

    x_mean = np.mean(x)
    sxx = np.sum((x - x_mean)**2)

    slope_std_err = np.sqrt(mse / sxx)

    alpha = 1 - confidence
    t_val = stats.t.ppf(1 - alpha/2, n - 2)

    slope_ci_lower = slope - t_val * slope_std_err
    slope_ci_upper = slope + t_val * slope_std_err
    
    return slope, slope_ci_lower, slope_ci_upper, p_value




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
            elif year >= 2022: 
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
        if np.sum(valid_mask) > 1:
            x_vals = unique_years[valid_mask]
            y_vals = np.array(weak_data['running_mean'])[valid_mask]
            
            slope_weak, intercept_weak, r_weak, p_weak, _ = stats.linregress(x_vals, y_vals)
            slope, ci_lower_weak, ci_upper_weak, p_value_weak = calculate_confidence_interval(x_vals, y_vals)
            line_weak = slope_weak * unique_years + intercept_weak
            
            ci_low, ci_high = calculate_bootstrap_ci(x_vals, y_vals, unique_years)

            plt.plot(unique_years, line_weak, '--', color='.7',)
            plt.plot(unique_years, ci_low, '--', color='.7', alpha=0.8, linewidth=1)
            plt.plot(unique_years, ci_high, '--', color='.7', alpha=0.8,  linewidth=1)
            plt.plot(unique_years, weak_data['running_mean'], '-', color='.7', 
                 label=f'Weak R={r_weak:.2f}, P={p_weak:.2f}')
            #plt.plot(unique_years, weak_data['running_mean'], '-', color='.7',
            #    label=f'Weak R={r_weak:.2f}**')
    if len(strong_data['running_mean']) > 0:
    
        valid_mask = ~np.isnan(strong_data['running_mean'])
        if np.sum(valid_mask) > 1:
            x_vals = unique_years[valid_mask]
            y_vals = np.array(strong_data['running_mean'])[valid_mask]
            
            slope_strong, intercept_strong, r_strong, p_strong, _ = stats.linregress(x_vals, y_vals)
            line_strong = slope_strong * unique_years + intercept_strong
            slope, ci_lower_strong, ci_upper_strong, p_value_strong = calculate_confidence_interval(x_vals, y_vals)
            ci_low, ci_high = calculate_bootstrap_ci(x_vals, y_vals, unique_years)
            
            plt.plot(unique_years, line_strong, '--', color='black')
            plt.plot(unique_years, ci_low, '--', color='black', alpha=0.8,  linewidth=1)
            plt.plot(unique_years, ci_high, '--', color='black', alpha=0.8,  linewidth=1)
            plt.plot(unique_years, strong_data['running_mean'], '-', color='black',
                 label=f'Strong R={r_strong:.2f}, P={p_strong:.2f}')
            #plt.plot(unique_years, strong_data['running_mean'], '-', color='black',
            #    label=f'Strong R={r_strong:.2f}')
    plt.xlabel('Year')
    plt.ylabel('Absolute Landfall Latitude')
    #plt.title('Annual Mean Landfall Latitude', fontsize=14)
    
    plt.legend(loc='upper left')
    #plt.tick_params(axis='both', labelsize=58)
    
    plt.savefig(output_file, dpi=1000, bbox_inches='tight')
    plt.close()


def main():
    file_paths = [
        '../DATA/WP/TC_decay_timescale.nc',
        '../DATA/EP/TC_decay_timescale.nc',
        '../DATA/NI/TC_decay_timescale.nc',
        '../DATA/SI/TC_decay_timescale.nc',
        '../DATA/SP/TC_decay_timescale.nc',
        '../DATA/NA/TC_decay_timescale.nc'
    ]
    

    basin_names = ['WP', 'EP', 'NI', 'SI', 'SP', 'NA']

    for i, file_path in enumerate(file_paths):
        basin_name = basin_names[i]
        print(f"\nProcessing {basin_name} basin...")

        decay_timescale_single, V0_single, year_single, month_single, tc_lon_single, tc_lat_single = read_tc_data(file_path)

        all_lats_single = []
        all_years_single = []
        all_V0s_single = []

        for intensity in ['weak', 'strong']:
            if intensity == 'weak':
                mask = (V0_single >= 11) & (V0_single < 33)
            else:
                mask = V0_single >= 33
        
            valid_lats = np.abs(tc_lat_single[mask]) 
            valid_years = year_single[mask]
            valid_V0s = V0_single[mask]
        
            all_lats_single.extend(valid_lats)
            all_years_single.extend(valid_years)
            all_V0s_single.extend(valid_V0s)

            early_mask = (valid_years >= 1990) & (valid_years < 2008)
            late_mask = (valid_years >= 2008) & (valid_years < 2025)
        
        all_lats_single = np.array(all_lats_single)
        all_years_single = np.array(all_years_single)
        all_V0s_single = np.array(all_V0s_single)

        plot_annual_rh_trend(all_lats_single, all_years_single, all_V0s_single,
                        f'./FigureS4_{basin_name}.png')


if __name__ == "__main__":
    main()