import netCDF4 as nc
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from scipy import stats
from scipy.stats import bootstrap


def read_nc_data(file_path):
    with nc.Dataset(file_path, 'r') as ds:
        V0 = ds.variables['V0'][:]
        year = ds.variables['year'][:]
        t_11ms = ds.variables['decay_11ms'][:]
        d = ds.variables['decay_timescale'][:]
    return V0, year, t_11ms, d

def calculate_bootstrap_ci(x_vals, y_vals, unique_years, n_bootstrap=10000, confidence=0.95):

    unique_years = np.array(unique_years)
    
    def trend_func(x, y):
        slope, intercept = np.polyfit(x, y, 1)
        return slope * unique_years + intercept
    

    data = (x_vals, y_vals)

    rng = np.random.default_rng(42)  
    res = bootstrap(data, trend_func, n_resamples=n_bootstrap, 
                   confidence_level=confidence, random_state=rng,
                   paired=True)
    
    return res.confidence_interval.low, res.confidence_interval.high

def calculate_7year_running_mean(years, values):
    annual_data = defaultdict(list)
    for year, value in zip(years, values):
        if not np.isnan(value):
            annual_data[year].append(value)
    
    years_sorted = sorted(annual_data.keys())
    annual_means = [np.mean(annual_data[year]) for year in years_sorted]
    
    running_means = []
    for i, year in enumerate(years_sorted):
        if i <= 2: 
            start_idx = 0
            end_idx = min(i + 4, len(years_sorted))
        elif i >= len(years_sorted) - 3: 
            start_idx = max(i - 3, 0)
            end_idx = len(years_sorted)
        else:  
            start_idx = i - 3
            end_idx = i + 4
        
        values_in_window = annual_means[start_idx:end_idx]
        running_means.append(np.nanmean(values_in_window))
    
    return years_sorted, running_means

def plot_timescale_with_regression(years, values, title, ylabel, filename):
    annual_data = defaultdict(list)
    for year, value in zip(years, values):
        if not np.isnan(value):
            annual_data[year].append(value)
    
    years_sorted = sorted(annual_data.keys())
    percentile_90 = [np.max(annual_data[year]) for year in years_sorted]
    mean_values = [np.mean(annual_data[year]) for year in years_sorted]
    percentile_10 = [np.min(annual_data[year]) for year in years_sorted]
    
    def calculate_running_mean(values):
        running_means = []
        for i in range(len(values)):
            if i <= 2:
                start_idx = 0
                end_idx = min(i + 4, len(values))
            elif i >= len(values) - 3:
                start_idx = max(i - 3, 0)
                end_idx = len(values)
            else:
                start_idx = i - 3
                end_idx = i + 4
            
            window_values = values[start_idx:end_idx]
            running_means.append(np.nanmean(window_values))
        return running_means
    
    max_running = calculate_running_mean(percentile_90)
    mean_running = calculate_running_mean(mean_values)
    min_running = calculate_running_mean(percentile_10)
    
    plt.figure(figsize=(5, 4))
    
    p_values = []
    r_values = []
    
    for data, color, label in zip([max_running, mean_running, min_running], 
                              ['#D73027', '#2166AC', '#74ADD1'],
                              ['Max', 'Mean', 'Min']):
        valid_data = ~np.isnan(data)
        if np.sum(valid_data) > 1:
            x_vals = np.array(years_sorted)[valid_data]
            y_vals = np.array(data)[valid_data]
            
            slope, intercept, r_value, p_value, _ = stats.linregress(x_vals, y_vals)
            regression_line = slope * np.array(years_sorted) + intercept
            
            try:
                ci_low, ci_high = calculate_bootstrap_ci(x_vals, y_vals, years_sorted)
                plt.plot(years_sorted, ci_low, '--', color=color, alpha=0.6, linewidth=1)
                plt.plot(years_sorted, ci_high, '--', color=color, alpha=0.6, linewidth=1)
            except Exception as e:
                print(f"Bootstrap CI calculation failed for {label}: {e}")

            plt.plot(years_sorted, regression_line, color=color, linestyle='--', linewidth=2)
            
            p_values.append(p_value)
            r_values.append(r_value)
        else:
            p_values.append(np.nan)
            r_values.append(np.nan)

    plt.plot(years_sorted, max_running, '-', color='#D73027', linewidth=2,
             label=f'Annual Max R={r_values[0]:.2f} P={p_values[0]:.2f}')
    plt.plot(years_sorted, mean_running, '-', color='#2166AC', linewidth=2,
             label=f'Annual Mean R={r_values[1]:.2f} P={p_values[1]:.2f}')
    plt.plot(years_sorted, min_running, '-', color='#74ADD1', linewidth=2,
             label=f'Annual Min R={r_values[2]:.2f} P={p_values[2]:.2f}')
    #plt.plot(years_sorted, max_running, '-', color='#D73027', linewidth=2,
    #         label=f'Annual Max R={r_values[0]:.2f}**')
    #plt.plot(years_sorted, mean_running, '-', color='#2166AC', linewidth=2,
    #         label=f'Annual Mean R={r_values[1]:.2f}**')
    #plt.plot(years_sorted, min_running, '-', color='#74ADD1', linewidth=2,
    #         label=f'Annual Min R={r_values[2]:.2f}**')
    plt.xlabel('Year')
    plt.ylabel(ylabel)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=1000, bbox_inches='tight')
    plt.close()




def get_basin_info(file_path):
    basin_info = {
        'WP/TC_decay_timescale.nc': ('WP', 33.8),
        'EP/TC_decay_timescale.nc': ('EP', 29.8),
        'NI/TC_decay_timescale.nc': ('NI', 29.7),
        'SI/TC_decay_timescale.nc': ('SI', 27.2),
        'SP/TC_decay_timescale.nc': ('SP', 16.3),
        'NA/TC_decay_timescale.nc': ('NA', 38.8)
    }
    
    for key, (name, threshold) in basin_info.items():
        if key in file_path:
            return name, threshold
    
    return 'Unknown', 32.8

def main():
    file_paths = [
        '../DATA/WP/TC_decay_timescale.nc',
        '../DATA/EP/TC_decay_timescale.nc',
        '../DATA/NI/TC_decay_timescale.nc',
        '../DATA/SI/TC_decay_timescale.nc',
        '../DATA/SP/TC_decay_timescale.nc',
        '../DATA/NA/TC_decay_timescale.nc'
    ]
    
    V0_all = []
    year_all = []
    t_11ms_all = []
    d_all = []
    
    for file_path in file_paths:
        V0, year, t_11ms, d = read_nc_data(file_path)
        V0_all.extend(V0)
        year_all.extend(year)
        t_11ms_all.extend(t_11ms)
        d_all.extend(d)
    
    V0_all = np.array(V0_all)
    year_all = np.array(year_all)
    t_11ms_all = np.array(t_11ms_all)
    d_all = np.array(d_all)
    
    for file_path in file_paths:
        basin_name, threshold = get_basin_info(file_path)
        V0, year, t_11ms, d = read_nc_data(file_path)
        
        
        plot_timescale_with_regression(year, d, 
                                       f'{basin_name} Annual Decay Timescale', 
                                       'Decay Timescale (h)', 
                                       f'./FigureS3_{basin_name}.png')
        
       

if __name__ == "__main__":
    main()