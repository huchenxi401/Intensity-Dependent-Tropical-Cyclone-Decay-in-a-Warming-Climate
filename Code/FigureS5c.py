import netCDF4 as nc
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from scipy import stats
from scipy.stats import bootstrap
def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c

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


def plot_intensity_categories_timeseries(years, V0, d, filename):
    unique_years = np.arange(1990, 2025)
    annual_data_strong = defaultdict(list)  # V0 >= 33
    annual_data_weak = defaultdict(list)    # 11 <= V0 < 33
    
    for year, v0, decay in zip(years, V0, d):
        if not np.isnan(v0) and not np.isnan(decay):
            if v0 >= 33:
                annual_data_strong[year].append(decay)
            elif 11 <= v0 < 33:
                annual_data_weak[year].append(decay)

    years_sorted = sorted(set(years))
    mean_strong = []
    mean_weak = []
    years_plot = []
    
    for year in years_sorted:
        if annual_data_strong[year] and annual_data_weak[year]:  
            mean_strong.append(np.mean(annual_data_strong[year]))
            mean_weak.append(np.mean(annual_data_weak[year]))
            years_plot.append(year)
    
    years_array = np.array(years_plot)
    
    weak_data = {
        'years': years_array,
        'values': np.array(mean_weak),
        'running_mean': []
    }
    
    strong_data = {
        'years': years_array,
        'values': np.array(mean_strong),
        'running_mean': []
    }

    unique_years = years_plot
    
    for data in [weak_data, strong_data]:
        for year in unique_years:
            year_idx = list(unique_years).index(year)
            
            if year_idx <= 2:
                start_idx = 0
                end_idx = min(year_idx + 4, len(unique_years))
            elif year_idx >= len(unique_years) - 3: 
                start_idx = max(year_idx - 3, 0)
                end_idx = len(unique_years)
            else:
                start_idx = year_idx - 3
                end_idx = year_idx + 4

            values_in_window = data['values'][start_idx:end_idx]
            
            if len(values_in_window) > 0:
                data['running_mean'].append(np.nanmean(values_in_window))
            else:
                data['running_mean'].append(np.nan)
    
    running_mean_weak = np.array(weak_data['running_mean'])
    running_mean_strong = np.array(strong_data['running_mean'])

    plt.figure(figsize=(5, 4))
    valid_idx_strong = ~np.isnan(running_mean_strong)
    valid_idx_weak = ~np.isnan(running_mean_weak)
    
    if np.sum(valid_idx_weak) > 1:
        slope, intercept, r_value, p_value, _ = stats.linregress(
            np.array(years_plot)[valid_idx_weak], 
            running_mean_weak[valid_idx_weak]
        )
        regression_line = slope * np.array(years_plot) + intercept
        x_weak = np.array(years_plot)[valid_idx_weak]
        y_weak = running_mean_weak[valid_idx_weak]
        ci_low, ci_high = calculate_bootstrap_ci(x_weak, y_weak, unique_years)
        plt.plot(unique_years, regression_line, color='.7', linestyle='--')
        plt.plot(unique_years, ci_low, '--', color='.7', alpha=0.8, linewidth=1)
        plt.plot(unique_years, ci_high, '--', color='.7', alpha=0.8,  linewidth=1)
        plt.plot(unique_years, running_mean_weak, '-', color='.7', label=f'Weak R={r_value:.2f}**')   
    if np.sum(valid_idx_strong) > 1:
        slope, intercept, r_value, p_value, _ = stats.linregress(
            np.array(years_plot)[valid_idx_strong], 
            running_mean_strong[valid_idx_strong]
        )
        regression_line = slope * np.array(years_plot) + intercept
        x_strong = np.array(years_plot)[valid_idx_strong]
        y_strong = running_mean_strong[valid_idx_strong]
        ci_low, ci_high = calculate_bootstrap_ci(x_strong, y_strong, unique_years)
        plt.plot(unique_years, regression_line, color='black', linestyle='--')
        plt.plot(unique_years, ci_low, '--', color='black', alpha=0.8, linewidth=1)
        plt.plot(unique_years, ci_high, '--', color='black', alpha=0.8,  linewidth=1)
        plt.plot(unique_years, running_mean_strong, '-', color='black', label=f'Strong R={r_value:.2f}**')
    
    plt.xlabel('Year')
    plt.ylabel('Decay Timescale (h)')
    plt.legend(loc='upper left') 
    
    plt.tight_layout()
    plt.savefig(filename, dpi=1000, bbox_inches='tight')
    plt.close()






def main():
    file_paths = [
        '../DATA/RAFT/TC_decay_timescale_raft.nc'
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


    
    plot_intensity_categories_timeseries(year_all, V0_all, d_all, 
                                       './Figure5c.png')



if __name__ == "__main__":
    main()