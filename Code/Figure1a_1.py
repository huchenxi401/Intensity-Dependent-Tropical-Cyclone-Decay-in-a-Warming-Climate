import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as stats

def read_nc_data(file_paths):
    all_V0 = []
    all_years = []
    all_basins = []
    
    for file_path in file_paths:
        basin_name = get_basin_name(file_path)
        with nc.Dataset(file_path, 'r') as ds:
            V0 = ds.variables['V0'][:]
            years = ds.variables['year'][:]
            all_V0.extend(V0)
            all_years.extend(years)
            all_basins.extend([basin_name] * len(V0))
    
    return np.array(all_V0), np.array(all_years), np.array(all_basins)

def read_single_nc_data(file_path):
    with nc.Dataset(file_path, 'r') as ds:
        V0 = ds.variables['V0'][:]
        years = ds.variables['year'][:]
    return np.array(V0), np.array(years)

def get_basin_name(file_path):
    if 'WP' in file_path:
        return 'WP'
    elif 'SI' in file_path:
        return 'SI'
    elif 'EP' in file_path:
        return 'EP'
    elif 'NI' in file_path:
        return 'NI'
    elif 'SP' in file_path:
        return 'SP'
    elif 'NA' in file_path:
        return 'NA'
    else:
        return 'Unknown'

def manual_bootstrap_ci(data, n_bootstrap=10000, confidence_level=0.95):
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    ci_lower = np.percentile(bootstrap_means, lower_percentile)
    ci_upper = np.percentile(bootstrap_means, upper_percentile)
    
    return ci_lower, ci_upper

def calculate_frequency_stats(years):
    unique_years = np.arange(int(years.min()), int(years.max()) + 1)
    annual_counts = np.array([np.sum(years == year) for year in unique_years])
    
    early_counts = annual_counts[:18]  # 1990-2007
    late_counts = annual_counts[-17:]  # 2008-2024
    
    early_mean = np.mean(early_counts)
    late_mean = np.mean(late_counts)
    
    early_ci = manual_bootstrap_ci(early_counts)
    late_ci = manual_bootstrap_ci(late_counts)

    t_stat, p_value = stats.ttest_ind(late_counts, early_counts)
    
    return {
        'early_mean': early_mean,
        'late_mean': late_mean,
        'early_ci': early_ci,
        'late_ci': late_ci,
        'p_value': p_value,
        't_stat': t_stat
    }
def plot_simple_bars_with_stats(years, title, output_file):
    stats_result = calculate_frequency_stats(years)
    
    t1 = stats_result['early_mean']
    t2 = stats_result['late_mean']
    early_ci = stats_result['early_ci']
    late_ci = stats_result['late_ci']
    p_value = stats_result['p_value']
    
    fig, ax = plt.subplots(figsize=(0.8, 0.6))
    plt.bar([0, 1], [t1, t2], width=0.65, color=('#2166AC', '#B2182B'))
    
    
    if title == 'Global':
        y_max = 18
        label_offset = 1.8
    else:
        y_max = 7
        label_offset = 1.0
    plt.ylim(0, y_max)
    plt.xticks([])
    plt.xlim(-0.5, 1.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title(title, fontweight='bold', pad=0)
    if title == 'Global':
        
        ax.set_title(title, fontweight='bold')
    ax.text(0, t1 + label_offset, round(t1, 1), ha='center', va='center')
    ax.text(1, t2 + label_offset, round(t2, 1), ha='center', va='center')
    

    fig.set_facecolor('none')
    ax.set_facecolor('none')
    

    plt.savefig(output_file, dpi=1000, bbox_inches='tight', facecolor='none')
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
    
    output_dir = './'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Processing global data...")
    V0_global, years_global, basins_global = read_nc_data(file_paths)
    
    plot_simple_bars_with_stats(years_global, 'Global', 
                     os.path.join(output_dir, 'global_tc_frequency.png'))
    
    print("\n" + "="*50 + "\n")
    
    for file_path in file_paths:
        basin_name = get_basin_name(file_path)
        print(f"Processing {basin_name} basin...")
        
        V0, years = read_single_nc_data(file_path)

        plot_simple_bars_with_stats(years, basin_name, 
                         os.path.join(output_dir, f'{basin_name}_tc_frequency.png'))
        
        print()

if __name__ == "__main__":
    main()