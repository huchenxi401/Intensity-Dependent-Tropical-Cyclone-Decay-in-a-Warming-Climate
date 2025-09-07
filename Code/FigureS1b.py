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






def calc_exp_r_squared(x, y, popt):
    residuals = y - exp_func(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

def plot_scatter_V0_decay(V0_values, decay_timescales, years, filename, threshold=32.8, basin_name="Global"):
    plt.figure(figsize=(4, 4))
    
    valid_mask = ~np.isnan(V0_values) & ~np.isnan(decay_timescales)
    V0_valid = V0_values[valid_mask]
    decay_valid = decay_timescales[valid_mask]
    years_valid = years[valid_mask]
    
    ranges = [(1990, 2007), (2008, 2024)]
    colors = ['#2166AC', '#B2182B']
    
    for (start, end), color in zip(ranges, colors):
        mask = (years_valid >= start) & (years_valid <= end)
        plt.scatter(V0_valid[mask], decay_valid[mask], c=color, s=0.2, alpha=0.5)
    from matplotlib.lines import Line2D

    from matplotlib.patches import Patch
    import matplotlib.patches as mpatches

    legend_elements = []
    for (start, end), color in zip(ranges, colors):
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, alpha=0.7, linestyle='None',
                label=f'{start}-{end}')
        )


    
    popt, pcov = curve_fit(exp_func, V0_valid, decay_valid, p0=[100, 0.01, 10])
    r_squared = calc_exp_r_squared(V0_valid, decay_valid, popt)
    r_squared = r_squared**0.5
    x_smooth = np.linspace(V0_valid.min(), V0_valid.max(), 300)
    plt.plot(x_smooth, exp_func(x_smooth, *popt), color='black')

    legend_elements.append(
        Line2D([0], [0], color='black',
            label=f'All: R={r_squared:.2f}**')
    )

    for (start, end), color in zip(ranges, colors):
        mask = (years_valid >= start) & (years_valid <= end)
        if np.sum(mask) > 3:  
            popt, _ = curve_fit(exp_func, V0_valid[mask], decay_valid[mask], p0=[100, 0.01, 10])
            r_squared = calc_exp_r_squared(V0_valid[mask], decay_valid[mask], popt)
            r_squared = r_squared**0.5
            plt.plot(x_smooth, exp_func(x_smooth, *popt), color=color)
        
            legend_elements.append(
                Line2D([0], [0], color=color,
                    label=f'R={r_squared:.2f}**')
            )
    
    plt.xlabel('Landfall intensity (m/s)')
    plt.ylabel('Decay Timescale (h)')
    plt.legend(handles=legend_elements, loc='upper right', fontsize=8)
    plt.tick_params(axis='both')
    plt.xlim(10, 75)
    plt.ylim(5, 160)
    plt.xticks([25,50,75])
    plt.yticks([50,100,150])
    plt.tight_layout()
    plt.savefig(filename, dpi=1000, bbox_inches='tight')
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
    
    plot_scatter_V0_decay(V0_all, d_all, year_all, 
                          './Figure1b.png',
                          threshold=32.8, basin_name="Global")
    
if __name__ == "__main__":
    main()