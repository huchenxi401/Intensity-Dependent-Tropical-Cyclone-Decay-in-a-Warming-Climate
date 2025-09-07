import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize
from scipy import stats
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.ndimage import distance_transform_edt
import matplotlib.gridspec as gridspec
from shapely.geometry import Point
import cartopy.io.shapereader as shpreader
from scipy.interpolate import griddata
import netCDF4 as nc
import pandas as pd


def load_combined_data_from_nc(filename='combined_PDI_data.nc'):
    try:
        with nc.Dataset(filename, 'r') as ds:
            loaded_data = {}
            for var_name in ds.variables:
                if var_name != 'n_points': 
                    loaded_data[var_name] = ds.variables[var_name][:]
            
        
        return loaded_data
        
    except Exception as e:
        raise


combined_data = load_combined_data_from_nc('../DATA/PDI/combined_PDI_data.nc')

def plot_boxplot_weak():
    fig, ax = plt.subplots(figsize=(18, 16))
    
    ax2 = ax.twinx()
    bins = np.arange(0, 501, 50)

    x_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    x_positions = np.arange(len(x_labels)) 

    valid_early = combined_data['weak_early'] > 0
    valid_late = combined_data['weak_late'] > 0
    
    dist_early = combined_data['distance'][valid_early]
    precip_early = combined_data['weak_early'][valid_early]
    
    dist_late = combined_data['distance'][valid_late]
    precip_late = combined_data['weak_late'][valid_late]

    bin_precip_early = [[] for _ in range(len(bins)-1)]
    bin_precip_late = [[] for _ in range(len(bins)-1)]
    
    for d, p in zip(dist_early, precip_early):
        bin_idx = np.digitize(d, bins) - 1
        if 0 <= bin_idx < len(bin_precip_early):
            bin_precip_early[bin_idx].append(p)
    
    for d, p in zip(dist_late, precip_late):
        bin_idx = np.digitize(d, bins) - 1
        if 0 <= bin_idx < len(bin_precip_late):
            bin_precip_late[bin_idx].append(p)

    mean_precip_early = []
    mean_precip_late = []
    
    for early_data, late_data in zip(bin_precip_early, bin_precip_late):
        if len(early_data) > 0:
            mean_precip_early.append(np.mean(early_data))
        else:
            mean_precip_early.append(np.nan)
            
        if len(late_data) > 0:
            mean_precip_late.append(np.mean(late_data))
        else:
            mean_precip_late.append(np.nan)

    mean_early_filled = pd.Series(mean_precip_early).interpolate().values
    mean_late_filled = pd.Series(mean_precip_late).interpolate().values

    precip_change_early = np.zeros_like(mean_early_filled)
    precip_change_late = np.zeros_like(mean_late_filled)

    for i in range(1, len(mean_early_filled)-1):
        precip_change_early[i] = (mean_early_filled[i+1] - mean_early_filled[i-1]) / 2
        precip_change_late[i] = (mean_late_filled[i+1] - mean_late_filled[i-1]) / 2

    if len(mean_early_filled) > 1:
        precip_change_early[0] = mean_early_filled[1] - mean_early_filled[0]
        precip_change_early[-1] = mean_early_filled[-1] - mean_early_filled[-2]
        
        precip_change_late[0] = mean_late_filled[1] - mean_late_filled[0]
        precip_change_late[-1] = mean_late_filled[-1] - mean_late_filled[-2]

    box_data = []
    positions = []
    box_colors = []
    
    for i, (early_data, late_data) in enumerate(zip(bin_precip_early, bin_precip_late)):
        if len(early_data) > 0:
            box_data.append(early_data)
            positions.append(i - 0.2)  
            box_colors.append('#2166AC')
        
        if len(late_data) > 0:
            box_data.append(late_data)
            positions.append(i + 0.2)  
            box_colors.append('#B2182B')
    
    if box_data:
        bplot = ax.boxplot(box_data, positions=positions, widths=0.3, showfliers=False, 
                  patch_artist=True, zorder=1)

        for i, box in enumerate(bplot['boxes']):
            color = box_colors[i]
            box.set(color=color, linewidth=4)
            box.set_facecolor('none')

        for i in range(len(box_data)):
            cap_idx = i * 2
            color = box_colors[i]

            if cap_idx < len(bplot['caps']):
                bplot['caps'][cap_idx].set(color=color, linewidth=4)
            if cap_idx + 1 < len(bplot['caps']):
                bplot['caps'][cap_idx + 1].set(color=color, linewidth=4)

        for i in range(len(box_data)):
            whisker_idx = i * 2
            color = box_colors[i]

            if whisker_idx < len(bplot['whiskers']):
                bplot['whiskers'][whisker_idx].set(color=color, linewidth=4)
            if whisker_idx + 1 < len(bplot['whiskers']):
                bplot['whiskers'][whisker_idx + 1].set(color=color, linewidth=4)

        for i, median in enumerate(bplot['medians']):
            median.set(color=box_colors[i], linewidth=4)

    for i, (early_data, late_data) in enumerate(zip(bin_precip_early, bin_precip_late)):
        if len(early_data) > 0 and len(late_data) > 0:
            t_stat, p_val = stats.ttest_ind(early_data, late_data, equal_var=False)

            if p_val < 0.01:
                ax.text(i, 0.05, '**', 
                       ha='center', va='bottom', fontsize=32, fontweight='bold')
            elif p_val < 0.05:
                ax.text(i, 0.05, '*', 
                       ha='center', va='bottom', fontsize=32, fontweight='bold')
    ax.plot(x_positions, mean_precip_early, color='#2166AC', linewidth=4, label='1990-2007 Mean')
    ax.plot(x_positions, mean_precip_late, color='#B2182B', linewidth=4, label='2008-2024 Mean')

    ax2.plot(x_positions, precip_change_early, color='#2166AC', linestyle='--', linewidth=2, label='1990-2007 Decay Rate')
    ax2.plot(x_positions, precip_change_late, color='#B2182B', linestyle='--', linewidth=2, label='2008-2024 Decay Rate')

    ax2.fill_between(x_positions, 0, precip_change_early, color='#2166AC', alpha=0.3, label='1990-2007 Decay')
    ax2.fill_between(x_positions, 0, precip_change_late, color='#B2182B', alpha=0.3, label='2008-2024 Decay')
    ax.set_xlabel('   ', fontsize=36)
    ax.set_ylabel('    ', fontsize=36)
    ax2.set_ylabel('    ', fontsize=36)
    ax.set_xlabel('   ', fontsize=36)
    ax.set_ylabel('    ', fontsize=36)
    ax2.set_ylabel('    ', fontsize=36)
    
   

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=30, ha='right')
    ax.tick_params(axis='x', which='major', labelsize=36)
    ax.tick_params(axis='y', which='major', labelsize=36)  
    ax.set_ylabel('PDI ((m³/s³))', fontsize=36)
    ax.tick_params(axis='y', which='major', labelsize=36)
    ax.set_xlabel('Distance to Coastline (km)', fontsize=36)
    ax2.tick_params(axis='y', which='major', labelsize=36)  
    ax2.set_ylabel('PDI Decay Rate (mm/50km)', fontsize=36)
    
    ax.set_ylim(bottom=0, top=20000)
    ax2.set_ylim(bottom=-6000, top=0)

    from matplotlib.patches import Rectangle

    left_legend_elements = [
        Line2D([0], [0], color='black', linewidth=3, label='Mean'),
        Rectangle((0, 0), 1, 1, edgecolor='black', facecolor='white', linewidth=3, label='Distribution'),
        Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.3, edgecolor='black', linestyle='--', linewidth=3, label='Decay Rate')
    ]

    right_legend_elements = [
        Line2D([0], [0], color='#2166AC', linewidth=3, label='1990-2007'),
        Line2D([0], [0], color='#B2182B', linewidth=3, label='2008-2024'),
    ]
    
    legend2 = ax.legend(handles=right_legend_elements, loc='center right', 
                       bbox_to_anchor=(1.0, 0.7), fontsize=36, framealpha=0.8,
                       ) 
    
    plt.tight_layout()
    plt.savefig('./FigureS8b.png', dpi=300)
    plt.close()

def plot_boxplot_strong():
    fig, ax = plt.subplots(figsize=(18, 16))

    ax2 = ax.twinx()
    bins = np.arange(0, 501, 50)

    x_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    x_positions = np.arange(len(x_labels))  
    valid_early = combined_data['strong_early'] > 0
    valid_late = combined_data['strong_late'] > 0
    
    dist_early = combined_data['distance'][valid_early]
    precip_early = combined_data['strong_early'][valid_early]
    
    dist_late = combined_data['distance'][valid_late]
    precip_late = combined_data['strong_late'][valid_late]

    bin_precip_early = [[] for _ in range(len(bins)-1)]
    bin_precip_late = [[] for _ in range(len(bins)-1)]

    for d, p in zip(dist_early, precip_early):
        bin_idx = np.digitize(d, bins) - 1
        if 0 <= bin_idx < len(bin_precip_early):
            bin_precip_early[bin_idx].append(p)
    
    for d, p in zip(dist_late, precip_late):
        bin_idx = np.digitize(d, bins) - 1
        if 0 <= bin_idx < len(bin_precip_late):
            bin_precip_late[bin_idx].append(p)

    mean_precip_early = []
    mean_precip_late = []
    
    for early_data, late_data in zip(bin_precip_early, bin_precip_late):
        if len(early_data) > 0:
            mean_precip_early.append(np.mean(early_data))
        else:
            mean_precip_early.append(np.nan)
            
        if len(late_data) > 0:
            mean_precip_late.append(np.mean(late_data))
        else:
            mean_precip_late.append(np.nan)

    mean_early_filled = pd.Series(mean_precip_early).interpolate().values
    mean_late_filled = pd.Series(mean_precip_late).interpolate().values

    precip_change_early = np.zeros_like(mean_early_filled)
    precip_change_late = np.zeros_like(mean_late_filled)

    for i in range(1, len(mean_early_filled)-1):
        precip_change_early[i] = (mean_early_filled[i+1] - mean_early_filled[i-1]) / 2
        precip_change_late[i] = (mean_late_filled[i+1] - mean_late_filled[i-1]) / 2

    if len(mean_early_filled) > 1:
        precip_change_early[0] = mean_early_filled[1] - mean_early_filled[0]
        precip_change_early[-1] = mean_early_filled[-1] - mean_early_filled[-2]
        
        precip_change_late[0] = mean_late_filled[1] - mean_late_filled[0]
        precip_change_late[-1] = mean_late_filled[-1] - mean_late_filled[-2]
    box_data = []
    positions = []
    box_colors = []
    
    for i, (early_data, late_data) in enumerate(zip(bin_precip_early, bin_precip_late)):
        if len(early_data) > 0:
            box_data.append(early_data)
            positions.append(i - 0.2)  
            box_colors.append('#2166AC')
        
        if len(late_data) > 0:
            box_data.append(late_data)
            positions.append(i + 0.2) 
            box_colors.append('#B2182B')
    
    if box_data:
        bplot = ax.boxplot(box_data, positions=positions, widths=0.3, showfliers=False, 
                  patch_artist=True, zorder=1)
    
        for i, box in enumerate(bplot['boxes']):
            color = box_colors[i]
            box.set(color=color, linewidth=4)
            box.set_facecolor('none')

        for i in range(len(box_data)):
            cap_idx = i * 2
            color = box_colors[i]

            if cap_idx < len(bplot['caps']):
                bplot['caps'][cap_idx].set(color=color, linewidth=4)
            if cap_idx + 1 < len(bplot['caps']):
                bplot['caps'][cap_idx + 1].set(color=color, linewidth=4)
    
        for i in range(len(box_data)):
            whisker_idx = i * 2
            color = box_colors[i]

            if whisker_idx < len(bplot['whiskers']):
                bplot['whiskers'][whisker_idx].set(color=color, linewidth=4)
            if whisker_idx + 1 < len(bplot['whiskers']):
                bplot['whiskers'][whisker_idx + 1].set(color=color, linewidth=4)
        for i, median in enumerate(bplot['medians']):
            median.set(color=box_colors[i], linewidth=4)

    for i, (early_data, late_data) in enumerate(zip(bin_precip_early, bin_precip_late)):
        if len(early_data) > 0 and len(late_data) > 0:
            t_stat, p_val = stats.ttest_ind(early_data, late_data, equal_var=False)
            
            if p_val < 0.01:
                ax.text(i, 0.05, '**', 
                       ha='center', va='bottom', fontsize=32, fontweight='bold')
            elif p_val < 0.05:
                ax.text(i, 0.05, '*', 
                       ha='center', va='bottom', fontsize=32, fontweight='bold')

    ax.plot(x_positions, mean_precip_early, color='#2166AC', linewidth=4, label='1990-2007 Mean')
    ax.plot(x_positions, mean_precip_late, color='#B2182B', linewidth=4, label='2008-2024 Mean')
    
    ax2.plot(x_positions, precip_change_early, color='#2166AC', linestyle='--', linewidth=2, label='1990-2007 Decay Rate')
    ax2.plot(x_positions, precip_change_late, color='#B2182B', linestyle='--', linewidth=2, label='2008-2024 Decay Rate')

    ax2.fill_between(x_positions, 0, precip_change_early, color='#2166AC', alpha=0.3, label='1990-2007 Decay')
    ax2.fill_between(x_positions, 0, precip_change_late, color='#B2182B', alpha=0.3, label='2008-2024 Decay')
    

    ax.set_xlabel('Distance to Coastline (km)', fontsize=36)
    ax.set_ylabel('PDI ((m³/s³))', fontsize=36)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax2.set_ylabel('PDI Decay Rate (mm/50km)', fontsize=36)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=30, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=36)
    

    ax2.tick_params(axis='y', which='major', labelsize=36)

    ax.set_ylim(bottom=0, top=120000)

    ax2.set_ylim(bottom=-50000, top=0)

    from matplotlib.patches import Rectangle
    
    left_legend_elements = [
        Line2D([0], [0], color='black', linewidth=3, label='Mean'),
        Rectangle((0, 0), 1, 1, edgecolor='black', facecolor='white', linewidth=3, label='Distribution'),
        Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.3, edgecolor='black', linestyle='--', linewidth=3, label='Decay Rate')
    ]
    
    right_legend_elements = [
        Line2D([0], [0], color='#2166AC', linewidth=3, label='1990-2007'),
        Line2D([0], [0], color='#B2182B', linewidth=3, label='2008-2024'),
        Line2D([0], [0], color='none'),
        Line2D([0], [0], color='none')
    ]
    
    legend1 = ax.legend(handles=left_legend_elements, loc='center right', 
                       bbox_to_anchor=(1.0, 0.7), fontsize=36, framealpha=0.8,
                       )
    

    plt.tight_layout()
    plt.savefig('./FigureS8a.png', dpi=300)
    plt.close()



plot_boxplot_weak()
plot_boxplot_strong()

