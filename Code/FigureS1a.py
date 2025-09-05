import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os

def read_nc_data(file_paths):
    all_V0 = []
    all_years = []
    for file_path in file_paths:
        with nc.Dataset(file_path, 'r') as ds:
            all_V0.extend(ds.variables['V0'][:])
            all_years.extend(ds.variables['year'][:])
    return np.array(all_V0), np.array(all_years)

def read_single_nc_data(file_path):
    with nc.Dataset(file_path, 'r') as ds:
        V0 = ds.variables['V0'][:]
        years = ds.variables['year'][:]
    return np.array(V0), np.array(years)

def calculate_occurrence_per_year(data, bins, num_years):
    hist, _ = np.histogram(data, bins=bins)
    return hist / num_years 

def plot_occurrence(occurrence_data, bin_edges, colors, labels, title, output_file, grouped_data, num_years_list, y_min=None, y_max=None):
    fig, ax = plt.subplots(figsize=(5, 4))
   
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    for i, (occurrence, color, label) in enumerate(zip(occurrence_data, colors, labels)):
        ax.plot(bin_centers, occurrence, color=color, linewidth=2, 
                marker='o', markersize=4, alpha=0.8, label=label)
        
        for j, value in enumerate(occurrence):
            if value > 0: 
                ax.annotate(f'{value:.1f}', 
                           (bin_centers[j], value), 
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center',
                           color=color)
    
    ax.set_xlabel('Landfall intensity (m/s)')
    ax.set_ylabel('Events/Yr')
    
    period_totals = [len(group) for group in grouped_data]
    period_avg_per_year = [period_totals[i] / num_years_list[i] for i in range(len(period_totals))]
    
    legend_labels = [f"{labels[i]}" 
                    for i in range(len(labels))]
    
    ax.legend(legend_labels, loc='upper right') #prop={'size': 32})

    ax.set_xticks(bin_centers)
    tick_labels = [f'{int(bin_edges[i])}-{int(bin_edges[i+1])}' for i in range(len(bin_edges)-1)]
    ax.set_xticklabels(tick_labels)
    
    if y_max is not None:
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=1000, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as '{output_file}'")

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

def main():
    file_paths = [
        '../DATA/WP/TC_decay_timescale.nc',
        '../DATA/EP/TC_decay_timescale.nc',
        '../DATA/NI/TC_decay_timescale.nc',
        '../DATA/SI/TC_decay_timescale.nc',
        '../DATA/SP/TC_decay_timescale.nc',
        '../DATA/NA/TC_decay_timescale.nc'
    ]
    
    groups = [(1990, 2007), (2008, 2024)]
    colors = ['#2166AC', '#B2182B']
    labels = ['1990-2007', '2008-2024']

    num_years_list = [18, 17] 
    

    bin_edges = np.arange(10, 70, 10)  # [10, 20, 30, 40, 50, 60, 70]
    

    
    V0_global, years_global = read_nc_data(file_paths)
    
    grouped_global_data = [V0_global[(years_global >= start) & (years_global <= end)] for start, end in groups]
    
    occurrence_global_data = [calculate_occurrence_per_year(grouped_global_data[i], bin_edges, num_years_list[i]) 
                             for i in range(len(grouped_global_data))]
    
    plot_occurrence(occurrence_global_data, bin_edges, colors, labels, 
               'Global Typhoon Landfall Intensity Occurrence',
               './FigureS1a.png',
               grouped_global_data, num_years_list, y_min=1, y_max=7)
    

    print("\n=== Statistics Summary ===")
    print(f"Global total typhoons (1990-2007): {len(grouped_global_data[0])} (avg: {len(grouped_global_data[0])/18:.1f}/year)")
    print(f"Global total typhoons (2008-2024): {len(grouped_global_data[1])} (avg: {len(grouped_global_data[1])/17:.1f}/year)")
    
    for file_path in file_paths:
        basin_name = get_basin_name(file_path)
        V0_single, years_single = read_single_nc_data(file_path)
        grouped_single_data = [V0_single[(years_single >= start) & (years_single <= end)] for start, end in groups]
        print(f"{basin_name} typhoons (1990-2007): {len(grouped_single_data[0])} (avg: {len(grouped_single_data[0])/18:.1f}/year)")
        print(f"{basin_name} typhoons (2008-2024): {len(grouped_single_data[1])} (avg: {len(grouped_single_data[1])/17:.1f}/year)")

if __name__ == "__main__":
    main()