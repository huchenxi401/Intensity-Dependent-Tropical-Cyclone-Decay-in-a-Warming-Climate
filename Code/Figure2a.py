import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def read_nc_data(file_path):
    with nc.Dataset(file_path, 'r') as ds:
        wind = ds.variables['wind'][:, 8:17] * 0.51  
        years = ds.variables['year'][:, 8]  
    return wind, years

def filter_tc_data(wind, years):
    valid_data = []
    valid_years = []
    for w, y in zip(wind, years):
        is_monotonic = True
        for i in range(1, len(w)):
            if w[i] > w[i-1]:
                is_monotonic = False
                break
        
        if  is_monotonic and  w[0] >= 11 and np.sum(~np.isnan(w)) >= 5:
            valid_data.append(w)
            valid_years.append(y)
    return np.array(valid_data), np.array(valid_years)

def calculate_average_timeseries(wind_data, years, year_range):
    mask = (years >= year_range[0]) & (years <= year_range[1])
    selected_data = wind_data[mask]
    avg_timeseries = np.nanmean(selected_data, axis=0)
    std_timeseries = np.nanstd(selected_data, axis=0)
    return avg_timeseries, std_timeseries

def plot_timeseries(ts1, ts2, std1, std2, output_file):
    time_points = np.arange(0, 24, 3) 
    
    fig, ax = plt.subplots(figsize=(22, 16))
    
    ax.plot(time_points, ts1, color='blue', label='1990-2007',linewidth=5)
    ax.fill_between(time_points, ts1 - std1, ts1 + std1, color='blue', alpha=0.2)
    
    ax.plot(time_points, ts2, color='red', label='2008-2024',linewidth=5)
    ax.fill_between(time_points, ts2 - std2, ts2 + std2, color='red', alpha=0.2)
    
    ax.set_xlabel('Time after landfall (hours)', fontsize=64)
    ax.set_ylabel('Maximum Wind Speed (m/s)', fontsize=64)
    ax.legend(fontsize=58)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', labelsize=64)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()



def main():
    file_paths = [
        r'../DATA/WP/landfall_TC_WP.nc',
        r'../DATA/NA/landfall_TC_NA.nc',
        r'../DATA/EP/landfall_TC_EP.nc',
        r'../DATA/NI/landfall_TC_NI.nc',
        r'../DATA/SI/landfall_TC_SI.nc',
        r'../DATA/SP/landfall_TC_SP.nc'
    ]

    all_wind_data = []
    all_years = []

    for file_path in file_paths:
        try:
            wind, years = read_nc_data(file_path)
            all_wind_data.append(wind)
            all_years.append(years)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    combined_wind = np.concatenate(all_wind_data)
    combined_years = np.concatenate(all_years)

    filtered_wind, filtered_years = filter_tc_data(combined_wind, combined_years)

    ts_1990_2007, std_1990_2007 = calculate_average_timeseries(filtered_wind, filtered_years, (1990, 2007))
    ts_2008_2024, std_2008_2024 = calculate_average_timeseries(filtered_wind, filtered_years, (2008, 2024))

        
    ts1, ts2, std1, std2 = ts_1990_2007, ts_2008_2024, std_1990_2007, std_2008_2024
    
    time_points = np.arange(0, 27, 3) 

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(time_points, ts1, color='#2166AC', label='Early (1990-2007)') #,linewidth=5
    ax.fill_between(time_points, ts1 - std1, ts1 + std1, color='#2166AC', alpha=0.1)

    ax.plot(time_points, ts2, color='#B2182B', label='Late (2008-2024)') #,linewidth=5
    ax.fill_between(time_points, ts2 - std2, ts2 + std2, color='#B2182B', alpha=0.1)
    plt.vlines(0, 33, 42.5, 'k')
    plt.vlines(0, 16.5, 33, 'k', ls='dashed')
    plt.hlines(33, 0, 0.5, 'k')
    plt.text(0.2, 37.5, 'Strong',  ha='left', va='center')  
    plt.text(0.2, 25, 'Weak',  ha='left', va='center')      
    plt.text(0.6, 32.15, '33m/s',  ha='left', va='bottom')   
    plt.xlim(-0.5, 22)
    plt.xticks(np.arange(0, 27, 5))
    ax.set_xlabel('Time after landfall (h)') #, fontsize=64
    ax.set_ylabel('Maximum Wind Speed (m/s)') #, fontsize=64
    ax.legend() #fontsize=58
    plt.tight_layout()
    plt.savefig('./Figure2a.png', dpi=1000, bbox_inches='tight')

if __name__ == "__main__":
    main()