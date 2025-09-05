import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import make_interp_spline
import os


def read_threshold_nc(nc_file_path):
    try:
        with nc.Dataset(nc_file_path, 'r') as ds:
            data = {}
            for var_name in ds.variables:
                if var_name != 'nrows': 
                    var = ds.variables[var_name]
                    if var.dtype.char == 'S':
                        str_data = []
                        for i in range(var.shape[1]):
                            chars = var[:, i]
                            str_val = ''.join([c.decode('utf-8') for c in chars if c != b''])
                            str_data.append(str_val)
                        data[var_name] = str_data
                    else:
                        data[var_name] = var[:]
            
            return pd.DataFrame(data)
            
    except Exception as e:
        print(f"Error reading NC file {nc_file_path}: {str(e)}")
        raise



def smooth_curve(x, y, smooth_factor=50):
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean, y_clean = x[mask], y[mask]
   
    if len(x_clean) > 3:
        unique_indices = np.unique(x_clean, return_index=True)[1]
        x_unique = x_clean[unique_indices]
        y_unique = y_clean[unique_indices]
        
        
        if len(x_unique) > 3:
            x_smooth = np.linspace(x_unique.min(), x_unique.max(), smooth_factor)
            spl = make_interp_spline(x_unique, y_unique, k=min(3, len(x_unique)-1))
            y_smooth = spl(x_smooth)
            return x_smooth, y_smooth
    
    return x_clean, y_clean

def read_nc_data(file_path):
    with nc.Dataset(file_path, 'r') as ds:
        V0 = ds.variables['V0'][:]
        year = ds.variables['year'][:]
        decay = ds.variables['decay_timescale'][:]
    return V0, year, decay

def get_basin_name(file_path):
    if 'RAFT' in file_path:
        return 'RAFT'
    else:
        return 'Unknown'

def plot_global_scatter(df_global, thres_global, output_file='./Figure5b.png'):
    plt.figure(figsize=(5, 4))
    ind = df_global.year < 2008
    plt.scatter(df_global.V0[ind], df_global.decay[ind], c='#2166AC', s=0.2)
    ind = df_global.year > 2007
    plt.scatter(df_global.V0[ind], df_global.decay[ind], c='#B2182B', s=0.2)
    
    x_early_smooth, y_early_smooth = smooth_curve(thres_global.LI_threshold, thres_global.early_period_mean)
    x_late_smooth, y_late_smooth = smooth_curve(thres_global.LI_threshold, thres_global.late_period_mean)
    
    plt.plot(x_early_smooth, y_early_smooth, c='#2166AC', label='1990-2007', lw=2)
    plt.plot(x_late_smooth, y_late_smooth, c='#B2182B', label='2008-2024', lw=2)
    
    thres = 36.5
    plt.vlines(thres, 0, 160, color='k', ls='dashed', lw=0.8)
    plt.text(thres+1, 120, str(thres)+'m/s')
    plt.ylim(5, 160)
    plt.xlim(10, 75)
    plt.xticks([25,50,75])
    plt.yticks([50,100,150])
    plt.legend()
    plt.xlabel('Landfall intensity (m/s)')
    plt.ylabel('Decay timescale (h)')
    plt.tight_layout()
      
    plt.savefig(output_file, dpi=1000, bbox_inches='tight')
    plt.close()
    print(f"Global plot saved as {output_file}")

def main():

    file_paths = [
        '../DATA/RAFT/TC_decay_timescale_raft.nc'
    ]
    
    basin_names = ['RAFT']
    thresholds = {
        'RAFT': 36.5
    }
    

    all_data = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
            
        V0, year, decay = read_nc_data(file_path)
        basin = get_basin_name(file_path)

        basin_df = pd.DataFrame({
            'V0': V0,
            'year': year,
            'decay': decay,
            'ocean': basin
        })
        
        all_data.append(basin_df)
    
    if not all_data:
        print("No valid data files found, exiting...")
        return

    df_all = pd.concat(all_data, ignore_index=True)
    df_all = df_all.dropna()
    df_global = df_all[['V0', 'year', 'decay']].copy()    
    thres_global = read_threshold_nc("../DATA/DT_diff_nc/RAFT_NA_threshold_analysis.nc")
    plot_global_scatter(df_global, thres_global)
    
    print("\nAll operations completed successfully!")

if __name__ == "__main__":
    main()