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
    
    print(f"After removing NaN: {len(x_clean)} points")
    
    if len(x_clean) > 3:
        unique_indices = np.unique(x_clean, return_index=True)[1]
        x_unique = x_clean[unique_indices]
        y_unique = y_clean[unique_indices]

        
        if len(x_unique) > 3:
            x_smooth = np.linspace(x_unique.min(), x_unique.max(), smooth_factor)
            spl = make_interp_spline(x_unique, y_unique, k=min(3, len(x_unique)-1))
            y_smooth = spl(x_smooth)
            print("Smoothing applied successfully")
            return x_smooth, y_smooth

    return x_clean, y_clean

def read_nc_data(file_path):
    with nc.Dataset(file_path, 'r') as ds:
        V0 = ds.variables['V0'][:]
        year = ds.variables['year'][:]
        decay = ds.variables['decay_timescale'][:]
    return V0, year, decay

def get_basin_name(file_path):
    if 'WP' in file_path:
        return 'WP'
    elif 'EP' in file_path:
        return 'EP'
    elif 'NI' in file_path:
        return 'NI'
    elif 'SI' in file_path:
        return 'SI'
    elif 'SP' in file_path:
        return 'SP'
    elif 'NA' in file_path:
        return 'NA'
    else:
        return 'Unknown'

def plot_global_scatter(df_global, thres_global, output_file='Figure1b_global.png'):
    plt.figure(figsize=(3.5, 3.5))
    
    ind = df_global.year < 2008
    plt.scatter(df_global.V0[ind], df_global.decay[ind], c='#2166AC', s=0.2, label='before')
    ind = df_global.year > 2007
    plt.scatter(df_global.V0[ind], df_global.decay[ind], c='#B2182B', s=0.2, label='after')
    
    x_early_smooth, y_early_smooth = smooth_curve(thres_global.LI_threshold, thres_global.early_period_mean)
    x_late_smooth, y_late_smooth = smooth_curve(thres_global.LI_threshold, thres_global.late_period_mean)
    
    plt.plot(x_early_smooth, y_early_smooth, c='#2166AC', lw=2)
    plt.plot(x_late_smooth, y_late_smooth, c='#B2182B', lw=2)
    
    thres = 32.8
    plt.vlines(thres, 0, 160, color='k', ls='dashed', lw=0.8)
    plt.text(thres+1, 120, str(thres)+'m/s')
    plt.text(0.75, 0.85, 'Global', fontweight='bold', transform=plt.gca().transAxes)
    plt.ylim(5, 160)
    plt.xlim(10, 75)
    plt.xticks([25,50,75])
    plt.yticks([50,100,150])
    plt.xlabel('Landfall intensity (m/s)')
    plt.ylabel('Decay timescale (h)')
    plt.tight_layout()
      
    plt.savefig(output_file, dpi=1000, bbox_inches='tight')
    plt.close()
    print(f"Global plot saved as {output_file}")

def plot_basin_subplots(df_all, basin_names, thresholds, nc_threshold_dir='../DATA/DT_diff_nc/', 
                       output_file='Figure1b_basins.png'):
    plt.figure(figsize=(11, 2))
    
    for i, basin in enumerate(basin_names):
        thres_nc_file = os.path.join(nc_threshold_dir, f'Obs_{basin}_threshold_analysis.nc')
        try:
            thres_data = read_threshold_nc(thres_nc_file)
        except FileNotFoundError:
            print(f"Warning: {thres_nc_file} not found, skipping {basin}")
            continue
        
        plt.subplot(1, 6, i+1)
        
        basin_data = df_all[df_all.ocean == basin]
        
        ind = basin_data.year < 2008
        plt.scatter(basin_data.V0[ind], basin_data.decay[ind], c='#2166AC', s=0.2, label='before')
        ind = basin_data.year > 2007
        plt.scatter(basin_data.V0[ind], basin_data.decay[ind], c='#B2182B', s=0.2, label='after')
        
        x_early_smooth, y_early_smooth = smooth_curve(thres_data.LI_threshold, thres_data.early_period_mean)
        x_late_smooth, y_late_smooth = smooth_curve(thres_data.LI_threshold, thres_data.late_period_mean)
        
        plt.plot(x_early_smooth, y_early_smooth, c='#2166AC', lw=2)
        plt.plot(x_late_smooth, y_late_smooth, c='#B2182B', lw=2)
        
        thres = thresholds[basin]
        plt.vlines(thres, 0, 160, color='k', ls='dashed', lw=0.8)
        plt.text(thres+1, 120, str(thres)+'m/s')
        plt.ylim(5, 160)
        plt.xlim(10, 75)
        plt.xticks([25,50,75]) 
        plt.yticks([50,100,150])   
        x_pos = 0.75 if basin == 'WP' else 0.8
        plt.text(x_pos, 0.85, basin, fontweight='bold', transform=plt.gca().transAxes)
        
        if i == 0:
            plt.ylabel('Decay timescale (h)')
        else:
            plt.yticks([])
    
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=1000, bbox_inches='tight')
    plt.close()
    print(f"Basin subplots saved as {output_file}")

def main():
    nc_threshold_dir = '../DATA/DT_diff_nc' 
    

    file_paths = [
        '../DATA/WP/TC_decay_timescale.nc',
        '../DATA/EP/TC_decay_timescale.nc',
        '../DATA/NI/TC_decay_timescale.nc',
        '../DATA/SI/TC_decay_timescale.nc',
        '../DATA/SP/TC_decay_timescale.nc',
        '../DATA/NA/TC_decay_timescale.nc'
    ]

    basin_names = ['WP', 'EP', 'NI', 'SI', 'SP', 'NA']
    thresholds = {
        'WP': 33.8,
        'EP': 29.8,
        'NI': 29.7,
        'SI': 27.2,
        'SP': 16.3,
        'NA': 38.8
    }
    
    all_data = []
    
    for file_path in file_paths:
        V0, year, decay = read_nc_data(file_path)
        basin = get_basin_name(file_path)

        basin_df = pd.DataFrame({
            'V0': V0,
            'year': year,
            'decay': decay,
            'ocean': basin
        })
        
        all_data.append(basin_df)

    df_all = pd.concat(all_data, ignore_index=True)
    
    df_all = df_all.dropna()

    df_global = df_all[['V0', 'year', 'decay']].copy()
    
    try:
        thres_global_nc = os.path.join(nc_threshold_dir, 'Obs_global_threshold_analysis.nc')
        thres_global = read_threshold_nc(thres_global_nc)
    except FileNotFoundError:
        print(f"Error: {thres_global_nc} not found")
        return
    
    plot_global_scatter(df_global, thres_global)
    
    plot_basin_subplots(df_all, basin_names, thresholds, nc_threshold_dir)


if __name__ == "__main__":
    main()