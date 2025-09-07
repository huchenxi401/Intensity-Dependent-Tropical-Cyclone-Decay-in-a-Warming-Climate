import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
import os



def read_ablation_nc(nc_file_path):
    try:
        with nc.Dataset(nc_file_path, 'r') as ds:
            exp_names_var = ds.variables['experiment_names']
            experiment_names = []
            for i in range(exp_names_var.shape[1]):
                chars = exp_names_var[:, i]
                name = ''.join([c.decode('utf-8') for c in chars if c != b''])
                experiment_names.append(name)

            r2_differences = ds.variables['r2_differences'][:]
            
            ablation_results = pd.DataFrame({
                'Experiment': experiment_names,
                'R2_difference': r2_differences
            })
            
            return ablation_results
            
    except Exception as e:
        print(f"Error reading NC file {nc_file_path}: {str(e)}")
        raise

nc_file_path = '../DATA/ML_output/Alation_weak.nc'

ablation_results = read_ablation_nc(nc_file_path)

ablation_results = ablation_results.dropna()

plt.figure(figsize=(6, 3))

group_colors = {
    'No Characteristic': '#2E7D32',
    'No Dynamic': '#1565C0', 
    'No Thermodynamic': '#E65100',
}

mask = ablation_results['Experiment'] != 'All Features'
bars = plt.bar(ablation_results[mask]['Experiment'], 
               ablation_results[mask]['R2_difference'])

green_features = ['Intensity', 'Speed', 'Ocean%', 'Elevation', 'Ruggedness']
red_features = ['Temperature', 'SST', '500 Moisture', '850 Moisture', '850 RH', 'Soil Moisture']
blue_features = ['200 Divergence', '850 Vorticity', 'Vertical Motion', 'VWS']

for bar, experiment in zip(bars, ablation_results[mask]['Experiment']):
    if experiment in group_colors:
        bar.set_color(group_colors[experiment])
    else:
        feature = experiment.replace('Without ', '')
        if feature in green_features:
            bar.set_color('#2E7D32')
        elif feature in blue_features:
            bar.set_color('#1565C0')
        elif feature in red_features:
            bar.set_color('#E65100')

ax = plt.gca()

xticks = ax.get_xticklabels()
for xtick in xticks:
    text = xtick.get_text()
    if text in group_colors:
        xtick.set_color(group_colors[text])
    else:
        feature = text.replace('Without ', '')
        if feature in green_features:
            xtick.set_color('#2E7D32')
        elif feature in blue_features:
            xtick.set_color('#1565C0')
        elif feature in red_features:
            xtick.set_color('#E65100')

plt.axhline(y=0, color='black', linestyle='-', linewidth=1.5)

plt.xticks(rotation=30, ha='right')
plt.ylabel('Precentage Change (%)')

plt.tight_layout()

plt.savefig('./FigureS8b.png', 
            dpi=1000, bbox_inches='tight')
plt.close()
