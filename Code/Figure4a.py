import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
import netCDF4 as nc
import os


def read_feature_importance_nc(nc_file_path):

    try:
        with nc.Dataset(nc_file_path, 'r') as ds:
            col_names_var = ds.variables['column_names']
            column_names = []
            for i in range(col_names_var.shape[1]):
                chars = col_names_var[:, i]
                col_name = ''.join([c.decode('utf-8') for c in chars if c != b''])
                column_names.append(col_name)

            data = {}
            for var_name in ds.variables:
                if var_name.startswith('col_') and var_name != 'column_names':
                    var = ds.variables[var_name]
                    col_idx = var.column_index
                    original_name = column_names[col_idx]
                    
                    if var.dtype.char == 'S':
                        str_data = []
                        for i in range(var.shape[1]):
                            chars = var[:, i]
                            str_val = ''.join([c.decode('utf-8') for c in chars if c != b''])
                            str_data.append(str_val)
                        data[original_name] = str_data
                    else:
                        values = var[:]
                        if hasattr(var, 'missing_value'):
                            values = np.where(values == var.missing_value, np.nan, values)
                        data[original_name] = values
            
            sorted_data = {}
            for col_name in column_names:
                if col_name in data:
                    sorted_data[col_name] = data[col_name]
            
            return pd.DataFrame(sorted_data)
            
    except Exception as e:
        print(f"Error reading NC file {nc_file_path}: {str(e)}")
        raise


feature_categories = {
    'TC Characteristics': ['Intensity', 'Speed', 'Ocean%', 'Elevation', 'Ruggedness'],
    'Thermodynamic features': ['Temperature', 'SST', '500 Moisture', '850 Moisture', '850 RH', 'Soil Moisture'],
    'Dynamic features': ['200 Divergence', '850 Vorticity', 'Vertical Motion', 'VWS']
}

colors = {
    'TC Characteristics': '#2E7D32',     
    'Thermodynamic features': "#E65100",  
    'Dynamic features': '#1565C0'        
}
colors_text = {
    'TC Characteristics': 'white',      
    'Thermodynamic features': 'cyan',  
    'Dynamic features': 'lime'        
}

from matplotlib.colors import LinearSegmentedColormap
methods = ['Correlation Coefficient', 'Decision Tree', 'SHAP Value', 'Causal Forest', 'Ablation Experiment']

def get_feature_category(feature):
    for category, features in feature_categories.items():
        if feature in features:
            return category
    return None



def process_data(df):
    features = df.columns.tolist()
    
    processed_data = {}
    
    for i, method in enumerate(methods):

        values = df.iloc[i].abs().values
        

        total = values.sum()
        relative_importance = (values / total * 100) if total > 0 else values * 0

        feature_importance = list(zip(features, relative_importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        processed_data[method] = feature_importance
    
    return processed_data

def main():
    data_dir = '../DATA/ML_output'
    

    weak_nc_path = '../DATA/ML_output/Feature importance list_weak.nc'
    strong_nc_path = '../DATA/ML_output/Feature importance list_strong.nc'
    
    
    weak_df = read_feature_importance_nc(weak_nc_path)
    strong_df = read_feature_importance_nc(strong_nc_path)
       
    weak_data = process_data(weak_df)
    strong_data = process_data(strong_df)

    fig2, ax2 = plt.subplots(figsize=(5, 6))

    def calculate_rankings(data):

        feature_rankings = {}
        all_features = set()

        for method_data in data.values():
            for feature, _ in method_data:
                all_features.add(feature)

        for feature in all_features:
            rankings = []
            for method in methods:
                method_features = [f for f, _ in data[method]]
                if feature in method_features:
                    rank = method_features.index(feature) + 1
                else:
                    rank = len(method_features) + 1
                rankings.append(rank)
            
            feature_rankings[feature] = {
                'rankings': rankings,
                'avg_rank': np.mean(rankings)
            }
        
        return feature_rankings

    strong_rankings = calculate_rankings(strong_data)
    weak_rankings = calculate_rankings(weak_data)

    strong_sorted = sorted(strong_rankings.items(), key=lambda x: x[1]['avg_rank'])
    weak_sorted = sorted(weak_rankings.items(), key=lambda x: x[1]['avg_rank'])

    max_features = 15  
    y_positions = np.arange(max_features)

    strong_features = [feature for feature, _ in strong_sorted]
    weak_features = [feature for feature, _ in weak_sorted]

    strong_y_pos = list(range(len(strong_sorted)-1, -1, -1))
    weak_y_pos = list(range(len(weak_sorted)-1, -1, -1))

    ax2.set_xlim(0, 32)
    method_markers = ['o', 's', '^', '|', '_']
    
    for i, (feature, data) in enumerate(strong_sorted):
        y_pos = len(strong_sorted) - 1 - i
        category = get_feature_category(feature)
        feature_color = colors.get(category, 'black')
        
        for j, (rank, marker) in enumerate(zip(data['rankings'], method_markers)):
            x_pos = 0 + rank
            ax2.scatter(x_pos, y_pos, color=feature_color, s=20, alpha=0.8, 
                       marker=marker, zorder=2, edgecolor='white', linewidth=1)

    strong_avg_ranks = [0 + data['avg_rank'] for _, data in strong_sorted]
    ax2.plot(strong_avg_ranks, strong_y_pos,  color='black', linewidth=1, alpha=0.7, zorder=1)
    
    for i, (feature, data) in enumerate(weak_sorted):
        y_pos = len(weak_sorted) - 1 - i
        category = get_feature_category(feature)
        feature_color = colors.get(category, 'black') 
        
        for j, (rank, marker) in enumerate(zip(data['rankings'], method_markers)):
            x_pos = 16 + rank
            ax2.scatter(x_pos, y_pos, color=feature_color, s=20, alpha=0.8, 
                       marker=marker, zorder=2, edgecolor='white', linewidth=1)

    weak_avg_ranks = [16 + data['avg_rank'] for _, data in weak_sorted]
    ax2.plot(weak_avg_ranks, weak_y_pos, color='black', linewidth=1, alpha=0.7, zorder=1)

    ax2.set_yticks(strong_y_pos)
    ax2.set_yticklabels(strong_features)

    ax2_right = ax2.twinx()
    ax2_right.set_yticks(weak_y_pos)
    ax2_right.set_yticklabels(weak_features)

    ax2.set_ylim(-0.5, max_features - 0.5)
    ax2_right.set_ylim(-0.5, max_features - 0.5)

    ax2.axvline(x=16, color='black', linestyle='-', linewidth=0.5)

    for i, feature in enumerate(strong_features):
        category = get_feature_category(feature)
        color = colors.get(category, 'black')
        ax2.get_yticklabels()[i].set_color(color)

    for i, feature in enumerate(weak_features):
        category = get_feature_category(feature)
        color = colors.get(category, 'black')
        ax2_right.get_yticklabels()[i].set_color(color)

    left_ticks = [1, 5, 10, 15]
    right_ticks = [17, 22, 27, 31] 
    ax2.set_xticks(left_ticks + right_ticks)

    labels = ['1', '5', '10', '15', '1', '5', '10', '15']
    ax2.set_xticklabels(labels)

    n_features = max(len(strong_sorted), len(weak_sorted))

    ax2.text(8, n_features-1.5, 'Strong TC', ha='center', va='bottom', 
               color='#333333')
    ax2.text(24, n_features-1.5, 'Weak TC', ha='center', va='bottom', 
               color='#333333')

    ax2.set_xlabel('Feature Ranking')
    ax2.tick_params(axis='both')
    ax2_right.tick_params(axis='y')

    legend_elements = []
    for i, (method, marker) in enumerate(zip(methods, method_markers)):
        legend_elements.append(plt.scatter([], [], color='black', 
                                         marker=marker, s=20, label=method, alpha=0.8,
                                         edgecolor='white', linewidth=1))

    legend_elements.append(plt.Line2D([0], [0], color='black', linewidth=1,
                                     label='Average'))

    ax2.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.3),
              ncol=2, frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig('./Figure4a', dpi=1000, bbox_inches='tight')



if __name__ == "__main__":
    main()