import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import netCDF4 as nc
data_dir = r"../DATA/ML_output"
strong_file = "Feature importance list_strong.nc"
weak_file = "Feature importance list_weak.nc"

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

strong_df = read_feature_importance_nc(os.path.join(data_dir, strong_file))
weak_df = read_feature_importance_nc(os.path.join(data_dir, weak_file))

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

x_labels = {
    'Decision Tree': 'Tree-based Importance',
    'SHAP Value': 'SHAP Value',
    'Causal Forest': 'Treatment Effect Value'
}

x_limits = {
    'Decision Tree': (0, 0.3),
    'SHAP Value': (0, 6),
    'Causal Forest': (0, 14)
}

methods = ['Decision Tree', 'SHAP Value', 'Causal Forest']

def get_feature_color(feature):
    for category, features in feature_categories.items():
        if feature in features:
            return colors[category]
    return 'gray'  

def plot_feature_importance(data, features, title,  filename, x_label, xlim=None):
    abs_values = np.abs(data)
    sorted_indices = np.argsort(abs_values)

    sorted_features = [features[i] for i in sorted_indices]
    sorted_values = [abs(data[i]) for i in sorted_indices]
    sorted_colors = [get_feature_color(feat) for feat in sorted_features]

    fig, ax = plt.subplots(figsize=(5, 5))

    bars = ax.barh(range(len(sorted_features)), sorted_values, color=sorted_colors, alpha=0.8)

    ax.set_yticks(range(len(sorted_features)))
    #ax.set_yticklabels(sorted_features)
    ax.set_yticks(range(len(sorted_features)))
    for i, feature in enumerate(sorted_features):
        ax.text(-0.02, i, feature, va='center', ha='right', 
            color=get_feature_color(feature), fontsize=12,
            transform=ax.get_yaxis_transform())
    ax.set_yticklabels([]) 
    ax.set_xlabel(x_label)
    

    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[cat], alpha=0.8, label=cat) 
                      for cat in feature_categories.keys()]
    ax.legend(handles=legend_elements, loc='lower right')
    if xlim is not None:
        ax.set_xlim(xlim)
    plt.tight_layout()

    plt.savefig(filename, dpi=1000, bbox_inches='tight')
    plt.close()

def main():
    features = strong_df.columns.tolist()

    strong_data = strong_df.iloc[1:4].values  
    weak_data = weak_df.iloc[1:4].values     
    
    # 为每个方法和台风类型绘制图表
    for i, method in enumerate(methods):
        strong_values = strong_data[i]  
        title_strong = f'Feature Importance - Strong TC - {method}'
        filename_strong = f'FigureS9_strong_{method.replace(" ", "_").lower()}.png'
        plot_feature_importance(strong_values, features, title_strong, filename_strong, 
                       x_labels[method], x_limits[method])
        
        weak_values = weak_data[i] 
        title_weak = f'Feature Importance - Weak TC - {method}'
        filename_weak = f'FigureS9__weak_{method.replace(" ", "_").lower()}.png'
        plot_feature_importance(weak_values, features, title_weak, filename_weak, 
                       x_labels[method], x_limits[method])

if __name__ == "__main__":
    main()
