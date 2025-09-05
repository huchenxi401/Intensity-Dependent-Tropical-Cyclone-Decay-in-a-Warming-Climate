import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from matplotlib.patches import PathPatch, Polygon
from matplotlib.path import Path
import netCDF4 as nc

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

def read_and_calculate_importance():

    
    tc_features = ['Intensity', 'Speed', 'Ocean%', 'Elevation', 'Ruggedness']  # TC Characteristics
    thermodynamic_features = ['Temperature', 'SST', '500 Moisture', '850 Moisture', '850 RH', 'Soil Moisture']  # Thermodynamic features
    dynamic_features = ['200 Divergence', '850 Vorticity', 'Vertical Motion', 'VWS']  # Dynamic features

    weak_nc_path = '../DATA/ML_output/Feature importance list_weak.nc'
    strong_nc_path = '../DATA/ML_output/Feature importance list_strong.nc'
    
    methods = ['Correlation Coefficient', 'Decision Tree', 'SHAP', 'Causal Forest']
    
    def process_file(file_path, tc_type):

        df = read_feature_importance_nc(file_path)

        

        if tc_type == 'strong':
            tc_features_filtered = ['Speed', 'Ocean%', 'Elevation', 'Ruggedness']  # 移除LI
            thermodynamic_features_filtered = ['SST', '850 Moisture', '850 RH', 'Soil Moisture']  # 移除2m-T和500hPa Q
            dynamic_features_filtered = ['200 Divergence', '850 Vorticity', 'Vertical Motion', 'VWS']
        else: 
            tc_features_filtered = ['Intensity', 'Ocean%', 'Elevation', 'Ruggedness']  # 移除Speed
            thermodynamic_features_filtered = ['SST', '850 Moisture', '850 RH', 'Soil Moisture']  # 移除2m-T和500hPa Q
            dynamic_features_filtered = ['200 Divergence', '850 Vorticity', 'Vertical Motion', 'VWS']
        
        method_data = {}
        
        for i, method in enumerate(methods):
            row_data = df.iloc[i] 
            
            abs_values = row_data.abs()
            
            tc_importance = abs_values[tc_features_filtered].sum()  
            thermo_importance = abs_values[thermodynamic_features_filtered].sum()   
            dynamic_importance = abs_values[dynamic_features_filtered].sum()   
            
            total = tc_importance + thermo_importance + dynamic_importance
            
            if total > 0:
                method_data[method] = {
                    'TC': (tc_importance / total) * 100,
                    'Thermodynamic': (thermo_importance / total) * 100,
                    'Dynamic': (dynamic_importance / total) * 100
                }
            else:
                method_data[method] = {'TC': 0, 'Thermodynamic': 0, 'Dynamic': 0}
        
        return method_data
    
    strong_data = process_file(strong_nc_path, 'strong')
    weak_data = process_file(weak_nc_path, 'weak')
    

    strong_data['Ablation Experiment'] = {
        'TC': 56.0,
        'Thermodynamic': 4.0,
        'Dynamic': 40.0
    }
    
    weak_data['Ablation Experiment'] = {
        'TC': 68.20,
        'Thermodynamic': 16.68,
        'Dynamic': 15.12
    }
    
    return strong_data, weak_data

def create_sankey_diagram(strong_data, weak_data, tc_type='Strong'):

    data = strong_data if tc_type == 'Strong' else weak_data

    methods = ['Correlation Coefficient', 'Decision Tree', 'SHAP', 'Causal Forest', 'Ablation Experiment']
    method_abbrev = ['Correlation\nCoefficient', 'Decision\nTree', 'SHAP\nValue', 'Causal\nForest', 'Ablation\nExperiment']
    method_colors = [
        '#2C3E50',  
        '#34495E',  
        '#5D6D7E',  
        '#85929E',  
        '#AEB6BF'   
    ]
    
    categories = ['TC', 'Thermodynamic', 'Dynamic']
    category_colors = ['#2E7D32', '#E65100', '#1565C0'] 
    category_names = ['TC\nCharacteristics', 'Thermo', 'Dynamic']
    
    fig, ax = plt.subplots(figsize=(5, 3))  
    ax.set_xlim(0, 8)  
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    left_x = 0.8
    method_width = 2.2  
    method_height = 2.0  
    method_spacing = 0.2  
    start_y = 10.5

    right_x = 5.5  
    category_width = 2.0  
    
    category_totals = {cat: 0 for cat in categories}
    for method in methods:
        for cat in categories:
            category_totals[cat] += data[method][cat]
    
    total_method_height = len(methods) * method_height + (len(methods) - 1) * method_spacing
    total_category_importance = sum(category_totals.values())
    
    category_positions = {}
    current_y = start_y + 1
    
    for i, (cat, color, name) in enumerate(zip(categories, category_colors, category_names)):
        height = (category_totals[cat] / total_category_importance) * total_method_height
        y_bottom = current_y - height
        y_center = (current_y + y_bottom) / 2
        
        rect = FancyBboxPatch((right_x, y_bottom), category_width, height,
                             boxstyle="round,pad=0.05", facecolor=color, alpha=0.8,
                             edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        
        ax.text(right_x + category_width/2, y_center, f'{name}\n{category_totals[cat]/5.0:.1f}%',
            ha='center', va='center', color='white')
        
        category_positions[cat] = (right_x, y_bottom, y_center, current_y, category_width, height)
        current_y = y_bottom -0.1 
    
    for i, (method, abbrev, method_color) in enumerate(zip(methods, method_abbrev, method_colors)):
        y_pos = start_y - i * (method_height + method_spacing)
        
        rect = FancyBboxPatch((left_x, y_pos - method_height/2), method_width, method_height,
                             boxstyle="round,pad=0.08", facecolor=method_color, alpha=0.8,
                             edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        
        ax.text(left_x + method_width/2, y_pos, abbrev,
                ha='center', va='center', color='white')
        
        method_right = left_x + method_width
        
        method_top = y_pos + method_height/2
        method_bottom = y_pos - method_height/2
        
        current_y_start = method_top
        
        for j, (cat, color) in enumerate(zip(categories, category_colors)):
            importance = data[method][cat]
            
            if importance > 0:  
                band_height_left = (importance / 100) * method_height
                
                start_y_top = current_y_start
                start_y_bottom = current_y_start - band_height_left
                start_y_center = (start_y_top + start_y_bottom) / 2
                current_y_start = start_y_bottom
                
                target_x, target_bottom, target_center, target_top, target_width, target_height = category_positions[cat]
                
                method_contribution = importance / category_totals[cat]
                band_height_right = method_contribution * target_height

                if not hasattr(create_sankey_diagram, 'used_heights'):
                    create_sankey_diagram.used_heights = {cat: 0 for cat in categories}
                
                target_y_top = target_top - create_sankey_diagram.used_heights[cat]
                target_y_bottom = target_y_top - band_height_right
                target_y_center = (target_y_top + target_y_bottom) / 2


                create_sankey_diagram.used_heights[cat] += band_height_right
                

                ctrl1_x = method_right + 0.5
                ctrl2_x = target_x - 0.3
                
                x_coords = [method_right+0.1, ctrl1_x, ctrl2_x, target_x, 
                           target_x, ctrl2_x, ctrl1_x, method_right+0.1]
                y_coords = [start_y_top, start_y_top, target_y_top, target_y_top,
                           target_y_bottom, target_y_bottom, start_y_bottom, start_y_bottom]
                
                polygon_coords = list(zip(x_coords, y_coords))
                polygon = Polygon(polygon_coords, facecolor=color, alpha=0.5, 
                                edgecolor=color, linewidth=0.3)
                ax.add_patch(polygon)
    
    if hasattr(create_sankey_diagram, 'used_heights'):
        delattr(create_sankey_diagram, 'used_heights')
    
    plt.tight_layout()
    return fig

def main():
    try:
        print("Reading data from NetCDF files...")
        strong_data, weak_data = read_and_calculate_importance()
        fig1 = create_sankey_diagram(strong_data, weak_data, 'Strong')
        plt.savefig('./Figure4b.png', dpi=1000, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', pad_inches=0.1)

        fig2 = create_sankey_diagram(strong_data, weak_data, 'Weak')
        plt.savefig('./Figure4c.png', dpi=1000, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', pad_inches=0.1)

        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()