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



def plot_violin_intensity_period(years, V0, d, filename):
    data = []
    
    for year, v0, decay in zip(years, V0, d):
        if not np.isnan(v0) and not np.isnan(decay):
            period = "1990-2007" if 1990 <= year <= 2007 else "2008-2024" if 2008 <= year <= 2024 else None
            if period:
                if v0 >= 33:
                    intensity = "Strong TC"
                elif 11 <= v0 < 33:
                    intensity = "Weak TC"
                else:
                    continue
                    
                data.append({
                    "Period": period,
                    "Intensity": intensity,
                    "Decay Timescale": decay
                })
    
    df = pd.DataFrame(data)

    df['Group'] = df['Intensity'] + " " + df['Period']

    groups = [
        "Weak TC 1990-2007",
        "Weak TC 2008-2024",
        "Strong TC 1990-2007",
        "Strong TC 2008-2024"
    ]
    groups1 = [
        "Weak TC \n1990-2007",
        "Weak TC \n2008-2024",
        "Strong TC \n1990-2007",
        "Strong TC \n2008-2024"
    ]
    colors = ['#2166AC', '#B2182B', '#2166AC', '#B2182B']
    df['Group'] = pd.Categorical(df['Group'], categories=groups, ordered=True)
    
    df = df.sort_values('Group')
    for group in groups:
        if group not in df['Group'].unique():
            dummy = pd.DataFrame({
                'Period': [group.split()[-1]],
                'Intensity': [' '.join(group.split()[:-1])],
                'Decay Timescale': [np.nan],
                'Group': [group]
            })
            df = pd.concat([df, dummy], ignore_index=True)

    all_data = {}
    for group in groups:
        all_data[group] = df[df['Group'] == group]['Decay Timescale'].dropna().tolist()

    fig, ax = plt.subplots(figsize=(5, 4))

    sns.violinplot(
        x="Group", 
        y="Decay Timescale", 
        data=df,
        order=groups,  
        palette=dict(zip(groups, colors)),
        inner=None,
        fill=False,
        cut=0
    )

    boxplot = plt.boxplot(
        [all_data[group] for group in groups],
        positions=range(len(groups)),
        # widths=0.3,
        patch_artist=False,
        showfliers=False,
        boxprops={'color': 'black'}, #, 'linewidth': 5, 'linestyle': '-'
        whiskerprops={'color': 'black'}, #, 'linewidth': 5, 'linestyle': '-'
        capprops={'color': 'black'}, #, 'linewidth': 5, 'linestyle': '-'
        medianprops={'color': 'black'} #, 'linewidth': 6, 'linestyle': '-'
    )

    plt.ylabel("Decay Timescale (h)")
    plt.xlabel("")  


    plt.xticks(range(len(groups1)), [g.replace(" (", "\n(") for g in groups1])#, fontsize=58
    plt.tick_params(axis='y')#, labelsize=58

    y_max = df['Decay Timescale'].max() * 1.1
    plt.ylim(0, max(100, y_max))  

    plt.vlines(1.5, 0, max(100, y_max), color='k', linestyle='dashed', linewidth=0.5)

    weak_early = all_data[groups[0]]
    weak_late = all_data[groups[1]]
    if weak_early and weak_late:
        t_stat, p_val = stats.ttest_ind(weak_early, weak_late, equal_var=False)
        plt.text(0.5, 125, 'Weak\n**', ha='center')#, fontsize=58, color='blue'
    print(np.mean(weak_early))
    print(np.mean(weak_late))
    strong_early = all_data[groups[2]]
    strong_late = all_data[groups[3]]
    if strong_early and strong_late:
        t_stat, p_val = stats.ttest_ind(strong_early, strong_late, equal_var=False)
        plt.text(2.5, 125, 'Strong\n**', ha='center')#, fontsize=58， P<0.01, color='red'
    print(np.mean(strong_early))
    print(np.mean(strong_late))
    positions = [0, 1, 2, 3]
    counts = np.array([366, 236, 144, 109])

    for pos, count in zip(positions, counts):
        plt.text(pos, 8, f'n={count}', ha='center', va='top', fontsize=10) 

    plt.tight_layout()
    plt.savefig(filename, dpi=1000, bbox_inches='tight')
    plt.close()



def main():
    file_paths = [
        'D:/py/TC-decay-test/RAFT/TC_decay_timescale_raft.nc'
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

    plot_violin_intensity_period(year_all, V0_all, d_all,
                                './Figure5b.png')
    
if __name__ == "__main__":
    main()