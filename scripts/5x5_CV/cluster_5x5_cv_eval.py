import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import AnovaRM
import scipy.stats as stats
import warnings
import os

# Ignore warnings to keep the output clean
warnings.filterwarnings('ignore')

# ==========================================
# Helper Functions
# ==========================================
def _load_and_prepare_data(method_csv_map):
    dfs = []
    for name, path in method_csv_map.items():
        df = pd.read_csv(path) 
        df['Method'] = name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def _print_summary_stats(df, metric, methods_order):
    """
    Calculate and print the mean, standard deviation (Std), and 95% confidence interval (CI) for each method.
    Strictly display in the order passed.
    """
    print(f"\n--- [Step 0] Data Statistics Summary ({metric}) ---")
    # Print header
    print(f"{'Method':<15} | {'Mean':<8} | {'Std':<8} | {'95% CI'}")
    print("-" * 55)
    
    for method in methods_order:
        data = df[df['Method'] == method][metric].values
        n = len(data)
        
        # Calculate mean and standard deviation
        mean_val = np.mean(data)
        std_val = np.std(data, ddof=1) # ddof=1 to calculate sample standard deviation
        
        # Calculate Standard Error and 95% CI (based on t-distribution)
        se = stats.sem(data)
        ci_lower, ci_upper = stats.t.interval(0.95, n-1, loc=mean_val, scale=se)
        
        # Format output
        print(f"{method:<15} | {mean_val:<8.4f} | {std_val:<8.4f} | [{ci_lower:.4f}, {ci_upper:.4f}]")

def _calculate_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def _plot_simultaneous_ci(df, tukey_result, metric_name, output_dir, methods_order):
    """
    Plot and save confidence interval plots.
    Fixed API compatibility issue: manually calculate 95% CI half-width, no longer relying on tukey_result.halfwidths.
    """
    import scipy.stats as stats
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Find the best method (Lower RMSE is better, higher R2 is better)
    mean_scores = df.groupby('Method')[metric_name].mean()
    ascending_order = True if metric_name.upper() == 'RMSE' else False
    best_method = mean_scores.idxmin() if ascending_order else mean_scores.idxmax()
    
    # 2. Extract Tukey data for color determination (significance)
    tukey_data = pd.DataFrame(data=tukey_result._results_table.data[1:], 
                              columns=tukey_result._results_table.data[0])
    
    best_lower = best_upper = 0
    
    # 3. Calculate 95% CI in the given order and plot error bars
    for i, method in enumerate(methods_order):
        # Extract data for the current method
        data = df[df['Method'] == method][metric_name].values
        n = len(data)
        mean_val = np.mean(data)
        
        # Calculate 95% confidence interval half-width (Margin of Error)
        se = stats.sem(data)
        margin_of_error = se * stats.t.ppf(0.975, n - 1)
        
        # Determine color and status
        if method == best_method:
            color = '#1f77b4'  # Best method marked in blue
            best_lower = mean_val - margin_of_error
            best_upper = mean_val + margin_of_error
        else:
            # Find the comparison record between the current method and best_method in the Tukey table
            row = tukey_data[((tukey_data['group1'] == method) & (tukey_data['group2'] == best_method)) | 
                             ((tukey_data['group1'] == best_method) & (tukey_data['group2'] == method))]
            reject = row['reject'].values[0] if not row.empty else False
            # Mark red if null hypothesis is rejected (significant difference), otherwise gray
            color = '#d62728' if reject else '#7f7f7f'  
            
        # Plot the confidence interval for this method
        ax.errorbar(mean_val, i, xerr=margin_of_error, fmt='o', color=color, markersize=8, capsize=6, capthick=2, elinewidth=2)

    # 4. Draw dashed reference lines through the best method's confidence interval
    ax.axvline(best_lower, color='#7f7f7f', linestyle='--', alpha=0.6)
    ax.axvline(best_upper, color='#7f7f7f', linestyle='--', alpha=0.6)
    
    # Set axes and labels (according to the given order)
    ax.set_yticks(np.arange(len(methods_order)))
    # ax.set_yticklabels([])
    ax.set_yticklabels(methods_order, fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # Invert Y-axis so the first passed method is at the top
    
    ax.tick_params(axis='x', labelsize=14) 
    ax.set_xlabel(metric_name, fontsize=16, fontweight='bold')
    # ax.set_title(f'95% Confidence Intervals ({metric_name})')
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    
    # Save image
    save_path = os.path.join(output_dir, f'Cluster_{property_name}_Simultaneous_CI_{metric_name}.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📈 Confidence interval plot saved to: {save_path}")
    plt.close(fig)

def _plot_mcsim(df, tukey_results, metric_col, output_dir, methods_order):
    """Plot and save Multiple Comparisons Similarity (MCSim) Plot"""
    tukey_data = pd.DataFrame(data=tukey_results._results_table.data[1:], 
                              columns=tukey_results._results_table.data[0])
    
    mean_scores = df.groupby('Method')[metric_col].mean()
    n = len(methods_order)
    diff_matrix = np.zeros((n, n))
    annot_matrix = np.empty((n, n), dtype=object)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                diff_matrix[i, j] = 0.0
                annot_matrix[i, j] = "0.0"
                continue
                
            m1, m2 = methods_order[i], methods_order[j]
            row = tukey_data[((tukey_data['group1'] == m1) & (tukey_data['group2'] == m2)) | 
                             ((tukey_data['group1'] == m2) & (tukey_data['group2'] == m1))]
            
            if not row.empty:
                diff = mean_scores[m1] - mean_scores[m2]
                diff_matrix[i, j] = diff
                reject = row['reject'].values[0]
                stars = "*" if reject else ""
                annot_matrix[i, j] = f"{diff:.3f}{stars}"
                
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(diff_matrix, annot=annot_matrix, fmt="", cmap="coolwarm", 
                xticklabels=methods_order, yticklabels=methods_order,
                center=0, ax=ax, 
                annot_kws={"size": 14, "weight": "bold"})
    
    ax.set_xticklabels([f"{m}\n{mean_scores[m]:.3f}" for m in methods_order], 
                       fontsize=13, fontweight='bold')
    ax.set_yticklabels([f"{m}\n{mean_scores[m]:.3f}" for m in methods_order], 
                       fontsize=12, fontweight='bold', multialignment='center')

    cbar = ax.collections[0].colorbar
    cbar.set_label(f'Mean Difference ({metric_col})', size=13, weight='bold')
    cbar.ax.tick_params(labelsize=10)
    plt.setp(cbar.ax.get_yticklabels())

    plt.xticks(rotation=0)
    plt.yticks(rotation=90)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'Cluster_{property_name}_MCSim_Plot_{metric_col}.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 MCSim plot saved to: {save_path}")
    plt.close(fig)


# ==========================================
# Main Control Function
# ==========================================
def compare_models(method_csv_map, target_metric='RMSE', output_dir='./results'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"\n" + "="*50)
    print(f"🚀 Starting evaluation process | Target metric: {target_metric} | Images saved to: {output_dir}")
    print("="*50)

    # 1. Load data and fix order
    df_all = _load_and_prepare_data(method_csv_map)
    methods_order = list(method_csv_map.keys())

    # New: Step 0. Print descriptive statistics summary (Mean, Std, 95% CI)
    _print_summary_stats(df_all, target_metric, methods_order)

    # 2. ANOVA
    print("\n--- [Step 1] Repeated Measures ANOVA ---")
    anova = AnovaRM(data=df_all, depvar=target_metric, subject='Test_Set', within=['Method'])
    anova_results = anova.fit()
    print(anova_results.summary())
    anova_pvalue = anova_results.anova_table['Pr > F'][0]

    # 3. Tukey HSD
    print("\n--- [Step 2] Tukey HSD Post-hoc Test ---")
    tukey = pairwise_tukeyhsd(endog=df_all[target_metric], groups=df_all['Method'], alpha=0.05)
    print(tukey.summary())

    # 4. Cohen's d
    print("\n--- [Step 3] Effect Size Analysis (Cohen's d) ---")
    for i in range(len(methods_order)):
        for j in range(i+1, len(methods_order)):
            m1 = df_all[df_all['Method'] == methods_order[i]][target_metric].values
            m2 = df_all[df_all['Method'] == methods_order[j]][target_metric].values
            d = _calculate_cohens_d(m1, m2)
            print(f"👉 {methods_order[i]} vs {methods_order[j]} - Cohen's d: {d:.3f}")

    # 5. Visualization and Saving
    print("\n--- [Step 4] Generating and saving charts ---")
    _plot_simultaneous_ci(df_all, tukey, target_metric, output_dir, methods_order)
    _plot_mcsim(df_all, tukey, target_metric, output_dir, methods_order)
    
    print("\n✅ Evaluation process completed!")


# ==========================================
# Execution Entry Point
# ==========================================
if __name__ == "__main__":
    
    property_name_list = ["Egc", "Egb", "Eea", "Ei", "Xc", "EPS", "Nc", "Eat"]

    for property_name in property_name_list:

        my_models = {
            'Uni-Mol': f'./cluster_5x5_cv_results/{property_name}/UniMol_cluster_5x5_cv_{property_name}_results.csv',
            'Transpolymer': f'./cluster_5x5_cv_results/{property_name}/Transpolymer_cluster_5x5_cv_{property_name}_results.csv',
            'MMPolymer': f'./cluster_5x5_cv_results/{property_name}/MMPolymer_cluster_5x5_cv_{property_name}_results.csv',
            'PolyConFM': f'./cluster_5x5_cv_results/{property_name}/PolyConFM_cluster_5x5_cv_{property_name}_results.csv'
        }
        
        compare_models(method_csv_map=my_models, target_metric='RMSE', output_dir='./evaluation_results')
        compare_models(method_csv_map=my_models, target_metric='R2', output_dir='./evaluation_results')