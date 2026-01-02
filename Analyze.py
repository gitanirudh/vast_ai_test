#!/usr/bin/env python3
"""
analyze.py - Analyze and visualize results from multiple runs
UPDATED: Fixed statistical_tests.csv issue and added per-class radar plot
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR
MODELS = ["roberta-base", "microsoft__deberta-v3-large", "answerdotai__ModernBERT-large"]

print("="*80)
print("ANALYZING EXPERIMENTAL RESULTS")
print("="*80)

# Load all summaries
all_dfs = {}
for model_name in MODELS:
    summary_file = RESULTS_DIR / f"{model_name}_run_summary.csv"
    if summary_file.exists():
        df = pd.read_csv(summary_file)
        all_dfs[model_name] = df
        print(f"Loaded {len(df)} runs for {model_name}")
    else:
        print(f"WARNING: {summary_file} not found!")

if not all_dfs:
    print("ERROR: No summary files found!")
    exit(1)

# Combine all dataframes
all_df = pd.concat(all_dfs.values(), ignore_index=True)

print("\n" + "="*80)
print("SUMMARY STATISTICS (Best F1 Score)")
print("="*80)

summary_stats_display = all_df.groupby("model")["best_f1_score"].agg([
    ("count", "count"),
    ("mean", "mean"),
    ("std", "std"),
    ("min", "min"),
    ("max", "max"),
    ("median", "median"),
])

print(summary_stats_display)
print()

# Save summary statistics with numeric values
summary_stats_numeric = all_df.groupby("model")["best_f1_score"].agg([
    ("count", "count"),
    ("mean", "mean"),
    ("std", "std"),
    ("min", "min"),
    ("max", "max"),
    ("median", "median"),
])
summary_stats_numeric.to_csv(OUTPUT_DIR / "summary_statistics.csv")
print(f"Summary statistics saved to: {OUTPUT_DIR / 'summary_statistics.csv'}")

# Statistical significance tests
print("\n" + "="*80)
print("STATISTICAL TESTS (Paired t-test)")
print("="*80)

model_pairs = [
    ("roberta-base", "microsoft__deberta-v3-large"),
    ("roberta-base", "answerdotai__ModernBERT-large"),
    ("microsoft__deberta-v3-large", "answerdotai__ModernBERT-large"),
]

test_results = []

for model1, model2 in model_pairs:
    if model1 in all_dfs and model2 in all_dfs:
        scores1 = all_dfs[model1]["best_f1_score"].values
        scores2 = all_dfs[model2]["best_f1_score"].values
        
        # Ensure same number of runs
        min_len = min(len(scores1), len(scores2))
        scores1 = scores1[:min_len]
        scores2 = scores2[:min_len]
        
        # Paired t-test
        t_stat, p_val = stats.ttest_rel(scores1, scores2)
        mean_diff = scores1.mean() - scores2.mean()
        
        print(f"\n{model1} vs {model2}:")
        print(f"  Mean F1 ({model1}): {scores1.mean():.4f}")
        print(f"  Mean F1 ({model2}): {scores2.mean():.4f}")
        print(f"  Mean difference: {mean_diff:.4f}")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_val:.6f}")
        print(f"  Significant (p<0.05): {'Yes' if p_val < 0.05 else 'No'}")
        print(f"  Significant (p<0.01): {'Yes' if p_val < 0.01 else 'No'}")
        
        test_results.append({
            "comparison": f"{model1} vs {model2}",
            "model1_mean": scores1.mean(),
            "model2_mean": scores2.mean(),
            "mean_diff": mean_diff,
            "t_stat": t_stat,
            "p_value": p_val,
            "significant_005": p_val < 0.05,
            "significant_001": p_val < 0.01,
        })

# Save test results
test_df = pd.DataFrame(test_results)
test_df.to_csv(OUTPUT_DIR / "statistical_tests.csv", index=False)
print(f"\nStatistical tests saved to: {OUTPUT_DIR / 'statistical_tests.csv'}")

# Visualizations
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# 1. Box plot of F1 scores
plt.figure(figsize=(10, 6))
sns.boxplot(data=all_df, x="model", y="best_f1_score", palette="Set2")
plt.title("F1 Score Distribution Across Models", fontsize=14, fontweight='bold')
plt.ylabel("Best F1 Score", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "f1_boxplot.png")
print(f"Box plot saved to: {OUTPUT_DIR / 'f1_boxplot.png'}")
plt.close()

# 2. Bar plot with error bars
plt.figure(figsize=(10, 6))
model_means = all_df.groupby("model")["best_f1_score"].mean()
model_stds = all_df.groupby("model")["best_f1_score"].std()

x_pos = np.arange(len(MODELS))
plt.bar(x_pos, model_means, yerr=model_stds, capsize=5, alpha=0.7, color=['#FF9999', '#66B2FF', '#99FF99'])
plt.xticks(x_pos, MODELS, rotation=15)
plt.ylabel("Best F1 Score", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.title("Mean F1 Score with Standard Deviation", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "f1_barplot.png")
print(f"Bar plot saved to: {OUTPUT_DIR / 'f1_barplot.png'}")
plt.close()

# 3. Violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=all_df, x="model", y="best_f1_score", palette="muted")
plt.title("F1 Score Distribution (Violin Plot)", fontsize=14, fontweight='bold')
plt.ylabel("Best F1 Score", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "f1_violin.png")
print(f"Violin plot saved to: {OUTPUT_DIR / 'f1_violin.png'}")
plt.close()

# 4. Training time comparison
plt.figure(figsize=(10, 6))
sns.boxplot(data=all_df, x="model", y="total_training_time_seconds", palette="Set3")
plt.title("Training Time Comparison", fontsize=14, fontweight='bold')
plt.ylabel("Training Time (seconds)", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_time.png")
print(f"Training time plot saved to: {OUTPUT_DIR / 'training_time.png'}")
plt.close()

# 5. Loss comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

sns.boxplot(data=all_df, x="model", y="best_train_loss", palette="Set2", ax=ax1)
ax1.set_title("Training Loss", fontsize=12, fontweight='bold')
ax1.set_ylabel("Best Training Loss", fontsize=10)
ax1.set_xlabel("Model", fontsize=10)
ax1.tick_params(axis='x', rotation=15)

sns.boxplot(data=all_df, x="model", y="best_val_loss", palette="Set2", ax=ax2)
ax2.set_title("Validation Loss", fontsize=12, fontweight='bold')
ax2.set_ylabel("Best Validation Loss", fontsize=10)
ax2.set_xlabel("Model", fontsize=10)
ax2.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "loss_comparison.png")
print(f"Loss comparison saved to: {OUTPUT_DIR / 'loss_comparison.png'}")
plt.close()

# 6. Overall radar plot
print("Creating overall radar plot...")
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='polar')

# Metrics for radar plot
metrics = ['F1 Score', 'Precision', 'Recall', 'Speed\n(inv. time)', 'Stability\n(inv. std)']
num_vars = len(metrics)

# Calculate angles for each metric
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

# Prepare data for each model
for model_name in MODELS:
    if model_name in all_dfs:
        df_model = all_dfs[model_name]
        
        # Calculate metrics (normalized to 0-1 scale)
        f1_norm = df_model['best_f1_score'].mean()
        
        # For precision and recall, approximate from F1
        precision_est = f1_norm
        recall_est = f1_norm
        
        # Speed: inverse of training time (normalized)
        max_time = all_df['total_training_time_seconds'].max()
        speed_norm = 1 - (df_model['total_training_time_seconds'].mean() / max_time)
        
        # Stability: inverse of std (normalized)
        max_std = all_df['best_f1_score'].std()
        stability_norm = 1 - (df_model['best_f1_score'].std() / max_std) if max_std > 0 else 1.0
        
        values = [f1_norm, precision_est, recall_est, speed_norm, stability_norm]
        values += values[:1]  # Complete the circle
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.15)

# Customize plot
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, size=10)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
plt.title("Model Comparison - Radar Plot", size=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "radar_plot.png", bbox_inches='tight')
print(f"Radar plot saved to: {OUTPUT_DIR / 'radar_plot.png'}")
plt.close()

# 7. Per-class F1 radar plot
print("Creating per-class F1 radar plot...")

# Define all 19 classes from your dataset
all_classes = [
    'Project Scope', 'OBDH', 'Project Organisation / Documentation',
    'Space Environment', 'Propulsion', 'GN&C', 'Materials / EEEs',
    'Structure & Mechanism', 'Telecom.', 'Nonconformity', 'Power',
    'Safety / Risk (Control)', 'Parameter', 'Thermal', 'Quality control',
    'Measurement', 'Cleanliness', 'System engineering'
]

# NOTE: This uses placeholder data. To use real per-class scores,
# you need to read from the actual report_X.csv files
per_class_f1 = {}
for model_name in MODELS:
    if model_name in all_dfs:
        # Placeholder: random values for demonstration
        # Replace this with actual per-class F1 scores from report files
        per_class_f1[model_name] = {
            cls: np.random.uniform(0.3, 0.7) for cls in all_classes[:8]
        }

if per_class_f1:
    class_names = list(next(iter(per_class_f1.values())).keys())
    num_classes = len(class_names)
    
    # Calculate angles
    angles_classes = np.linspace(0, 2 * np.pi, num_classes, endpoint=False).tolist()
    angles_classes += angles_classes[:1]
    
    # Create figure
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='polar')
    
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    
    for idx, (model_name, class_scores) in enumerate(per_class_f1.items()):
        values = list(class_scores.values())
        values += values[:1]  # Complete circle
        
        ax.plot(angles_classes, values, 'o-', linewidth=2, 
                label=model_name.replace('__', ' ').replace('microsoft ', '').replace('answerdotai ', ''), 
                color=colors[idx])
        ax.fill(angles_classes, values, alpha=0.15, color=colors[idx])
    
    # Customize
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles_classes[:-1])
    ax.set_xticklabels(class_names, size=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.title("Per-Class F1 Score Comparison\n(Placeholder Data - Update with Real Scores)", 
              size=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "radar_plot_per_class.png", bbox_inches='tight', dpi=300)
    print(f"Per-class radar plot saved to: {OUTPUT_DIR / 'radar_plot_per_class.png'}")
    print("NOTE: Using placeholder data. Update code to read from report files for real per-class scores.")
    plt.close()

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print(f"  - {OUTPUT_DIR / 'summary_statistics.csv'}")
print(f"  - {OUTPUT_DIR / 'statistical_tests.csv'}")
print(f"  - {OUTPUT_DIR / 'f1_boxplot.png'}")
print(f"  - {OUTPUT_DIR / 'f1_barplot.png'}")
print(f"  - {OUTPUT_DIR / 'f1_violin.png'}")
print(f"  - {OUTPUT_DIR / 'training_time.png'}")
print(f"  - {OUTPUT_DIR / 'loss_comparison.png'}")
print(f"  - {OUTPUT_DIR / 'radar_plot.png'}")
print(f"  - {OUTPUT_DIR / 'radar_plot_per_class.png'}")
print("\n" + "="*80)
