#!/usr/bin/env python3
"""
analyze_results.py
Analyze and visualize results from multiple runs
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

summary_stats = all_df.groupby("model")["best_f1_score"].agg([
    ("count", "count"),
    ("mean", lambda x: f"{x.mean():.4f}"),
    ("std", lambda x: f"{x.std():.4f}"),
    ("min", lambda x: f"{x.min():.4f}"),
    ("max", lambda x: f"{x.max():.4f}"),
    ("median", lambda x: f"{x.median():.4f}"),
])

print(summary_stats)
print()

# Save summary statistics
summary_stats.to_csv(OUTPUT_DIR / "summary_statistics.csv")
print(f"Summary statistics saved to: {OUTPUT_DIR / 'summary_statistics.csv'}")

# Statistical significance tests
print("\n" + "="*80)
print("STATISTICAL TESTS (Paired t-test)")
print("="*80)

model_pairs = [
    ("roberta-base", "deberta-v3-large"),
    ("roberta-base", "modernbert-large"),
    ("deberta-v3-large", "modernbert-large"),
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
print("\n" + "="*80)
