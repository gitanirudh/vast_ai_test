#!/usr/bin/env python3
"""
create_comprehensive_perclass_radar.py
Generate per-class radar plot using REAL F1 scores from ALL report files
Shows all 18 classes (excluding averages)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# Configuration
RESULTS_DIR = Path("results_30runs")
REPORTS_DIR = RESULTS_DIR / "reports"
OUTPUT_FILE = RESULTS_DIR / "radar_plot_ALL_classes_REAL.png"

MODELS = ["roberta-base", "microsoft__deberta-v3-large", "answerdotai__ModernBERT-large"]

print("="*80)
print("CREATING COMPREHENSIVE PER-CLASS RADAR PLOT (ALL CLASSES)")
print("="*80)

# Calculate average F1 scores across all 30 runs for each model
# Report mapping: RoBERTa (1-30), DeBERTa (31-60), ModernBERT (61-90)
model_report_ranges = {
    "roberta-base": (1, 30),
    "microsoft__deberta-v3-large": (31, 60),
    "answerdotai__ModernBERT-large": (61, 90),
}

per_class_avg_f1 = {}

for model_name, (start, end) in model_report_ranges.items():
    print(f"\nProcessing {model_name}...")
    print(f"  Reading reports {start} to {end}")
    
    all_runs_data = []
    
    for report_num in range(start, end + 1):
        report_file = REPORTS_DIR / f"report_{report_num}.csv"
        
        if report_file.exists():
            df = pd.read_csv(report_file, index_col=0)
            
            # Extract F1 scores for all classes (exclude avg rows)
            class_f1 = {}
            for idx in df.index:
                if idx not in ['micro avg', 'macro avg', 'weighted avg']:
                    f1_score = df.loc[idx, 'f1-score']
                    class_f1[idx] = f1_score
            
            all_runs_data.append(class_f1)
        else:
            print(f"  WARNING: {report_file} not found!")
    
    print(f"  Successfully read {len(all_runs_data)} reports")
    
    # Calculate average F1 for each class across all runs
    if all_runs_data:
        # Get all unique classes
        all_classes = set()
        for run_data in all_runs_data:
            all_classes.update(run_data.keys())
        
        avg_f1 = {}
        for cls in all_classes:
            scores = [run_data.get(cls, 0.0) for run_data in all_runs_data]
            avg_f1[cls] = np.mean(scores)
        
        per_class_avg_f1[model_name] = avg_f1
        print(f"  Computed averages for {len(avg_f1)} classes")

if not per_class_avg_f1:
    print("\nERROR: No report files found!")
    print("Make sure report files are in:", REPORTS_DIR)
    exit(1)

# Get all unique classes across all models
all_classes = set()
for classes in per_class_avg_f1.values():
    all_classes.update(classes.keys())
all_classes = sorted(list(all_classes))

print(f"\n{'='*80}")
print(f"Total classes found: {len(all_classes)}")
print(f"Classes: {', '.join(all_classes)}")
print(f"{'='*80}")

# Use ALL classes
selected_classes = all_classes

# Create radar plot
num_vars = len(selected_classes)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# Create larger figure for all classes
fig = plt.figure(figsize=(16, 16))
ax = fig.add_subplot(111, projection='polar')

colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
labels = {
    "roberta-base": "RoBERTa-base",
    "microsoft__deberta-v3-large": "DeBERTa-v3-large",
    "answerdotai__ModernBERT-large": "ModernBERT-large"
}

for idx, (model_name, class_scores) in enumerate(per_class_avg_f1.items()):
    # Get F1 scores for all classes
    values = [class_scores.get(cls, 0.0) for cls in selected_classes]
    values += values[:1]  # Complete circle
    
    label = labels.get(model_name, model_name)
    ax.plot(angles, values, 'o-', linewidth=2.5, label=label, color=colors[idx], markersize=6)
    ax.fill(angles, values, alpha=0.15, color=colors[idx])

# Customize
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(selected_classes, size=9)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=10)
ax.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=13, framealpha=0.9)
plt.title("Per-Class F1 Score Comparison - All Classes\n(Average across 30 runs per model)", 
          size=18, fontweight='bold', pad=40)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, bbox_inches='tight', dpi=300)
print(f"\n{'='*80}")
print(f"Comprehensive per-class radar plot saved to: {OUTPUT_FILE}")
print(f"{'='*80}\n")
plt.close()

# Print detailed statistics
print("\nPer-class F1 scores (average across 30 runs):")
print("="*80)

# Create a summary DataFrame
summary_data = []
for cls in selected_classes:
    row = {'Class': cls}
    for model_name in MODELS:
        score = per_class_avg_f1[model_name].get(cls, 0.0)
        label = labels.get(model_name, model_name)
        row[label] = f"{score:.4f}"
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Save summary to CSV
summary_csv = RESULTS_DIR / "per_class_f1_summary.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"\nPer-class summary saved to: {summary_csv}")

# Calculate which model wins for each class
print("\n" + "="*80)
print("Best Model per Class:")
print("="*80)
for cls in selected_classes:
    scores = {labels[m]: per_class_avg_f1[m].get(cls, 0.0) for m in MODELS}
    best_model = max(scores, key=scores.get)
    best_score = scores[best_model]
    print(f"{cls:40s}: {best_model:20s} (F1={best_score:.4f})")

print("\n" + "="*80)
print("DONE!")
print("="*80)
