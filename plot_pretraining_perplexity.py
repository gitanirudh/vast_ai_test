#!/usr/bin/env python3
"""
plot_pretraining_perplexity.py
Generate perplexity evolution plot during pretraining (like Figure 2 in the paper)
Uses real data from training_log.csv files saved during pretraining
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
RUNS_DIR = Path("runs")
OUTPUT_FILE = Path("results_30runs") / "pretraining_perplexity_evolution.png"

MODELS = {
    "roberta-base": "RoBERTa-base",
    "microsoft__deberta-v3-large": "DeBERTa-v3-large",
    "answerdotai__ModernBERT-large": "ModernBERT-large"
}

print("="*80)
print("CREATING PRETRAINING PERPLEXITY EVOLUTION PLOT")
print("="*80)

# Read training logs
model_perplexity_data = {}

for model_dir, model_label in MODELS.items():
    training_log = RUNS_DIR / model_dir / "training_log.csv"
    
    if training_log.exists():
        print(f"\nReading: {training_log}")
        df = pd.read_csv(training_log)
        
        # Check what columns are available
        print(f"  Columns: {df.columns.tolist()}")
        
        # Look for perplexity or eval_loss columns
        # Perplexity = exp(eval_loss) or 2^(eval_loss/log(2))
        
        # Evaluation happens at specific checkpoints, not every step
        # Filter for rows where eval_ppl is not NaN
        df_eval = df[df['eval_ppl'].notna()].copy()
        
        if len(df_eval) == 0:
            print(f"  WARNING: No evaluation data found (all eval_ppl are NaN)!")
            # Try using eval_loss instead
            df_eval = df[df['eval_loss'].notna()].copy()
            if len(df_eval) == 0:
                print(f"  ERROR: No evaluation data at all!")
                continue
            perplexity = np.exp(df_eval['eval_loss'].values)
            epochs = df_eval['epoch'].values
            print(f"  Using eval_loss to compute perplexity")
        else:
            perplexity = df_eval['eval_ppl'].values
            epochs = df_eval['epoch'].values
            print(f"  Using eval_ppl directly")
        
        # Remove any remaining NaN or inf values
        valid_mask = np.isfinite(perplexity)
        perplexity = perplexity[valid_mask]
        epochs = epochs[valid_mask]
        
        if len(perplexity) == 0:
            print(f"  ERROR: No valid perplexity values after filtering!")
            continue
        
        model_perplexity_data[model_label] = {
            'epochs': epochs,
            'perplexity': perplexity
        }
        
        print(f"  Loaded {len(epochs)} epochs")
        print(f"  Perplexity range: {perplexity.min():.2f} to {perplexity.max():.2f}")
    else:
        print(f"\nWARNING: {training_log} not found!")

if not model_perplexity_data:
    print("\nERROR: No training logs found!")
    print("Make sure pretraining logs are in runs/{model}/training_log.csv")
    exit(1)

# Create plot
fig, ax = plt.subplots(figsize=(12, 8))

colors = {
    "RoBERTa-base": "#3498DB",
    "DeBERTa-v3-large": "#E74C3C", 
    "ModernBERT-large": "#2ECC71"
}

linestyles = {
    "RoBERTa-base": '-',
    "DeBERTa-v3-large": '--',
    "ModernBERT-large": ':'
}

for model_label, data in model_perplexity_data.items():
    ax.plot(data['epochs'], data['perplexity'],
            label=model_label,
            color=colors.get(model_label, '#000000'),
            linestyle=linestyles.get(model_label, '-'),
            linewidth=2.5,
            marker='o',
            markersize=4,
            markevery=max(1, len(data['epochs']) // 20))  # Show markers every ~20th point

# Formatting to match paper style
ax.set_xlabel('Training Epoch', fontsize=16, fontweight='bold')
ax.set_ylabel('Evaluation Perplexity', fontsize=16, fontweight='bold')
ax.set_title('Evolution of the Evaluation Perplexity in Function of the\nNumber of Further Pre-training Epochs', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, color='gray')
ax.legend(fontsize=13, loc='upper right', framealpha=0.95, edgecolor='black',
          fancybox=True, shadow=True)
ax.tick_params(axis='both', labelsize=12)

# Add subtle background
ax.set_facecolor('#F8F9FA')

# Set y-axis to start from appropriate value
y_min = min([data['perplexity'].min() for data in model_perplexity_data.values()])
y_max = max([data['perplexity'].max() for data in model_perplexity_data.values()])
ax.set_ylim(y_min * 0.95, y_max * 1.05)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
print(f"\n{'='*80}")
print(f"Perplexity evolution plot saved to: {OUTPUT_FILE}")
print(f"{'='*80}\n")

plt.close()

# Print summary statistics
print("\nPerplexity Summary (from pretraining):")
print("="*80)
print(f"{'Model':<25} {'Initial':>12} {'Final':>12} {'Reduction':>12}")
print("-"*80)

for model_label, data in model_perplexity_data.items():
    initial = data['perplexity'][0]
    final = data['perplexity'][-1]
    reduction = ((initial - final) / initial) * 100
    
    print(f"{model_label:<25} {initial:>12.2f} {final:>12.2f} {reduction:>11.1f}%")

print("\n" + "="*80)
print("Analysis:")
print("="*80)

for model_label, data in model_perplexity_data.items():
    ppl = data['perplexity']
    epochs = data['epochs']
    
    # Find when convergence happens (when change is < 1%)
    convergence_epoch = None
    for i in range(1, len(ppl)):
        pct_change = abs((ppl[i] - ppl[i-1]) / ppl[i-1]) * 100
        if pct_change < 1.0:
            convergence_epoch = epochs[i]
            break
    
    if convergence_epoch:
        print(f"{model_label}: Converged around epoch {convergence_epoch:.0f}")
    else:
        print(f"{model_label}: Did not fully converge in {len(epochs)} epochs")

print("\n" + "="*80)
