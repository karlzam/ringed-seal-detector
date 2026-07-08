import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

file_path_xlsx = "detections-pos.xlsx"

df = pd.read_excel(file_path_xlsx)

# Define the 10 model columns
model_cols = ['1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9']

# 2. Compute the spread and median metrics
df['ensemble_median'] = df[model_cols].median(axis=1)
df['ensemble_sd'] = df[model_cols].std(axis=1)
df['ensemble_range'] = df[model_cols].max(axis=1) - df[model_cols].min(axis=1)

# Set decision filter thresholds
min_median_score = 0.5   # Initial detection threshold
max_allowed_sd = 0.15    # Maximum allowed spread (SD)

plt.style.use('default') 

# Initialize a figure with 1 row and 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
sc = ax1.scatter(
    df['ensemble_median'], 
    df['ensemble_sd'], 
    c=df['ensemble_range'], 
    cmap='viridis', 
    alpha=0.5, 
    s=12
)

# Draw decision boundary lines
ax1.axvline(x=min_median_score, color='red', linestyle='--', alpha=0.7, label=f'Score Cut-off ({min_median_score})')
ax1.axhline(y=max_allowed_sd, color='green', linestyle='--', alpha=0.7, label=f'Max Allowed SD ({max_allowed_sd})')

ax1.set_xlabel('Ensemble Median (Prediction Score)')
ax1.set_ylabel('Ensemble Standard Deviation (Spread)')
ax1.set_title('A: Model Uncertainty vs. Prediction Score')
ax1.legend(loc='upper left')

# Add the colorbar attached to the first subplot
cbar = fig.colorbar(sc, ax=ax1)
cbar.set_label('Ensemble Range (Max - Min)')

high_median_df = df[df['ensemble_median'] > min_median_score]

ax2.hist(high_median_df['ensemble_sd'], bins=30, color='#3b528b', edgecolor='black', alpha=0.8)

# Vertical uncertainty threshold boundary
ax2.axvline(x=max_allowed_sd, color='green', linestyle='--', linewidth=2, label=f'Filter Threshold ({max_allowed_sd})')

ax2.set_xlabel('Ensemble Standard Deviation (Spread)')
ax2.set_ylabel('Frequency')
ax2.set_title(f'B: Distribution of Spread (Median > {min_median_score})')
ax2.legend()

# Final layout adjustments and display
plt.tight_layout()
plt.savefig('ensemble_spread_analysis.png', dpi=300)
plt.show()
