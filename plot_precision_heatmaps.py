#!/usr/bin/env python3
"""
Generate heatmaps for cross-precision comparisons from embedding precision analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def create_precision_heatmaps():
    """Create heatmaps for deterministic and non-deterministic cross-precision comparisons."""

    # Define precision types
    precisions = ['FP32', 'FP16', 'BF16', 'TF32']

    # Deterministic cross-precision L2 distances (from precision_comparison_report.md)
    det_data = {
        ('FP32', 'FP16'): 5.74e-04,
        ('FP32', 'BF16'): 6.31e-03,
        ('FP32', 'TF32'): 4.09e-04,
        ('FP16', 'BF16'): 6.39e-03,
        ('FP16', 'TF32'): 5.53e-04,
        ('BF16', 'TF32'): 6.36e-03,
    }

    # Non-deterministic cross-precision L2 distances (identical to deterministic)
    nondet_data = {
        ('FP32', 'FP16'): 5.74e-04,
        ('FP32', 'BF16'): 6.31e-03,
        ('FP32', 'TF32'): 4.09e-04,
        ('FP16', 'BF16'): 6.39e-03,
        ('FP16', 'TF32'): 5.53e-04,
        ('BF16', 'TF32'): 6.36e-03,
    }

    # Create symmetric matrices
    def create_distance_matrix(data, precisions):
        n = len(precisions)
        matrix = np.zeros((n, n))

        # Fill upper triangle with data
        for (p1, p2), distance in data.items():
            i = precisions.index(p1)
            j = precisions.index(p2)
            matrix[i, j] = distance
            matrix[j, i] = distance  # Make symmetric

        return matrix

    det_matrix = create_distance_matrix(det_data, precisions)
    nondet_matrix = create_distance_matrix(nondet_data, precisions)

    # Create the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Common color map settings
    vmin = 0
    vmax = max(det_matrix.max(), nondet_matrix.max())

    # Deterministic heatmap
    im1 = ax1.imshow(det_matrix, cmap='YlOrRd', vmin=vmin, vmax=vmax)
    ax1.set_title('Deterministic Cross-Precision Comparisons\n(L2 Distance)', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(precisions)))
    ax1.set_yticks(range(len(precisions)))
    ax1.set_xticklabels(precisions)
    ax1.set_yticklabels(precisions)

    # Add text annotations for deterministic
    for i in range(len(precisions)):
        for j in range(len(precisions)):
            if i != j:  # Don't annotate diagonal (self-comparisons)
                text = f'{det_matrix[i, j]:.2e}'
                ax1.text(j, i, text, ha='center', va='center', fontsize=10, fontweight='bold')
            else:
                ax1.text(j, i, '0.00e+00', ha='center', va='center', fontsize=10,
                        fontweight='bold', color='gray')

    # Non-deterministic heatmap
    im2 = ax2.imshow(nondet_matrix, cmap='YlOrRd', vmin=vmin, vmax=vmax)
    ax2.set_title('Non-Deterministic Cross-Precision Comparisons\n(L2 Distance)', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(precisions)))
    ax2.set_yticks(range(len(precisions)))
    ax2.set_xticklabels(precisions)
    ax2.set_yticklabels(precisions)

    # Add text annotations for non-deterministic
    for i in range(len(precisions)):
        for j in range(len(precisions)):
            if i != j:  # Don't annotate diagonal (self-comparisons)
                text = f'{nondet_matrix[i, j]:.2e}'
                ax2.text(j, i, text, ha='center', va='center', fontsize=10, fontweight='bold')
            else:
                ax2.text(j, i, '0.00e+00', ha='center', va='center', fontsize=10,
                        fontweight='bold', color='gray')

    # Add colorbar
    cbar = plt.colorbar(im2, ax=[ax1, ax2], shrink=0.6, aspect=30, pad=0.02)
    cbar.set_label('L2 Distance', rotation=270, labelpad=20, fontsize=12)

    # Style improvements
    for ax in [ax1, ax2]:
        ax.set_xlabel('Precision Type', fontsize=12)
        ax.set_ylabel('Precision Type', fontsize=12)

        # Add grid
        ax.set_xticks(np.arange(len(precisions)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(precisions)) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
        ax.tick_params(which='minor', size=0)

    plt.tight_layout()

    # Save the plot
    output_dir = Path('results/analyze')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'precision_cross_comparison_heatmaps.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    print(f"📊 Heatmaps saved to: {output_path}")

    # Also create individual heatmaps with seaborn for better styling
    plt.figure(figsize=(16, 6))

    # Create DataFrames for seaborn
    det_df = pd.DataFrame(det_matrix, index=precisions, columns=precisions)
    nondet_df = pd.DataFrame(nondet_matrix, index=precisions, columns=precisions)

    # Deterministic heatmap with seaborn
    plt.subplot(1, 2, 1)
    sns.heatmap(det_df, annot=True, fmt='.2e', cmap='YlOrRd',
                square=True, linewidths=0.5, cbar=False,
                annot_kws={'fontsize': 10, 'fontweight': 'bold'})
    plt.title('Deterministic Cross-Precision Comparisons\n(L2 Distance)',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Precision Type', fontsize=12)
    plt.ylabel('Precision Type', fontsize=12)

    # Non-deterministic heatmap with seaborn
    plt.subplot(1, 2, 2)
    sns.heatmap(nondet_df, annot=True, fmt='.2e', cmap='YlOrRd',
                square=True, linewidths=0.5, cbar_kws={'label': 'L2 Distance'},
                annot_kws={'fontsize': 10, 'fontweight': 'bold'})
    plt.title('Non-Deterministic Cross-Precision Comparisons\n(L2 Distance)',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Precision Type', fontsize=12)
    plt.ylabel('Precision Type', fontsize=12)

    plt.tight_layout()

    # Save seaborn version
    output_path_sns = output_dir / 'precision_cross_comparison_heatmaps_seaborn.png'
    plt.savefig(output_path_sns, dpi=300, bbox_inches='tight', facecolor='white')

    print(f"📊 Seaborn heatmaps saved to: {output_path_sns}")

    # Create a summary comparison plot
    plt.figure(figsize=(12, 8))

    # Calculate difference matrix (should be zero since det and nondet are identical)
    diff_matrix = nondet_matrix - det_matrix
    diff_df = pd.DataFrame(diff_matrix, index=precisions, columns=precisions)

    plt.subplot(2, 2, 1)
    sns.heatmap(det_df, annot=True, fmt='.2e', cmap='YlOrRd',
                square=True, linewidths=0.5, cbar=True,
                annot_kws={'fontsize': 9})
    plt.title('Deterministic Mode', fontweight='bold')

    plt.subplot(2, 2, 2)
    sns.heatmap(nondet_df, annot=True, fmt='.2e', cmap='YlOrRd',
                square=True, linewidths=0.5, cbar=True,
                annot_kws={'fontsize': 9})
    plt.title('Non-Deterministic Mode', fontweight='bold')

    plt.subplot(2, 2, 3)
    sns.heatmap(diff_df, annot=True, fmt='.2e', cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar=True,
                annot_kws={'fontsize': 9})
    plt.title('Difference (NonDet - Det)', fontweight='bold')

    # Summary statistics subplot
    plt.subplot(2, 2, 4)
    plt.axis('off')

    # Calculate some statistics
    det_mean = np.mean(det_matrix[det_matrix > 0])  # Exclude zeros (diagonal)
    det_std = np.std(det_matrix[det_matrix > 0])
    det_max = np.max(det_matrix)
    det_min = np.min(det_matrix[det_matrix > 0])

    stats_text = f"""
Cross-Precision Comparison Statistics

Deterministic Mode:
• Mean L2 Distance: {det_mean:.2e}
• Std L2 Distance: {det_std:.2e}
• Max L2 Distance: {det_max:.2e}
• Min L2 Distance: {det_min:.2e}

Non-Deterministic Mode:
• Identical to Deterministic

Key Observations:
• Most similar: FP32 ↔ TF32 ({det_data[('FP32', 'TF32')]:.2e})
• Least similar: FP16 ↔ BF16 ({det_data[('FP16', 'BF16')]:.2e})
• Det/NonDet modes are identical
• This suggests non-deterministic mode
  is not working as expected
    """

    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.suptitle('Embedding Precision Cross-Comparison Analysis',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    # Save summary plot
    output_path_summary = output_dir / 'precision_comparison_summary.png'
    plt.savefig(output_path_summary, dpi=300, bbox_inches='tight', facecolor='white')

    print(f"📊 Summary plot saved to: {output_path_summary}")

    plt.show()

if __name__ == "__main__":
    print("🎨 Generating precision comparison heatmaps...")
    create_precision_heatmaps()
    print("✅ All plots generated successfully!")
