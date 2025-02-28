#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for comparing two TI field scans and analyzing specific cortical regions.

This script allows users to:
1. Load two different TI field scans (NIfTI files)
2. Compare the field values within specific cortical regions
3. Generate visualizations and reports of the differences
4. Identify regions with the greatest differences between scans

Example usage:
    python compare_field_scans.py --field1 scan1.nii.gz --field2 scan2.nii.gz \
        --atlas HCP_parcellation.nii.gz --labels HCP.txt \
        --regions 1 2 3 4 5 --output comparison_results

Authors: TI Field Analysis Team
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import nibabel as nib
from datetime import datetime
from ti_field_core import TIFieldAnalyzer

def compare_field_scans(field1, field2, atlas, labels, regions, output_dir, t1_mni=None, visualize=True, top_n=20):
    """
    Compare two field scans focusing on specific cortical regions.
    
    Parameters
    ----------
    field1 : str
        Path to first NIfTI field file
    field2 : str
        Path to second NIfTI field file
    atlas : str
        Path to atlas NIfTI file
    labels : str
        Path to labels file
    regions : list
        List of region IDs or names to analyze. If empty, all regions will be analyzed.
    output_dir : str
        Directory to save outputs
    t1_mni : str, optional
        Path to T1 MNI reference for visualization
    visualize : bool, optional
        Whether to generate visualizations
    top_n : int, optional
        Number of top regions to show in visualizations for full analysis
        
    Returns
    -------
    pandas.DataFrame
        Comparison results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Analyzing fields:\n  Field 1: {field1}\n  Field 2: {field2}")
    print(f"Using atlas: {atlas}")
    print(f"Output directory: {output_dir}")
    
    # Initialize analyzers for both field scans
    print("\nInitializing Field 1 analyzer...")
    analyzer1 = TIFieldAnalyzer(field1, atlas, labels, os.path.join(output_dir, 'field1'), t1_mni)
    
    print("\nInitializing Field 2 analyzer...")
    analyzer2 = TIFieldAnalyzer(field2, atlas, labels, os.path.join(output_dir, 'field2'), t1_mni)
    
    # Check if field dimensions match
    if analyzer1.field_data.shape != analyzer2.field_data.shape:
        print(f"WARNING: Field dimensions don't match: {analyzer1.field_data.shape} vs {analyzer2.field_data.shape}")
        print("Attempting to proceed, but results may be unreliable.")
    
    # Run analysis for both
    print("\nAnalyzing Field 1 by region...")
    results1 = analyzer1.analyze_by_region()
    
    print("\nAnalyzing Field 2 by region...")
    results2 = analyzer2.analyze_by_region()
    
    # Check if we're doing full analysis or region-specific
    is_full_analysis = not regions
    
    # Filter for regions of interest or use all regions
    if regions:
        print(f"\nPerforming targeted analysis on {len(regions)} specified regions")
        # Check if regions are IDs or names
        if all(isinstance(r, int) or (isinstance(r, str) and r.isdigit()) for r in regions):
            # Convert string digits to integers
            region_ids = [int(r) if isinstance(r, str) else r for r in regions]
            results1_filtered = results1[results1['RegionID'].isin(region_ids)]
            results2_filtered = results2[results2['RegionID'].isin(region_ids)]
            
            # Check if all specified regions were found
            found_ids = set(results1_filtered['RegionID'].tolist())
            missing_ids = set(region_ids) - found_ids
            if missing_ids:
                print(f"WARNING: The following region IDs were not found in the atlas: {missing_ids}")
        else:
            # Assume region names
            results1_filtered = results1[results1['RegionName'].isin(regions)]
            results2_filtered = results2[results2['RegionName'].isin(regions)]
            
            # Check if all specified regions were found
            found_names = set(results1_filtered['RegionName'].tolist())
            missing_names = set(regions) - found_names
            if missing_names:
                print(f"WARNING: The following region names were not found in the atlas: {missing_names}")
    else:
        print("\nPerforming full analysis on all atlas regions")
        # Use all regions
        results1_filtered = results1
        results2_filtered = results2
    
    print(f"Analyzing {len(results1_filtered)} regions...")
    
    # Create comparison dataframe
    comparison = []
    
    # Match regions from both results
    for _, row1 in results1_filtered.iterrows():
        region_id = row1['RegionID']
        region_name = row1['RegionName']
        
        # Find corresponding row in results2
        row2 = results2_filtered[results2_filtered['RegionID'] == region_id]
        
        if not row2.empty:
            row2 = row2.iloc[0]
            
            # Calculate differences
            mean_diff = row2['MeanValue'] - row1['MeanValue']
            mean_diff_pct = (mean_diff / row1['MeanValue']) * 100 if row1['MeanValue'] != 0 else float('inf')
            max_diff = row2['MaxValue'] - row1['MaxValue']
            max_diff_pct = (max_diff / row1['MaxValue']) * 100 if row1['MaxValue'] != 0 else float('inf')
            
            comparison.append({
                'RegionID': region_id,
                'RegionName': region_name,
                'Field1_Mean': row1['MeanValue'],
                'Field2_Mean': row2['MeanValue'],
                'Mean_Diff': mean_diff,
                'Mean_Diff_Pct': mean_diff_pct,
                'Field1_Max': row1['MaxValue'],
                'Field2_Max': row2['MaxValue'],
                'Max_Diff': max_diff,
                'Max_Diff_Pct': max_diff_pct,
                'Volume_mm3': row1['Volume_mm3'],
                'Color': row1.get('Color', (0.5, 0.5, 0.5))
            })
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison)
    
    # Sort by absolute mean difference
    comparison_df['Abs_Mean_Diff'] = comparison_df['Mean_Diff'].abs()
    comparison_df = comparison_df.sort_values('Abs_Mean_Diff', ascending=False)
    
    # Save comparison to CSV
    csv_path = os.path.join(output_dir, 'field_comparison.csv')
    # Remove color column for CSV
    comparison_df.drop(columns=['Color', 'Abs_Mean_Diff']).to_csv(csv_path, index=False)
    print(f"\nComparison results saved to {csv_path}")
    
    # Also create a difference NIfTI file
    if analyzer1.field_data.shape == analyzer2.field_data.shape:
        diff_data = analyzer2.field_data - analyzer1.field_data
        diff_img = nib.Nifti1Image(diff_data, analyzer1.field_img.affine)
        diff_path = os.path.join(output_dir, 'field_difference.nii.gz')
        nib.save(diff_img, diff_path)
        print(f"Field difference volume saved to {diff_path}")
    
    # Generate visualizations if requested
    if visualize:
        create_comparison_visualizations(comparison_df, analyzer1, analyzer2, output_dir, is_full_analysis, top_n)
        # Generate HTML report
        generate_comparison_report(
            comparison_df, 
            output_dir, 
            os.path.basename(analyzer1.field_nifti), 
            os.path.basename(analyzer2.field_nifti),
            is_full_analysis
        )
    
    return comparison_df

def create_comparison_visualizations(comparison_df, analyzer1, analyzer2, output_dir, is_full_analysis=False, top_n=20):
    """Create visualizations for the field comparison.
    
    Parameters
    ----------
    comparison_df : pandas.DataFrame
        DataFrame with comparison results
    analyzer1 : TIFieldAnalyzer
        Analyzer for first field
    analyzer2 : TIFieldAnalyzer
        Analyzer for second field
    output_dir : str
        Directory to save visualizations
    is_full_analysis : bool
        Whether this is a full analysis of all regions
    top_n : int
        Number of top regions to show in visualizations
    """
    print("\nGenerating comparison visualizations...")
    
    # For full analysis, show limited number of regions in bar chart
    if is_full_analysis and len(comparison_df) > top_n:
        # Get top regions by absolute difference
        display_df = comparison_df.nlargest(top_n, 'Abs_Mean_Diff')
    else:
        display_df = comparison_df
    
    # Bar chart of mean differences
    plt.figure(figsize=(12, max(8, len(display_df) * 0.4)))
    bars = plt.barh(display_df['RegionName'][::-1], 
                    display_df['Mean_Diff'][::-1])
    
    # Color bars based on sign
    for i, bar in enumerate(bars):
        if bar.get_width() < 0:
            bar.set_color('blue')
        else:
            bar.set_color('red')
    
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Mean Field Difference (Field2 - Field1)')
    plt.title(f'Mean Field Differences by Region')
    plt.tight_layout()
    
    # Save the figure
    figure_path = os.path.join(output_dir, 'mean_diff_bar.png')
    plt.savefig(figure_path, dpi=300)
    plt.close()
    print(f"Mean difference bar chart saved to {figure_path}")
    
    # Scatter plot of means
    plt.figure(figsize=(10, 8))
    # Use simple colors
    plt.scatter(comparison_df['Field1_Mean'], comparison_df['Field2_Mean'], 
                alpha=0.7, s=comparison_df['Volume_mm3']/100, 
                c='blue')
    
    # Add region labels for top regions
    for i, row in comparison_df.nlargest(min(15, len(comparison_df)), 'Abs_Mean_Diff').iterrows():
        plt.annotate(row['RegionName'], 
                     (row['Field1_Mean'], row['Field2_Mean']),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8)
    
    # Add diagonal line
    max_val = max(comparison_df['Field1_Mean'].max(), comparison_df['Field2_Mean'].max())
    min_val = min(comparison_df['Field1_Mean'].min(), comparison_df['Field2_Mean'].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=0.8)
    
    plt.xlabel('Field 1 Mean Value')
    plt.ylabel('Field 2 Mean Value')
    plt.title('Comparison of Mean Field Values')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    figure_path = os.path.join(output_dir, 'means_scatter.png')
    plt.savefig(figure_path, dpi=300)
    plt.close()
    print(f"Means scatter plot saved to {figure_path}")
    
    # For full analysis, add histogram of differences
    if is_full_analysis:
        plt.figure(figsize=(10, 6))
        plt.hist(comparison_df['Mean_Diff'], bins=30, color='skyblue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
        plt.xlabel('Mean Difference (Field2 - Field1)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Field Differences Across Regions')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the figure
        figure_path = os.path.join(output_dir, 'difference_histogram.png')
        plt.savefig(figure_path, dpi=300)
        plt.close()
        print(f"Difference histogram saved to {figure_path}")
    
    # Create a field difference visualization using slices
    try:
        # Load both field data
        field1_data = analyzer1.field_data
        field2_data = analyzer2.field_data
        
        # Calculate difference volume
        if field1_data.shape == field2_data.shape:
            diff_data = field2_data - field1_data
            
            # Create figure with 3 columns (sagittal, coronal, axial)
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            
            # Get middle slices for each dimension
            x_mid = field1_data.shape[0] // 2
            y_mid = field1_data.shape[1] // 2
            z_mid = field1_data.shape[2] // 2
            
            # Background image for visualization
            if analyzer1.t1_data is not None:
                # If T1 reference is available, use it as background
                background_data = analyzer1.t1_data
                cmap_bg = 'gray'
            else:
                # If no T1, use zeros
                background_data = np.zeros_like(field1_data)
                cmap_bg = 'gray'
            
            # Normalize for visualization
            vmax = max(np.max(field1_data), np.max(field2_data))
            vmin = min(np.min(field1_data), np.min(field2_data))
            diff_vmax = np.max(np.abs(diff_data))
            
            # First row: Field 1
            # Sagittal view (x_mid)
            axes[0, 0].imshow(np.rot90(background_data[x_mid, :, :]), cmap=cmap_bg, alpha=0.5)
            axes[0, 0].imshow(np.rot90(field1_data[x_mid, :, :]), cmap='hot', vmin=vmin, vmax=vmax, alpha=0.7)
            axes[0, 0].set_title(f'Field 1 - Sagittal (x={x_mid})')
            axes[0, 0].axis('off')
            
            # Coronal view (y_mid)
            axes[0, 1].imshow(np.rot90(background_data[:, y_mid, :]), cmap=cmap_bg, alpha=0.5)
            axes[0, 1].imshow(np.rot90(field1_data[:, y_mid, :]), cmap='hot', vmin=vmin, vmax=vmax, alpha=0.7)
            axes[0, 1].set_title(f'Field 1 - Coronal (y={y_mid})')
            axes[0, 1].axis('off')
            
            # Axial view (z_mid)
            axes[0, 2].imshow(np.rot90(background_data[:, :, z_mid]), cmap=cmap_bg, alpha=0.5)
            im1 = axes[0, 2].imshow(np.rot90(field1_data[:, :, z_mid]), cmap='hot', vmin=vmin, vmax=vmax, alpha=0.7)
            axes[0, 2].set_title(f'Field 1 - Axial (z={z_mid})')
            axes[0, 2].axis('off')
            
            # Second row: Field 2
            # Sagittal view (x_mid)
            axes[1, 0].imshow(np.rot90(background_data[x_mid, :, :]), cmap=cmap_bg, alpha=0.5)
            axes[1, 0].imshow(np.rot90(field2_data[x_mid, :, :]), cmap='hot', vmin=vmin, vmax=vmax, alpha=0.7)
            axes[1, 0].set_title(f'Field 2 - Sagittal (x={x_mid})')
            axes[1, 0].axis('off')
            
            # Coronal view (y_mid)
            axes[1, 1].imshow(np.rot90(background_data[:, y_mid, :]), cmap=cmap_bg, alpha=0.5)
            axes[1, 1].imshow(np.rot90(field2_data[:, y_mid, :]), cmap='hot', vmin=vmin, vmax=vmax, alpha=0.7)
            axes[1, 1].set_title(f'Field 2 - Coronal (y={y_mid})')
            axes[1, 1].axis('off')
            
            # Axial view (z_mid)
            axes[1, 2].imshow(np.rot90(background_data[:, :, z_mid]), cmap=cmap_bg, alpha=0.5)
            axes[1, 2].imshow(np.rot90(field2_data[:, :, z_mid]), cmap='hot', vmin=vmin, vmax=vmax, alpha=0.7)
            axes[1, 2].set_title(f'Field 2 - Axial (z={z_mid})')
            axes[1, 2].axis('off')
            
            # Third row: Difference (Field2 - Field1)
            # Sagittal view (x_mid)
            axes[2, 0].imshow(np.rot90(background_data[x_mid, :, :]), cmap=cmap_bg, alpha=0.5)
            axes[2, 0].imshow(np.rot90(diff_data[x_mid, :, :]), cmap='coolwarm', vmin=-diff_vmax, vmax=diff_vmax, alpha=0.7)
            axes[2, 0].set_title(f'Difference - Sagittal (x={x_mid})')
            axes[2, 0].axis('off')
            
            # Coronal view (y_mid)
            axes[2, 1].imshow(np.rot90(background_data[:, y_mid, :]), cmap=cmap_bg, alpha=0.5)
            axes[2, 1].imshow(np.rot90(diff_data[:, y_mid, :]), cmap='coolwarm', vmin=-diff_vmax, vmax=diff_vmax, alpha=0.7)
            axes[2, 1].set_title(f'Difference - Coronal (y={y_mid})')
            axes[2, 1].axis('off')
            
            # Axial view (z_mid)
            axes[2, 2].imshow(np.rot90(background_data[:, :, z_mid]), cmap=cmap_bg, alpha=0.5)
            im3 = axes[2, 2].imshow(np.rot90(diff_data[:, :, z_mid]), cmap='coolwarm', vmin=-diff_vmax, vmax=diff_vmax, alpha=0.7)
            axes[2, 2].set_title(f'Difference - Axial (z={z_mid})')
            axes[2, 2].axis('off')
            
            # Add colorbars
            cbar_ax1 = fig.add_axes([0.92, 0.7, 0.02, 0.2])
            fig.colorbar(im1, cax=cbar_ax1, label='Field Intensity')
            
            cbar_ax3 = fig.add_axes([0.92, 0.1, 0.02, 0.2])
            fig.colorbar(im3, cax=cbar_ax3, label='Field Difference')
            
            plt.tight_layout(rect=[0, 0, 0.9, 1])
            
            # Save the figure
            figure_path = os.path.join(output_dir, 'field_difference_slices.png')
            plt.savefig(figure_path, dpi=300)
            plt.close()
            print(f"Field difference slices saved to {figure_path}")
        else:
            print("WARNING: Field dimensions don't match, skipping slice visualization")
    
    except Exception as e:
        print(f"WARNING: Could not create field difference visualization: {str(e)}")
        
    # Create visualization of analyzed regions overlaid on field data
    try:
        print("\nCreating region overlay visualization...")
        
        # Get the atlas data and the regions we're comparing
        atlas_data = analyzer1.atlas_data
        region_ids = comparison_df['RegionID'].unique()
        
        # Create a region mask highlighting the analyzed regions
        region_mask = np.zeros_like(atlas_data)
        region_colors = np.zeros((*atlas_data.shape, 4), dtype=np.float32)  # RGBA (with alpha)
        
        # For each region, assign a unique bright color with high contrast
        # Use a bright, high-contrast colormap
        bright_colors = [
            (1.0, 0.0, 0.0, 0.9),    # Bright red
            (0.0, 1.0, 0.0, 0.9),    # Bright green
            (0.0, 0.0, 1.0, 0.9),    # Bright blue
            (1.0, 1.0, 0.0, 0.9),    # Bright yellow
            (1.0, 0.0, 1.0, 0.9),    # Bright magenta
            (0.0, 1.0, 1.0, 0.9),    # Bright cyan
            (1.0, 0.5, 0.0, 0.9),    # Bright orange
            (0.5, 0.0, 1.0, 0.9),    # Bright purple
            (0.0, 0.8, 0.4, 0.9),    # Bright teal
            (1.0, 0.0, 0.5, 0.9),    # Bright pink
        ]
        
        # For each region, assign a unique color
        for i, region_id in enumerate(region_ids):
            region_row = comparison_df[comparison_df['RegionID'] == region_id].iloc[0]
            region_mask[atlas_data == region_id] = 1
            
            # Get bright color from our list, cycling if needed
            color_idx = i % len(bright_colors)
            color = bright_colors[color_idx]
            
            # Apply color to the region
            mask = atlas_data == region_id
            for c in range(4):  # RGBA
                region_colors[mask, c] = color[c]
        
        # Load field data for overlay
        field1_data = analyzer1.field_data
        field2_data = analyzer2.field_data
        
        # Calculate difference volume
        if field1_data.shape == field2_data.shape:
            diff_data = field2_data - field1_data
        else:
            # If shapes don't match, use a dummy array
            diff_data = np.zeros_like(field1_data)
        
        # Get slices with maximum region presence
        x_counts = np.sum(region_mask, axis=(1, 2))
        y_counts = np.sum(region_mask, axis=(0, 2))
        z_counts = np.sum(region_mask, axis=(0, 1))
        
        # Find slice with maximum region presence
        x_mid = np.argmax(x_counts) if np.max(x_counts) > 0 else field1_data.shape[0] // 2
        y_mid = np.argmax(y_counts) if np.max(y_counts) > 0 else field1_data.shape[1] // 2
        z_mid = np.argmax(z_counts) if np.max(z_counts) > 0 else field1_data.shape[2] // 2
        
        # Background image for visualization
        if analyzer1.t1_data is not None:
            # If T1 reference is available, use it as background
            background_data = analyzer1.t1_data
            cmap_bg = 'gray'
        else:
            # If no T1, use zeros
            background_data = np.zeros_like(field1_data)
            cmap_bg = 'gray'
        
        # Create field overlay with regions
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        
        # Normalize for visualization
        vmax = max(np.max(field1_data), np.max(field2_data))
        vmin = min(np.min(field1_data), np.min(field2_data))
        diff_vmax = np.max(np.abs(diff_data))
        
        # Row 1: Field 1 with regions
        # Sagittal view
        axes[0, 0].imshow(np.rot90(background_data[x_mid, :, :]), cmap=cmap_bg, alpha=0.5)
        axes[0, 0].imshow(np.rot90(field1_data[x_mid, :, :]), cmap='hot', vmin=vmin, vmax=vmax, alpha=0.7)
        axes[0, 0].imshow(np.rot90(region_colors[x_mid, :, :]), alpha=np.rot90(region_mask[x_mid, :, :])*0.8)
        axes[0, 0].set_title(f'Field 1 with Regions - Sagittal (x={x_mid})')
        axes[0, 0].axis('off')
        
        # Coronal view
        axes[0, 1].imshow(np.rot90(background_data[:, y_mid, :]), cmap=cmap_bg, alpha=0.5)
        axes[0, 1].imshow(np.rot90(field1_data[:, y_mid, :]), cmap='hot', vmin=vmin, vmax=vmax, alpha=0.7)
        axes[0, 1].imshow(np.rot90(region_colors[:, y_mid, :]), alpha=np.rot90(region_mask[:, y_mid, :])*0.8)
        axes[0, 1].set_title(f'Field 1 with Regions - Coronal (y={y_mid})')
        axes[0, 1].axis('off')
        
        # Axial view
        axes[0, 2].imshow(np.rot90(background_data[:, :, z_mid]), cmap=cmap_bg, alpha=0.5)
        axes[0, 2].imshow(np.rot90(field1_data[:, :, z_mid]), cmap='hot', vmin=vmin, vmax=vmax, alpha=0.7)
        axes[0, 2].imshow(np.rot90(region_colors[:, :, z_mid]), alpha=np.rot90(region_mask[:, :, z_mid])*0.8)
        axes[0, 2].set_title(f'Field 1 with Regions - Axial (z={z_mid})')
        axes[0, 2].axis('off')
        
        # Row 2: Field 2 with regions
        # Sagittal view
        axes[1, 0].imshow(np.rot90(background_data[x_mid, :, :]), cmap=cmap_bg, alpha=0.5)
        axes[1, 0].imshow(np.rot90(field2_data[x_mid, :, :]), cmap='hot', vmin=vmin, vmax=vmax, alpha=0.7)
        axes[1, 0].imshow(np.rot90(region_colors[x_mid, :, :]), alpha=np.rot90(region_mask[x_mid, :, :])*0.8)
        axes[1, 0].set_title(f'Field 2 with Regions - Sagittal (x={x_mid})')
        axes[1, 0].axis('off')
        
        # Coronal view
        axes[1, 1].imshow(np.rot90(background_data[:, y_mid, :]), cmap=cmap_bg, alpha=0.5)
        axes[1, 1].imshow(np.rot90(field2_data[:, y_mid, :]), cmap='hot', vmin=vmin, vmax=vmax, alpha=0.7)
        axes[1, 1].imshow(np.rot90(region_colors[:, y_mid, :]), alpha=np.rot90(region_mask[:, y_mid, :])*0.8)
        axes[1, 1].set_title(f'Field 2 with Regions - Coronal (y={y_mid})')
        axes[1, 1].axis('off')
        
        # Axial view
        axes[1, 2].imshow(np.rot90(background_data[:, :, z_mid]), cmap=cmap_bg, alpha=0.5)
        axes[1, 2].imshow(np.rot90(field2_data[:, :, z_mid]), cmap='hot', vmin=vmin, vmax=vmax, alpha=0.7)
        axes[1, 2].imshow(np.rot90(region_colors[:, :, z_mid]), alpha=np.rot90(region_mask[:, :, z_mid])*0.8)
        axes[1, 2].set_title(f'Field 2 with Regions - Axial (z={z_mid})')
        axes[1, 2].axis('off')
        
        # Row 3: Difference with regions
        # Sagittal view
        axes[2, 0].imshow(np.rot90(background_data[x_mid, :, :]), cmap=cmap_bg, alpha=0.5)
        axes[2, 0].imshow(np.rot90(diff_data[x_mid, :, :]), cmap='coolwarm', vmin=-diff_vmax, vmax=diff_vmax, alpha=0.7)
        axes[2, 0].imshow(np.rot90(region_colors[x_mid, :, :]), alpha=np.rot90(region_mask[x_mid, :, :])*0.8)
        axes[2, 0].set_title(f'Difference with Regions - Sagittal (x={x_mid})')
        axes[2, 0].axis('off')
        
        # Coronal view
        axes[2, 1].imshow(np.rot90(background_data[:, y_mid, :]), cmap=cmap_bg, alpha=0.5)
        axes[2, 1].imshow(np.rot90(diff_data[:, y_mid, :]), cmap='coolwarm', vmin=-diff_vmax, vmax=diff_vmax, alpha=0.7)
        axes[2, 1].imshow(np.rot90(region_colors[:, y_mid, :]), alpha=np.rot90(region_mask[:, y_mid, :])*0.8)
        axes[2, 1].set_title(f'Difference with Regions - Coronal (y={y_mid})')
        axes[2, 1].axis('off')
        
        # Axial view
        axes[2, 2].imshow(np.rot90(background_data[:, :, z_mid]), cmap=cmap_bg, alpha=0.5)
        im3 = axes[2, 2].imshow(np.rot90(diff_data[:, :, z_mid]), cmap='coolwarm', vmin=-diff_vmax, vmax=diff_vmax, alpha=0.7)
        axes[2, 2].imshow(np.rot90(region_colors[:, :, z_mid]), alpha=np.rot90(region_mask[:, :, z_mid])*0.8)
        axes[2, 2].set_title(f'Difference with Regions - Axial (z={z_mid})')
        axes[2, 2].axis('off')
        
        # Add colorbar for difference
        cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        fig.colorbar(im3, cax=cbar_ax, label='Field Value/Difference')
        
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        
        # Save the figure
        figure_path = os.path.join(output_dir, 'field_regions_overlay.png')
        fig.savefig(figure_path, dpi=300)
        plt.close(fig)
        print(f"Field-region overlay visualization saved to {figure_path}")
        
        # Create a legend for the regions
        if len(region_ids) > 0:
            legend_fig, legend_ax = plt.subplots(figsize=(10, max(5, len(region_ids)*0.25)))
            legend_ax.axis('off')
            
            # Create legend patches
            patches = []
            labels = []
            
            for i, region_id in enumerate(region_ids):
                region_row = comparison_df[comparison_df['RegionID'] == region_id].iloc[0]
                region_name = region_row['RegionName']
                
                # Use the same bright colors as above
                color_idx = i % len(bright_colors)
                color = bright_colors[color_idx]
                
                # Create a patch and label for the legend
                patch = plt.Rectangle((0, 0), 1, 1, fc=color[:3], alpha=color[3])
                patches.append(patch)
                labels.append(f"{region_name} (ID: {region_id})")
            
            # Add the legend
            legend_ax.legend(patches, labels, loc='center', ncol=max(1, len(region_ids)//20))
            
            # Save the legend figure
            legend_path = os.path.join(output_dir, 'region_legend.png')
            legend_fig.savefig(legend_path, dpi=300, bbox_inches='tight')
            plt.close(legend_fig)
            print(f"Region legend saved to {legend_path}")
        
    except Exception as e:
        print(f"WARNING: Could not create field-region overlay visualization: {str(e)}")
        import traceback
        traceback.print_exc()

def generate_comparison_report(comparison_df, output_dir, field1_name, field2_name, is_full_analysis=False):
    """Generate an HTML report for the field comparison.
    
    Parameters
    ----------
    comparison_df : pandas.DataFrame
        DataFrame with comparison results
    output_dir : str
        Directory to save report
    field1_name : str
        Name of first field
    field2_name : str
        Name of second field
    is_full_analysis : bool
        Whether this is a full analysis of all regions
    """
    print("\nGenerating comparison report...")
    
    # Create HTML head and style section
    html_head = """<!DOCTYPE html>
<html>
<head>
    <title>TI Field Comparison Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2, h3 { color: #2c3e50; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        img { max-width: 100%; height: auto; margin: 20px 0; }
        .container { max-width: 1200px; margin: 0 auto; }
        .positive { color: red; }
        .negative { color: blue; }
        .summary-box { border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; background-color: #f9f9f9; }
    </style>
</head>
<body>
    <div class="container">
"""
    
    # Format the highest and lowest difference regions
    try:
        max_region = comparison_df.loc[comparison_df['Mean_Diff'].idxmax(), 'RegionName']
        max_diff = comparison_df['Mean_Diff'].max()
        max_region_text = f"{max_region} ({max_diff:.4f})"
    except:
        max_region_text = "N/A"
        
    try:
        min_region = comparison_df.loc[comparison_df['Mean_Diff'].idxmin(), 'RegionName']
        min_diff = comparison_df['Mean_Diff'].min()
        min_region_text = f"{min_region} ({min_diff:.4f})"
    except:
        min_region_text = "N/A"
    
    # Create the report content section
    html_content = f"""
        <h1>TI Field Comparison Report</h1>
        <p><strong>Field 1:</strong> {field1_name}</p>
        <p><strong>Field 2:</strong> {field2_name}</p>
        <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        
        <h2>Summary Statistics</h2>
    """
    
    # Add analysis type note
    if is_full_analysis:
        html_content += f"""
        <div class="summary-box">
            <h3>Full Analysis Mode</h3>
            <p>This report compares all {len(comparison_df)} atlas regions between the two field scans.</p>
            <p>Key findings:</p>
            <ul>
                <li>{len(comparison_df[comparison_df['Mean_Diff'] > 0])} regions show increased field in Field 2</li>
                <li>{len(comparison_df[comparison_df['Mean_Diff'] < 0])} regions show decreased field in Field 2</li>
                <li>Average absolute difference: {comparison_df['Mean_Diff'].abs().mean():.4f}</li>
                <li>Maximum positive difference: {comparison_df['Mean_Diff'].max():.4f}</li>
                <li>Maximum negative difference: {comparison_df['Mean_Diff'].min():.4f}</li>
            </ul>
        </div>
        """
    
    html_content += f"""
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Number of Analyzed Regions</td>
                <td>{len(comparison_df)}</td>
            </tr>
            <tr>
                <td>Region with Largest Positive Difference</td>
                <td>{max_region_text}</td>
            </tr>
            <tr>
                <td>Region with Largest Negative Difference</td>
                <td>{min_region_text}</td>
            </tr>
            <tr>
                <td>Average Absolute Difference</td>
                <td>{comparison_df['Mean_Diff'].abs().mean():.4f}</td>
            </tr>
        </table>
    """
    
    # For full analysis, limit to top regions in table
    if is_full_analysis and len(comparison_df) > 40:
        # Get top 20 positive and 20 negative difference regions
        top_pos = comparison_df.nlargest(20, 'Mean_Diff')
        top_neg = comparison_df.nsmallest(20, 'Mean_Diff')
        table_df = pd.concat([top_pos, top_neg])
        
        html_content += f"""
        <h2>Top Regions by Difference (40 of {len(comparison_df)} total regions)</h2>
        <p>Showing the 20 regions with largest positive and 20 with largest negative differences.</p>
        """
    else:
        table_df = comparison_df
        html_content += f"""
        <h2>Region Comparison</h2>
        """
    
    html_content += f"""
        <table>
            <tr>
                <th>Region Name</th>
                <th>Field 1 Mean</th>
                <th>Field 2 Mean</th>
                <th>Difference</th>
                <th>Diff %</th>
            </tr>
    """
    
    # Generate rows for each region
    region_rows = ""
    for _, row in table_df.iterrows():
        # Format difference values with color
        diff_class = 'positive' if row['Mean_Diff'] > 0 else 'negative'
        diff_formatted = f"<span class='{diff_class}'>{row['Mean_Diff']:.4f}</span>"
        diff_pct_formatted = f"<span class='{diff_class}'>{row['Mean_Diff_Pct']:.2f}%</span>" if row['Mean_Diff_Pct'] != float('inf') else f"<span class='{diff_class}'>∞%</span>"
        
        region_rows += f"""
            <tr>
                <td>{row['RegionName']}</td>
                <td>{row['Field1_Mean']:.4f}</td>
                <td>{row['Field2_Mean']:.4f}</td>
                <td>{diff_formatted}</td>
                <td>{diff_pct_formatted}</td>
            </tr>
        """
    
    # Add visualizations section
    visualizations = """
        </table>
        
        <h2>Visualizations</h2>
        <div>
            <h3>Mean Field Differences by Region</h3>
            <img src="mean_diff_bar.png" alt="Mean Difference Bar Chart">
        </div>
        <div>
            <h3>Comparison of Mean Field Values</h3>
            <img src="means_scatter.png" alt="Means Scatter Plot">
        </div>
    """
    
    # Add field difference visualization if it exists
    field_diff_section = ""
    if os.path.exists(os.path.join(output_dir, 'field_difference_slices.png')):
        field_diff_section = """
        <div>
            <h3>Field Difference Visualization</h3>
            <img src="field_difference_slices.png" alt="Field Difference Slices">
            <p>Top row: Field 1, Middle row: Field 2, Bottom row: Difference (Field 2 - Field 1)</p>
            <p>Red indicates positive difference (Field 2 > Field 1), blue indicates negative difference (Field 1 > Field 2).</p>
        </div>
        """
    
    # Add region visualization section
    region_vis_section = ""
    if os.path.exists(os.path.join(output_dir, 'field_regions_overlay.png')):
        region_vis_section = """
        <div>
            <h3>Brain Regions Overlaid on TI Fields</h3>
            <img src="field_regions_overlay.png" alt="Field and Region Overlay">
            <p>Shows the analyzed brain regions highlighted with distinct colors overlaid on the field data.</p>
        """
        
        if os.path.exists(os.path.join(output_dir, 'region_legend.png')):
            region_vis_section += """
            <h4>Region Legend</h4>
            <img src="region_legend.png" alt="Region Legend">
            """
                
        region_vis_section += """
        </div>
        """
    
    # Add full analysis visualizations if they exist
    full_analysis_section = ""
    if is_full_analysis:
        if os.path.exists(os.path.join(output_dir, 'difference_histogram.png')):
            full_analysis_section += """
            <div>
                <h3>Distribution of Differences</h3>
                <img src="difference_histogram.png" alt="Difference Histogram">
            </div>
            """
    
    # Close HTML tags
    html_closing = """
    </div>
</body>
</html>
    """
    
    # Combine all parts
    full_html = html_head + html_content + region_rows + visualizations + field_diff_section + region_vis_section + full_analysis_section + html_closing
    
    # Write HTML file
    html_path = os.path.join(output_dir, 'comparison_report.html')
    with open(html_path, 'w') as f:
        f.write(full_html)
        
    print(f"HTML comparison report saved to {html_path}")
    print("\nGenerating comparison report...")
    
    # Create HTML head and style section
    html_head = """<!DOCTYPE html>
<html>
<head>
    <title>TI Field Comparison Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2, h3 { color: #2c3e50; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        img { max-width: 100%; height: auto; margin: 20px 0; }
        .container { max-width: 1200px; margin: 0 auto; }
        .positive { color: red; }
        .negative { color: blue; }
        .summary-box { border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; background-color: #f9f9f9; }
    </style>
</head>
<body>
    <div class="container">
"""
    
    # Format the highest and lowest difference regions
    try:
        max_region = comparison_df.loc[comparison_df['Mean_Diff'].idxmax(), 'RegionName']
        max_diff = comparison_df['Mean_Diff'].max()
        max_region_text = f"{max_region} ({max_diff:.4f})"
    except:
        max_region_text = "N/A"
        
    try:
        min_region = comparison_df.loc[comparison_df['Mean_Diff'].idxmin(), 'RegionName']
        min_diff = comparison_df['Mean_Diff'].min()
        min_region_text = f"{min_region} ({min_diff:.4f})"
    except:
        min_region_text = "N/A"
    
    # Create the report content section
    html_content = f"""
        <h1>TI Field Comparison Report</h1>
        <p><strong>Field 1:</strong> {field1_name}</p>
        <p><strong>Field 2:</strong> {field2_name}</p>
        <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        
        <h2>Summary Statistics</h2>
    """
    
    # Add analysis type note
    if is_full_analysis:
        html_content += f"""
        <div class="summary-box">
            <h3>Full Analysis Mode</h3>
            <p>This report compares all {len(comparison_df)} atlas regions between the two field scans.</p>
            <p>Key findings:</p>
            <ul>
                <li>{len(comparison_df[comparison_df['Mean_Diff'] > 0])} regions show increased field in Field 2</li>
                <li>{len(comparison_df[comparison_df['Mean_Diff'] < 0])} regions show decreased field in Field 2</li>
                <li>Average absolute difference: {comparison_df['Mean_Diff'].abs().mean():.4f}</li>
                <li>Maximum positive difference: {comparison_df['Mean_Diff'].max():.4f}</li>
                <li>Maximum negative difference: {comparison_df['Mean_Diff'].min():.4f}</li>
            </ul>
        </div>
        """
    
    html_content += f"""
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Number of Analyzed Regions</td>
                <td>{len(comparison_df)}</td>
            </tr>
            <tr>
                <td>Region with Largest Positive Difference</td>
                <td>{max_region_text}</td>
            </tr>
            <tr>
                <td>Region with Largest Negative Difference</td>
                <td>{min_region_text}</td>
            </tr>
            <tr>
                <td>Average Absolute Difference</td>
                <td>{comparison_df['Mean_Diff'].abs().mean():.4f}</td>
            </tr>
        </table>
    """
    
    # For full analysis, limit to top regions in table
    if is_full_analysis and len(comparison_df) > 40:
        # Get top 20 positive and 20 negative difference regions
        top_pos = comparison_df.nlargest(20, 'Mean_Diff')
        top_neg = comparison_df.nsmallest(20, 'Mean_Diff')
        table_df = pd.concat([top_pos, top_neg])
        
        html_content += f"""
        <h2>Top Regions by Difference (40 of {len(comparison_df)} total regions)</h2>
        <p>Showing the 20 regions with largest positive and 20 with largest negative differences.</p>
        """
    else:
        table_df = comparison_df
        html_content += f"""
        <h2>Region Comparison</h2>
        """
    
    html_content += f"""
        <table>
            <tr>
                <th>Region Name</th>
                <th>Field 1 Mean</th>
                <th>Field 2 Mean</th>
                <th>Difference</th>
                <th>Diff %</th>
            </tr>
    """
    
    # Generate rows for each region
    region_rows = ""
    for _, row in table_df.iterrows():
        # Format difference values with color
        diff_class = 'positive' if row['Mean_Diff'] > 0 else 'negative'
        diff_formatted = f"<span class='{diff_class}'>{row['Mean_Diff']:.4f}</span>"
        diff_pct_formatted = f"<span class='{diff_class}'>{row['Mean_Diff_Pct']:.2f}%</span>" if row['Mean_Diff_Pct'] != float('inf') else "<span class='{diff_class}'>∞%</span>"
        
        region_rows += f"""
            <tr>
                <td>{row['RegionName']}</td>
                <td>{row['Field1_Mean']:.4f}</td>
                <td>{row['Field2_Mean']:.4f}</td>
                <td>{diff_formatted}</td>
                <td>{diff_pct_formatted}</td>
            </tr>
        """
    
    # Add visualizations section
    visualizations = """
        </table>
        
        <h2>Visualizations</h2>
        <div>
            <h3>Mean Field Differences by Region</h3>
            <img src="mean_diff_bar.png" alt="Mean Difference Bar Chart">
        </div>
        <div>
            <h3>Comparison of Mean Field Values</h3>
            <img src="means_scatter.png" alt="Means Scatter Plot">
        </div>
    """
    
    # Add field difference visualization if it exists
    field_diff_section = ""
    if os.path.exists(os.path.join(output_dir, 'field_difference_slices.png')):
        field_diff_section = """
        <div>
            <h3>Field Difference Visualization</h3>
            <img src="field_difference_slices.png" alt="Field Difference Slices">
            <p>Top row: Field 1, Middle row: Field 2, Bottom row: Difference (Field 2 - Field 1)</p>
            <p>Red indicates positive difference (Field 2 > Field 1), blue indicates negative difference (Field 1 > Field 2).</p>
        </div>
        """
    
    # Add region visualization section
    region_vis_section = ""
    if os.path.exists(os.path.join(output_dir, 'field_regions_overlay.png')):
        region_vis_section = """
        <div>
            <h3>Brain Regions Overlaid on TI Fields</h3>
            <img src="field_regions_overlay.png" alt="Field and Region Overlay">
            <p>Shows the analyzed brain regions highlighted with distinct colors overlaid on the field data.</p>
        """
        
        if os.path.exists(os.path.join(output_dir, 'region_legend.png')):
            region_vis_section += """
            <h4>Region Legend</h4>
            <img src="region_legend.png" alt="Region Legend">
            """
                
        region_vis_section += """
        </div>
        """
    
    # Add full analysis visualizations if they exist
    full_analysis_section = ""
    if is_full_analysis:
        if os.path.exists(os.path.join(output_dir, 'difference_histogram.png')):
            full_analysis_section += """
            <div>
                <h3>Distribution of Differences</h3>
                <img src="difference_histogram.png" alt="Difference Histogram">
            </div>
            """
            full_analysis_section += """
            <div>
                <h3>Distribution of Differences</h3>
                <img src="difference_histogram.png" alt="Difference Histogram">
            </div>
            """
    
    # Close HTML tags
    html_closing = """
    </div>
</body>
</html>
    """
    
    # Combine all parts
    full_html = html_head + html_content + region_rows + visualizations + field_diff_section + region_vis_section + full_analysis_section + html_closing
    
    # Write HTML file
    html_path = os.path.join(output_dir, 'comparison_report.html')
    with open(html_path, 'w') as f:
        f.write(full_html)
        
    print(f"HTML comparison report saved to {html_path}")
    
    # Close HTML tags
    html_closing = """
    </div>
</body>
</html>
    """
    
    # Combine all parts
    full_html = html_head + html_content + region_rows + visualizations + field_diff_section + full_analysis_section + html_closing
    
    # Write HTML file
    html_path = os.path.join(output_dir, 'comparison_report.html')
    with open(html_path, 'w') as f:
        f.write(full_html)
        
    print(f"HTML comparison report saved to {html_path}")

def main():
    """Parse command-line arguments and run comparison."""
    parser = argparse.ArgumentParser(description='Compare two TI field scans and analyze specific cortical regions')
    parser.add_argument('--field1', required=True, help='First NIfTI file containing field values')
    parser.add_argument('--field2', required=True, help='Second NIfTI file containing field values')
    parser.add_argument('--atlas', required=True, help='NIfTI file containing the atlas parcellation')
    parser.add_argument('--labels', required=True, help='Text file with region labels (format: ID Name R G B A)')
    parser.add_argument('--regions', nargs='+', help='Region IDs or names to focus on (if omitted, all regions will be analyzed)')
    parser.add_argument('--output', default='ti_field_comparison', help='Output directory for results')
    parser.add_argument('--t1-mni', help='Optional T1 MNI reference image for visualization')
    parser.add_argument('--no-visualizations', action='store_true', help='Skip generating visualizations')
    parser.add_argument('--top-n', type=int, default=20, help='Number of top regions to show in visualizations for full analysis')
    
    args = parser.parse_args()
    
    # Check if input files exist
    for file_path in [args.field1, args.field2, args.atlas, args.labels]:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
    
    if args.t1_mni and not os.path.exists(args.t1_mni):
        print(f"Error: T1 MNI file not found: {args.t1_mni}")
        sys.exit(1)
    
    try:
        # Determine analysis mode
        if args.regions:
            print(f"Targeted analysis mode: comparing {len(args.regions)} specific regions")
        else:
            print(f"Full analysis mode: comparing all regions in the atlas")
        
        # Run the comparison
        compare_field_scans(
            field1=args.field1,
            field2=args.field2,
            atlas=args.atlas,
            labels=args.labels,
            regions=args.regions,
            output_dir=args.output,
            t1_mni=args.t1_mni,
            visualize=not args.no_visualizations,
            top_n=args.top_n
        )
        
        print(f"Comparison completed successfully!")
        
    except Exception as e:
        print(f"Error during comparison: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()