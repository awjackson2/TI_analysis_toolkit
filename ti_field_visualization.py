#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization module for TI field analysis with functions for creating plots and reports.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
from papaya_utils import add_papaya_viewer, add_papaya_comparison, add_papaya_to_multiple_fields

def create_bar_chart(analyzer, n_regions=20, output_dir=None):
    """Create bar chart of top regions by mean value.
    
    Parameters
    ----------
    analyzer : TIFieldAnalyzer
        Analyzer object with results_df attribute
    n_regions : int, optional
        Number of top regions to display
    output_dir : str, optional
        Output directory (defaults to analyzer's output_dir)
        
    Returns
    -------
    str
        Path to saved figure
    """
    if output_dir is None:
        output_dir = analyzer.output_dir
        
    top_regions = analyzer.results_df.nlargest(n_regions, 'MeanValue')
    
    plt.figure(figsize=(12, 10))
    
    # Create bars with the corresponding region colors
    bars = plt.barh(top_regions['RegionName'][::-1], 
                     top_regions['MeanValue'][::-1])
    
    # Set bar colors according to region colors
    for i, bar in enumerate(bars):
        bar.set_color(top_regions['Color'].iloc[-(i+1)])
    
    plt.xlabel('Mean Field Value')
    plt.title(f'Top {n_regions} Regions by Mean Field Value')
    plt.tight_layout()
    
    # Save the figure
    figure_path = os.path.join(output_dir, 'top_regions_bar.png')
    plt.savefig(figure_path, dpi=300)
    plt.close()
    print(f"Bar chart saved to {figure_path}")
    return figure_path

def create_slices_visualization(analyzer, n_regions=8, output_dir=None):
    """Create visualization of the top regions on anatomical slices.
    
    Parameters
    ----------
    analyzer : TIFieldAnalyzer
        Analyzer object
    n_regions : int, optional
        Number of top regions to highlight
    output_dir : str, optional
        Output directory (defaults to analyzer's output_dir)
        
    Returns
    -------
    str
        Path to saved figure
    """
    if output_dir is None:
        output_dir = analyzer.output_dir
        
    # Create figure with 3 columns (sagittal, coronal, axial) and 2 rows (T1+field, T1+atlas)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Get middle slices for each dimension
    x_mid = analyzer.field_data.shape[0] // 2
    y_mid = analyzer.field_data.shape[1] // 2
    z_mid = analyzer.field_data.shape[2] // 2
    
    # Background image for visualization
    if analyzer.t1_data is not None:
        # If T1 reference is available, use it as background
        background_data = analyzer.t1_data
        cmap_bg = 'gray'
    else:
        # If no T1, use field data but use a different colormap to distinguish
        background_data = np.zeros_like(analyzer.field_data)
        cmap_bg = 'gray'
    
    # Create a mask of top regions
    top_regions = analyzer.results_df.nlargest(n_regions, 'MeanValue')
    top_mask = np.zeros_like(analyzer.atlas_data)
    region_colors = {}
    
    for _, row in top_regions.iterrows():
        region_id = row['RegionID']
        region_color = row['Color']
        top_mask[analyzer.atlas_data == region_id] = region_id
        region_colors[region_id] = region_color
    
    # First row: T1 + field overlay
    # Normalize field data for overlay
    field_norm = Normalize(vmin=0, vmax=np.max(analyzer.field_data))

    # Sagittal view (x_mid)
    axes[0, 0].imshow(np.rot90(background_data[x_mid, :, :]), cmap=cmap_bg)
    field_overlay = axes[0, 0].imshow(np.rot90(analyzer.field_data[x_mid, :, :]), 
                                    cmap='hot', alpha=0.7, norm=field_norm)
    axes[0, 0].set_title(f'Field - Sagittal (x={x_mid})')
    axes[0, 0].axis('off')
    
    # Coronal view (y_mid)
    axes[0, 1].imshow(np.rot90(background_data[:, y_mid, :]), cmap=cmap_bg)
    axes[0, 1].imshow(np.rot90(analyzer.field_data[:, y_mid, :]), 
                     cmap='hot', alpha=0.7, norm=field_norm)
    axes[0, 1].set_title(f'Field - Coronal (y={y_mid})')
    axes[0, 1].axis('off')
    
    # Axial view (z_mid)
    axes[0, 2].imshow(np.rot90(background_data[:, :, z_mid]), cmap=cmap_bg)
    axes[0, 2].imshow(np.rot90(analyzer.field_data[:, :, z_mid]), 
                     cmap='hot', alpha=0.7, norm=field_norm)
    axes[0, 2].set_title(f'Field - Axial (z={z_mid})')
    axes[0, 2].axis('off')
    
    # Add colorbar for field overlay
    cbar_ax = fig.add_axes([0.92, 0.55, 0.02, 0.3])
    fig.colorbar(field_overlay, cax=cbar_ax, label='Field Intensity')
    
    # Second row: T1 + atlas overlay
    # Create a custom colormap for atlas overlay
    colors = []
    for region_id in range(int(np.max(top_mask)) + 1):
        if region_id == 0:
            colors.append((0, 0, 0, 0))  # Transparent for background
        elif region_id in region_colors:
            r, g, b = region_colors[region_id]
            colors.append((r, g, b, 0.7))  # Semi-transparent
        else:
            colors.append((0, 0, 0, 0))  # Transparent for unused IDs
    
    atlas_cmap = mcolors.ListedColormap(colors)
    atlas_norm = Normalize(vmin=0, vmax=len(colors)-1)
    
    # Sagittal view (x_mid)
    axes[1, 0].imshow(np.rot90(background_data[x_mid, :, :]), cmap=cmap_bg)
    atlas_overlay = axes[1, 0].imshow(np.rot90(top_mask[x_mid, :, :]),
                                    cmap=atlas_cmap, norm=atlas_norm)
    axes[1, 0].set_title(f'Atlas - Sagittal (x={x_mid})')
    axes[1, 0].axis('off')
    
    # Coronal view (y_mid)
    axes[1, 1].imshow(np.rot90(background_data[:, y_mid, :]), cmap=cmap_bg)
    axes[1, 1].imshow(np.rot90(top_mask[:, y_mid, :]),
                      cmap=atlas_cmap, norm=atlas_norm)
    axes[1, 1].set_title(f'Atlas - Coronal (y={y_mid})')
    axes[1, 1].axis('off')
    
    # Axial view (z_mid)
    axes[1, 2].imshow(np.rot90(background_data[:, :, z_mid]), cmap=cmap_bg)
    axes[1, 2].imshow(np.rot90(top_mask[:, :, z_mid]),
                      cmap=atlas_cmap, norm=atlas_norm)
    axes[1, 2].set_title(f'Atlas - Axial (z={z_mid})')
    axes[1, 2].axis('off')
    
    # Add legend for regions
    patches = []
    for _, row in top_regions.iterrows():
        patch = plt.Rectangle((0, 0), 1, 1, fc=row['Color'])
        patches.append(patch)
    
    leg_ax = fig.add_axes([0.92, 0.1, 0.06, 0.3])
    leg_ax.axis('off')
    leg_ax.legend(patches, top_regions['RegionName'], loc='center')
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    # Save the figure
    figure_path = os.path.join(output_dir, 'field_and_regions_slices.png')
    plt.savefig(figure_path, dpi=300)
    plt.close()
    print(f"Slice visualization saved to {figure_path}")
    return figure_path

def generate_histogram(analyzer, n_regions=5, output_dir=None):
    """Generate histograms of field values for top regions.
    
    Parameters
    ----------
    analyzer : TIFieldAnalyzer
        Analyzer object
    n_regions : int, optional
        Number of top regions to display
    output_dir : str, optional
        Output directory (defaults to analyzer's output_dir)
        
    Returns
    -------
    str
        Path to saved figure
    """
    if output_dir is None:
        output_dir = analyzer.output_dir
        
    top_regions = analyzer.results_df.nlargest(n_regions, 'MeanValue')
    
    plt.figure(figsize=(12, 8))
    
    for _, row in top_regions.iterrows():
        region_id = row['RegionID']
        region_name = row['RegionName']
        region_color = row['Color']
        
        # Extract field values for this region
        mask = (analyzer.atlas_data == region_id)
        values = analyzer.field_data[mask]
        
        # Plot histogram
        plt.hist(values, bins=30, alpha=0.5, 
                 label=f"{region_name} (Mean: {row['MeanValue']:.4f})",
                 color=region_color)
    
    plt.xlabel('Field Value')
    plt.ylabel('Frequency')
    plt.title(f'Field Value Distribution in Top {n_regions} Regions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    figure_path = os.path.join(output_dir, 'region_histograms.png')
    plt.savefig(figure_path, dpi=300)
    plt.close()
    print(f"Histograms saved to {figure_path}")
    return figure_path
    
def generate_report(analyzer, include_papaya=True):
    """Generate a comprehensive HTML report.
    
    Parameters
    ----------
    analyzer : TIFieldAnalyzer
        Analyzer object with results
    visualizations : bool, optional
        Whether to generate visualizations
        
    Returns
    -------
    str
        Path to HTML report
    """
    # Save results to CSV
    csv_path = analyzer.save_results()
    
    # Create visualizations if requested
    
    try:
        bar_chart_path = create_bar_chart(analyzer)
    except Exception as e:
        print(f"Warning: Could not create bar chart: {str(e)}")
        bar_chart_path = None
        
    try:
        slices_path = create_slices_visualization(analyzer)
    except Exception as e:
        print(f"Warning: Could not create slice visualization: {str(e)}")
        slices_path = None
        
    try:
        histogram_path = generate_histogram(analyzer)
    except Exception as e:
        print(f"Warning: Could not create histograms: {str(e)}")
        histogram_path = None
   
    
    # Load HTML template
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report_template.txt')
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Generate top regions HTML rows
    top_regions_rows = ""
    for _, row in analyzer.results_df.head(10).iterrows():
        top_regions_rows += f"""
            <tr>
                <td>{row['RegionID']}</td>
                <td>{row['RegionName']}</td>
                <td>{row['MeanValue']:.6f}</td>
                <td>{row['MaxValue']:.6f}</td>
                <td>{row['Volume_mm3']:.2f}</td>
            </tr>
        """
    
    # Generate visualizations section if available
    visualizations_section = "<h2>Visualizations</h2>"
    
    if bar_chart_path:
        visualizations_section += f"""
        <h3>Top Regions by Mean Field Value</h3>
        <img src="{os.path.basename(bar_chart_path)}" alt="Bar Chart of Top Regions">
        """
        
    if slices_path:
        visualizations_section += f"""
        <h3>Field Distribution and Top Regions</h3>
        <img src="{os.path.basename(slices_path)}" alt="Field and Region Slices">
        """
        
    if histogram_path:
        visualizations_section += f"""
        <h3>Field Value Distribution in Top Regions</h3>
        <img src="{os.path.basename(histogram_path)}" alt="Field Value Histograms">
        """
    
    # Fill template with values
    html_content = template.format(
        field_data=os.path.basename(analyzer.field_nifti),
        atlas=os.path.basename(analyzer.atlas_nifti),
        analysis_date=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
        total_field_mean=f"{np.mean(analyzer.field_data):.6f}",
        total_field_max=f"{np.max(analyzer.field_data):.6f}",
        num_regions=str(len(analyzer.results_df)),
        highest_mean_region=f"{analyzer.results_df.iloc[0]['RegionName']} ({analyzer.results_df.iloc[0]['MeanValue']:.6f})",
        highest_max_region=f"{analyzer.results_df.sort_values('MaxValue', ascending=False).iloc[0]['RegionName']} ({analyzer.results_df.sort_values('MaxValue', ascending=False).iloc[0]['MaxValue']:.6f})",
        top_regions_rows=top_regions_rows,
        visualizations_section=visualizations_section
    )
    
    # Write HTML file
    html_path = os.path.join(analyzer.output_dir, 'analysis_report.html')
    with open(html_path, 'w') as f:
        f.write(html_content)
        
    print(f"HTML report saved to {html_path}")
    
    # Add Papaya viewer if requested and files are provided
    add_papaya_viewer(html_path, analyzer.t1_mni, analyzer.field_nifti)
    
    return html_path