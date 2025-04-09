#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for cortical region analysis of TI fields.
This script analyzes TI field data within specific cortical regions defined in the HCP atlas.
It can be used to compare field statistics across different cortical regions.

Now includes voxel-level distribution visualization and threshold filtering.
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from ti_field_core import TIFieldAnalyzer

# Try to import the RegionVoxelPlotter from the added module
try:
    from region_voxel_plot import RegionVoxelPlotter
    HAS_VOXEL_PLOTTER = True
except ImportError:
    print("Warning: RegionVoxelPlotter module not found. Voxel distribution plotting will be disabled.")
    HAS_VOXEL_PLOTTER = False

def load_hcp_labels(hcp_labels_file):
    """Load HCP region labels from the specified file.
    
    Parameters
    ----------
    hcp_labels_file : str
        Path to the HCP labels file
        
    Returns
    -------
    dict
        Dictionary mapping region IDs to region information
    """
    region_info = {}
    
    # Try multiple encodings with error handling
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
    
    for encoding in encodings_to_try:
        try:
            with open(hcp_labels_file, 'r', encoding=encoding) as f:
                # Skip header line if it starts with #
                first_line = f.readline()
                if not first_line.startswith('#'):
                    # If it's not a header, process it as a data line
                    process_region_line(first_line, region_info)
                
                # Process remaining lines
                for line in f:
                    process_region_line(line, region_info)
                    
            print(f"Loaded information for {len(region_info)} cortical regions using {encoding} encoding")
            return region_info
                
        except UnicodeDecodeError:
            # If this encoding failed, try the next one
            continue
        except Exception as e:
            # For other errors, print a warning and continue with the next encoding
            print(f"Warning: Error loading region info with {encoding} encoding: {str(e)}")
            continue
    
    # If we get here, none of the encodings worked
    print(f"Warning: Could not load region info from {hcp_labels_file} with any encoding")
    print("Will continue analysis with region IDs only (no names or colors)")
    return region_info

def process_region_line(line, region_info):
    """Process a single line from the HCP labels file.
    
    Parameters
    ----------
    line : str
        Line from the HCP labels file
    region_info : dict
        Dictionary to add the region information to
    """
    # Try different delimiters (tab, space, comma)
    for delimiter in ['\t', ' ', ',']:
        parts = line.strip().split(delimiter)
        # Filter out empty strings that might come from multiple spaces
        parts = [p for p in parts if p]
        
        if len(parts) >= 2:  # At minimum we need ID and name
            try:
                region_id = int(parts[0])
                region_name = parts[1]
                
                # Try to extract colors if available
                if len(parts) >= 5:
                    try:
                        r = int(parts[2]) / 255.0
                        g = int(parts[3]) / 255.0
                        b = int(parts[4]) / 255.0
                        color = (r, g, b)
                    except (ValueError, IndexError):
                        color = (0.5, 0.5, 0.5)  # Default gray
                else:
                    color = (0.5, 0.5, 0.5)  # Default gray
                
                region_info[region_id] = {
                    'name': region_name,
                    'color': color
                }
                return  # Successfully processed this line
            except ValueError:
                # If we can't convert the ID to int, try the next delimiter
                continue
    
    # If we get here, none of the delimiter options worked for this line
    # Just ignore this line and continue

def find_region_ids(region_inputs, region_info):
    """Find region IDs based on user input.
    
    Parameters
    ----------
    region_inputs : list
        List of region inputs (names or IDs)
    region_info : dict
        Dictionary with region information
        
    Returns
    -------
    dict
        Dictionary mapping region IDs to region names
    """
    selected_regions = {}
    for input_val in region_inputs:
        # Check if the input is a number (region ID)
        try:
            region_id = int(input_val)
            if region_id in region_info:
                selected_regions[region_id] = region_info[region_id]['name']
            else:
                print(f"Warning: Region ID {region_id} not found in HCP labels")
        except ValueError:
            # Input is a string, search by name (case-insensitive)
            input_lower = input_val.lower()
            found = False
            for region_id, info in region_info.items():
                if input_lower in info['name'].lower():
                    selected_regions[region_id] = info['name']
                    found = True
            
            if not found:
                print(f"Warning: No region found matching '{input_val}'")
    
    return selected_regions

def analyze_cortical_region(analyzer, region_id, region_name, threshold=0.2613):
    """Analyze a specific cortical region with threshold filtering.
    
    Parameters
    ----------
    analyzer : TIFieldAnalyzer
        Analyzer object
    region_id : int
        ID of the region to analyze
    region_name : str
        Name of the region
    threshold : float, optional
        Minimum field value threshold (default: 0.0)
        
    Returns
    -------
    dict
        Dictionary with region statistics
    """
    # Create mask for this region
    region_mask = (analyzer.atlas_data == region_id)
    
    # Check if the mask contains any voxels
    mask_count = np.sum(region_mask)
    if mask_count == 0:
        print(f"Warning: Region {region_name} (ID: {region_id}) contains 0 voxels in the atlas")
        # Return zeros for all metrics
        results = {
            'RegionID': region_id,
            'RegionName': region_name,
            'MeanValue': 0,
            'MaxValue': 0,
            'MinValue': 0,
            'MedianValue': 0, 
            'StdValue': 0,
            'VoxelCount': 0,
            'Volume_mm3': 0,
            'Error': 'No voxels in mask'
        }
        return results
    
    # Apply threshold filter
    value_mask = (analyzer.field_data > threshold)
    
    # Combine masks - only voxels that are both in the region AND above threshold
    combined_mask = region_mask & value_mask
    
    # Check if any voxels remain after threshold filtering
    threshold_count = np.sum(combined_mask)
    if threshold_count == 0:
        print(f"Warning: Region {region_name} (ID: {region_id}) has no voxels above threshold {threshold}")
        # Return zeros with error note
        results = {
            'RegionID': region_id,
            'RegionName': region_name,
            'MeanValue': 0,
            'MaxValue': 0,
            'MinValue': 0,
            'MedianValue': 0, 
            'StdValue': 0,
            'VoxelCount': 0,
            'Volume_mm3': 0,
            'Error': f'No voxels above threshold {threshold}'
        }
        return results
    
    # Extract all field values within region mask (for comparison)
    all_field_values = analyzer.field_data[region_mask]
    
    # Extract field values after threshold filtering
    field_values = analyzer.field_data[combined_mask]
    
    # Double-check minimum value
    min_value = np.min(field_values)
    if min_value <= threshold:
        print(f"Warning: Found min value {min_value:.8e} <= threshold {threshold} in region {region_name}")
    
    # Debug information
    print(f"Region {region_name} (ID: {region_id}): {threshold_count}/{mask_count} voxels above threshold {threshold}")
    print(f"Field values range: {min_value:.8e} to {np.max(field_values):.8e}")
    
    # Calculate voxel volume
    voxel_sizes = analyzer.field_img.header.get_zooms()[:3]
    voxel_volume = np.prod(voxel_sizes)
    
    # Calculate statistics
    mean_value = np.mean(field_values)
    max_value = np.max(field_values)
    median_value = np.median(field_values)
    std_value = np.std(field_values)
    volume_mm3 = threshold_count * voxel_volume
    
    # Save detailed voxel value information for debugging
    debug_dir = os.path.join(analyzer.output_dir, 'voxel_debug')
    os.makedirs(debug_dir, exist_ok=True)
    
    # Save detailed voxel values to CSV for inspection
    safe_name = region_name.replace(" ", "_").replace("/", "_")
    voxel_csv_path = os.path.join(debug_dir, f'voxel_values_{region_id}_{safe_name}.csv')
    
    # Create a DataFrame with all voxel values in this region
    voxel_df = pd.DataFrame({
        'Value': all_field_values,
        'AboveThreshold': all_field_values > threshold
    })
    voxel_df.to_csv(voxel_csv_path, index=False, float_format='%.10e')
    print(f"Saved voxel values to {voxel_csv_path}")
    
    # Return statistics as dictionary with additional info
    results = {
        'RegionID': region_id,
        'RegionName': region_name,
        'MeanValue': mean_value,
        'MaxValue': max_value,
        'MinValue': min_value,
        'MedianValue': median_value, 
        'StdValue': std_value,
        'VoxelCount': threshold_count,
        'TotalVoxelsInRegion': mask_count,
        'VoxelsAboveThreshold': threshold_count,
        'PercentAboveThreshold': (threshold_count / mask_count * 100) if mask_count > 0 else 0,
        'Volume_mm3': volume_mm3,
        'Threshold': threshold
    }
    
    # Save region mask for visualization
    mask_img = nib.Nifti1Image(combined_mask.astype(np.int16), analyzer.field_img.affine)
    mask_path = os.path.join(analyzer.output_dir, f'region_mask_{region_id}_{safe_name}.nii.gz')
    nib.save(mask_img, mask_path)
    print(f"Saved thresholded region mask to {mask_path}")
    
    return results

def create_voxel_distributions(field_file, atlas_file, labels_file, output_dir, region_ids, 
                              plot_types=None, max_n_regions=5, threshold=None):
    """Create voxel-level distribution plots for the specified regions.
    
    Parameters
    ----------
    field_file : str
        Path to field NIfTI file
    atlas_file : str
        Path to atlas NIfTI file
    labels_file : str
        Path to labels file
    output_dir : str
        Path to output directory
    region_ids : list
        List of region IDs to plot
    plot_types : list, optional
        List of plot types to create ('violin', 'scatter', 'box')
    max_n_regions : int, optional
        Maximum number of regions to include in plots
    threshold : float, optional
        Minimum field value threshold for voxel plotting
        
    Returns
    -------
    list
        List of paths to created plots
    """
    if not HAS_VOXEL_PLOTTER:
        print("WARNING: RegionVoxelPlotter module not found. Skipping voxel distribution plots.")
        return []
        
    # Limit to maximum number of regions
    if len(region_ids) > max_n_regions:
        print(f"Too many regions ({len(region_ids)}) for voxel distribution visualization.")
        print(f"Limiting to {max_n_regions} regions with highest region IDs.")
        # Sort by region ID and take the last max_n_regions
        region_ids = sorted(region_ids)[-max_n_regions:]
    
    # Default plot types if not specified
    if plot_types is None:
        plot_types = ['violin', 'box', 'scatter']
        
    # Create voxel distribution plots subdirectory
    voxel_plots_dir = os.path.join(output_dir, 'voxel_distributions')
    os.makedirs(voxel_plots_dir, exist_ok=True)
    
    plot_outputs = []
    
    try:
        print("\nGenerating voxel-level distribution plots...")
        
        # Initialize the voxel plotter
        plotter = RegionVoxelPlotter(
            field_file=field_file,
            atlas_file=atlas_file,
            labels_file=labels_file,
            output_dir=voxel_plots_dir
        )
        
        # Apply threshold filter if provided
        if threshold is not None and threshold != 0:
            # Create a thresholded field copy for plotting
            print(f"Applying threshold {threshold} to field data for voxel distribution plots")
            
            # Load the original field
            field_img = nib.load(field_file)
            field_data = field_img.get_fdata()
            
            # Apply threshold
            field_data_thresholded = np.copy(field_data)
            field_data_thresholded[field_data_thresholded <= threshold] = 0
            
            # Save thresholded field to temporary file
            thresholded_field_file = os.path.join(voxel_plots_dir, 'temp_thresholded_field.nii.gz')
            thresholded_img = nib.Nifti1Image(field_data_thresholded, field_img.affine)
            nib.save(thresholded_img, thresholded_field_file)
            
            # Use thresholded field for plotting
            plotter = RegionVoxelPlotter(
                field_file=thresholded_field_file,
                atlas_file=atlas_file,
                labels_file=labels_file,
                output_dir=voxel_plots_dir
            )
            
        # Generate each plot type
        for plot_type in plot_types:
            print(f"Creating {plot_type} plot for regions: {region_ids}")
            
            try:
                fig = plotter.plot_region_voxels(
                    region_ids=region_ids,
                    plot_type=plot_type,
                    save=True,
                    show=False
                )
                
                # Add threshold info to plot filename if applicable
                threshold_str = f"_thresh{threshold}" if threshold is not None and threshold != 0 else ""
                plot_filename = f'{plot_type}_plot_regions_{"-".join(str(r) for r in region_ids)}{threshold_str}.png'
                plot_path = os.path.join(voxel_plots_dir, plot_filename)
                
                # Add plot info to outputs
                plot_outputs.append({
                    'type': plot_type,
                    'path': plot_path,
                    'threshold': threshold
                })
                
            except Exception as e:
                print(f"Warning: Could not create {plot_type} plot: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Export voxel values to CSV for further analysis
        try:
            plotter.export_region_values_to_csv(region_ids)
            print(f"Exported voxel values to CSV files in {voxel_plots_dir}")
        except Exception as e:
            print(f"Warning: Could not export voxel values to CSV: {str(e)}")
        
        # Clean up temporary file if created
        if threshold is not None and threshold != 0:
            if os.path.exists(thresholded_field_file):
                os.remove(thresholded_field_file)
                print(f"Removed temporary thresholded field file")
        
        return plot_outputs
        
    except Exception as e:
        print(f"Warning: Could not create voxel distribution plots: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def update_html_report(report_path, voxel_plots_info, threshold=None):
    """Add voxel distribution plots to HTML report.
    
    Parameters
    ----------
    report_path : str
        Path to HTML report
    voxel_plots_info : list
        List of dictionaries with plot information
    threshold : float, optional
        Threshold value used for analysis
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    if not voxel_plots_info:
        return False
    
    try:
        # Read existing HTML
        with open(report_path, 'r') as f:
            html_content = f.read()
        
        # Create voxel distributions section
        threshold_info = f" (with threshold {threshold})" if threshold is not None and threshold != 0 else ""
        voxel_section = f"""
        <div class="voxel-distributions">
            <h2>Voxel-Level Distribution Analysis{threshold_info}</h2>
            <p>The following plots show the distribution of individual voxel values within each region:</p>
        """
        
        # Group plots by type
        plot_types = set(plot['type'] for plot in voxel_plots_info)
        
        for plot_type in plot_types:
            plots = [plot for plot in voxel_plots_info if plot['type'] == plot_type]
            
            if plots:
                # Use relative path from report dir to plot
                report_dir = os.path.dirname(report_path)
                rel_path = os.path.relpath(plots[0]['path'], report_dir)
                
                # Add description based on plot type
                if plot_type == 'violin':
                    voxel_section += f"""
                    <h3>Violin Plot</h3>
                    <p>Violin plots show the full distribution shape of voxel values within each region, 
                    making it easy to compare distributions across regions.</p>
                    <img src="{rel_path}" alt="Violin plot of voxel distributions">
                    """
                elif plot_type == 'box':
                    voxel_section += f"""
                    <h3>Box Plot</h3>
                    <p>Box plots show the quartiles and outliers of voxel distributions, 
                    offering a statistical summary of each region.</p>
                    <img src="{rel_path}" alt="Box plot of voxel distributions">
                    """
                elif plot_type == 'scatter':
                    voxel_section += f"""
                    <h3>Scatter Plot</h3>
                    <p>Scatter plots show the full distribution of voxel values, 
                    with density coloring to highlight clusters.</p>
                    <img src="{rel_path}" alt="Scatter plot of voxel distributions">
                    """
        
        voxel_section += """
        </div>
        """
        
        # Insert before the closing body tag
        if '</body>' in html_content:
            new_html = html_content.replace('</body>', f'{voxel_section}</body>')
            
            # Write updated HTML
            with open(report_path, 'w') as f:
                f.write(new_html)
                
            print(f"Updated HTML report with voxel distribution plots")
            return True
        else:
            print(f"Warning: Could not find closing body tag in HTML report")
            return False
        
    except Exception as e:
        print(f"Warning: Could not update HTML report: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Analyze TI field in specific cortical regions')
    parser.add_argument('--field', required=True, help='NIfTI file containing the field values')
    parser.add_argument('--atlas', required=True, help='NIfTI file containing the HCP atlas parcellation')
    parser.add_argument('--labels', required=True, help='Text file with HCP region labels (format: ID Name R G B A)')
    parser.add_argument('--output', default='ti_cortex_analysis', help='Output directory for results')
    parser.add_argument('--regions', required=True, nargs='+', 
                      help='Region IDs or names to analyze (can specify multiple)')
    parser.add_argument('--compare', action='store_true', 
                      help='Compare multiple regions and calculate differential values')
    parser.add_argument('--t1-mni', help='Optional T1 MNI reference image for visualization')
    parser.add_argument('--no-visualizations', action='store_true', help='Skip generating visualizations')
    parser.add_argument('--threshold', type=float, default=0.0,
                      help='Minimum field value threshold (default: 0.0)')
    parser.add_argument('--voxel-plots', action='store_true', 
                      help='Generate voxel-level distribution plots (default: True)')
    parser.add_argument('--voxel-plot-types', nargs='+', choices=['violin', 'scatter', 'box'],
                      default=['violin', 'box'],
                      help='Types of voxel plots to generate')
    
    args = parser.parse_args()
    
    # Check if input files exist
    for file_path in [args.field, args.atlas, args.labels]:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
    
    try:
        # Load HCP labels
        region_info = load_hcp_labels(args.labels)
        
        # Find region IDs based on user input
        if not args.regions:
            print("Error: No regions specified for analysis")
            sys.exit(1)
            
        selected_regions = find_region_ids(args.regions, region_info)
        
        if not selected_regions:
            print("Error: None of the specified regions were found")
            sys.exit(1)
            
        print(f"Selected regions for analysis:")
        for region_id, region_name in selected_regions.items():
            print(f"  - {region_name} (ID: {region_id})")
        
        # Initialize the output directory
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize the analyzer
        analyzer = TIFieldAnalyzer(
            field_nifti=args.field,
            atlas_nifti=args.atlas,
            hcp_labels_file=args.labels,
            output_dir=output_dir,
            t1_mni=args.t1_mni
        )
        
        # Analyze each region
        region_results = []
        for region_id, region_name in selected_regions.items():
            print(f"Analyzing region: {region_name} (ID: {region_id})")
            
            # Run the analysis with threshold
            result = analyze_cortical_region(analyzer, region_id, region_name, args.threshold)
            region_results.append(result)
        
        # Create a DataFrame with the results
        df = pd.DataFrame(region_results)
        
        # Reorder columns to put RegionName first
        cols = ['RegionName', 'RegionID'] + [col for col in df.columns if col not in ['RegionName', 'RegionID']]
        df = df[cols]
        
        # Save results to CSV
        csv_path = os.path.join(output_dir, 'cortical_region_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        # Calculate and display differential values if requested
        if args.compare and len(region_results) > 1:
            print("\nDifferential Values:")
            diff_results = []
            
            for i in range(len(region_results)):
                for j in range(i+1, len(region_results)):
                    region1 = region_results[i]
                    region2 = region_results[j]
                    
                    diff = abs(region1['MeanValue'] - region2['MeanValue'])
                    
                    print(f"{region1['RegionName']} vs {region2['RegionName']}: {diff:.6f}")
                    
                    diff_results.append({
                        'Region_1': region1['RegionName'],
                        'Region_1_ID': region1['RegionID'],
                        'Region_2': region2['RegionName'],
                        'Region_2_ID': region2['RegionID'],
                        'Diff_Mean': diff,
                        'Region_1_Mean': region1['MeanValue'],
                        'Region_2_Mean': region2['MeanValue']
                    })
            
            # Save differential results to CSV
            diff_df = pd.DataFrame(diff_results)
            diff_csv_path = os.path.join(output_dir, 'cortical_region_differentials.csv')
            diff_df.to_csv(diff_csv_path, index=False)
            print(f"Differential results saved to {diff_csv_path}")
            
            # Create visualization of differentials if not disabled
            if not args.no_visualizations:
                try:
                    plt.figure(figsize=(12, 6))
                    
                    # Format labels with region names and their mean values
                    comparison_labels = [
                        f"{r['Region_1']} ({r['Region_1_Mean']:.4f}) vs\n{r['Region_2']} ({r['Region_2_Mean']:.4f})"
                        for r in diff_results
                    ]
                    
                    # Create the bar plot
                    bars = plt.bar(
                        range(len(diff_results)),
                        [r['Diff_Mean'] for r in diff_results],
                        color='skyblue'
                    )
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(
                            bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.4f}',
                            ha='center', va='bottom', rotation=0
                        )
                    
                    plt.ylabel('Absolute Difference in Mean Field Value')
                    plt.title('Differential Field Values Between Cortical Regions')
                    plt.xticks(range(len(diff_results)), comparison_labels, rotation=45, ha='right')
                    plt.tight_layout()
                    plt.subplots_adjust(bottom=0.3)  # Make more room for labels
                    
                    diff_plot_path = os.path.join(output_dir, 'differential_values.png')
                    plt.savefig(diff_plot_path, dpi=300)
                    plt.close()
                    print(f"Differential plot saved to {diff_plot_path}")
                except Exception as e:
                    print(f"Warning: Could not create differential plot: {str(e)}")
        
        # Generate bar plot of mean values if not disabled
        if not args.no_visualizations:
            try:
                plt.figure(figsize=(12, 6))
                
                # Sort by mean value for better visualization
                sorted_indices = np.argsort([r['MeanValue'] for r in region_results])[::-1]  # Descending
                sorted_results = [region_results[i] for i in sorted_indices]
                
                # Get colors from region_info if available
                colors = []
                for r in sorted_results:
                    region_id = r['RegionID']
                    if region_id in region_info and 'color' in region_info[region_id]:
                        colors.append(region_info[region_id]['color'])
                    else:
                        colors.append((0.5, 0.5, 0.5))  # Default gray
                
                bars = plt.bar(
                    range(len(sorted_results)),
                    [r['MeanValue'] for r in sorted_results],
                    color=colors
                )
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{height:.4f}',
                        ha='center', va='bottom', rotation=0
                    )
                
                threshold_str = f" (Threshold: {args.threshold})" if args.threshold != 0 else ""
                plt.ylabel('Mean Field Value')
                plt.title(f'Mean Field Values in Cortical Regions{threshold_str}')
                plt.xticks(range(len(sorted_results)), 
                         [f"{r['RegionName']} (ID:{r['RegionID']})" for r in sorted_results], 
                         rotation=45, ha='right')
                plt.tight_layout()
                plt.subplots_adjust(bottom=0.3)  # Make more room for labels
                
                mean_plot_path = os.path.join(output_dir, 'mean_values.png')
                plt.savefig(mean_plot_path, dpi=300)
                plt.close()
                print(f"Mean values plot saved to {mean_plot_path}")
                
                # Create a box plot or violin plot to show distribution
                plt.figure(figsize=(12, 6))
                
                # Gather the field values for each region
                distributions = []
                labels = []
                
                for r in sorted_results:
                    region_id = r['RegionID']
                    region_name = r['RegionName']
                    
                    # Create mask with threshold applied
                    region_mask = (analyzer.atlas_data == region_id)
                    value_mask = (analyzer.field_data > args.threshold)
                    combined_mask = region_mask & value_mask
                    
                    # Extract values using the mask
                    values = analyzer.field_data[combined_mask]
                    
                    if len(values) > 0:
                        distributions.append(values)
                        labels.append(f"{region_name}\n(ID:{region_id})")
                
                if distributions:
                    plt.boxplot(distributions, labels=labels)
                    plt.ylabel('Field Value')
                    threshold_str = f" (Threshold: {args.threshold})" if args.threshold != 0 else ""
                    plt.title(f'Field Value Distribution in Cortical Regions{threshold_str}')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.subplots_adjust(bottom=0.3)  # Make more room for labels
                    
                    dist_plot_path = os.path.join(output_dir, 'value_distributions.png')
                    plt.savefig(dist_plot_path, dpi=300)
                    plt.close()
                    print(f"Distribution plot saved to {dist_plot_path}")
                
            except Exception as e:
                print(f"Warning: Could not create mean values plot: {str(e)}")

        # Generate voxel-level distribution plots if requested
        voxel_plots_info = None
        if not args.no_visualizations and (args.voxel_plots or HAS_VOXEL_PLOTTER):
            region_ids = [r['RegionID'] for r in region_results]
            voxel_plots_info = create_voxel_distributions(
                field_file=args.field,
                atlas_file=args.atlas,
                labels_file=args.labels,
                output_dir=output_dir,
                region_ids=region_ids,
                plot_types=args.voxel_plot_types,
                threshold=args.threshold
            )
            
            # Create simple HTML report
            html_path = os.path.join(output_dir, 'analysis_report.html')
            if os.path.exists(html_path):
                # Update existing report with voxel plots
                update_html_report(html_path, voxel_plots_info, args.threshold)
            else:
                # Create basic HTML report
                try:
                    print("\nGenerating HTML report...")
                    
                    threshold_str = f" (Threshold: {args.threshold})" if args.threshold != 0 else ""
                    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Cortical Region Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Cortical Region Analysis Report{threshold_str}</h1>
        <p><strong>Field:</strong> {os.path.basename(args.field)}</p>
        <p><strong>Atlas:</strong> {os.path.basename(args.atlas)}</p>
        <p><strong>Analysis Date:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
        
        <h2>Region Statistics</h2>
        <table>
            <tr>
                <th>Region Name</th>
                <th>Region ID</th>
                <th>Mean Value</th>
                <th>Max Value</th>
                <th>Median Value</th>
                <th>Std Value</th>
                <th>Voxels > Threshold</th>
                <th>Total Voxels</th>
                <th>% Above Threshold</th>
                <th>Volume (mm³)</th>
            </tr>
    """
                    
                    # Add rows for each region
                    for r in region_results:
                        percent_above = r.get('PercentAboveThreshold', 0)
                        html_content += f"""
            <tr>
                <td>{r['RegionName']}</td>
                <td>{r['RegionID']}</td>
                <td>{r['MeanValue']:.6f}</td>
                <td>{r['MaxValue']:.6f}</td>
                <td>{r.get('MedianValue', 0):.6f}</td>
                <td>{r.get('StdValue', 0):.6f}</td>
                <td>{r.get('VoxelsAboveThreshold', 0)}</td>
                <td>{r.get('TotalVoxelsInRegion', 0)}</td>
                <td>{percent_above:.2f}%</td>
                <td>{r['Volume_mm3']:.2f}</td>
            </tr>
                        """
                    
                    html_content += """
        </table>
        
        <h2>Visualizations</h2>
    """
                    
                    # Add standard visualizations
                    if os.path.exists(os.path.join(output_dir, 'mean_values.png')):
                        html_content += f"""
        <h3>Mean Field Values</h3>
        <img src="mean_values.png" alt="Mean Field Values">
                        """
                        
                    if os.path.exists(os.path.join(output_dir, 'value_distributions.png')):
                        html_content += f"""
        <h3>Field Value Distribution</h3>
        <img src="value_distributions.png" alt="Field Value Distribution">
                        """
                        
                    if args.compare and os.path.exists(os.path.join(output_dir, 'differential_values.png')):
                        html_content += f"""
        <h3>Differential Values</h3>
        <img src="differential_values.png" alt="Differential Values">
                        """
                    
                    # Close HTML tags
                    html_content += """
    </div>
</body>
</html>
                    """
                    
                    # Write HTML file
                    with open(html_path, 'w') as f:
                        f.write(html_content)
                    
                    print(f"HTML report saved to {html_path}")
                    
                    # Update report with voxel plots if available
                    if voxel_plots_info:
                        update_html_report(html_path, voxel_plots_info, args.threshold)
                
                except Exception as e:
                    print(f"Warning: Could not create HTML report: {str(e)}")
        
        print(f"Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()