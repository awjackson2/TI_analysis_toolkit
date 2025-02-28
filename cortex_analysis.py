#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for cortical region analysis of TI fields.
This script analyzes TI field data within specific cortical regions defined in the HCP atlas.
It can be used to compare field statistics across different cortical regions.
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

def analyze_cortical_region(analyzer, region_id, region_name):
    """Analyze a specific cortical region.
    
    Parameters
    ----------
    analyzer : TIFieldAnalyzer
        Analyzer object
    region_id : int
        ID of the region to analyze
    region_name : str
        Name of the region
        
    Returns
    -------
    dict
        Dictionary with region statistics
    """
    # Create mask for this region
    mask = (analyzer.atlas_data == region_id)
    
    # Check if the mask contains any voxels
    mask_count = np.sum(mask)
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
    
    # Extract field values within the mask
    field_values = analyzer.field_data[mask]
    
    # Debug information
    print(f"Mask for {region_name} contains {mask_count} voxels")
    print(f"Field values range: {np.min(field_values)} to {np.max(field_values)}")
    
    # Calculate statistics
    voxel_sizes = analyzer.field_img.header.get_zooms()[:3]
    
    mean_value = np.mean(field_values) if len(field_values) > 0 else 0
    max_value = np.max(field_values) if len(field_values) > 0 else 0
    min_value = np.min(field_values) if len(field_values) > 0 else 0
    median_value = np.median(field_values) if len(field_values) > 0 else 0
    std_value = np.std(field_values) if len(field_values) > 0 else 0
    voxel_count = len(field_values)
    volume_mm3 = voxel_count * np.prod(voxel_sizes)
    
    # Return statistics as dictionary
    results = {
        'RegionID': region_id,
        'RegionName': region_name,
        'MeanValue': mean_value,
        'MaxValue': max_value,
        'MinValue': min_value,
        'MedianValue': median_value, 
        'StdValue': std_value,
        'VoxelCount': voxel_count,
        'Volume_mm3': volume_mm3
    }
    
    # Save region mask for visualization
    mask_img = nib.Nifti1Image(mask.astype(np.int16), analyzer.field_img.affine)
    mask_path = os.path.join(analyzer.output_dir, f'region_mask_{region_id}_{region_name.replace(" ", "_")}.nii.gz')
    nib.save(mask_img, mask_path)
    print(f"Saved region mask to {mask_path}")
    
    return results

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
            
            # Run the analysis
            result = analyze_cortical_region(analyzer, region_id, region_name)
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
                
                plt.ylabel('Mean Field Value')
                plt.title('Mean Field Values in Cortical Regions')
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
                    mask = (analyzer.atlas_data == region_id)
                    values = analyzer.field_data[mask]
                    
                    if len(values) > 0:
                        distributions.append(values)
                        labels.append(f"{region_name}\n(ID:{region_id})")
                
                if distributions:
                    plt.boxplot(distributions, labels=labels)
                    plt.ylabel('Field Value')
                    plt.title('Field Value Distribution in Cortical Regions')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.subplots_adjust(bottom=0.3)  # Make more room for labels
                    
                    dist_plot_path = os.path.join(output_dir, 'value_distributions.png')
                    plt.savefig(dist_plot_path, dpi=300)
                    plt.close()
                    print(f"Distribution plot saved to {dist_plot_path}")
                
            except Exception as e:
                print(f"Warning: Could not create mean values plot: {str(e)}")
        
        print(f"Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
