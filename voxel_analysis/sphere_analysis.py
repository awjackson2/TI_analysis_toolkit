#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for spherical ROI analysis of TI fields.
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ti_field_core import TIFieldAnalyzer

def main():
    parser = argparse.ArgumentParser(description='Analyze TI field in spherical ROIs')
    parser.add_argument('--field', required=True, help='NIfTI file containing the field values')
    parser.add_argument('--output', default='ti_sphere_analysis', help='Output directory for results')
    parser.add_argument('--coords', required=True, 
                      help='Coordinates of sphere center as x,y,z or path to JSON file with multiple coordinates')
    parser.add_argument('--radius', type=float, default=5.0, 
                      help='Radius of sphere in mm (default: 5.0)')
    parser.add_argument('--compare', action='store_true', 
                      help='Compare multiple ROIs and calculate differential values')
    parser.add_argument('--t1-mni', help='Optional T1 MNI reference image for visualization')
    parser.add_argument('--no-visualizations', action='store_true', help='Skip generating visualizations')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.field):
        print(f"Error: Field file not found: {args.field}")
        sys.exit(1)
    
    # Process coordinates input
    roi_coordinates = []
    roi_names = []
    
    # Check if coords is a path to a JSON file
    if args.coords.endswith('.json') and os.path.exists(args.coords):
        try:
            with open(args.coords, 'r') as f:
                coords_data = json.load(f)
                
            # Support different JSON formats
            if isinstance(coords_data, dict):
                # Format: {"name1": [x, y, z], "name2": [x, y, z], ...}
                for name, coords in coords_data.items():
                    roi_names.append(name)
                    roi_coordinates.append(tuple(coords))
            elif isinstance(coords_data, list):
                # Format: [{"name": "name1", "coords": [x, y, z]}, ...]
                for item in coords_data:
                    if isinstance(item, dict) and 'coords' in item:
                        coords = item['coords']
                        name = item.get('name', f"ROI_{len(roi_names)}")
                        roi_names.append(name)
                        roi_coordinates.append(tuple(coords))
        except Exception as e:
            print(f"Error parsing JSON file: {str(e)}")
            sys.exit(1)
    else:
        # Parse as single coordinate set "x,y,z"
        try:
            coords = tuple(map(int, args.coords.split(',')))
            if len(coords) != 3:
                raise ValueError("Coordinates must be 3 values (x,y,z)")
            roi_names.append(f"ROI_x{coords[0]}_y{coords[1]}_z{coords[2]}")
            roi_coordinates.append(coords)
        except Exception as e:
            print(f"Error parsing coordinates: {str(e)}")
            print("Format should be: x,y,z (e.g., 80,90,75)")
            sys.exit(1)
    
    try:
        # Initialize the analyzer
        analyzer = TIFieldAnalyzer(
            field_nifti=args.field,
            atlas_nifti=args.field,  # Use field as atlas (not used for sphere analysis)
            hcp_labels_file=args.field,  # Use field as labels (not used for sphere analysis)
            output_dir=args.output,
            t1_mni=args.t1_mni
        )
        
        # Analyze each ROI
        roi_results = []
        for i, coords in enumerate(roi_coordinates):
            print(f"Analyzing ROI: {roi_names[i]} at coordinates {coords}")
            
            # Run the analysis
            result = analyzer.analyze_spherical_roi(coords, args.radius)
            result['ROI_Name'] = roi_names[i]
            roi_results.append(result)
        
        # Create a DataFrame with the results
        df = pd.DataFrame(roi_results)
        
        # Reorder columns to put ROI_Name first
        cols = ['ROI_Name'] + [col for col in df.columns if col != 'ROI_Name']
        df = df[cols]
        
        # Save results to CSV
        csv_path = os.path.join(args.output, 'sphere_roi_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        # Calculate and display differential values if requested
        if args.compare and len(roi_results) > 1:
            print("\nDifferential Values:")
            diff_results = []
            
            for i in range(len(roi_results)):
                for j in range(i+1, len(roi_results)):
                    roi1 = roi_results[i]
                    roi2 = roi_results[j]
                    
                    diff = abs(roi1['MeanValue'] - roi2['MeanValue'])
                    
                    print(f"{roi1['ROI_Name']} vs {roi2['ROI_Name']}: {diff:.6f}")
                    
                    diff_results.append({
                        'ROI_1': roi1['ROI_Name'],
                        'ROI_2': roi2['ROI_Name'],
                        'Diff_Mean': diff,
                        'ROI_1_Mean': roi1['MeanValue'],
                        'ROI_2_Mean': roi2['MeanValue']
                    })
            
            # Save differential results to CSV
            diff_df = pd.DataFrame(diff_results)
            diff_csv_path = os.path.join(args.output, 'sphere_roi_differentials.csv')
            diff_df.to_csv(diff_csv_path, index=False)
            print(f"Differential results saved to {diff_csv_path}")
            
            # Create visualization of differentials if not disabled
            if not args.no_visualizations:
                try:
                    plt.figure(figsize=(10, 6))
                    bars = plt.bar(
                        [f"{r['ROI_1']} vs {r['ROI_2']}" for r in diff_results],
                        [r['Diff_Mean'] for r in diff_results]
                    )
                    plt.ylabel('Absolute Difference in Mean Value')
                    plt.title('Differential Values Between ROIs')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    diff_plot_path = os.path.join(args.output, 'differential_values.png')
                    plt.savefig(diff_plot_path, dpi=300)
                    plt.close()
                    print(f"Differential plot saved to {diff_plot_path}")
                except Exception as e:
                    print(f"Warning: Could not create differential plot: {str(e)}")
        
        # Generate bar plot of mean values if not disabled
        if not args.no_visualizations and len(roi_results) > 1:
            try:
                plt.figure(figsize=(10, 6))
                bars = plt.bar(
                    [r['ROI_Name'] for r in roi_results],
                    [r['MeanValue'] for r in roi_results]
                )
                plt.ylabel('Mean Field Value')
                plt.title('Mean Field Values in ROIs')
                plt.xticks(rotation=45, ha='right')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.4f}',
                        ha='center', va='bottom', rotation=0
                    )
                
                plt.tight_layout()
                
                mean_plot_path = os.path.join(args.output, 'mean_values.png')
                plt.savefig(mean_plot_path, dpi=300)
                plt.close()
                print(f"Mean values plot saved to {mean_plot_path}")
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
