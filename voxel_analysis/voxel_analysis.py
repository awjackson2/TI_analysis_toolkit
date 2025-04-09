#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for atlas-based voxel analysis of TI fields.
"""

import os
import sys
import argparse
import datetime
import numpy as np
from ti_field_core import TIFieldAnalyzer
import ti_field_visualization as visu

def get_analysis_name():
    """Gets user preference for analysis name"""
    print("\nOutput Options:")
    analysis_name = input("Enter a name for this analysis (press Enter for default name with timestamp): ").strip()
    return analysis_name

def main():
    parser = argparse.ArgumentParser(description='ROI analysis of TI field data using atlas parcellation.')
    parser.add_argument('--field', required=True, help='TI field NIfTI file')
    parser.add_argument('--atlas', required=True, help='Atlas parcellation NIfTI file')
    parser.add_argument('--labels', required=True, help='Region labels text file')
    parser.add_argument('--output', default='ti_field_analysis', help='Output directory')
    parser.add_argument('--t1-mni', help='Optional T1 MNI reference image')
    parser.add_argument('--top-n', type=int, default=20, help='Number of top regions to visualize')
    parser.add_argument('--no-visualizations', action='store_true', help='Skip generating visualizations')
    parser.add_argument('--threshold', type=float, default=0.0, help='Minimum field value threshold (default: 0.0)')
    parser.add_argument('--include-papaya', action='store_true', help='Include Papaya viewer in HTML report')
    
    args = parser.parse_args()
    
    try:
        # Get user input for analysis name
        analysis_name = get_analysis_name()
        
        # Create timestamp for unique folder naming
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        if analysis_name:
            output_dir = analysis_name
        else:
            output_dir = f'ti_field_analysis_{timestamp}'
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize analyzer
        analyzer = TIFieldAnalyzer(
            field_nifti=args.field,
            atlas_nifti=args.atlas,
            hcp_labels_file=args.labels,
            output_dir=output_dir,
            t1_mni=args.t1_mni
        )
        
        # Run analysis with specified threshold
        results_df = analyzer.analyze_by_region(min_threshold=args.threshold)
        
        # Save results
        csv_path = analyzer.save_results()
        print(f"Results saved to {csv_path}")
        
        # Generate visualizations
        if not args.no_visualizations:
            try:
                visu.create_bar_chart(analyzer, n_regions=args.top_n)
            except Exception as e:
                print(f"Warning: Could not create bar chart: {str(e)}")
                
            try:
                visu.create_slices_visualization(analyzer, n_regions=min(args.top_n, 8))
            except Exception as e:
                print(f"Warning: Could not create slice visualization: {str(e)}")
                
            try:
                visu.generate_histogram(analyzer, n_regions=min(args.top_n, 5))
            except Exception as e:
                print(f"Warning: Could not create histograms: {str(e)}")
                
            try:
                visu.generate_report(analyzer, include_papaya=args.include_papaya)
            except Exception as e:
                print(f"Warning: Could not generate HTML report: {str(e)}")
        
        print(f"Analysis completed successfully!")
    
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()