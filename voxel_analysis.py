#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for atlas-based voxel analysis of TI fields.
"""

import os
import sys
import argparse
import numpy as np
from ti_field_core import TIFieldAnalyzer
import ti_field_visualization as visu

def main():
    parser = argparse.ArgumentParser(description='Analyze TI field by cortical regions in voxel space')
    parser.add_argument('--field', required=True, help='NIfTI file containing the field values')
    parser.add_argument('--atlas', required=True, help='NIfTI file containing the atlas parcellation')
    parser.add_argument('--labels', required=True, help='Text file with region labels (format: ID Name R G B A)')
    parser.add_argument('--output', default='ti_field_analysis', help='Output directory for results')
    parser.add_argument('--t1-mni', help='Optional T1 MNI reference image for visualization')
    parser.add_argument('--top-n', type=int, default=20, help='Number of top regions to visualize')
    parser.add_argument('--no-visualizations', action='store_true', help='Skip generating visualizations')
    
    args = parser.parse_args()
    
    # Check if input files exist
    for file_path in [args.field, args.atlas, args.labels]:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
    
    if args.t1_mni and not os.path.exists(args.t1_mni):
        print(f"Error: T1 MNI file not found: {args.t1_mni}")
        sys.exit(1)
    
    try:
        # Initialize the analyzer
        analyzer = TIFieldAnalyzer(
            field_nifti=args.field,
            atlas_nifti=args.atlas,
            hcp_labels_file=args.labels,
            output_dir=args.output,
            t1_mni=args.t1_mni
        )
        
        # Run the analysis
        analyzer.analyze_by_region()
        
        # Save results to CSV
        analyzer.save_results()
        
        # Generate visualizations if not disabled
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
                visu.generate_report(analyzer)
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
