#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core module for TI field analysis containing the main TIFieldAnalyzer class.
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors

class TIFieldAnalyzer:
    def __init__(self, field_nifti, atlas_nifti, hcp_labels_file, output_dir, t1_mni=None):
        """Initialize the analyzer with input files and parameters.
        
        Parameters
        ----------
        field_nifti : str
            Path to NIfTI file containing field values
        atlas_nifti : str
            Path to NIfTI file containing atlas parcellation
        hcp_labels_file : str
            Path to text file with HCP region labels
        output_dir : str
            Directory to save output files
        t1_mni : str, optional
            Path to T1 MNI reference image for visualization
        """
        self.field_nifti = field_nifti
        self.atlas_nifti = atlas_nifti
        self.hcp_labels_file = hcp_labels_file
        self.output_dir = output_dir
        self.t1_mni = t1_mni
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        print("Loading field data...")
        self.field_img = nib.load(field_nifti)
        self.field_data = self.field_img.get_fdata()
        
        # Handle 4D data (take the first volume if it's 4D)
        if len(self.field_data.shape) > 3:
            print(f"Field data is {len(self.field_data.shape)}D, using first volume...")
            self.field_data = self.field_data[..., 0]
        
        print("Loading atlas data...")
        self.atlas_img = nib.load(atlas_nifti)
        self.atlas_data_orig = self.atlas_img.get_fdata()
        
        # Add this line to squeeze out any extra dimensions (the fix)
        self.atlas_data_orig = np.squeeze(self.atlas_data_orig)
        
        # Load T1 reference if provided
        if t1_mni:
            print("Loading T1 reference...")
            self.t1_img = nib.load(t1_mni)
            self.t1_data = self.t1_img.get_fdata()
            # Handle 4D data (take the first volume if it's 4D)
            if len(self.t1_data.shape) > 3:
                print(f"T1 data is {len(self.t1_data.shape)}D, using first volume...")
                self.t1_data = self.t1_data[..., 0]
        else:
            self.t1_img = None
            self.t1_data = None
        
        # Align the atlas to the field space
        print("Aligning atlas to field space...")
        self.atlas_data = self.align_atlas_to_field()
        
        # Load region labels and colors
        self.load_region_info()
        
    def align_atlas_to_field(self):
        """Align atlas to field space without resizing."""
        # Get dimensions
        field_dims = self.field_data.shape
        atlas_dims = self.atlas_data_orig.shape
        
        print(f"Field dimensions: {field_dims}")
        print(f"Atlas dimensions: {atlas_dims}")
        
        # Create output array with field dimensions
        aligned_atlas = np.zeros(field_dims, dtype=self.atlas_data_orig.dtype)
        
        # Calculate the offsets to center the atlas in the field space
        # Negative offsets mean the atlas is larger than the field
        offsets = [(f_dim - a_dim) // 2 for f_dim, a_dim in zip(field_dims, atlas_dims)]
        print(f"Centering offsets: {offsets}")
        
        # Define slices for the aligned atlas and original atlas
        aligned_slices = []
        atlas_slices = []
        
        for i, offset in enumerate(offsets):
            if offset >= 0:
                # Atlas is smaller than or equal to field in this dimension
                # Place it centered in the field
                aligned_start = offset
                aligned_end = offset + atlas_dims[i]
                atlas_start = 0
                atlas_end = atlas_dims[i]
            else:
                # Atlas is larger than field in this dimension
                # Center it and crop to field size
                aligned_start = 0
                aligned_end = field_dims[i]
                atlas_start = -offset
                atlas_end = atlas_start + field_dims[i]
            
            aligned_slices.append(slice(aligned_start, aligned_end))
            atlas_slices.append(slice(atlas_start, atlas_end))
        
        # Copy the atlas data into the aligned array
        aligned_atlas[tuple(aligned_slices)] = self.atlas_data_orig[tuple(atlas_slices)]
        
        # Save the aligned atlas for inspection
        aligned_atlas_img = nib.Nifti1Image(aligned_atlas, self.field_img.affine)
        aligned_path = os.path.join(self.output_dir, 'aligned_atlas.nii.gz')
        nib.save(aligned_atlas_img, aligned_path)
        print(f"Saved aligned atlas to {aligned_path}")
        
        return aligned_atlas
    
    def load_region_info(self):
        """Load region labels and colors from HCP_labels file."""
        self.region_info = {}
        
        # Try multiple encodings with error handling
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings_to_try:
            try:
                with open(self.hcp_labels_file, 'r', encoding=encoding) as f:
                    # Skip header line if it starts with #
                    first_line = f.readline()
                    if not first_line.startswith('#'):
                        # If it's not a header, process it as a data line
                        self.process_region_line(first_line)
                    
                    # Process remaining lines
                    for line in f:
                        self.process_region_line(line)
                        
                print(f"Loaded information for {len(self.region_info)} regions using {encoding} encoding")
                return  # Successfully loaded the file, exit the function
                    
            except UnicodeDecodeError:
                # If this encoding failed, try the next one
                continue
            except Exception as e:
                # For other errors, print a warning and continue with the next encoding
                print(f"Warning: Error loading region info with {encoding} encoding: {str(e)}")
                continue
        
        # If we get here, none of the encodings worked
        print(f"Warning: Could not load region info from {self.hcp_labels_file} with any encoding")
        print("Will continue analysis with region IDs only (no names or colors)")
    
    def process_region_line(self, line):
        """Process a single line from the HCP labels file."""
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
                    
                    self.region_info[region_id] = {
                        'name': region_name,
                        'color': color
                    }
                    return  # Successfully processed this line
                except ValueError:
                    # If we can't convert the ID to int, try the next delimiter
                    continue
        
        # If we get here, none of the delimiter options worked for this line
        # Just ignore this line and continue
    
    def analyze_by_region(self):
        """Calculate statistics for each atlas region."""
        # Find unique regions in atlas
        unique_regions = np.unique(self.atlas_data)
        # Remove 0 (typically background)
        unique_regions = unique_regions[unique_regions > 0]
        
        print(f"Found {len(unique_regions)} unique regions in the atlas")
        
        results = []
        
        # Process each region
        for i, region_id in enumerate(unique_regions):
            region_id = int(region_id)
            print(f"Processing region {i+1}/{len(unique_regions)}: ID={region_id}", end="\r")
            
            # Create mask for this region
            mask = (self.atlas_data == region_id)
            
            # Extract field values within the mask
            field_values = self.field_data[mask]
            
            if len(field_values) > 0:
                # Calculate statistics
                mean_value = np.mean(field_values)
                max_value = np.max(field_values)
                min_value = np.min(field_values)
                std_value = np.std(field_values)
                median_value = np.median(field_values)
                voxel_count = len(field_values)
                volume_mm3 = voxel_count * np.prod(self.field_img.header.get_zooms()[:3])
                
                # Get region info
                if region_id in self.region_info:
                    region_name = self.region_info[region_id]['name']
                    region_color = self.region_info[region_id]['color']
                else:
                    region_name = f"Unknown_{region_id}"
                    region_color = (0.5, 0.5, 0.5)
                
                # Add to results
                results.append({
                    'RegionID': region_id,
                    'RegionName': region_name,
                    'MeanValue': mean_value,
                    'MaxValue': max_value,
                    'MinValue': min_value,
                    'MedianValue': median_value,
                    'StdValue': std_value,
                    'VoxelCount': voxel_count,
                    'Volume_mm3': volume_mm3,
                    'Color': region_color
                })
        
        print("\nAnalysis complete.")
        
        # Convert to DataFrame
        self.results_df = pd.DataFrame(results)
        self.results_df.sort_values('MeanValue', ascending=False, inplace=True)
        
        return self.results_df
    
    def save_results(self):
        """Save results to CSV file."""
        csv_path = os.path.join(self.output_dir, 'region_stats.csv')
        # Save without color column (not CSV compatible)
        save_df = self.results_df.drop(columns=['Color'])
        save_df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        return csv_path
    
    def analyze_spherical_roi(self, center_coords, radius_mm, is_ras=True):
        """Analyze field within a spherical ROI.
        
        Parameters
        ----------
        center_coords : tuple
            (x, y, z) coordinates of sphere center
        radius_mm : float
            Radius of sphere in millimeters
        is_ras : bool, optional
            If True, center_coords are in RAS coordinates and need to be converted
            If False, center_coords are already in voxel coordinates
            
        Returns
        -------
        dict
            Dictionary with ROI statistics
        """
        # Convert RAS to voxel coordinates if needed
        if is_ras:
            print(f"Converting RAS coordinates {center_coords} to voxel space...")
            voxel_coords = self.ras_to_voxel(center_coords)
            print(f"Converted to voxel coordinates: {voxel_coords}")
        else:
            voxel_coords = center_coords
        
        # Convert radius from mm to voxels
        voxel_sizes = self.field_img.header.get_zooms()[:3]
        radius_voxels = [radius_mm / size for size in voxel_sizes]
        
        # Create spherical mask
        x, y, z = np.indices(self.field_data.shape)
        x_center, y_center, z_center = voxel_coords
        
        # Calculate normalized distance from center
        dist_normalized = np.sqrt(
            ((x - x_center) / radius_voxels[0]) ** 2 +
            ((y - y_center) / radius_voxels[1]) ** 2 +
            ((z - z_center) / radius_voxels[2]) ** 2
        )
        
        # Create mask where distance <= 1.0 (inside sphere)
        mask = dist_normalized <= 1.0
        
        # Check if the mask contains any voxels
        mask_count = np.sum(mask)
        if mask_count == 0:
            print(f"Warning: Sphere mask contains 0 voxels. Check coordinates and radius.")
            # Return zeros for all metrics
            results = {
                'CenterCoords_RAS': center_coords if is_ras else None,
                'CenterCoords_Voxel': voxel_coords,
                'RadiusMM': radius_mm,
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
        
        # Extract field values within mask
        field_values = self.field_data[mask]
        
        # Debug information
        print(f"Mask contains {mask_count} voxels")
        print(f"Field values shape: {field_values.shape}")
        print(f"Field values range: {np.min(field_values)} to {np.max(field_values)}")
        
        # Calculate statistics
        mean_value = np.mean(field_values) if len(field_values) > 0 else 0
        max_value = np.max(field_values) if len(field_values) > 0 else 0
        min_value = np.min(field_values) if len(field_values) > 0 else 0
        median_value = np.median(field_values) if len(field_values) > 0 else 0
        std_value = np.std(field_values) if len(field_values) > 0 else 0
        voxel_count = len(field_values)
        volume_mm3 = voxel_count * np.prod(voxel_sizes)
        
        # Return statistics as dictionary
        results = {
            'CenterCoords_RAS': center_coords if is_ras else None,
            'CenterCoords_Voxel': voxel_coords,
            'RadiusMM': radius_mm,
            'MeanValue': mean_value,
            'MaxValue': max_value,
            'MinValue': min_value,
            'MedianValue': median_value, 
            'StdValue': std_value,
            'VoxelCount': voxel_count,
            'Volume_mm3': volume_mm3
        }
        
        # Save sphere mask for visualization
        sphere_img = nib.Nifti1Image(mask.astype(np.int16), self.field_img.affine)
        mask_path = os.path.join(self.output_dir, f'sphere_mask_x{center_coords[0]}_y{center_coords[1]}_z{center_coords[2]}_r{radius_mm}.nii.gz')
        nib.save(sphere_img, mask_path)
        print(f"Saved sphere mask to {mask_path}")
        
        return results
    
    def ras_to_voxel(self, ras_coords):
        """Convert RAS coordinates to voxel coordinates.
        
        Parameters
        ----------
        ras_coords : tuple or list
            (x, y, z) coordinates in RAS space
            
        Returns
        -------
        tuple
            (x, y, z) coordinates in voxel space
        """
        # Get the image affine matrix
        affine = self.field_img.affine
        
        # Invert the affine matrix
        inv_affine = np.linalg.inv(affine)
        
        # Convert RAS coordinates to homogeneous coordinates
        homogeneous_coords = np.array([ras_coords[0], ras_coords[1], ras_coords[2], 1.0])
        
        # Transform to voxel coordinates
        voxel_coords = np.dot(inv_affine, homogeneous_coords)
        
        # Return the voxel coordinates (without the homogeneous component)
        return tuple(voxel_coords[:3])