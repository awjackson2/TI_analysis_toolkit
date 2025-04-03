'''
ROI analysis of the electric field from a simulation using an atlas.
Calculates the mean and max electric field in all gray matter ROIs defined using an atlas 
and saves results to a CSV file.
'''

import os
import numpy as np
import simnibs
import pandas as pd
from cortex_extractor import extract_cortex_roi

## Input ##
mesh_path = os.path.join('example_data', 'subject_overlays', 'ernie_TI_Parallel_Horizontal_central.msh')
atlas_type = 'HCP_MMP1'
subject_dir = 'example_data/m2m_ernie'

# Field to analyze
field_name = 'TI_max'

# Create output directory structure
output_dir = f'{field_name}_{atlas_type}_results'
os.makedirs(output_dir, exist_ok=True)

# Create a dedicated folder for cortex visualizations
cortex_visuals_dir = os.path.join(output_dir, 'cortex_visuals')
os.makedirs(cortex_visuals_dir, exist_ok=True)

# Load the atlas for getting the list of regions
atlas = simnibs.subject_atlas(atlas_type, subject_dir)

# Create a list to store results
results = []

# Iterate through all regions in the atlas
for region_name in atlas.keys():
    print(f"Processing region: {region_name}")
    
    try:
        # Use the extract_cortex_roi function to get the field values for this region
        roi_data = extract_cortex_roi(
            mesh_path=mesh_path,
            atlas_type=atlas_type,
            subject_dir=subject_dir,
            target_region=region_name,
            field_name=field_name,
            output_dir=cortex_visuals_dir,  # Save to the dedicated visualizations folder
            save_output=False  # Don't save individual region files by default
        )
        
        # Check if the region exists in this subject (has any values)
        if np.any(roi_data['roi_mask']):
            # Store results using values from the extraction
            results.append({
                'region_name': region_name,
                f'mean_{field_name}': roi_data['mean_value'],
                f'max_{field_name}': roi_data['max_value'],
                'min_value': roi_data['min_value']
            })
            
            print(f'Region {region_name}: mean {field_name} = {roi_data["mean_value"]:.6f}, max = {roi_data["max_value"]:.6f}')
    except Exception as e:
        print(f"Error processing {region_name}: {str(e)}")

# Convert results to a pandas DataFrame
results_df = pd.DataFrame(results)

# Save results to CSV in the output directory
output_file = os.path.join(output_dir, 'roi_analysis_results.csv')
results_df.to_csv(output_file, index=False)
print(f'Results saved to {output_file}')

# Get top 10 regions by mean field value
top_mean = results_df.sort_values(by=f'mean_{field_name}', ascending=False).head(10)
print("\nTop 10 regions by mean field value:")
for idx, row in top_mean.iterrows():
    print(f"{row['region_name']}: {row[f'mean_{field_name}']:.6f}")

# Get top 10 regions by max field value
top_max = results_df.sort_values(by=f'max_{field_name}', ascending=False).head(10)
print("\nTop 10 regions by max field value:")
for idx, row in top_max.iterrows():
    print(f"{row['region_name']}: {row[f'max_{field_name}']:.6f}")

# Save top regions to text file
with open(os.path.join(output_dir, 'top_regions.txt'), 'w') as f:
    f.write(f"Top 10 regions by mean {field_name}:\n")
    for idx, row in top_mean.iterrows():
        f.write(f"{row['region_name']}: {row[f'mean_{field_name}']:.6f}\n")
    
    f.write(f"\nTop 10 regions by max {field_name}:\n")
    for idx, row in top_max.iterrows():
        f.write(f"{row['region_name']}: {row[f'max_{field_name}']:.6f}\n")

print(f"Top regions saved to {os.path.join(output_dir, 'top_regions.txt')}")

# Generate visualizations for top regions
print("\nGenerating visualizations for top regions...")

# Create visualization for top regions by mean
print("Generating top regions by mean field value...")
for idx, row in top_mean.head(5).iterrows():
    region_name = row['region_name']
    print(f"Creating visualization for {region_name}")
    
    try:
        # Generate and save visualization for this region
        extract_cortex_roi(
            mesh_path=mesh_path,
            atlas_type=atlas_type,
            subject_dir=subject_dir,
            target_region=region_name,
            field_name=field_name,
            output_dir=cortex_visuals_dir,
            save_output=True
        )
    except Exception as e:
        print(f"  Error creating visualization for {region_name}: {str(e)}")

# Create visualization for top regions by max (if not already created)
print("\nGenerating top regions by max field value...")
for idx, row in top_max.head(5).iterrows():
    region_name = row['region_name']
    # Skip if already created (may overlap with top mean regions)
    mesh_path = os.path.join(cortex_visuals_dir, f"brain_with_{region_name}_ROI.msh")
    if os.path.exists(mesh_path):
        print(f"Visualization for {region_name} already exists, skipping")
        continue
        
    print(f"Creating visualization for {region_name}")
    
    try:
        # Generate and save visualization for this region
        extract_cortex_roi(
            mesh_path=mesh_path,
            atlas_type=atlas_type,
            subject_dir=subject_dir,
            target_region=region_name,
            field_name=field_name,
            output_dir=cortex_visuals_dir,
            save_output=True
        )
    except Exception as e:
        print(f"  Error creating visualization for {region_name}: {str(e)}")

print(f"Visualizations saved to: {cortex_visuals_dir}")
print("Analysis complete!")