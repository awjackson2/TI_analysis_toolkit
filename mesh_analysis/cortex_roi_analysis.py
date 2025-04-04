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
import datetime

def get_user_input():
    # Valid atlas types
    valid_atlases = ['DK40', 'a2009s', 'HCP_MMP1']
    
    # Ask user to select atlas type
    print("\nSelect Atlas Type:")
    print("1. DK40")
    print("2. a2009s")
    print("3. HCP_MMP1")
    print("Or simply type the atlas name directly (DK40, a2009s, or HCP_MMP1)")
    
    while True:
        atlas_choice = input("\nEnter your choice: ").strip()
        
        # Check if the input is a number
        if atlas_choice.isdigit():
            choice_num = int(atlas_choice)
            if choice_num == 1:
                atlas_type = 'DK40'
                break
            elif choice_num == 2:
                atlas_type = 'a2009s'
                break
            elif choice_num == 3:
                atlas_type = 'HCP_MMP1'
                break
            else:
                print("Invalid choice number. Please enter 1, 2, 3, or the atlas name.")
        
        # Check if the input is a valid atlas name
        elif atlas_choice.upper() in [a.upper() for a in valid_atlases]:
            # Find the actual atlas name with correct case
            for atlas in valid_atlases:
                if atlas.upper() == atlas_choice.upper():
                    atlas_type = atlas
                    break
            break
        else:
            print(f"Invalid atlas. Please choose from: {', '.join(valid_atlases)}")
    
    print(f"Selected atlas: {atlas_type}")
    return atlas_type

def select_regions(atlas):
    """Allows users to select specific regions from the atlas"""
    print("\nAvailable regions in the selected atlas:")
    regions = list(atlas.keys())
    
    # Display all regions with numbers
    for i, region in enumerate(regions, 1):
        print(f"{i}. {region}")
    
    print("\nSelect regions to analyze:")
    print("- Enter comma-separated numbers (e.g., 1,3,5)")
    print("- Enter region names directly, separated by commas")
    print("- Enter 'all' to analyze all regions")
    print("- Enter 'range' to select a range (e.g., 1-10)")
    
    choice = input("\nYour selection: ").strip()
    
    # Handle 'all' option
    if choice.lower() == 'all':
        return regions
    
    # Handle range option
    elif choice.lower() == 'range':
        range_input = input("Enter range (e.g., 1-10): ")
        try:
            start, end = map(int, range_input.split('-'))
            if 1 <= start <= len(regions) and 1 <= end <= len(regions):
                selected_regions = regions[start-1:end]
            else:
                print("Invalid range. Using all regions.")
                return regions
        except:
            print("Invalid format. Using all regions.")
            return regions
    
    # Handle direct input
    else:
        selected_regions = []
        
        # Split by commas
        items = [item.strip() for item in choice.split(',')]
        
        for item in items:
            # Check if item is a number
            if item.isdigit():
                idx = int(item)
                if 1 <= idx <= len(regions):
                    selected_regions.append(regions[idx-1])
                else:
                    print(f"Index {idx} is out of range, ignoring.")
            
            # Check if item is a region name
            elif item in regions:
                selected_regions.append(item)
            else:
                print(f"'{item}' is not a valid region name or index, ignoring.")
        
        if not selected_regions:
            print("No valid regions selected. Using all regions.")
            return regions
    
    return selected_regions

def get_analysis_name():
    """Gets user preference for analysis name"""
    print("\nOutput Options:")
    analysis_name = input("Enter a name for this analysis (press Enter for default name with timestamp): ").strip()
    return analysis_name

## Main Script ##

# Input paths
mesh_path = os.path.join('example_data', 'subject_overlays', 'ernie_TI_Parallel_Horizontal_central.msh')
subject_dir = 'example_data/m2m_ernie'

# Field to analyze
field_name = 'TI_max'

# Get user input for atlas type
atlas_type = get_user_input()

# Get analysis name
analysis_name = get_analysis_name()

# Create timestamp for unique folder naming
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Create output directory structure
if analysis_name:
    output_dir = analysis_name
else:
    output_dir = f'{field_name}_{atlas_type}_{timestamp}'

os.makedirs(output_dir, exist_ok=True)

# Create a dedicated folder for cortex visualizations
cortex_visuals_dir = os.path.join(output_dir, 'cortex_visuals')
os.makedirs(cortex_visuals_dir, exist_ok=True)

# Log analysis parameters to a file
with open(os.path.join(output_dir, 'analysis_info.txt'), 'w') as f:
    f.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Field Name: {field_name}\n")
    f.write(f"Atlas Type: {atlas_type}\n")
    f.write(f"Mesh Path: {mesh_path}\n")
    f.write(f"Subject Directory: {subject_dir}\n")
    if analysis_name:
        f.write(f"Analysis Name: {analysis_name}\n")

# Load the atlas for getting the list of regions
atlas = simnibs.subject_atlas(atlas_type, subject_dir)

# Let user select regions
selected_regions = select_regions(atlas)
print(f"\nSelected {len(selected_regions)} regions for analysis.")

# Update the analysis info file with selected regions
with open(os.path.join(output_dir, 'analysis_info.txt'), 'a') as f:
    f.write(f"\nSelected Regions ({len(selected_regions)}):\n")
    for region in selected_regions:
        f.write(f"- {region}\n")

# Create a list to store results
results = []

# Iterate through selected regions in the atlas
for region_name in selected_regions:
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

# Check if we have any results
if not results:
    print("No valid regions were processed. Check your atlas and region selections.")
    exit(1)

# Convert results to a pandas DataFrame
results_df = pd.DataFrame(results)

# Save results to CSV in the output directory (using standard filename)
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

# Ask user if they want to generate visualizations
visualize = input("\nDo you want to generate visualizations for top regions? (y/n): ").strip().lower()
if visualize != 'y':
    print("Skipping visualizations. Analysis complete!")
    exit(0)

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
    vis_file = os.path.join(cortex_visuals_dir, f"brain_with_{region_name}_ROI.msh")
    if os.path.exists(vis_file):
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