# cortex_extractor.py
import os
import numpy as np
import simnibs

def extract_cortex_roi(mesh_path, atlas_type, subject_dir, target_region, field_name='TI_max', 
                      output_dir=None, save_output=False):
    """
    Extract field values from a specific cortical region.
    
    Parameters
    ----------
    mesh_path : str
        Path to the mesh file
    atlas_type : str
        Type of atlas to use (e.g., 'HCP_MMP1', 'DK40')
    subject_dir : str
        Path to the subject directory
    target_region : str
        Name of the target region in the atlas
    field_name : str, optional
        Name of the field to extract (default: 'TI_max')
    output_dir : str, optional
        Directory to save output files (default: current directory)
    save_output : bool, optional
        Whether to save the output mesh and options file (default: False)
        
    Returns
    -------
    dict
        Dictionary containing the roi_mask, gm_surf, min_value, max_value, and mean_value
    """
    # 1. Load the original mesh and atlas
    gm_surf = simnibs.read_msh(mesh_path)
    
    # 2. Load the atlas
    atlas = simnibs.subject_atlas(atlas_type, subject_dir)
    
    # 3. Select the target region
    roi_mask = atlas[target_region]
    
    # 4. Calculate the min and max values in the specific cortex region
    field_values_in_roi = gm_surf.field[field_name].value[roi_mask]
    min_value = np.min(field_values_in_roi)
    max_value = np.max(field_values_in_roi)
    
    # Calculate mean value using node areas for proper averaging
    node_areas = gm_surf.nodes_areas()
    mean_value = np.average(gm_surf.field[field_name].value[roi_mask], weights=node_areas[roi_mask])
    
    # 5. Create a new field with field values only in ROI (zeros elsewhere)
    masked_field = np.zeros(gm_surf.nodes.nr)
    # Copy the field values for nodes in our ROI
    masked_field[roi_mask] = gm_surf.field[field_name].value[roi_mask]
    
    # 6. Add this as a new field to the original mesh
    gm_surf.add_node_field(masked_field, 'ROI_field')
    
    if save_output:
        # Create the output directory if it doesn't exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_filename = os.path.join(output_dir, f"brain_with_{target_region}_ROI.msh")
        else:
            output_filename = f"brain_with_{target_region}_ROI.msh"
            
        # 7. Save the modified original mesh
        gm_surf.write(output_filename)
        
        # 8. Create the .msh.opt file with custom color map and alpha settings
        with open(f"{output_filename}.opt", 'w') as f:
            f.write(f"""
// Make View[1] (ROI_field) visible with custom colormap
View[1].Visible = 1;
View[1].ColormapNumber = 1;  // Use the first predefined colormap
View[1].RangeType = 2;       // Custom range
View[1].CustomMin = 0;   // Specific minimum value for this cortex
View[1].CustomMax = {max_value};   // Specific maximum value for this cortex
View[1].ShowScale = 1;       // Show the color scale

// Add alpha/transparency based on value
View[1].ColormapAlpha = 1;
View[1].ColormapAlphaPower = 0.08;
""")
        
        print(f"Created mesh with ROI field: {output_filename}")
        print(f"Visualization settings saved to: {output_filename}.opt")
    
    return {
        'roi_mask': roi_mask,
        'gm_surf': gm_surf,
        'min_value': min_value,
        'max_value': max_value,
        'mean_value': mean_value
    }

# If this script is run directly
if __name__ == "__main__":
    # Example usage
    mesh_path = os.path.join('example_data', 'subject_overlays', 'ernie_TI_Parallel_Horizontal_central.msh')
    atlas_type = 'HCP_MMP1'
    subject_dir = 'example_data/m2m_ernie'
    target_region = "rh.V1"  # Change to your region of interest
    
    result = extract_cortex_roi(mesh_path, atlas_type, subject_dir, target_region, save_output=True)
    
    print(f"Field values in {target_region}:")
    print(f"Minimum value: {result['min_value']:.6f}")
    print(f"Maximum value: {result['max_value']:.6f}")
    print(f"Mean value: {result['mean_value']:.6f}")