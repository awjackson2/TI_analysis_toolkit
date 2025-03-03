#!/bin/bash

# TI Analysis Toolkit Interactive CLI
# This script provides an interactive menu-driven interface to all the analysis tools

# Default paths 
DATA_DIR="./data"
OUTPUT_DIR="./results"
# Colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to display header
show_header() {
    clear
    echo -e "${BLUE}======================================================${NC}"
    echo -e "${BLUE}             TI Analysis Toolkit Interface            ${NC}"
    echo -e "${BLUE}======================================================${NC}"
    echo ""
}

# Function to scan for .nii.gz files in data directory
scan_nifti_files() {
    # Create an array to store the .nii.gz files
    nifti_files=()
    
    # Check if data directory exists
    if [ ! -d "$DATA_DIR" ]; then
        echo -e "${RED}Error: Data directory not found: $DATA_DIR${NC}"
        return 1
    fi
    
    # Loop through files in data directory
    for file in "$DATA_DIR"/*.nii; do
        if [ -f "$file" ]; then
            nifti_files+=("$file")
        fi
    done
    
    # Check if any .nii.gz files were found
    if [ ${#nifti_files[@]} -eq 0 ]; then
        echo -e "${RED}Error: No .nii files found in $DATA_DIR${NC}"
        return 1
    fi
    
    return 0
}

# Function to find HCP atlas and labels - ENHANCED to allow manual T1 selection
find_hcp_files() {
    hcp_atlas=""
    hcp_labels=""
    t1_mni=""
    
    # Look for HCP atlas file
    for file in "$DATA_DIR"/*parcellation*.nii.gz "$DATA_DIR"/*atlas*.nii "$DATA_DIR"/HCP*.nii; do
        if [ -f "$file" ]; then
            hcp_atlas="$file"
            break
        fi
    done
    
    # Look for HCP labels file
    for file in "$DATA_DIR"/*.txt; do
        if [ -f "$file" ] && grep -q "HCP" "$file" 2>/dev/null; then
            hcp_labels="$file"
            break
        fi
    done
    
    # Look for T1 MNI file
    for file in "$DATA_DIR"/*T1*.nii "$DATA_DIR"/*mni*.nii "$DATA_DIR"/*brain*.nii; do
        if [ -f "$file" ]; then
            t1_mni="$file"
            break
        fi
    done
}

# Function to look for ROI JSON file
find_roi_json() {
    roi_json=""
    for file in "$DATA_DIR"/*.json; do
        if [ -f "$file" ]; then
            roi_json="$file"
            break
        fi
    done
}

# Function to select a file from a list
select_file() {
    local title="$1"
    shift
    local files=("$@")
    
    echo -e "${YELLOW}$title${NC}"
    echo ""
    
    for i in "${!files[@]}"; do
        echo "$((i+1)). $(basename "${files[$i]}")"
    done
    
    echo ""
    echo -e "${YELLOW}Enter selection (1-${#files[@]}):${NC}"
    read -r selection
    
    # Validate input
    if ! [[ "$selection" =~ ^[0-9]+$ ]] || [ "$selection" -lt 1 ] || [ "$selection" -gt "${#files[@]}" ]; then
        echo -e "${RED}Invalid selection. Please try again.${NC}"
        return 1
    fi
    
    selected_file="${files[$((selection-1))]}"
    echo -e "${GREEN}Selected: $(basename "$selected_file")${NC}"
    echo ""
    return 0
}

# Check if Python and required scripts exist
check_requirements() {
    # Check if Python is available
    if ! command -v python &> /dev/null; then
        echo -e "${RED}Error: Python is not installed or not in PATH${NC}"
        return 1
    fi
    
    # Check if analysis scripts exist
    local scripts=("voxel_analysis.py" "sphere_analysis.py" "cortex_analysis.py" "compare_field_scans.py")
    for script in "${scripts[@]}"; do
        if [ ! -f "$script" ]; then
            echo -e "${RED}Error: Required script not found: $script${NC}"
            return 1
        fi
    done
    
    return 0
}

# Function for voxel analysis
run_voxel_analysis() {
    show_header
    echo -e "${BLUE}Voxel-based Analysis${NC}"
    echo ""
    
    # Scan for NIFTI files
    scan_nifti_files
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to find NIfTI files. Make sure they are in $DATA_DIR${NC}"
        echo ""
        read -n 1 -s -r -p "Press any key to return to main menu..."
        return
    fi
    
    # Find HCP files
    find_hcp_files
    
    # Select field file
    echo -e "${YELLOW}Select TI field file:${NC}"
    select_file "Available field files:" "${nifti_files[@]}"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to select field file${NC}"
        echo ""
        read -n 1 -s -r -p "Press any key to return to main menu..."
        return
    fi
    field_file="$selected_file"
    
    # Select atlas file if not found automatically
    if [ -z "$hcp_atlas" ]; then
        echo -e "${YELLOW}Select atlas file:${NC}"
        select_file "Available NIfTI files:" "${nifti_files[@]}"
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to select atlas file${NC}"
            echo ""
            read -n 1 -s -r -p "Press any key to return to main menu..."
            return
        fi
        hcp_atlas="$selected_file"
    else
        echo -e "${GREEN}Using atlas file: $(basename "$hcp_atlas")${NC}"
        echo ""
    fi
    
    # Handle labels file
    if [ -z "$hcp_labels" ]; then
        # Look for text files in data directory
        txt_files=()
        for file in "$DATA_DIR"/*.txt; do
            if [ -f "$file" ]; then
                txt_files+=("$file")
            fi
        done
        
        if [ ${#txt_files[@]} -eq 0 ]; then
            echo -e "${RED}Error: No .txt files found for labels in $DATA_DIR${NC}"
            echo ""
            read -n 1 -s -r -p "Press any key to return to main menu..."
            return
        fi
        
        echo -e "${YELLOW}Select labels file:${NC}"
        select_file "Available text files:" "${txt_files[@]}"
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to select labels file${NC}"
            echo ""
            read -n 1 -s -r -p "Press any key to return to main menu..."
            return
        fi
        hcp_labels="$selected_file"
    else
        echo -e "${GREEN}Using labels file: $(basename "$hcp_labels")${NC}"
        echo ""
    fi
    
    # Prompt for output directory
    echo -e "${YELLOW}Enter output directory name (default: voxel_analysis):${NC}"
    read -r output_name
    if [ -z "$output_name" ]; then
        output_name="voxel_analysis"
    fi
    output_dir="$OUTPUT_DIR/$output_name"
    
    # Prompt for number of top regions to visualize
    echo -e "${YELLOW}Enter number of top regions to visualize (default: 20):${NC}"
    read -r top_n
    if [ -z "$top_n" ] || ! [[ "$top_n" =~ ^[0-9]+$ ]]; then
        top_n=20
    fi
    
    # Ask if T1 MNI should be used (if found)
    if [ -n "$t1_mni" ]; then
        echo -e "${YELLOW}Use T1 MNI reference ($(basename "$t1_mni")) as background for visualization?${NC}"
        echo -e "${YELLOW}This will overlay the field data on anatomical MRI for better context. (y/n, default: y):${NC}"
        read -r use_t1_response
        if [ -z "$use_t1_response" ] || [[ "$use_t1_response" =~ ^[Yy]$ ]]; then
            use_t1=true
        fi
    else
        # Try to let the user select a T1 file manually
        select_t1_mni
        if [ $? -eq 0 ] && [ -n "$t1_mni" ]; then
            use_t1=true
        fi
    fi
    
    # Ask if visualizations should be generated
    echo -e "${YELLOW}Generate visualizations? (y/n, default: y):${NC}"
    read -r gen_viz
    no_viz=false
    if [[ "$gen_viz" =~ ^[Nn]$ ]]; then
        no_viz=true
    fi
    
    # Build and run command
    echo ""
    echo -e "${BLUE}Running voxel analysis...${NC}"
    echo ""
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Build command
    cmd="python voxel_analysis.py --field \"$field_file\" --atlas \"$hcp_atlas\" --labels \"$hcp_labels\" --output \"$output_dir\" --top-n $top_n"
    
    if [ "$use_t1" = true ]; then
        cmd="$cmd --t1-mni \"$t1_mni\""
    fi
    
    if [ "$no_viz" = true ]; then
        cmd="$cmd --no-visualizations"
    fi
    
    echo "Command: $cmd"
    
    # Run the command and capture both stdout and stderr
    output=$(eval $cmd 2>&1)
    exit_code=$?
    
    echo "$output"
    
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo -e "${RED}Error: Analysis failed with exit code $exit_code${NC}"
    else
        echo ""
        echo -e "${GREEN}Analysis complete. Results saved to: $output_dir${NC}"
    fi
    
    echo ""
    read -n 1 -s -r -p "Press any key to return to main menu..."
    return
}

# Function for sphere analysis
run_sphere_analysis() {
    show_header
    echo -e "${BLUE}Spherical ROI Analysis${NC}"
    echo ""
    
    # Scan for NIFTI files
    scan_nifti_files
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to find NIfTI files. Make sure they are in $DATA_DIR${NC}"
        echo ""
        read -n 1 -s -r -p "Press any key to return to main menu..."
        return
    fi
    
    # Find HCP files and ROI JSON
    find_hcp_files
    find_roi_json
    
    # Select field file
    echo -e "${YELLOW}Select TI field file:${NC}"
    select_file "Available field files:" "${nifti_files[@]}"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to select field file${NC}"
        echo ""
        read -n 1 -s -r -p "Press any key to return to main menu..."
        return
    fi
    field_file="$selected_file"
    
    # Handle coordinates
    use_json=false
    coords=""
    
    if [ -n "$roi_json" ]; then
        echo -e "${YELLOW}ROI JSON file found: $(basename "$roi_json")${NC}"
        echo -e "${YELLOW}Use this file for multiple ROIs? (y/n, default: y):${NC}"
        read -r use_json_response
        if [ -z "$use_json_response" ] || [[ "$use_json_response" =~ ^[Yy]$ ]]; then
            use_json=true
            coords="$roi_json"
        fi
    fi
    
    if [ "$use_json" = false ]; then
        echo -e "${YELLOW}Enter coordinates as 'x,y,z' (e.g., 80,90,75):${NC}"
        read -r coords
        
        # Validate input
        if ! [[ "$coords" =~ ^[0-9]+,[0-9]+,[0-9]+$ ]]; then
            echo -e "${RED}Invalid coordinates format. Please use format: x,y,z${NC}"
            echo ""
            read -n 1 -s -r -p "Press any key to return to main menu..."
            return
        fi
    fi
    
    # Prompt for radius
    echo -e "${YELLOW}Enter sphere radius in mm (default: 5.0):${NC}"
    read -r radius
    if [ -z "$radius" ] || ! [[ "$radius" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        radius=5.0
    fi
    
    # Prompt for output directory
    echo -e "${YELLOW}Enter output directory name (default: sphere_analysis):${NC}"
    read -r output_name
    if [ -z "$output_name" ]; then
        output_name="sphere_analysis"
    fi
    output_dir="$OUTPUT_DIR/$output_name"
    
    # Ask if T1 MNI should be used (if found)
    use_t1=false
    if [ -n "$t1_mni" ]; then
        echo -e "${YELLOW}Use T1 MNI reference for visualization? (y/n, default: y):${NC}"
        read -r use_t1_response
        if [ -z "$use_t1_response" ] || [[ "$use_t1_response" =~ ^[Yy]$ ]]; then
            use_t1=true
        fi
    fi
    
    # Ask if multiple ROIs should be compared (only if using JSON)
    compare=false
    if [ "$use_json" = true ]; then
        echo -e "${YELLOW}Compare multiple ROIs? (y/n, default: y):${NC}"
        read -r compare_response
        if [ -z "$compare_response" ] || [[ "$compare_response" =~ ^[Yy]$ ]]; then
            compare=true
        fi
    fi
    
    # Ask if visualizations should be generated
    echo -e "${YELLOW}Generate visualizations? (y/n, default: y):${NC}"
    read -r gen_viz
    no_viz=false
    if [[ "$gen_viz" =~ ^[Nn]$ ]]; then
        no_viz=true
    fi
    
    # Build and run command
    echo ""
    echo -e "${BLUE}Running spherical ROI analysis...${NC}"
    echo ""
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Build command
    cmd="python sphere_analysis.py --field \"$field_file\" --coords \"$coords\" --radius $radius --output \"$output_dir\""
    
    if [ "$use_t1" = true ]; then
        cmd="$cmd --t1-mni \"$t1_mni\""
    fi
    
    if [ "$compare" = true ]; then
        cmd="$cmd --compare"
    fi
    
    if [ "$no_viz" = true ]; then
        cmd="$cmd --no-visualizations"
    fi
    
    echo "Command: $cmd"
    
    # Run the command and capture both stdout and stderr
    output=$(eval $cmd 2>&1)
    exit_code=$?
    
    echo "$output"
    
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo -e "${RED}Error: Analysis failed with exit code $exit_code${NC}"
    else
        echo ""
        echo -e "${GREEN}Analysis complete. Results saved to: $output_dir${NC}"
    fi
    
    echo ""
    read -n 1 -s -r -p "Press any key to return to main menu..."
    return
}

# Function for cortex analysis
run_cortex_analysis() {
    show_header
    echo -e "${BLUE}Cortical Region Analysis${NC}"
    echo ""
    
    # Scan for NIFTI files
    scan_nifti_files
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to find NIfTI files. Make sure they are in $DATA_DIR${NC}"
        echo ""
        read -n 1 -s -r -p "Press any key to return to main menu..."
        return
    fi
    
    # Find HCP files
    find_hcp_files
    
    # Select field file
    echo -e "${YELLOW}Select TI field file:${NC}"
    select_file "Available field files:" "${nifti_files[@]}"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to select field file${NC}"
        echo ""
        read -n 1 -s -r -p "Press any key to return to main menu..."
        return
    fi
    field_file="$selected_file"
    
    # Select atlas file if not found automatically
    if [ -z "$hcp_atlas" ]; then
        echo -e "${YELLOW}Select atlas file:${NC}"
        select_file "Available NIfTI files:" "${nifti_files[@]}"
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to select atlas file${NC}"
            echo ""
            read -n 1 -s -r -p "Press any key to return to main menu..."
            return
        fi
        hcp_atlas="$selected_file"
    else
        echo -e "${GREEN}Using atlas file: $(basename "$hcp_atlas")${NC}"
        echo ""
    fi
    
    # Handle labels file
    if [ -z "$hcp_labels" ]; then
        # Look for text files in data directory
        txt_files=()
        for file in "$DATA_DIR"/*.txt; do
            if [ -f "$file" ]; then
                txt_files+=("$file")
            fi
        done
        
        if [ ${#txt_files[@]} -eq 0 ]; then
            echo -e "${RED}Error: No .txt files found for labels in $DATA_DIR${NC}"
            echo ""
            read -n 1 -s -r -p "Press any key to return to main menu..."
            return
        fi
        
        echo -e "${YELLOW}Select labels file:${NC}"
        select_file "Available text files:" "${txt_files[@]}"
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to select labels file${NC}"
            echo ""
            read -n 1 -s -r -p "Press any key to return to main menu..."
            return
        fi
        hcp_labels="$selected_file"
    else
        echo -e "${GREEN}Using labels file: $(basename "$hcp_labels")${NC}"
        echo ""
    fi
    
    # Enter regions to analyze
    echo -e "${YELLOW}Enter region names or IDs to analyze (space-separated, use quotes for multi-word regions):${NC}"
    echo -e "${YELLOW}Example: \"Left Precentral\" \"Right Precentral\" 20 21${NC}"
    read -r regions
    
    if [ -z "$regions" ]; then
        echo -e "${RED}Error: No regions specified${NC}"
        echo ""
        read -n 1 -s -r -p "Press any key to return to main menu..."
        return
    fi
    
    # Prompt for output directory
    echo -e "${YELLOW}Enter output directory name (default: cortex_analysis):${NC}"
    read -r output_name
    if [ -z "$output_name" ]; then
        output_name="cortex_analysis"
    fi
    output_dir="$OUTPUT_DIR/$output_name"
    
    # Ask if T1 MNI should be used (if found)
    use_t1=false
    if [ -n "$t1_mni" ]; then
        echo -e "${YELLOW}Use T1 MNI reference for visualization? (y/n, default: y):${NC}"
        read -r use_t1_response
        if [ -z "$use_t1_response" ] || [[ "$use_t1_response" =~ ^[Yy]$ ]]; then
            use_t1=true
        fi
    fi
    
    # Ask if regions should be compared
    echo -e "${YELLOW}Compare multiple regions? (y/n, default: y):${NC}"
    read -r compare_response
    compare=false
    if [ -z "$compare_response" ] || [[ "$compare_response" =~ ^[Yy]$ ]]; then
        compare=true
    fi
    
    # Ask if visualizations should be generated
    echo -e "${YELLOW}Generate visualizations? (y/n, default: y):${NC}"
    read -r gen_viz
    no_viz=false
    if [[ "$gen_viz" =~ ^[Nn]$ ]]; then
        no_viz=true
    fi
    
    # Build and run command
    echo ""
    echo -e "${BLUE}Running cortical region analysis...${NC}"
    echo ""
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Build command
    cmd="python cortex_analysis.py --field \"$field_file\" --atlas \"$hcp_atlas\" --labels \"$hcp_labels\" --output \"$output_dir\" --regions $regions"
    
    if [ "$use_t1" = true ]; then
        cmd="$cmd --t1-mni \"$t1_mni\""
    fi
    
    if [ "$compare" = true ]; then
        cmd="$cmd --compare"
    fi
    
    if [ "$no_viz" = true ]; then
        cmd="$cmd --no-visualizations"
    fi
    
    echo "Command: $cmd"
    
    # Run the command and capture both stdout and stderr
    output=$(eval $cmd 2>&1)
    exit_code=$?
    
    echo "$output"
    
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo -e "${RED}Error: Analysis failed with exit code $exit_code${NC}"
    else
        echo ""
        echo -e "${GREEN}Analysis complete. Results saved to: $output_dir${NC}"
    fi
    
    echo ""
    read -n 1 -s -r -p "Press any key to return to main menu..."
    return
}

# Function for comparing field scans
run_compare_analysis() {
    show_header
    echo -e "${BLUE}Field Comparison Analysis${NC}"
    echo ""
    
    # Scan for NIFTI files
    scan_nifti_files
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to find NIfTI files. Make sure they are in $DATA_DIR${NC}"
        echo ""
        read -n 1 -s -r -p "Press any key to return to main menu..."
        return
    fi
    
    # Find HCP files
    find_hcp_files
    
    # Check if we have at least 2 NIFTI files
    if [ ${#nifti_files[@]} -lt 2 ]; then
        echo -e "${RED}Error: Need at least 2 .nii.gz files for comparison${NC}"
        echo ""
        read -n 1 -s -r -p "Press any key to return to main menu..."
        return
    fi
    
    # Select first field file
    echo -e "${YELLOW}Select FIRST field file:${NC}"
    select_file "Available field files:" "${nifti_files[@]}"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to select first field file${NC}"
        echo ""
        read -n 1 -s -r -p "Press any key to return to main menu..."
        return
    fi
    field1_file="$selected_file"
    
    # Select second field file (excluding the first one)
    nifti_files_2=()
    for file in "${nifti_files[@]}"; do
        if [ "$file" != "$field1_file" ]; then
            nifti_files_2+=("$file")
        fi
    done
    
    echo -e "${YELLOW}Select SECOND field file:${NC}"
    select_file "Available field files:" "${nifti_files_2[@]}"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to select second field file${NC}"
        echo ""
        read -n 1 -s -r -p "Press any key to return to main menu..."
        return
    fi
    field2_file="$selected_file"
    
    # Select atlas file if not found automatically
    if [ -z "$hcp_atlas" ]; then
        echo -e "${YELLOW}Select atlas file:${NC}"
        select_file "Available NIfTI files:" "${nifti_files[@]}"
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to select atlas file${NC}"
            echo ""
            read -n 1 -s -r -p "Press any key to return to main menu..."
            return
        fi
        hcp_atlas="$selected_file"
    else
        echo -e "${GREEN}Using atlas file: $(basename "$hcp_atlas")${NC}"
        echo ""
    fi
    
    # Handle labels file
    if [ -z "$hcp_labels" ]; then
        # Look for text files in data directory
        txt_files=()
        for file in "$DATA_DIR"/*.txt; do
            if [ -f "$file" ]; then
                txt_files+=("$file")
            fi
        done
        
        if [ ${#txt_files[@]} -eq 0 ]; then
            echo -e "${RED}Error: No .txt files found for labels in $DATA_DIR${NC}"
            echo ""
            read -n 1 -s -r -p "Press any key to return to main menu..."
            return
        fi
        
        echo -e "${YELLOW}Select labels file:${NC}"
        select_file "Available text files:" "${txt_files[@]}"
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to select labels file${NC}"
            echo ""
            read -n 1 -s -r -p "Press any key to return to main menu..."
            return
        fi
        hcp_labels="$selected_file"
    else
        echo -e "${GREEN}Using labels file: $(basename "$hcp_labels")${NC}"
        echo ""
    fi
    
    # Ask if specific regions should be compared
    echo -e "${YELLOW}Compare specific regions? (y/n, default: n - compare all regions):${NC}"
    read -r specific_regions
    regions=""
    if [[ "$specific_regions" =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Enter region names or IDs to compare (space-separated, use quotes for multi-word regions):${NC}"
        echo -e "${YELLOW}Example: \"Left Precentral\" \"Right Precentral\" 20 21${NC}"
        read -r regions
    fi
    
    # Prompt for output directory
    echo -e "${YELLOW}Enter output directory name (default: compare_analysis):${NC}"
    read -r output_name
    if [ -z "$output_name" ]; then
        output_name="compare_analysis"
    fi
    output_dir="$OUTPUT_DIR/$output_name"
    
    # Ask if T1 MNI should be used (if found)
    use_t1=false
    if [ -n "$t1_mni" ]; then
        echo -e "${YELLOW}Use T1 MNI reference for visualization? (y/n, default: y):${NC}"
        read -r use_t1_response
        if [ -z "$use_t1_response" ] || [[ "$use_t1_response" =~ ^[Yy]$ ]]; then
            use_t1=true
        fi
    fi
    
    # Prompt for number of top regions to visualize
    echo -e "${YELLOW}Enter number of top regions to visualize (default: 20):${NC}"
    read -r top_n
    if [ -z "$top_n" ] || ! [[ "$top_n" =~ ^[0-9]+$ ]]; then
        top_n=20
    fi
    
    # Ask if visualizations should be generated
    echo -e "${YELLOW}Generate visualizations? (y/n, default: y):${NC}"
    read -r gen_viz
    no_viz=false
    if [[ "$gen_viz" =~ ^[Nn]$ ]]; then
        no_viz=true
    fi
    
    # Build and run command
    echo ""
    echo -e "${BLUE}Running field comparison analysis...${NC}"
    echo ""
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Build command
    cmd="python compare_field_scans.py --field1 \"$field1_file\" --field2 \"$field2_file\" --atlas \"$hcp_atlas\" --labels \"$hcp_labels\" --output \"$output_dir\" --top-n $top_n"
    
    if [ -n "$regions" ]; then
        cmd="$cmd --regions $regions"
    fi
    
    if [ "$use_t1" = true ]; then
        cmd="$cmd --t1-mni \"$t1_mni\""
    fi
    
    if [ "$no_viz" = true ]; then
        cmd="$cmd --no-visualizations"
    fi
    
    echo "Command: $cmd"
    
    # Run the command and capture both stdout and stderr
    output=$(eval $cmd 2>&1)
    exit_code=$?
    
    echo "$output"
    
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo -e "${RED}Error: Analysis failed with exit code $exit_code${NC}"
    else
        echo ""
        echo -e "${GREEN}Analysis complete. Results saved to: $output_dir${NC}"
    fi
    
    echo ""
    read -n 1 -s -r -p "Press any key to return to main menu..."
    return
}

# Add this helper function to let users manually select a T1 MNI file
select_t1_mni() {
    if [ -z "$T1" ]; then
        echo -e "${YELLOW}No T1 MNI reference found automatically. Would you like to select one? (y/n, default: n):${NC}"
        read -r select_t1_response
        if [[ "$select_t1_response" =~ ^[Yy]$ ]]; then
            # Scan for potential T1/MNI files
            t1_candidates=()
            for file in "$DATA_DIR"/*.nii.gz "$DATA_DIR"/*.nii; do
                if [ -f "$file" ]; then
                    t1_candidates+=("$file")
                fi
            done
            
            if [ ${#t1_candidates[@]} -eq 0 ]; then
                echo -e "${RED}No NIfTI files found for T1 reference${NC}"
                return 1
            fi
            
            echo -e "${YELLOW}Select T1 MNI reference file:${NC}"
            select_file "Available NIfTI files:" "${t1_candidates[@]}"
            if [ $? -eq 0 ]; then
                t1_mni="$selected_file"
                echo -e "${GREEN}Selected T1 MNI reference: $(basename "$t1_mni")${NC}"
                echo ""
                return 0
            else
                return 1
            fi
        fi
    else
        return 0
    fi
}

# Main menu
main_menu() {
    while true; do
        show_header
        echo -e "${YELLOW}Main Menu${NC}"
        echo ""
        echo "1. Voxel-based Analysis"
        echo "2. Spherical ROI Analysis"
        echo "3. Cortical Region Analysis"
        echo "4. Compare Field Scans"
        echo "5. Exit"
        echo ""
        echo -e "${YELLOW}Enter your choice (1-5):${NC}"
        read -r choice
        
        case $choice in
            1) run_voxel_analysis ;;
            2) run_sphere_analysis ;;
            3) run_cortex_analysis ;;
            4) run_compare_analysis ;;
            5) 
                echo ""
                echo -e "${GREEN}Exiting TI Analysis Toolkit. Goodbye!${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid choice. Please try again.${NC}"
                sleep 1
                ;;
        esac
    done
}

# Check requirements
check_requirements
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to satisfy requirements. Exiting.${NC}"
    exit 1
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}Error: Data directory not found: $DATA_DIR${NC}"
    echo -e "${YELLOW}Creating data directory...${NC}"
    mkdir -p "$DATA_DIR"
    echo -e "${GREEN}Data directory created. Please place your data files in $DATA_DIR${NC}"
    echo ""
    read -n 1 -s -r -p "Press any key to continue..."
fi

# Create results directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo -e "${GREEN}Created output directory: $OUTPUT_DIR${NC}"
fi

# Start the main menu
main_menu