#!/bin/bash

# Temporal Interference (TI) Field Analysis Toolkit
# Interactive menu script

# Set colors for better readability
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Directory settings - Update these to match your new organization
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR=$(dirname "$SCRIPT_DIR")
BIN_DIR="$BASE_DIR/bin"
DATA_DIR="$BASE_DIR/data"
OUTPUT_DIR="$BASE_DIR/output"

# Create directories if they don't exist
mkdir -p "$OUTPUT_DIR"

# Function to display the banner
display_banner() {
    clear
    echo -e "${BLUE}=========================================================${NC}"
    echo -e "${BLUE}    Temporal Interference (TI) Field Analysis Toolkit    ${NC}"
    echo -e "${BLUE}=========================================================${NC}"
    echo ""
}

# Function to display the main menu
display_menu() {
    echo -e "${GREEN}Please select an analysis option:${NC}"
    echo "1) Voxel-Based Analysis"
    echo "2) Spherical ROI Analysis"
    echo "3) Cortex ROI Analysis"
    echo "4) View Previous Results"
    echo "5) Help & Documentation"
    echo "0) Exit"
    echo ""
    echo -e "${YELLOW}Enter your choice [0-5]:${NC} "
}

# Function to run voxel-based analysis
run_voxel_analysis() {
    display_banner
    echo -e "${GREEN}Voxel-Based Analysis${NC}"
    echo -e "${BLUE}----------------------------------${NC}"
    
    # Get input parameters
    read -p "Enter TI field NIfTI file: " field_file
    read -p "Enter atlas parcellation NIfTI file: " atlas_file
    read -p "Enter region labels text file: " labels_file
    read -p "Enter T1 MNI reference image (optional, press Enter to skip): " t1_file
    read -p "Enter output directory name: " output_name
    read -p "Enter number of top regions to visualize (default: 20): " top_n
    
    # Set defaults if empty
    output_name=${output_name:-"voxel_results_$(date +%Y%m%d_%H%M%S)"}
    top_n=${top_n:-20}
    
    # Build command
    cmd="python $BIN_DIR/voxel_analysis.py --field $field_file --atlas $atlas_file --labels $labels_file"
    
    # Add optional parameters
    if [ ! -z "$t1_file" ]; then
        cmd="$cmd --t1-mni $t1_file"
    fi
    
    cmd="$cmd --output $OUTPUT_DIR/$output_name --top-n $top_n"
    
    # Ask for visualization
    read -p "Generate visualizations? (y/n, default: y): " vis_option
    vis_option=${vis_option:-"y"}
    if [ "$vis_option" = "n" ]; then
        cmd="$cmd --no-visualizations"
    fi
    
    # Execute command
    echo -e "\n${YELLOW}Executing:${NC} $cmd"
    eval $cmd
    
    echo -e "\n${GREEN}Analysis complete. Results saved to:${NC} $OUTPUT_DIR/$output_name"
    read -p "Press Enter to continue..."
}

# Function to run spherical ROI analysis
run_sphere_analysis() {
    display_banner
    echo -e "${GREEN}Spherical ROI Analysis${NC}"
    echo -e "${BLUE}----------------------------------${NC}"
    
    # Get input parameters
    read -p "Enter TI field NIfTI file: " field_file
    read -p "Enter ROI coordinates (x,y,z or JSON file path): " coords
    read -p "Enter sphere radius in mm (default: 5.0): " radius
    read -p "Enter T1 MNI reference image (optional, press Enter to skip): " t1_file
    read -p "Enter output directory name: " output_name
    
    # Set defaults if empty
    radius=${radius:-5.0}
    output_name=${output_name:-"sphere_results_$(date +%Y%m%d_%H%M%S)"}
    
    # Build command
    cmd="python $BIN_DIR/sphere_analysis.py --field $field_file --coords $coords --radius $radius"
    
    # Add optional parameters
    if [ ! -z "$t1_file" ]; then
        cmd="$cmd --t1-mni $t1_file"
    fi
    
    cmd="$cmd --output $OUTPUT_DIR/$output_name"
    
    # Ask for comparison
    read -p "Calculate differential values between ROIs? (y/n, default: n): " compare_option
    compare_option=${compare_option:-"n"}
    if [ "$compare_option" = "y" ]; then
        cmd="$cmd --compare"
    fi
    
    # Ask for visualization
    read -p "Generate visualizations? (y/n, default: y): " vis_option
    vis_option=${vis_option:-"y"}
    if [ "$vis_option" = "n" ]; then
        cmd="$cmd --no-visualizations"
    fi
    
    # Execute command
    echo -e "\n${YELLOW}Executing:${NC} $cmd"
    eval $cmd
    
    echo -e "\n${GREEN}Analysis complete. Results saved to:${NC} $OUTPUT_DIR/$output_name"
    read -p "Press Enter to continue..."
}

# Function to run cortex ROI analysis
run_cortex_analysis() {
    display_banner
    echo -e "${GREEN}Cortex ROI Analysis${NC}"
    echo -e "${BLUE}----------------------------------${NC}"
    
    # Get input parameters
    read -p "Enter TI field NIfTI file: " field_file
    
    # Atlas selection
    echo -e "\nSelect atlas type:"
    echo "1) HCP (Human Connectome Project)"
    echo "2) DK40 (Desikan-Killiany)"
    echo "3) Schaefer (Schaefer 2018)"
    echo "4) Custom"
    read -p "Enter atlas type [1-4]: " atlas_type
    
    case $atlas_type in
        1) atlas_name="HCP" ;;
        2) atlas_name="DK40" ;;
        3) atlas_name="Schaefer" ;;
        4) 
            read -p "Enter custom atlas name: " atlas_name
            ;;
        *) 
            echo "Invalid selection, using HCP as default."
            atlas_name="HCP"
            ;;
    esac
    
    # Region selection
    read -p "Enter region name or ID to analyze: " region_id
    
    # Output directory
    read -p "Enter output directory name: " output_name
    
    # Set defaults if empty
    output_name=${output_name:-"cortex_results_$(date +%Y%m%d_%H%M%S)"}
    
    # Build command
    cmd="python $BIN_DIR/cortex_roi_analysis.py --field $field_file --atlas $atlas_name --region $region_id"
    
    # Add output directory
    cmd="$cmd --output $OUTPUT_DIR/$output_name"
    
    # Ask for additional options
    read -p "Number of top regions to highlight (default: 10): " top_n
    top_n=${top_n:-10}
    cmd="$cmd --top-n $top_n"
    
    # Execute command
    echo -e "\n${YELLOW}Executing:${NC} $cmd"
    eval $cmd
    
    echo -e "\n${GREEN}Analysis complete. Results saved to:${NC} $OUTPUT_DIR/$output_name"
    read -p "Press Enter to continue..."
}

# Function to view previous results
view_results() {
    display_banner
    echo -e "${GREEN}View Previous Results${NC}"
    echo -e "${BLUE}----------------------------------${NC}"
    
    # List available result directories
    echo "Available result directories:"
    ls -lt "$OUTPUT_DIR" | grep -v "total" | awk '{print NR") "$9" ("$6" "$7")"}'
    
    echo ""
    read -p "Enter the number of the result directory to view (or 0 to cancel): " result_num
    
    if [ "$result_num" = "0" ]; then
        return
    fi
    
    # Get the selected directory name
    selected_dir=$(ls -lt "$OUTPUT_DIR" | grep -v "total" | awk '{print $9}' | sed -n "${result_num}p")
    
    if [ -z "$selected_dir" ]; then
        echo "Invalid selection."
        read -p "Press Enter to continue..."
        return
    fi
    
    # Check for report file
    if [ -f "$OUTPUT_DIR/$selected_dir/analysis_report.html" ]; then
        echo "Opening HTML report..."
        if command -v xdg-open &> /dev/null; then
            xdg-open "$OUTPUT_DIR/$selected_dir/analysis_report.html"
        elif command -v open &> /dev/null; then
            open "$OUTPUT_DIR/$selected_dir/analysis_report.html"
        else
            echo "Cannot open HTML file automatically. Please open manually at:"
            echo "$OUTPUT_DIR/$selected_dir/analysis_report.html"
        fi
    else
        echo "No HTML report found. Listing available files:"
        ls -l "$OUTPUT_DIR/$selected_dir"
    fi
    
    read -p "Press Enter to continue..."
}

# Function to display help
show_help() {
    display_banner
    echo -e "${GREEN}Help & Documentation${NC}"
    echo -e "${BLUE}----------------------------------${NC}"
    
    echo "This interactive tool helps you run various analyses on Temporal Interference (TI) fields:"
    echo ""
    echo "1) Voxel-Based Analysis - Analyze TI fields using atlas-based parcellation"
    echo "   Required inputs: TI field NIfTI, atlas parcellation NIfTI, region labels file"
    echo ""
    echo "2) Spherical ROI Analysis - Analyze specific regions using spherical ROIs"
    echo "   Required inputs: TI field NIfTI, coordinates (x,y,z or JSON file)"
    echo ""
    echo "3) Cortex ROI Analysis - Analyze TI fields on cortical surface using atlas-based regions"
    echo "   Required inputs: TI field NIfTI, atlas type, region name/ID"
    echo ""
    echo "4) View Previous Results - Browse and view previously generated results"
    echo ""
    echo "For detailed documentation, refer to the README.md file or visit the project repository."
    echo ""
    read -p "Press Enter to continue..."
}

# Main program loop
while true; do
    display_banner
    display_menu
    read choice
    
    case $choice in
        1) run_voxel_analysis ;;
        2) run_sphere_analysis ;;
        3) run_cortex_analysis ;;
        4) view_results ;;
        5) show_help ;;
        0) 
            echo "Exiting. Thank you for using the TI Field Analysis Toolkit!"
            exit 0
            ;;
        *) 
            echo "Invalid option. Please try again."
            read -p "Press Enter to continue..."
            ;;
    esac
done