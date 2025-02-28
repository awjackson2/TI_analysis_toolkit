# Temporal Interference (TI) Field Analysis Toolkit

A comprehensive Python toolkit for advanced analysis of Temporal Interference (TI) electric fields in neuroimaging, providing robust capabilities for quantitative assessment across voxel and mesh spaces.

## Key Features

- 🧠 **Multi-Modal Analysis**
  - Voxel-space analysis using atlas-based parcellation
  - Spherical region of interest (ROI) analysis
  - Mesh-space field quantification

- 📊 **Advanced Visualization**
  - Interactive region-based visualizations
  - Slice-based field distribution maps
  - Region-specific histograms
  - Comprehensive HTML reporting

- 🔬 **Flexible Input Support**
  - Supports various neuroimaging formats (NIfTI, Mesh)
  - Compatible with multiple atlases (HCP, DK40, etc.)
  - Handles complex field data structures

## Installation

### Prerequisites
- Python 3.6+
- pip package manager

### Dependencies
```bash
pip install numpy nibabel pandas matplotlib simnibs
```

### Installation Steps
```bash
git clone https://github.com/yourusername/ti-field-analysis.git
cd ti-field-analysis
pip install .
```

## Command-Line Tools

### Voxel-Based Analysis
```bash
python voxel_analysis.py \
    --field TI_max.nii.gz \
    --atlas HCP_parcellation.nii.gz \
    --labels HCP.txt \
    --t1-mni T1_mni.nii.gz \
    --output voxel_results \
    --top-n 20
```

#### Voxel Analysis Options
- `--field`: TI field NIfTI file (required)
- `--atlas`: Atlas parcellation NIfTI file (required)
- `--labels`: Region labels text file (required)
- `--t1-mni`: Optional T1 MNI reference image
- `--output`: Output directory (default: `ti_field_analysis`)
- `--top-n`: Number of top regions to visualize (default: 20)
- `--no-visualizations`: Skip generating visualizations

### Spherical ROI Analysis
```bash
python sphere_analysis.py \
    --field TI_max.nii.gz \
    --coords 80,90,75 \
    --radius 5 \
    --output sphere_results \
    --compare
```

#### Spherical Analysis Options
- `--field`: TI field NIfTI file (required)
- `--coords`: Sphere center coordinates or JSON file with multiple ROIs
- `--radius`: Sphere radius in mm (default: 5.0)
- `--output`: Output directory
- `--compare`: Calculate differential values between ROIs
- `--t1-mni`: Optional T1 MNI reference image
- `--no-visualizations`: Skip generating visualizations

### Coordinate Input Formats
1. Single Coordinate: `x,y,z` (e.g., `80,90,75`)
2. JSON File with Multiple ROIs:
   ```json
   {
     "ROI1": [80, 90, 75],
     "ROI2": [60, 70, 80]
   }
   ```

## Output Files

### Voxel Analysis Outputs
- `region_stats.csv`: Detailed region statistics
- `top_regions_bar.png`: Bar chart of top regions
- `field_and_regions_slices.png`: Field distribution visualization
- `region_histograms.png`: Field value distributions
- `analysis_report.html`: Comprehensive HTML report

### Spherical ROI Outputs
- `sphere_roi_results.csv`: ROI-specific statistics
- `sphere_roi_differentials.csv`: Inter-ROI differential values
- `mean_values.png`: ROI mean value comparison
- `differential_values.png`: Differential value visualization
- Sphere mask NIfTI files

## Examples

### Complex Multi-ROI Analysis
```bash
python sphere_analysis.py \
    --field TI_max.nii.gz \
    --coords rois.json \
    --radius 5 \
    --compare \
    --t1-mni T1_mni.nii.gz \
    --output multi_roi_analysis
```

## Advanced Usage

### Programmatic Access
```python
from ti_field_core import TIFieldAnalyzer
import ti_field_visualization as visu

# Initialize analyzer
analyzer = TIFieldAnalyzer(
    field_nifti='TI_max.nii.gz',
    atlas_nifti='HCP_parcellation.nii.gz',
    hcp_labels_file='HCP.txt',
    output_dir='custom_analysis'
)

# Perform region-based analysis
results_df = analyzer.analyze_by_region()

# Generate custom visualizations
visu.create_bar_chart(analyzer)
visu.generate_report(analyzer)
```

## Extending the Toolkit

- Add new analysis methods by extending `TIFieldAnalyzer`
- Create custom visualizations in `ti_field_visualization.py`
- Develop new command-line tools based on existing scripts

## Troubleshooting

- Ensure input files are in compatible neuroimaging formats
- Check file paths and permissions
- Verify input data dimensions and orientation
- For large files, consider downsampling or using memory-efficient processing

## Citing This Work

If you use this toolkit in your research, please cite:
[Your Publication Details Here]

## License

[Specify License, e.g., MIT, GPL]

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to the repository.