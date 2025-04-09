"""
Region Voxel Distribution Plotter

Extension for the TI Field Analysis Toolkit that generates violin and scatter plots
for voxel values within specified brain regions.

Usage:
    python region_voxel_plot.py --field TI_max.nii.gz --atlas HCP_parcellation.nii.gz 
    --labels HCP.txt --regions 5,10,15 --plot-type violin --output region_plots
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

# Try to import seaborn, but make it optional
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn package not found. Using matplotlib for visualization instead.")
    print("For enhanced visualizations, install seaborn with: pip install seaborn")

# Try to import the TIFieldAnalyzer from ti_field_core
try:
    from ti_field_core import TIFieldAnalyzer
    HAS_TI_FIELD_CORE = True
except ImportError:
    HAS_TI_FIELD_CORE = False
    print("Warning: TIFieldAnalyzer from ti_field_core could not be imported.")
    print("Will use internal methods for processing instead.")

# Attempt to import from the TI field analysis package
try:
    from ti_field_core import TIFieldAnalyzer
except ImportError:
    print("Warning: Could not import TIFieldAnalyzer. Running in standalone mode.")
    TIFieldAnalyzer = None


class RegionVoxelPlotter:
    """Class for plotting voxel values within specific brain regions."""

    def __init__(self, field_file, atlas_file, labels_file, output_dir='region_voxel_plots'):
        """
        Initialize the plotter with input files.
        
        Args:
            field_file (str): Path to the TI field NIfTI file
            atlas_file (str): Path to the atlas parcellation NIfTI file
            labels_file (str): Path to the region labels text file
            output_dir (str): Directory to save output plots
        """
        self.field_file = field_file
        self.atlas_file = atlas_file
        self.labels_file = labels_file
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Load field and atlas data
        print(f"Loading TI field data from {field_file}...")
        self.field_img = nib.load(field_file)
        self.field_data = self.field_img.get_fdata()
        
        # Handle 4D data (common in TI field files)
        if len(self.field_data.shape) == 4:
            print(f"Detected 4D field data with shape {self.field_data.shape}, extracting first volume...")
            self.field_data = self.field_data[:,:,:,0]
            print(f"New field data shape: {self.field_data.shape}")
        
        print(f"Loading atlas data from {atlas_file}...")
        self.atlas_img = nib.load(atlas_file)
        self.atlas_data_orig = self.atlas_img.get_fdata()
        self.atlas_data_orig = np.squeeze(self.atlas_data_orig)
        
        # Check if we have access to TIFieldAnalyzer for consistent alignment
        if HAS_TI_FIELD_CORE:
            print("Using TIFieldAnalyzer.align_atlas_to_field for consistent alignment...")
            # Create a temporary TIFieldAnalyzer object to use its alignment method
            temp_analyzer = TIFieldAnalyzer(
                field_nifti=field_file,
                atlas_nifti=atlas_file,
                hcp_labels_file=labels_file,
                output_dir=output_dir
            )
            # Use the align_atlas_to_field method to get aligned atlas data
            self.atlas_data = temp_analyzer.align_atlas_to_field()
        else:
            # Fall back to our own alignment method
            print("TIFieldAnalyzer not available, using internal alignment method...")
            self.atlas_data = self.align_atlas_to_field()
        
        # Load region labels
        self.region_labels = self._load_region_labels(labels_file)
        
        print(f"Initialized with {len(self.region_labels)} brain regions")

    def _load_region_labels(self, labels_file):
        """
        Load region labels from file.
        
        Args:
            labels_file (str): Path to the region labels file
            
        Returns:
            dict: Mapping of region IDs to region names
        """
        labels_dict = {}
        
        with open(labels_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        # First part is the region ID, rest is the name
                        region_id = int(parts[0])
                        region_name = ' '.join(parts[1:])
                        labels_dict[region_id] = region_name
        
        return labels_dict

    def get_voxel_values_for_region(self, region_id):
        """
        Extract all voxel values for a specific brain region.
        
        Args:
            region_id (int): Region ID in the atlas
            
        Returns:
            numpy.ndarray: Array of voxel values for the region
        """
        # Create mask for the region
        region_mask = (self.atlas_data == region_id)
        
        # Extract field values where the mask is True
        region_values = self.field_data[region_mask]
        
        # Filter out zero or negative values if needed
        region_values = region_values[region_values > 0]
        
        return region_values

    def create_violin_plot(self, region_ids, figsize=(10, 8), kde_bandwidth=0.2):
        """
        Create violin plots for the specified regions.
        
        Args:
            region_ids (list): List of region IDs to plot
            figsize (tuple): Figure size (width, height) in inches
            kde_bandwidth (float): Bandwidth for kernel density estimation
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Collect data for all regions
        region_data = []
        region_names = []
        
        for region_id in region_ids:
            if region_id in self.region_labels:
                values = self.get_voxel_values_for_region(region_id)
                if len(values) > 0:
                    region_data.append(values)
                    region_names.append(f"{region_id}: {self.region_labels[region_id]}")
                else:
                    print(f"Warning: Region {region_id} has no valid voxels")
            else:
                print(f"Warning: Region {region_id} not found in labels")
                
        if not region_data:
            raise ValueError("No valid regions with data to plot")
            
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create violin plot
        parts = ax.violinplot(region_data, points=100, vert=False, widths=0.7, 
                             showmeans=True, showmedians=True, bw_method=kde_bandwidth)
        
        # Customize appearance
        for pc in parts['bodies']:
            pc.set_facecolor('#3182bd')
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
            
        parts['cmeans'].set_color('red')
        parts['cmedians'].set_color('black')
        
        # Set labels and title
        ax.set_yticks(np.arange(1, len(region_names) + 1))
        ax.set_yticklabels(region_names)
        ax.set_xlabel('TI Field Value')
        ax.set_title('Distribution of Voxel Values by Brain Region')
        
        # Add grid lines
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Tight layout
        plt.tight_layout()
        
        return fig

    def create_scatter_plot(self, region_ids, figsize=(12, 10), max_points=5000):
        """
        Create scatter plots for the specified regions.
        
        Args:
            region_ids (list): List of region IDs to plot
            figsize (tuple): Figure size (width, height) in inches
            max_points (int): Maximum number of points to plot per region
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Determine the grid size
        n_regions = len(region_ids)
        cols = min(3, n_regions)
        rows = (n_regions + cols - 1) // cols
        
        # Create figure with gridspec
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(rows, cols, figure=fig)
        
        # Create a plot for each region
        for i, region_id in enumerate(region_ids):
            if region_id in self.region_labels:
                values = self.get_voxel_values_for_region(region_id)
                
                if len(values) > 0:
                    # Calculate row and column
                    row = i // cols
                    col = i % cols
                    
                    # Create subplot
                    ax = fig.add_subplot(gs[row, col])
                    
                    # Sample points if too many
                    if len(values) > max_points:
                        np.random.seed(42)  # For reproducibility
                        indices = np.random.choice(len(values), max_points, replace=False)
                        values = values[indices]
                    
                    # Generate jittered x-coordinates
                    x = np.random.normal(1, 0.04, size=len(values))
                    
                    # Calculate KDE for coloring
                    kde = gaussian_kde(values)
                    density = kde(values)
                    
                    # Create scatter plot
                    scatter = ax.scatter(x, values, c=density, cmap='viridis', 
                                        alpha=0.7, s=10, edgecolor='none')
                    
                    # Add violin plot
                    parts = ax.violinplot([values], positions=[1], vert=True, 
                                         widths=0.5, showmeans=False, showmedians=False)
                    
                    for pc in parts['bodies']:
                        pc.set_facecolor('none')
                        pc.set_edgecolor('black')
                        pc.set_alpha(0.4)
                    
                    # Add statistics
                    ax.axhline(np.median(values), color='red', linestyle='-', alpha=0.8)
                    ax.axhline(np.mean(values), color='blue', linestyle='--', alpha=0.8)
                    
                    # Remove x-axis ticks
                    ax.set_xticks([])
                    
                    # Set labels and title
                    ax.set_ylabel('TI Field Value')
                    ax.set_title(f"{region_id}: {self.region_labels[region_id]}", fontsize=9)
                    
                    # Add legend for statistics
                    ax.text(0.95, 0.05, f"Mean: {np.mean(values):.2f}\nMedian: {np.median(values):.2f}", 
                           transform=ax.transAxes, fontsize=8, va='bottom', ha='right',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                else:
                    print(f"Warning: Region {region_id} has no valid voxels")
            else:
                print(f"Warning: Region {region_id} not found in labels")
        
        # Add a colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_label('Density')
        
        # Add an overall title
        fig.suptitle('Distribution of Voxel Values by Brain Region', fontsize=14)
        
        # Adjust layout
        fig.tight_layout(rect=[0, 0, 0.9, 0.95])
        
        return fig

    def create_box_plot(self, region_ids, figsize=(10, 8)):
        """
        Create box plots for the specified regions.
        
        Args:
            region_ids (list): List of region IDs to plot
            figsize (tuple): Figure size (width, height) in inches
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Collect data for all regions
        data_dict = {}
        
        for region_id in region_ids:
            if region_id in self.region_labels:
                values = self.get_voxel_values_for_region(region_id)
                if len(values) > 0:
                    region_name = f"{region_id}: {self.region_labels[region_id]}"
                    data_dict[region_name] = values
                else:
                    print(f"Warning: Region {region_id} has no valid voxels")
            else:
                print(f"Warning: Region {region_id} not found in labels")
                
        if not data_dict:
            raise ValueError("No valid regions with data to plot")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot using seaborn if available, otherwise use matplotlib
        if HAS_SEABORN:
            # Convert to DataFrame for seaborn
            df = pd.DataFrame({region: pd.Series(values) for region, values in data_dict.items()})
            
            # Create box plot
            sns.boxplot(data=df, orient='h', palette='Set3', ax=ax)
            
            # Add swarm plot for individual points
            sns.stripplot(data=df, orient='h', color='black', alpha=0.5, size=3, ax=ax)
        else:
            # Use matplotlib's boxplot
            data_list = list(data_dict.values())
            region_names = list(data_dict.keys())
            
            # Create box plot
            bp = ax.boxplot(data_list, vert=False, patch_artist=True, 
                           labels=region_names)
            
            # Customize appearance
            for box in bp['boxes']:
                box.set(facecolor='lightblue', alpha=0.7)
            for whisker in bp['whiskers']:
                whisker.set(color='#333333', linewidth=1.5, linestyle=':')
            for cap in bp['caps']:
                cap.set(color='#333333', linewidth=1.5)
            for median in bp['medians']:
                median.set(color='red', linewidth=2)
            for flier in bp['fliers']:
                flier.set(marker='o', markerfacecolor='#FF9999', alpha=0.7, markersize=4)
                
            # Add individual points with jitter for each region
            for i, (name, values) in enumerate(data_dict.items()):
                # Limit the number of points to display
                max_points = 300
                if len(values) > max_points:
                    np.random.seed(42)  # For reproducibility
                    indices = np.random.choice(len(values), max_points, replace=False)
                    values_sample = values[indices]
                else:
                    values_sample = values
                
                # Create jittered y positions (centered at position i+1)
                y_positions = np.random.normal(i+1, 0.05, size=len(values_sample))
                
                # Scatter plot for individual points
                ax.scatter(values_sample, y_positions, alpha=0.3, s=3, c='black')
        
        # Set labels and title
        ax.set_xlabel('TI Field Value')
        ax.set_title('Distribution of Voxel Values by Brain Region')
        
        # Add grid lines
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Tight layout
        plt.tight_layout()
        
        return fig

    def plot_region_voxels(self, region_ids, plot_type='violin', save=True, show=True):
        """
        Create and save plots for the specified regions.
        
        Args:
            region_ids (list): List of region IDs to plot
            plot_type (str): Type of plot ('violin', 'scatter', or 'box')
            save (bool): Whether to save the plot to file
            show (bool): Whether to display the plot
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        print(f"Creating {plot_type} plot for regions: {region_ids}")
        
        if plot_type == 'violin':
            fig = self.create_violin_plot(region_ids)
        elif plot_type == 'scatter':
            fig = self.create_scatter_plot(region_ids)
        elif plot_type == 'box':
            fig = self.create_box_plot(region_ids)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
        
        if save:
            # Create filename from regions and plot type
            regions_str = '_'.join(str(r) for r in region_ids)
            filename = f"{plot_type}_plot_regions_{regions_str}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            print(f"Saving plot to {filepath}")
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
            
        return fig

    def compare_all_top_regions(self, n_top=10, metric='mean', plot_type='violin'):
        """
        Compare the top N regions based on a metric.
        
        Args:
            n_top (int): Number of top regions to compare
            metric (str): Metric to use for ranking ('mean', 'median', 'max')
            plot_type (str): Type of plot to create
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Calculate metrics for all regions
        region_metrics = {}
        
        for region_id in self.region_labels:
            values = self.get_voxel_values_for_region(region_id)
            if len(values) > 0:
                if metric == 'mean':
                    region_metrics[region_id] = np.mean(values)
                elif metric == 'median':
                    region_metrics[region_id] = np.median(values)
                elif metric == 'max':
                    region_metrics[region_id] = np.max(values)
                else:
                    raise ValueError(f"Unknown metric: {metric}")
        
        # Sort regions by metric and get top N
        top_regions = sorted(region_metrics.keys(), 
                             key=lambda r: region_metrics[r], 
                             reverse=True)[:n_top]
        
        # Create plot
        return self.plot_region_voxels(top_regions, plot_type=plot_type)

    def export_region_values_to_csv(self, region_ids):
        """
        Export voxel values for selected regions to CSV files.
        
        Args:
            region_ids (list): List of region IDs to export
        """
        for region_id in region_ids:
            if region_id in self.region_labels:
                values = self.get_voxel_values_for_region(region_id)
                if len(values) > 0:
                    # Create filename
                    region_name = self.region_labels[region_id].replace(' ', '_')
                    filename = f"region_{region_id}_{region_name}_voxels.csv"
                    filepath = os.path.join(self.output_dir, filename)
                    
                    # Save to CSV
                    df = pd.DataFrame({'voxel_value': values})
                    df.to_csv(filepath, index=False)
                    print(f"Saved {len(values)} voxel values for region {region_id} to {filepath}")
                else:
                    print(f"Warning: Region {region_id} has no valid voxels to export")
            else:
                print(f"Warning: Region {region_id} not found in labels")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Region Voxel Distribution Plotter')
    
    parser.add_argument('--field', required=True, help='TI field NIfTI file')
    parser.add_argument('--atlas', required=True, help='Atlas parcellation NIfTI file')
    parser.add_argument('--labels', required=True, help='Region labels text file')
    parser.add_argument('--regions', help='Comma-separated list of region IDs to plot')
    parser.add_argument('--top-n', type=int, default=10, 
                        help='Number of top regions to plot (if regions not specified)')
    parser.add_argument('--metric', choices=['mean', 'median', 'max'], default='mean',
                       help='Metric to use for ranking top regions')
    parser.add_argument('--plot-type', choices=['violin', 'scatter', 'box'], default='violin',
                       help='Type of plot to create')
    parser.add_argument('--output', default='region_voxel_plots', help='Output directory')
    parser.add_argument('--export-csv', action='store_true', help='Export voxel values to CSV')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Initialize plotter
    plotter = RegionVoxelPlotter(
        field_file=args.field,
        atlas_file=args.atlas,
        labels_file=args.labels,
        output_dir=args.output
    )
    
    # Determine regions to plot
    if args.regions:
        # Parse region IDs from argument
        region_ids = [int(r) for r in args.regions.split(',')]
        
        # Create plot for specified regions
        plotter.plot_region_voxels(
            region_ids=region_ids,
            plot_type=args.plot_type,
            show=not args.no_show
        )
        
        # Export to CSV if requested
        if args.export_csv:
            plotter.export_region_values_to_csv(region_ids)
    else:
        # Compare top regions
        plotter.compare_all_top_regions(
            n_top=args.top_n,
            metric=args.metric,
            plot_type=args.plot_type
        )


if __name__ == '__main__':
    main()