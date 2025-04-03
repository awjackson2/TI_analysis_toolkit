#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for integrating Papaya 3D neuroimaging viewer with TI field reports.
"""

import os
import subprocess
import shutil
import traceback

def add_papaya_viewer(html_path, t1_file, overlay_file):
    """
    Add a Papaya interactive 3D viewer to an HTML report.
    
    Parameters
    ----------
    html_path : str
        Path to the HTML report file
    t1_file : str
        Path to T1 reference NIfTI file
    overlay_file : str
        Path to field NIfTI file to visualize
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    # Get the directory of the HTML file
    report_dir = os.path.dirname(os.path.abspath(html_path))
    
    try:
        # Ensure files exist
        if not all(os.path.exists(f) for f in [html_path, t1_file, overlay_file]):
            print(f"WARNING: One or more required files not found")
            return False
        
        # Run the papaya-builder.sh command
        cmd = [
            "bash", 
            "Papaya/papaya-builder.sh", 
            "-images", 
            "../" + t1_file, 
            "../" + overlay_file, 
            "-local"
        ]
        
        print(f"Running Papaya builder: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            print(f"WARNING: Papaya builder failed: {result.stderr}")
            return False
        
        # Check if the papaya files were generated
        papaya_build = os.path.join("Papaya", "build")
        papaya_files = ["index.html", "papaya.js", "papaya.css"]
        print()
        if not all(os.path.exists(os.path.join(papaya_build, f)) for f in papaya_files):
            print(f"WARNING: Not all Papaya files were generated in {papaya_build}")
            return False
    
        # Copy the papaya files to the report directory
        for file in papaya_files:
            src = os.path.join(papaya_build, file)
            dst = os.path.join(report_dir, file)
            shutil.copy2(src, dst)
            print(f"Copied {src} to {dst}")
        
        # Read the report HTML
        with open(html_path, 'r') as f:
            html_content = f.read()
        
        # Generate the Papaya viewer section to add to the report
        papaya_section = """
        <div class="papaya-section">
            <h2>Interactive 3D Visualization</h2>
            <p>View the TI field data in an interactive 3D viewer:</p>
            <div style="width: 100%; height: 600px; border: 1px solid #ddd; margin: 20px 0;">
                <iframe src="index.html" width="100%" height="100%" frameborder="0"></iframe>
            </div>
            <p><a href="index.html" target="_blank" style="display: inline-block; padding: 8px 15px; background-color: #4285f4; color: white; text-decoration: none; border-radius: 4px;">Open in New Tab</a></p>
        </div>
        """
        
        # Insert the Papaya section before the closing body tag
        if "</body>" in html_content:
            html_content = html_content.replace("</body>", f"{papaya_section}</body>")
        else:
            # If no body closing tag, append at the end
            html_content += f"\n{papaya_section}\n"
        
        # Write the updated HTML back to the file
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"Successfully added Papaya viewer to report: {html_path}")
        return True
        
    except Exception as e:
        print(f"WARNING: Failed to add Papaya viewer: {str(e)}")
        traceback.print_exc()
        return False

def add_papaya_comparison(html_path, t1_file, field1_file, field2_file, diff_file=None):
    """
    Add a Papaya interactive 3D viewer to compare multiple field files.
    
    Parameters
    ----------
    html_path : str
        Path to the HTML report file
    t1_file : str
        Path to T1 reference NIfTI file
    field1_file : str
        Path to first field NIfTI file
    field2_file : str
        Path to second field NIfTI file
    diff_file : str, optional
        Path to difference field NIfTI file
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    # Get the directory of the HTML file
    report_dir = os.path.dirname(os.path.abspath(html_path))
    
    try:
        # Ensure files exist
        required_files = [html_path, t1_file, field1_file, field2_file]
        if diff_file:
            required_files.append(diff_file)
            
        if not all(os.path.exists(f) for f in required_files):
            print(f"WARNING: One or more required files not found")
            return False
        
        # Construct image list for papaya-builder
        images = ["../"+t1_file, "../"+field1_file, "../"+field2_file]
        if diff_file:
            images.append("../"+diff_file)
        
        # Run the papaya-builder.sh command
        cmd = ["bash", "Papaya/papaya-builder.sh", "-images"] + images + ["-local"]
        
        print(f"Running Papaya builder: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            print(f"WARNING: Papaya builder failed: {result.stderr}")
            return False
        
        # Check if the papaya files were generated
        papaya_build = os.path.join("Papaya", "build")
        papaya_files = ["index.html", "papaya.js", "papaya.css"]
        
        if not all(os.path.exists(os.path.join(papaya_build, f)) for f in papaya_files):
            print(f"WARNING: Not all Papaya files were generated in {papaya_build}")
            return False
        
        # Copy the papaya files to the report directory
        for file in papaya_files:
            src = os.path.join(papaya_build, file)
            dst = os.path.join(report_dir, file)
            shutil.copy2(src, dst)
            print(f"Copied {src} to {dst}")
        
        # Read the report HTML
        with open(html_path, 'r') as f:
            html_content = f.read()
        
        # Generate the Papaya viewer section to add to the report
        papaya_section = """
        <div class="papaya-section">
            <h2>Interactive Field Comparison</h2>
            <p>Compare the TI field scans in an interactive 3D viewer:</p>
            <p>The viewer contains the following overlays that can be toggled using the controls:</p>
            <ul>
                <li>T1 Reference</li>
                <li>Field 1</li>
                <li>Field 2</li>
                {diff_item}
            </ul>
            <div style="width: 100%; height: 600px; border: 1px solid #ddd; margin: 20px 0;">
                <iframe src="index.html" width="100%" height="100%" frameborder="0"></iframe>
            </div>
            <p><a href="index.html" target="_blank" style="display: inline-block; padding: 8px 15px; background-color: #4285f4; color: white; text-decoration: none; border-radius: 4px;">Open in New Tab</a></p>
        </div>
        """.format(diff_item="<li>Difference Field</li>" if diff_file else "")
        
        # Insert the Papaya section before the closing body tag
        if "</body>" in html_content:
            html_content = html_content.replace("</body>", f"{papaya_section}</body>")
        else:
            # If no body closing tag, append at the end
            html_content += f"\n{papaya_section}\n"
        
        # Write the updated HTML back to the file
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"Successfully added Papaya comparison viewer to report: {html_path}")
        return True
        
    except Exception as e:
        print(f"WARNING: Failed to add Papaya viewer: {str(e)}")
        traceback.print_exc()
        return False

def add_papaya_to_multiple_fields(html_path, t1_file, field_files, labels=None):
    """
    Add a Papaya interactive 3D viewer with multiple field overlays.
    
    Parameters
    ----------
    html_path : str
        Path to the HTML report file
    t1_file : str
        Path to T1 reference NIfTI file
    field_files : list
        List of field NIfTI files to visualize
    labels : list, optional
        List of labels for each field file
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    # Get the directory of the HTML file
    report_dir = os.path.dirname(os.path.abspath(html_path))
    
    try:
        # Ensure files exist
        required_files = [html_path, t1_file] + field_files
        if not all(os.path.exists(f) for f in required_files):
            print(f"WARNING: One or more required files not found")
            return False
        
        # Generate labels if not provided
        if not labels:
            labels = [f"Field {i+1}" for i in range(len(field_files))]
        elif len(labels) != len(field_files):
            print(f"WARNING: Number of labels ({len(labels)}) does not match number of field files ({len(field_files)})")
            labels = [f"Field {i+1}" for i in range(len(field_files))]
        
        # Run the papaya-builder.sh command
        cmd = ["bash", "Papaya/papaya-builder.sh", "-images", t1_file] + field_files + ["-local"]
        
        print(f"Running Papaya builder: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            print(f"WARNING: Papaya builder failed: {result.stderr}")
            return False
        
        # Check if the papaya files were generated
        papaya_build = os.path.join("Papaya", "build")
        papaya_files = ["index.html", "papaya.js", "papaya.css"]
        
        if not all(os.path.exists(os.path.join(papaya_build, f)) for f in papaya_files):
            print(f"WARNING: Not all Papaya files were generated in {papaya_build}")
            return False
        
        # Copy the papaya files to the report directory
        for file in papaya_files:
            src = os.path.join(papaya_build, file)
            dst = os.path.join(report_dir, file)
            shutil.copy2(src, dst)
            print(f"Copied {src} to {dst}")
        
        # Read the report HTML
        with open(html_path, 'r') as f:
            html_content = f.read()
        
        # Generate list items for each field
        field_items = "\n".join([f"<li>{label}</li>" for label in labels])
        
        # Generate the Papaya viewer section to add to the report
        papaya_section = f"""
        <div class="papaya-section">
            <h2>Interactive 3D Visualization</h2>
            <p>View multiple TI field datasets in an interactive 3D viewer:</p>
            <p>The viewer contains the following overlays that can be toggled using the controls:</p>
            <ul>
                <li>T1 Reference</li>
                {field_items}
            </ul>
            <div style="width: 100%; height: 600px; border: 1px solid #ddd; margin: 20px 0;">
                <iframe src="index.html" width="100%" height="100%" frameborder="0"></iframe>
            </div>
            <p><a href="index.html" target="_blank" style="display: inline-block; padding: 8px 15px; background-color: #4285f4; color: white; text-decoration: none; border-radius: 4px;">Open in New Tab</a></p>
        </div>
        """
        
        # Insert the Papaya section before the closing body tag
        if "</body>" in html_content:
            html_content = html_content.replace("</body>", f"{papaya_section}</body>")
        else:
            # If no body closing tag, append at the end
            html_content += f"\n{papaya_section}\n"
        
        # Write the updated HTML back to the file
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"Successfully added Papaya multi-field viewer to report: {html_path}")
        return True
        
    except Exception as e:
        print(f"WARNING: Failed to add Papaya viewer: {str(e)}")
        traceback.print_exc()
        return False