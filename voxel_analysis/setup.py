#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="ti_field_analyzer",
    version="1.0.0",
    description="Tools for analyzing TI field data in voxel and mesh space",
    author="TI Field Analysis Team",
    author_email="example@example.com",
    py_modules=["ti_field_core", "ti_field_visualization"],
    scripts=["voxel_analysis.py", "sphere_analysis.py", "cortext_analysis.py"],
    install_requires=[
        "numpy",
        "nibabel",
        "pandas",
        "matplotlib",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
