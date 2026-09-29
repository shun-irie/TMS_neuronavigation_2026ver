#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:39:29 2026

@author: Shun Irie, Ph.D. @ Dokkyo Medical University
License information should be referred to License.txt

This script was used for analysis of MEP mapping study written by Koyanagi et al. 
(corresponding to Shun Irie, Ph.D. @ Dokkyo Medical University).

This script is available for non-commercial purpose only.
"""
# Loading modules
import numpy as np
import tkinter.filedialog as filedialog
import xml.etree.ElementTree as ET
import pyvista as pv
from scipy.spatial import cKDTree
import networkx as nx
import sys
import time
import os
import nibabel as nib
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths anchored to this Python file (independent of the current working dir)
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

BRAIN_MODEL_PATH = PROJECT_ROOT / "volume data" / "MNI152_T1_1mm_brain.nii_world.ply"
BRODMANN_ATLAS_PATH = PROJECT_ROOT / "nidata" / "brodmann.nii.gz"


# Loading reference coordinates
mni_landmark = {
    "Nz": np.array([0.25    ,  81.49998 , -43.999985]),
    "A2": np.array([79. , -23. , -46.5]),
    "A1": np.array([-78.5, -23. , -46.5]),
    "Iz": np.array([0.25, -114 ,   -36.5]),
    "Cz": np.array([0.25     ,  -22.47035, 98.50108])
} #MNI coordinates for standard brain-skin model (MNI152)

# Functions

def load_obj_data() -> dict | None:
    """
    Load file paths associated with a 3D head model.

    Returns
    -------
    output : dict or None
        Dictionary containing file paths for the 3D head mesh
        and landmark coordinate data.
        Returns None if the file dialog is canceled.
    """

    fname = filedialog.askopenfilename(
        title="Load PLY file",
        filetypes=[("PLY files", "*.ply"), ("All files", "*.*")]
    )

    base, _ = os.path.splitext(fname)
    fname2 = base + "_picked_points.pp"

    return {
        "ply": fname,
        "pp": fname2
    }


def parse_xml(file_path: str) -> dict:
    """
    Parse landmark coordinate data from an XML file.

    Parameters
    ----------
    file_path : str
        Path to the XML file containing landmark coordinates.

    Returns
    -------
    data : dict
        Dictionary containing landmark names and their
        corresponding 3D coordinates as NumPy arrays.
    """

    data = {}

    try:

        tree = ET.parse(file_path)
        root = tree.getroot()

        for point in root.findall('.//point'):

            data[point.get('name')] = np.array([
                float(point.get('x')),
                float(point.get('y')),
                float(point.get('z'))
            ])

    except ET.ParseError:

        print("XML parse error.")
        return None

    return data

def compute_Cz_on_equal_distance_plane(
    landmarks: dict,
    skin: pv.PolyData,
    show: bool = True,
    rank_int: int = 1
) -> tuple[dict, np.ndarray]:
    """
    Compute 10–20 system midline landmarks from the intersection curve
    between the scalp mesh and the plane equidistant from A1 and A2.

    Nz and Iz are projected onto the same connected intersection component.
    Intermediate midline landmarks are then determined by arc-length division
    along the selected curve.

    Parameters
    ----------
    landmarks : dict
