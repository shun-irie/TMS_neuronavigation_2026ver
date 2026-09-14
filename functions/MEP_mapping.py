#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:18:09 2026

@author: Shun Irie, Ph.D. @ Dokkyo Medical University
License information should be referred to License.txt

This script was used for analysis of MEP mapping study written by Koyanagi et al. 
(corresponding to Shun Irie, Ph.D. @ Dokkyo Medical University).

This script is available for non-commercial purpose only.
"""
# Set Path
from pathlib import Path
import sys
import tkinter as tk

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Landmarks
landmark_list = {
    "600": "Nz",
    "601": "_Cz",
    "602": "A1",
    "603": "A2"
}

# Import module
import process_3d as p3d
import re
import json
import chardet
import numpy as np
import pandas as pd
import tkinter.filedialog as tf
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator
from sklearn.decomposition import PCA
import os
from scipy.spatial import ConvexHull
import nibabel as nib
from matplotlib.path import Path as MplPath

# Define functions
def converted_to_dict(df, i):
    """
    Convert a DataFrame row containing fragmented JSON-like text
    into a Python dictionary object.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing JSON-like text data.

    i : int
        Row index to be converted.

    Returns
    -------
    data : dict or list or None
        Parsed JSON object.
        Returns None if JSON parsing fails.
    """

    # Concatenate text columns into a single raw string
    raw_text = ''.join(df.iloc[i, 2:].astype(str).values)

    # Remove unnecessary characters and formatting artifacts
    txt = (
        raw_text.replace("nan", "")
                .replace("\\0A", "")
                .replace("\n", "")
                .replace("\r", "")
                .strip()
    )

    # Extract the JSON array portion
    start = txt.find('[')
    end = txt.rfind(']')

    if start != -1 and end != -1:
        txt = txt[start:end + 1]

    # Insert commas between adjacent JSON objects
    txt = re.sub(r'}\s*{', '},{', txt)

    # Insert missing commas before JSON keys
    txt = re.sub(
        r'([-0-9.eE]+)\s+("ID"|"x"|"y"|"z")',
        r'\1, \2',
        txt
    )

    # Fix duplicated quotation marks
    txt = txt.replace('""', '"')

    # Ensure the text starts and ends as a JSON array
    if not txt.startswith('['):
        txt = '[' + txt

    if not txt.endswith(']'):
        txt += ']'

    try:
        data = json.loads(txt)
        return data

    except json.JSONDecodeError as e:

        print("JSON parsing error:", e)
        print(txt[:300])

        return None

def read_csv_with_nan(fname):
    with open(fname, 'rb') as f:
        raw = f.read()

    enc = chardet.detect(raw)['encoding']
    print(f"detected encoding: {enc}")

    text = raw.decode(enc, errors='replace')
    rows = [r.split('\t') if '\t' in r else r.split(',') for r in text.splitlines()]
    max_len = max(len(r) for r in rows)
    rows = [r + [np.nan] * (max_len - len(r)) for r in rows]
    return pd.DataFrame(rows)

def extract_data(df):
    ind = [i * 1000 for i in range(int(len(df) / 1000))]
    dataset = [converted_to_dict(df, i) for i in ind]
    array = np.array(df.iloc[:, 1], dtype=float)
    MEPs = [array[i * 1000:i * 1000 + 1000] for i in range(len(ind))]
    MEP_amp = [np.max(i[300:400]) - np.min(i[300:400]) for i in MEPs]
    return {"dataset": dataset, "MEPs": MEPs, "MEP_amp": MEP_amp}

def calculate_mean_loc_from_ID(
    locations,
    idx: int,
    ignored
):
    """
    Compute the mean 3D coordinate for a specified object ID.

    Parameters
    ----------
    locations : list
        Nested list containing dictionaries with 3D coordinates
        and object IDs.

    idx : int
        Target object ID used for coordinate extraction.

    ignored : array-like
        Indices of coordinates to exclude before averaging.

    Returns
    -------
    mean_coords : np.ndarray
        Mean 3D coordinate after excluding the specified indices.
    """

    # Extract coordinates corresponding to the specified ID
    coords = np.array([
        [d["x"], d["y"], d["z"]]
        for sublist in locations
        for d in sublist
        if d["ID"] == idx
    ])

    # Remove ignored coordinates
    coords_filtered = np.delete(coords, ignored, axis=0)

    # Compute the mean coordinate
    mean_coords = np.mean(coords_filtered, axis=0)

    return mean_coords

def extract_coords_from_ID(
    locations,
    idx: int,
    ignored
):
    """
    Extract 3D coordinates corresponding to a specified object ID.

    Parameters
    ----------
    locations : list
        Nested list containing dictionaries with 3D coordinates
        and object IDs.

    idx : int
        Target object ID used for coordinate extraction.

    ignored : array-like
        Indices of coordinates to exclude from the output.

    Returns
    -------
    coords_filtered : np.ndarray
        Extracted 3D coordinates after removing the specified indices.
    """

    # Extract coordinates corresponding to the specified ID
    coords = np.array([
        [d["x"], d["y"], d["z"]]
        for sublist in locations
        for d in sublist
        if d["ID"] == idx
    ])

    # Remove ignored coordinates
    return np.delete(coords, ignored, axis=0)

def mask_inside_hull(mesh_points, obs_xyz):
    """
    Determine which mesh points lie inside the convex hull
    defined by the observed points after projection onto a 2D PCA plane.

    Parameters
    ----------
    mesh_points : array-like, shape (N, 3)
        Three-dimensional coordinates of the mesh points to be tested.
        Each row represents an [x, y, z] coordinate.

    obs_xyz : array-like, shape (M, 3)
        Three-dimensional coordinates of the observed points used to define
        the convex hull. The points are projected onto a two-dimensional
        plane using PCA before constructing the convex hull.

    Returns
    -------
    inside : ndarray of bool, shape (N,)
        Boolean array indicating whether each mesh point lies inside
        the convex hull of the observed points.
        True indicates that the point is inside the hull, and False
        indicates that it is outside.

        If fewer than three observed points are provided, all mesh points
        are returned as True because a 2D convex hull cannot be constructed.
    """
    mesh_points = np.asarray(mesh_points, float)
    obs_xyz = np.asarray(obs_xyz, float)

    if len(obs_xyz) < 3:
        return np.ones(len(mesh_points), dtype=bool)

    # PCAで2Dへ
    pca = PCA(n_components=2)
    obs_2d = pca.fit_transform(obs_xyz)
    mesh_2d = pca.transform(mesh_points)

    # convex hull
    hull = ConvexHull(obs_2d)
    poly = obs_2d[hull.vertices]

    path = MplPath(poly)
    inside = path.contains_points(mesh_2d)

    return inside

def pca_2d_interp_fast(
    mesh_points: np.ndarray,
    obs_xyz: np.ndarray,
    obs_amp: np.ndarray,
    max_dist: float = 30.0
) -> np.ndarray:
    """
    Project 3D point coordinates into 2D using PCA and perform linear interpolation.

    Points outside the convex hull are filled using the nearest observed values.
    Points that are too far from the observed points are set to NaN.

    Parameters
    ----------
    mesh_points : np.ndarray
        Mesh point coordinates with shape (N, 3).

    obs_xyz : np.ndarray
        Observed point coordinates with shape (M, 3).

    obs_amp : np.ndarray
        Observed values corresponding to obs_xyz, with shape (M,).

    max_dist : float, optional
        Maximum allowed distance from the nearest observed point.
        Mesh points farther than this distance are set to NaN.
        The default is 30.0.

    Returns
    -------
    vals : np.ndarray
        Interpolated values for mesh_points, with shape (N,).
    """

    mesh_points = np.asarray(mesh_points, dtype=float)
    obs_xyz = np.asarray(obs_xyz, dtype=float)
    obs_amp = np.asarray(obs_amp, dtype=float)

    valid = np.isfinite(obs_xyz).all(axis=1) & np.isfinite(obs_amp)
    obs_xyz = obs_xyz[valid]
    obs_amp = obs_amp[valid]

    if len(obs_xyz) == 0:
        return np.full(len(mesh_points), np.nan, dtype=float)

    if len(obs_xyz) == 1:
        out = np.full(len(mesh_points), np.nan, dtype=float)
        tree = cKDTree(obs_xyz)
        dist, idx = tree.query(mesh_points, k=1)
        near = dist <= max_dist
        out[near] = obs_amp[idx[near]]
        return out

    if len(obs_xyz) == 2:
        tree = cKDTree(obs_xyz)
        dist, idx = tree.query(mesh_points, k=1)
        out = np.full(len(mesh_points), np.nan, dtype=float)
        near = dist <= max_dist
        out[near] = obs_amp[idx[near]]
        return out

    pca = PCA(n_components=2)
    obs_2d = pca.fit_transform(obs_xyz)
    mesh_2d = pca.transform(mesh_points)

    interp = LinearNDInterpolator(obs_2d, obs_amp, fill_value=np.nan)
    vals = interp(mesh_2d[:, 0], mesh_2d[:, 1])
    vals = np.asarray(vals, dtype=float)

    # Fill points outside the convex hull using nearest observed values
    tree_obs = cKDTree(obs_xyz)
    dist3d, idx_nn = tree_obs.query(mesh_points, k=1)

    nan_mask = ~np.isfinite(vals)
    vals[nan_mask] = obs_amp[idx_nn[nan_mask]]

    # Mask points that are too far from observed points
    vals[dist3d > max_dist] = np.nan

    # Force original observed values onto their nearest mesh vertices
    mesh_tree = cKDTree(mesh_points)
    _, idx_force = mesh_tree.query(obs_xyz, k=1)
    vals[idx_force] = obs_amp

    return vals

def export_xyz_format(xyz, out_path="output.txt"):
    """
    Export 3D coordinates to a tab-delimited text file.

    Parameters
    ----------
    xyz : numpy.ndarray, shape (N, 3)
        Array containing the X, Y, and Z coordinates of N points.

    out_path : str, optional
        Path of the output text file. The default is "output.txt".

    Output Format
    -------------
    Each row is written in the following format:

        X    Y    Z    1    1    S{i}

    For the last point, the fourth column is set to 2:

        X    Y    Z    2    1    S{i}

    Point labels are numbered sequentially starting from S1.
    """
    N = xyz.shape[0]
    with open(out_path, "w") as f:
        for i in range(N):
            x, y, z = xyz[i]
            if i == N - 1:
                line = f"{x}\t{y}\t{z}\t2\t1\tS{i+1}\n"
            else:
                line = f"{x}\t{y}\t{z}\t1\t1\tS{i+1}\n"
            f.write(line)
    

    print("Saved:", out_path)

def weighted_centroid(points, weights):
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = np.isfinite(points).all(axis=1) & np.isfinite(weights)
    points = points[valid]
    weights = weights[valid]

    if len(points) == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)

    if np.sum(weights) == 0:
        return np.mean(points, axis=0)

    centroid = np.sum(points * weights[:, None], axis=0) / np.sum(weights)
    return centroid

def mesh_scalar_to_nifti(mesh, scalar_name, out_fname,
                         voxel_size=2.0,
                         padding=10.0):
    """
    Convert scalar values defined on a 3D mesh into a NIfTI volume
    using nearest-neighbor assignment.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Input mesh containing 3D point coordinates and associated
        scalar values.

    scalar_name : str
        Name of the scalar data array stored in the mesh.

    out_fname : str
        Path of the output NIfTI file.

    voxel_size : float, optional
        Isotropic voxel size of the output volume, in the same spatial
        units as the mesh coordinates. The default is 2.0.

    padding : float, optional
        Additional margin added around the minimum and maximum mesh
        coordinates when defining the output volume. The default is 10.0.

    Raises
    ------
    ValueError
        If no finite scalar values are found in the specified scalar array.

    Returns
    -------
    None
        The generated NIfTI image is saved directly to ``out_fname``.

    Notes
    -----
    Only mesh points with finite scalar values are used.

    A regular 3D voxel grid is generated over the mesh bounding box with
    the specified padding. Each voxel is assigned the scalar value of its
    nearest valid mesh point using a KD-tree search.

    Scalar values are assigned only when the distance between the voxel
    and its nearest mesh point is less than twice the voxel size.
    Voxels outside this distance remain NaN.

    The NIfTI affine matrix is constructed from the specified voxel size
    and the minimum coordinates of the generated volume.
    """
    pts = mesh.points
    vals = mesh[scalar_name]

    valid = np.isfinite(vals)
    pts = pts[valid]
    vals = vals[valid]

    if len(pts) == 0:
        raise ValueError("No valid scalar values")

    mins = pts.min(axis=0) - padding
    maxs = pts.max(axis=0) + padding

    xs = np.arange(mins[0], maxs[0], voxel_size)
    ys = np.arange(mins[1], maxs[1], voxel_size)
    zs = np.arange(mins[2], maxs[2], voxel_size)

    nx, ny, nz = len(xs), len(ys), len(zs)
    grid = np.full((nx, ny, nz), np.nan, dtype=np.float32)

    tree = cKDTree(pts)

    xv, yv, zv = np.meshgrid(xs, ys, zs, indexing='ij')
    vox_points = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T

    dist, idx = tree.query(vox_points, k=1)

    max_assign_dist = voxel_size * 2
    good = dist < max_assign_dist

    grid_flat = grid.ravel()
    grid_flat[good] = vals[idx[good]]
    grid = grid_flat.reshape(nx, ny, nz)

    affine = np.eye(4)
    affine[0, 0] = voxel_size
    affine[1, 1] = voxel_size
    affine[2, 2] = voxel_size
    affine[:3, 3] = mins

    nii = nib.Nifti1Image(grid, affine)
    nib.save(nii, out_fname)

    print(f"saved: {out_fname}")

# Define Class
class LVM_data:
    """
    Container for data loaded from a single LVM file.

    The class stores MEP waveforms, stimulation amplitudes, marker locations,
    coil positions, stimulation-target information, and anatomical landmark
    coordinates extracted from an LVM file.
    """
    def __init__(self, fname: str):
        """
        Initialize an LVM_data instance from an LVM file.

        Parameters
        ----------
        fname : str
            Path to the input LVM file.

        Notes
        -----
        MEP waveforms are assumed to contain 1000 samples per trial at a
        sampling frequency of 5000 Hz. The time axis is defined such that
        sample 200 corresponds to time zero.

        The first trial is initially excluded by setting ``ignored = [0]``.
        Target-marker information and anatomical landmark coordinates are
        identified automatically during initialization.
        """

        df = read_csv_with_nan(fname)
        data = extract_data(df)

        self.t = [i / 5000 - 200 / 5000 for i in range(1000)]
        self.fname = fname
        self.locations = data['dataset']

        waveforms = np.array(data['MEPs'], dtype=float)
        self.waveforms = waveforms.reshape(int(np.size(waveforms) / 1000), 1000).T

        self.amps = data['MEP_amp']
        self.IDs = list({d["ID"] for sublist in self.locations for d in sublist})
        self.ignored = [0]
        self.targetID = None

        self.define_targetTag()
        self.define_landmark_coord()

        self.amps = [a for i, a in enumerate(self.amps) if i not in self.ignored]

    def _input_ignored_trials(self, num: int):
        """
        Add a trial to the ignored-trial list and update derived coordinates.

        Parameters
        ----------
        num : int
            Index of the trial to exclude from subsequent calculations.

        Notes
        -----
        After adding the specified trial to ``self.ignored``, the coil
        positions, anatomical landmark coordinates, and target-marker
        information are recalculated.
        """

        self.ignored.append(num)

        self.Coil_pos = extract_coords_from_ID(self.locations, 500, self.ignored)

        self.define_landmark_coord()
        self.define_targetTag()

    def define_targetTag(self):
        """
        Identify the virtual-tracker ID corresponding to the stimulation target.

        The mean position of marker ID 502 is treated as the stimulation-target
        coordinate. Candidate virtual trackers are defined as marker IDs below
        100, and the tracker closest to the target coordinate is identified.

        If the minimum Euclidean distance is less than 0.001, the corresponding
        tracker ID is stored in ``self.targetID``.

        Notes
        -----
        Coil positions are also extracted from marker ID 500 and stored in
        ``self.Coil_pos``.
        """

        self.target_coords = calculate_mean_loc_from_ID(
            self.locations,
            502,
            self.ignored
        )

        self.Coil_pos = extract_coords_from_ID(
            self.locations,
            500,
            self.ignored
        )

        virtual_trackers = [
            calculate_mean_loc_from_ID(self.locations, idx, self.ignored)
            for idx in self.IDs if idx < 100
        ]

        virtual_trackers_id = [
            idx for idx in self.IDs if idx < 100
        ]

        dist = [
            np.linalg.norm(vt - self.target_coords)
            for vt in virtual_trackers
        ]

        minDistID = np.argmin(dist)

        if dist[minDistID] < 0.001:
            self.targetID = virtual_trackers_id[minDistID]

    def define_landmark_coord(self):
        """
        Extract anatomical landmark names and coordinates from marker IDs.

        Landmark markers are defined as IDs from 600 to 699. Their names are
        obtained from ``landmark_list``, and their coordinates are calculated
        as the mean marker positions across non-ignored trials.

        If no landmark markers are present in the current LVM file, the user
        is prompted to select another LVM file. Landmark information from that
        file is then used as a reference.

        Notes
        -----
        The resulting landmark names and coordinates are stored in
        ``self.landmark_names`` and ``self.landmark_pos``, respectively.
        A dictionary mapping landmark names to coordinates is stored in
        ``self.landmark_dict``.
        """

        landmarks = [
            idx for idx in self.IDs
            if 600 <= idx < 700
        ]

        if landmarks:

            self.landmark_names = [
                landmark_list[str(idx)]
                for idx in landmarks
            ]

            self.landmark_pos = np.array([
                calculate_mean_loc_from_ID(
                    self.locations,
                    idx,
                    self.ignored
                )
                for idx in landmarks
            ])

        else:

            print("Refer to another LVM file.")

            temp_fname = tf.askopenfilename(
                filetypes=[("lvm", "*.lvm")]
            )

            temp_lvm = LVM_data(temp_fname)

            landmarks = [
                idx for idx in temp_lvm.IDs
                if 600 <= idx < 700
            ]

            self.landmark_names = [
                landmark_list[str(idx)]
                for idx in landmarks
            ]

            self.landmark_pos = np.array([
                calculate_mean_loc_from_ID(
                    temp_lvm.locations,
                    idx,
                    temp_lvm.ignored
                )
                for idx in landmarks
            ])

        print(landmarks)

        self.landmark_dict = dict(
            zip(self.landmark_names, self.landmark_pos)
        )

class MEP_mapping:
    """
   Perform MEP mapping by integrating smartphone-based tracking data
   with 3D scanner coordinates and MNI-space anatomical models.

   This class loads MEP and marker data from LVM files, estimates the
   spatial transformation between smartphone and scanner coordinate
   systems, converts coil and control-point coordinates into scanner
   and MNI space, interpolates MEP amplitudes over scalp and brain
   surfaces, calculates a weighted stimulation centroid, and exports
   the resulting maps and coordinates.

   Depending on the selected mode, the spatial transformation is
   estimated using anatomical landmarks, stimulation control points,
   or a combination of both. The mode named ``"Both"`` in the code
   corresponds to the ``Hybrid`` method described in the manuscript.
    """
    
    def __init__(self,path:str,savePath:str,modes = "landmark", isFullmode = True,
                 subj_num = 1,obj_data = []):
        """
        Initialize the MEP mapping analysis.

        Parameters
        ----------
        path : str
            Directory containing the input LVM files. All LVM files in this
            directory are loaded and processed as individual stimulation
            measurements.

        savePath : str
            Directory in which the resulting NIfTI, node, and JSON files
            are saved.

        modes : str, optional
            Method used to estimate the transformation between scanner and
            smartphone coordinate systems. The default is ``"landmark"``.

            Available modes are:

            - ``"landmark"``:
              Estimate the transformation using anatomical landmarks only.

            - ``"ControlPoints"``:
              Estimate the transformation using stimulation control points
              only.

            - ``"Both"``:
              Estimate the transformation using both anatomical landmarks
              and stimulation control points. This combined approach is
              referred to as the ``Hybrid`` method in the manuscript.

            - ``"NoScanner"``:
              Use the standard MNI152 scalp and brain models instead of
              participant-specific 3D scanner data.

        isFullmode : bool, optional
            If True, use a full affine transformation when estimating the
            spatial mapping between scanner and smartphone coordinate systems.
            If False, use the restricted transformation implemented in
            ``estimate_axis_scaled_transform_from_dicts``.
            The default is True.

        subj_num : int, optional
            Subject number used when generating output filenames.
            The default is 1.

        obj_data : dict, optional
            Dictionary containing the 3D scanner-related input files and
            geometric data required by ``Scanner_Param``.

            When ``modes="NoScanner"``, this argument is replaced internally
            with the standard MNI152 scalp model and corresponding
            picked-point file.

        Notes
        -----
        The analysis performs the following main steps:

        1. Load participant-specific 3D scanner data or the standard
           MNI152 anatomical model.
        2. Load all LVM files and extract MEP amplitudes, coil positions,
           stimulation targets, and anatomical landmark coordinates.
        3. Estimate the spatial transformation between smartphone and
           scanner coordinate systems.
        4. Transform coil and control-point coordinates into scanner and
           MNI space.
        5. Interpolate MEP amplitudes over the scalp and brain surfaces.
        6. Restrict interpolated values to the convex hull defined by the
           measured stimulation locations.
        7. Calculate the weighted centroid of the interpolated MEP
           distribution on the brain surface.
        8. Export the resulting volumetric map, stimulation coordinates,
           centroid coordinates, and spatial error measures.

        Output Files
        ------------
        ``S{subj_num}_{modes}_{prefix}.nii.gz``
            NIfTI volume containing the interpolated MEP amplitude map in
            MNI space.

        ``S{subj_num}_{modes}_{prefix}.node``
            Tab-delimited coordinate file containing the mapped stimulation
            locations and the weighted centroid.

        ``S{subj_num}_{modes}_{prefix}.json``
            JSON file containing mapped coil coordinates, stimulation target
            coordinates, MEP amplitudes, centroid coordinates, and spatial
            error measures.

        Attributes
        ----------
        scanData : Scanner_Param
            Scanner data and anatomical-model information.

        all_data : list of LVM_data
            Data loaded from all input LVM files.

        landmark_from_sp : dict
            Anatomical landmark coordinates obtained from smartphone tracking.

        control_points_from_sp : dict
            Stimulation control-point coordinates obtained from smartphone
            tracking.

        scan_control_points : dict
            Control-point coordinates obtained from the 3D scanner data.

        scan_landmarks : dict
            Anatomical landmark coordinates obtained from the 3D scanner data.

        T_scan_to_sp : numpy.ndarray
            Transformation matrix from scanner coordinates to smartphone
            coordinates.

        T_sp_to_scan : numpy.ndarray
            Transformation matrix from smartphone coordinates to scanner
            coordinates.

        Coil_from_sp : list
            Coil coordinates obtained from smartphone tracking.

        Coil_converted_scan : list
            Coil coordinates transformed into scanner and MNI anatomical space.

        MEPs : list
            Mean MEP amplitude for each stimulation measurement.

        pts : numpy.ndarray
            Coil locations projected onto the scalp surface.

        pts2 : numpy.ndarray
            Coil locations mapped onto the brain surface in MNI coordinates.

        mep_vals : numpy.ndarray
            MEP amplitude associated with each stimulation location.

        interp_amp : numpy.ndarray
            Interpolated MEP amplitudes over the scalp surface.

        interp_amp_mni : numpy.ndarray
            Interpolated MEP amplitudes over the brain surface in MNI space.

        centroid : numpy.ndarray
            Weighted centroid of the interpolated MEP distribution on the
            brain surface in MNI coordinates.

        diff_control : list
            Spatial differences between scanner-derived control points and
            smartphone-derived control points after transformation.

        diff_from_target : list
            Spatial differences between mapped coil positions and the
            corresponding stimulation target positions.
        """

        # Importing scanner data
        if modes == "NoScanner":
            obj_data = {
                "ply":"../volume data/MNI152_T1_1mm_skin.nii.ply",
                "pp":"../volume data/MNI152_T1_1mm_skin.nii_picked_points.pp"
                }
            
        self.scanData = p3d.Scanner_Param(obj_data, rank_int=1, target=[4,6])
        if not self.scanData.isLoad:
            p3d.save_geometrical_data(self.scanData,obj_data)
        #landmarks = ["Nz","Cz","A1","A2"]
        # Loading MEP data and Smartphone Coordinates
        fnames = os.listdir(path)
        fpath = sorted(
            [os.path.join(path, fname) for fname in fnames
             if fname.lower().endswith(".lvm") and not fname.startswith("._")]
        )
        self.all_data = [LVM_data(fname) for fname in fpath]
        
        self.landmark_from_sp = {}
        for key in landmark_list.keys():
            temp = []
            for ad in self.all_data:
                for loc in ad.locations:
                    for l in loc:
                        if l["ID"] == int(key):
                            temp.append(np.array([l["x"], l["y"], l["z"]]))
            self.landmark_from_sp[landmark_list[key]] = np.mean(np.array(temp), axis=0)
        
        self.control_points_from_sp = {}
        key = "502"
        for i, ad in enumerate(self.all_data):
            temp = []
            for loc in ad.locations:
                for l in loc:
                    if l["ID"] == int(key):
                        temp.append(np.array([l["x"], l["y"], l["z"]]))
            self.control_points_from_sp[f"S{i+1}"] = np.mean(temp, axis=0)

        self.scan_control_points = {}
        self.scan_landmarks = {}
        for i in self.scanData.control_points.keys():
            self.scan_control_points[i] = self.scanData.control_points[i]
        for i in self.scanData.landmarks.keys():
            self.scan_landmarks[i] = self.scanData.landmarks[i]
        
        # Setting
        self.mode =modes
        self.affine_mode = isFullmode
        if self.mode == "ControlPoints":
            self.scan = self.scan_control_points
            self.sphone = self.control_points_from_sp
            self.T_scan_to_sp, self.T_sp_to_scan = p3d.estimate_axis_scaled_transform_from_dicts(
                self.scan,
                self.sphone,
                allow_reflection=True,
                use_full_affine=self.affine_mode
            )
        elif self.mode == "Both":
            self.scan = dict(**self.scan_landmarks, **self.scan_control_points)
            self.sphone = dict(**self.landmark_from_sp, **self.control_points_from_sp)
            self.T_scan_to_sp, self.T_sp_to_scan = p3d.estimate_axis_scaled_transform_from_dicts(
                self.scan,
                self.sphone,
                allow_reflection=True,
                use_full_affine=self.affine_mode
            )
        else: # No 3D Volume data or landmark
            self.scan = self.scan_landmarks
            self.sphone = self.landmark_from_sp
            print(self.scan_landmarks)
            print(self.landmark_from_sp)
            self.T_scan_to_sp, self.T_sp_to_scan = p3d.estimate_axis_scaled_transform_from_dicts(
                self.scan,
                self.sphone,
                allow_reflection=True,
                use_full_affine=self.affine_mode
            )
        self.Coil_from_sp = []
        for ad in self.all_data:
            temp = []
            key = 500
            for loc in ad.locations:
                for l in loc:
                    if l["ID"] == key:
                        temp.append(np.array([l["x"], l["y"], l["z"]]))
            self.Coil_from_sp.append(np.mean(np.array(temp), axis=0))
        
        self.Coil_converted_scan = [
            self.scanData.convert_scanner_coord_to_MNI(p3d.apply_affine(i, self.T_sp_to_scan))
            for i in self.Coil_from_sp
        ]

        self.scan_control_points_arr = [
            v for k, v in sorted(self.scan_control_points.items(), key=lambda kv: int(kv[0][1:]))
        ]

        self.MEPs = [np.mean(ad.amps) for ad in self.all_data]

        self.Control_points_sp_arr = [
            v for k, v in sorted(self.control_points_from_sp.items(), key=lambda kv: int(kv[0][1:]))
        ]

        self.Control_points_converted_from_sp = [
            self.scanData.convert_scanner_coord_to_MNI(p3d.apply_affine(i, self.T_sp_to_scan))["nearest_skin_scanner"]
            for i in self.Control_points_sp_arr
        ]
        
        self.diff_control = [j - i for i, j in zip(self.scan_control_points_arr, self.Control_points_converted_from_sp)]
        self.diff_from_target = [i["nearest_skin_scanner"] - j for j, i in zip(self.scan_control_points_arr, self.Coil_converted_scan)]
        
        self.pts = np.asarray([c["nearest_skin_scanner"] for c in self.Coil_converted_scan], dtype=float)
        self.pts2 = np.asarray([c["mapped_brain_mni"] for c in self.Coil_converted_scan], dtype=float)
        self.labels = [f"S{i+1}" for i in range(len(self.pts))]
        self.mep_vals = np.asarray(self.MEPs, dtype=float)
        
        if np.any(np.isfinite(self.mep_vals)):
            self.mep_clim = [np.nanmin(self.mep_vals), np.nanmax(self.mep_vals)]
            if self.mep_clim[0] == self.mep_clim[1]:
                self.mep_clim = [self.mep_clim[0] - 1e-6, self.mep_clim[1] + 1e-6]
        else:
            self.mep_clim = [0.0, 1.0]
        
        mesh_skin = self.scanData.skin.copy()
        mesh_brain = self.scanData.brain_model.copy()
        
        self.interp_amp = pca_2d_interp_fast(
            mesh_skin.points,
            self.pts,
            self.mep_vals,
        )

        self.interp_amp_mni = pca_2d_interp_fast(
            mesh_brain.points,
            self.pts2,
            self.mep_vals,
        )
        
        mask_skin = mask_inside_hull(mesh_skin.points, self.pts)
        self.interp_amp[~mask_skin] = np.nan
        
        mask_brain = mask_inside_hull(mesh_brain.points, self.pts2)
        self.interp_amp_mni[~mask_brain] = np.nan
        
        c = weighted_centroid(mesh_brain.points, self.interp_amp_mni)
        dist, idx = self.scanData.brain_tree_mni.query(c)
        self.centroid = self.scanData.brain_points_mni[idx]
        
        mesh_skin["amp"] = self.interp_amp
        mesh_brain["amp"] = self.interp_amp_mni
        mesh_skin.set_active_scalars("amp")
        mesh_brain.set_active_scalars("amp")
        
        valid_skin = self.interp_amp[np.isfinite(self.interp_amp)]
        valid_brain = self.interp_amp_mni[np.isfinite(self.interp_amp_mni)]
        valid_all = np.concatenate([valid_skin, valid_brain]) if (len(valid_skin) + len(valid_brain)) > 0 else np.array([])
        
        if len(valid_all) > 0:
            interp_clim = [np.nanmin(valid_all), np.nanmax(valid_all)]
            if interp_clim[0] == interp_clim[1]:
                interp_clim = [interp_clim[0] - 1e-6, interp_clim[1] + 1e-6]
        else:
            interp_clim = [0.0, 1.0]

        mesh_brain_save = self.scanData.brain_model.copy()
        mesh_brain_save["amp"] = self.interp_amp_mni
        mesh_brain_save.set_active_scalars("amp")
        
        
        
        if isFullmode:
            prefix = "Full"
        else:
            prefix = "notFull"
        fname = f"S{subj_num}_{modes}_{prefix}.nii.gz"
        mesh_scalar_to_nifti(
            mesh_brain_save,
            scalar_name="amp",
            out_fname=os.path.join(savePath,fname),
            voxel_size=1.0
        )
        
        export_xyz_format(
            np.vstack((self.pts2, self.centroid[np.newaxis, :])),
            out_path=os.path.join(savePath, f"S{subj_num}_{modes}_{prefix}.node")
        )
        if modes == "NoScanner":
            target_coords_on_scanner = []
        else:
            target_coords_on_scanner = self.scanData.convert_scanner_coord_to_MNI(np.asarray(self.scan_control_points_arr))
        
        target_coords_on_sp_MNI = self.scanData.convert_scanner_coord_to_MNI(np.asarray(self.Control_points_converted_from_sp))
        outputDict = {
            "coil_on_skin": self.pts.tolist(),
            "coil_on_MNI": self.pts2.tolist(),
            "centroid_MNI": self.centroid.tolist(),
            "MEPs": self.mep_vals.tolist(),
            "target_on_skin": np.asarray(self.scan_control_points_arr).tolist(),
            "target_coords_on_scanner": target_coords_on_scanner,
            "target_coords_on_sp": np.asarray(self.Control_points_converted_from_sp).tolist(),
            "target_coords_on_sp_MNI":target_coords_on_sp_MNI,
            "Estimate_Error": np.asarray(self.diff_control).tolist(),
            "Target_Error": np.asarray(self.diff_from_target).tolist()
        }
        
        with open(os.path.join(savePath,f"S{subj_num}_{modes}_{prefix}.json"), "w") as w:
            json.dump(outputDict, w)



#%%

if __name__ == "__main__":
    """
   Run the MEP mapping analysis for all transformation modes.

   The user is prompted to specify the subject number, select the directory
   containing the MEP data, load the corresponding 3D volume data, and select
   the output directory.

   The analysis is then performed for all available registration modes:

   - ``"landmark"``:
     Registration based on anatomical landmarks.

   - ``"ControlPoints"``:
     Registration based on stimulation control points.

   - ``"Both"``:
     Registration based on both anatomical landmarks and control points.
     This mode corresponds to the ``Hybrid`` method described in the
     manuscript.

   - ``"NoScanner"``:
     Analysis using the standard MNI152 anatomical model without
     participant-specific 3D scanner data.

   Each registration mode is evaluated with both the full affine
   transformation and the restricted transformation.
   """
    print("Input Subject number")
    subj_num = input("")
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()
    print("Select path for MEP data")
    MEP_path = tf.askdirectory()
    print("Select volume data")
    obj_data = p3d.load_obj_data()
    print("Select the folder for data saving")
    savePath = tf.askdirectory()
    modes = ["landmark","ControlPoints","Both","NoScanner"]
    affine_mode = [True,False]
    for mode in modes:
        for am in affine_mode:
            MEPs = MEP_mapping(MEP_path,savePath,subj_num=int(subj_num),
                           isFullmode=am,obj_data=obj_data,modes = mode)
    
    