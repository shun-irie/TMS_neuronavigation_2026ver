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
        Dictionary containing anatomical landmark coordinates loaded
        from the picked-points file.

    skin : pv.PolyData
        Scalp mesh data loaded with ``pv.read()``.

    show : bool, optional
        Whether to display the 3D data for visual inspection.
        The default is True.

    rank_int : int, optional
        Rank of the candidate path sorted by path length.
        The default is 1.

    Raises
    ------
    RuntimeError
        Raised when the intersection curve cannot be computed,
        when no valid Nz–Iz component is found, or when the requested
        ranked path does not exist.

    Returns
    -------
    out : dict
        Dictionary containing the computed 10–20 system landmark coordinates.

    arc_pts : np.ndarray
        Points composing the selected midline curve between Nz and Iz.
    """

    A1 = landmarks["A1"]
    A2 = landmarks["A2"]
    Nz = landmarks["Nz"]
    Iz = landmarks["Iz"]

    # --- Compute the normal vector of the plane equidistant from A1 and A2 ---
    M_mid = (A1 + A2) / 2.0
    n_eq = (A2 - A1)
    n_eq = n_eq / np.linalg.norm(n_eq)

    # --- Compute the intersection curve between the plane and scalp mesh ---
    cut = skin.slice(origin=M_mid, normal=n_eq).clean()

    if cut.n_points < 2:
        raise RuntimeError("No intersection curve could be computed.")

    # =========================
    # 1) Split the intersection curve into connected components
    # =========================
    cut_conn = cut.connectivity()
    region_ids = cut_conn["RegionId"]
    unique_regions = np.unique(region_ids)

    best_region = None
    best_score = np.inf
    best_info = None

    for rid in unique_regions:
        sub = cut_conn.extract_cells(region_ids == rid).extract_surface().clean()

        if sub.n_points < 2 or sub.n_lines < 1:
            continue

        pts = sub.points
        tree = cKDTree(pts)

        d_nz, idx_nz = tree.query(Nz)
        d_iz, idx_iz = tree.query(Iz)

        # Select the connected component closest to both Nz and Iz
        score = d_nz + d_iz

        if score < best_score:
            best_score = score
            best_region = rid
            best_info = {
                "sub": sub,
                "pts": pts,
                "idx_nz": idx_nz,
                "idx_iz": idx_iz,
                "d_nz": d_nz,
                "d_iz": d_iz,
            }

    if best_info is None:
        raise RuntimeError(
            "No valid intersection component associated with Nz and Iz was found."
        )

    cut = best_info["sub"]
    cut_pts = best_info["pts"]
    idx_Nz_on = best_info["idx_nz"]
    idx_Iz_on = best_info["idx_iz"]

    Nz_on_line = cut_pts[idx_Nz_on]
    Iz_on_line = cut_pts[idx_Iz_on]

    # =========================
    # 2) Construct the graph representation of the curve
    # =========================
    G = nx.Graph()
    lines = cut.lines.reshape(-1)

    i = 0
    while i < len(lines):
        npts_line = int(lines[i])
        ids = lines[i + 1:i + 1 + npts_line]

        for a, b in zip(ids[:-1], ids[1:]):
            dist = np.linalg.norm(cut_pts[a] - cut_pts[b])
            G.add_edge(a, b, weight=dist)

        i += 1 + npts_line

    if not nx.has_path(G, idx_Nz_on, idx_Iz_on):
        raise RuntimeError(
            "Nz and Iz appear to lie on the same intersection component, "
            "but are not connected in the graph representation."
        )

    # =========================
    # 3) Compute the k-th shortest path
    # =========================
    from itertools import islice

    rank = rank_int

    gen = nx.shortest_simple_paths(
        G,
        source=idx_Nz_on,
        target=idx_Iz_on,
        weight="weight"
    )

    path = next(islice(gen, rank - 1, None), None)

    if path is None:
        raise RuntimeError(
            f"The {rank}-th shortest path does not exist."
        )

    arc_pts = cut_pts[path]

    # =========================
    # 4) Compute 10–20 system midline landmarks
    # =========================
    seg_len = np.linalg.norm(np.diff(arc_pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])

    cut_length = [i / 10 + 0.1 for i in range(9)]
    cut_label = ["Fpz", "AFz", "Fz", "FCz", "Cz", "CPz", "Pz", "POz", "Oz"]

    out = {}

    for i in range(len(cut_label)):
        _len = s[-1] * cut_length[i]

        k = np.searchsorted(s, _len) - 1
        k = np.clip(k, 0, len(seg_len) - 1)

        t = (_len - s[k]) / (seg_len[k] + 1e-12)

        out[cut_label[i]] = (
            (1 - t) * arc_pts[k] + t * arc_pts[k + 1]
        )

    out["Nz"] = Nz_on_line
    out["Iz"] = Iz_on_line
    out["A1"] = A1
    out["A2"] = A2

    return out, arc_pts

def geodesic_two_points(
    mesh: pv.PolyData,
    A: np.ndarray,
    B: np.ndarray,
    cut_label: list[str] = ["T9", "T7", "C3", "C1"]
) -> dict:
    """
    Compute intermediate landmark coordinates along the geodesic path
    between two points on a scalp mesh.

    The input mesh is converted to a triangulated surface mesh, and only
    the largest connected component is retained before computing the
    geodesic path.

    Parameters
    ----------
    mesh : pv.PolyData
        Input scalp mesh.

    A : np.ndarray
        3D coordinate of the first endpoint.

    B : np.ndarray
        3D coordinate of the second endpoint.

    cut_label : list[str], optional
        Labels assigned to intermediate points evenly spaced along
        the geodesic path. The default is ["T9", "T7", "C3", "C1"].

    Raises
    ------
    RuntimeError
        Raised when the reconstructed mesh still contains non-triangular faces.

    Returns
    -------
    out : dict
        Dictionary containing intermediate landmark labels and their
        corresponding 3D coordinates.
    """

    # Convert the mesh surface into a fully triangulated surface mesh
    surf = mesh.extract_surface().triangulate().clean()

    # Reconstruct PolyData using only face information
    mesh_tri = pv.PolyData(surf.points, surf.faces).clean()

    # Retain only the largest connected component
    conn = mesh_tri.connectivity()
    region_ids = conn["RegionId"]
    largest_region = np.bincount(region_ids).argmax()

    tmp = (
        conn.extract_cells(region_ids == largest_region)
        .extract_surface()
        .triangulate()
        .clean()
    )

    # Reconstruct the mesh again using only triangular faces
    mesh_tri = pv.PolyData(tmp.points, tmp.faces).clean()

    print("All faces are triangles:", mesh_tri.is_all_triangles)
    print(
        "Number of vertices =", mesh_tri.n_verts,
        "Number of lines =", mesh_tri.n_lines,
        "Number of strips =", mesh_tri.n_strips,
        "Number of faces =", mesh_tri.n_cells
    )

    if not mesh_tri.is_all_triangles:
        raise RuntimeError("mesh_tri still contains non-triangular faces.")

    # Find the closest mesh vertices to points A and B
    id_A = mesh_tri.find_closest_point(A)
    id_B = mesh_tri.find_closest_point(B)

    # Compute the geodesic path on the mesh surface
    path = mesh_tri.geodesic(id_A, id_B, keep_order=True)

    pts = path.points

    # Compute cumulative arc length along the geodesic path
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])

    # Determine normalized positions for intermediate landmarks
    cut_length = [
        (i + 1) / (len(cut_label) + 1)
        for i in range(len(cut_label))
    ]

    out = {}

    for i in range(len(cut_label)):

        _len = s[-1] * cut_length[i]

        k = np.searchsorted(s, _len) - 1
        k = np.clip(k, 0, len(seg_len) - 1)

        t = (_len - s[k]) / (seg_len[k] + 1e-12)

        out[cut_label[i]] = (
            (1 - t) * pts[k] + t * pts[k + 1]
        )

    return out

def calculate_midpoint(
    mesh: pv.PolyData,
    pair1: tuple[str, str],
    pair2: tuple[str, str],
    dict_land: dict
) -> np.ndarray:
    """
    Compute a landmark coordinate as the midpoint between two
    geodesic midpoints.

    Parameters
    ----------
    mesh : pv.PolyData
        Input scalp mesh.

    pair1 : tuple[str, str]
        Pair of landmark labels defining the first geodesic path.

    pair2 : tuple[str, str]
        Pair of landmark labels defining the second geodesic path.

    dict_land : dict
        Dictionary containing landmark labels and their corresponding
        3D coordinates.

    Returns
    -------
    midpoint : np.ndarray
        3D coordinate of the midpoint between the two geodesic midpoints.
    """

    # Compute the midpoint along the geodesic path for the first landmark pair
    p1 = geodesic_two_points(
        mesh,
        dict_land[pair1[0]],
        dict_land[pair1[1]],
        cut_label=["t1"]
    )["t1"]

    # Compute the midpoint along the geodesic path for the second landmark pair
    p2 = geodesic_two_points(
        mesh,
        dict_land[pair2[0]],
        dict_land[pair2[1]],
        cut_label=["t2"]
    )["t2"]

    # Return the midpoint between the two computed points
    return (p1 + p2) / 2

def auto_calculate_montage(obj_data: dict, rank_int=1):
    """
    Perform batch processing from loading a 3D scalp mesh
    to computing International 10–20 system electrode coordinates.

    Parameters
    ----------
    obj_data : dict
        Dictionary containing file paths and metadata required
        for montage computation.

    rank_int : int, optional
        Rank of the shortest geodesic path used for Cz estimation.
        The default is 1.

    Returns
    -------
    skin : pv.PolyData
        Triangulated scalp mesh with computed surface normals.

    montage : dict
        Dictionary containing estimated 10–20 electrode coordinates.

    arc_pts : np.ndarray
        Points composing the Nz–Iz midline geodesic arc.

    landmarks : dict
        Dictionary containing anatomical landmark coordinates.

    control_points : dict
        Dictionary containing additional control point coordinates.
    """

    # Import landmark and control point coordinates from the XML file
    loc_origin = parse_xml(obj_data["pp"])

    landmarks = {
        i: loc_origin[i]
        for i in ["Nz", "Cz", "Iz", "A1", "A2"]
    }

    control_points = {
        i: loc_origin[i]
        for i in loc_origin.keys()
        if i not in ["Nz", "Cz", "Iz", "A1", "A2"]
    }

    # Load and triangulate the scalp mesh
    skin = pv.read(obj_data["ply"]).triangulate()

    # Compute surface normals
    skin = skin.compute_normals(
        cell_normals=True,
        point_normals=False,
        auto_orient_normals=True,
        consistent_normals=True,
        inplace=False
    )

    # Estimate the midline montage points
    montage, arc_pts = compute_Cz_on_equal_distance_plane(
        landmarks,
        skin,
        rank_int=rank_int
    )

    # Define geodesic interpolation paths for lateral electrodes
    lis = [
        ["A1", "Cz", ["T7", "T5", "C3", "C1"]],
        ["A2", "Cz", ["T8", "T6", "C4", "C2"]],
        ["Fpz", "T7", ["Fp1", "AF7", "F7", "FT7"]],
        ["Fpz", "T8", ["Fp2", "AF8", "F8", "FT8"]],
        ["T7", "Oz", ["TP7", "P7", "PO7", "O1"]],
        ["T8", "Oz", ["TP8", "P8", "PO8", "O2"]]
    ]

    # Compute geodesic interpolation points
    for l in lis:

        temp = geodesic_two_points(
            skin,
            montage[l[0]],
            montage[l[1]],
            cut_label=l[2]
        )

        for i in temp.keys():

            if i not in montage.keys():
                montage[i] = np.array(temp[i])

    # Define midpoint-based electrode estimation rules
    lis2 = [
        ["F3", ["Fp1", "C3"], ["F7", "Fz"]],
        ["F4", ["Fp2", "C4"], ["F8", "Fz"]],
        ["P3", ["O1", "C3"], ["T7", "Pz"]],
        ["P4", ["O2", "C4"], ["T8", "Pz"]],
    ]

    # Compute midpoint-derived electrode positions
    for l in lis2:
        montage[l[0]] = calculate_midpoint(
            skin,
            l[1],
            l[2],
            montage
        )

    # Store the original Cz coordinate
    landmarks["_Cz"] = loc_origin["Cz"]

    # Replace Cz with the estimated montage Cz
    landmarks["Cz"] = np.asarray(montage["Cz"])

    return skin, montage, arc_pts, landmarks, control_points

def robust_normal_at_point(
    mesh: pv.PolyData,
    query_point: np.ndarray,
    radius: float = 10.0,
    min_points: int = 20,
    fallback_k: int = 30,
    orient_mode: str = "center",
    return_debug: bool = False
) -> tuple:
    """
    Estimate a robust surface normal at a query point using local PCA.

    Neighboring mesh vertices are collected around the closest mesh point.
    If too few points are found within the specified radius, the nearest
    fallback_k vertices are used instead. The estimated normal is then
    oriented either away from the mesh center or according to the existing
    point normal.

    Parameters
    ----------
    mesh : pv.PolyData
        Input mesh used for local surface normal estimation.

    query_point : np.ndarray
        3D coordinate at which the surface normal is estimated.

    radius : float, optional
        Search radius for collecting neighboring vertices.
        The default is 10.0.

    min_points : int, optional
        Minimum number of neighboring points required for radius-based
        estimation. The default is 20.

    fallback_k : int, optional
        Number of nearest vertices used when the radius-based neighborhood
        contains fewer than min_points. The default is 30.

    orient_mode : str, optional
        Method used to orient the estimated normal.
        Use "center" to orient the normal away from the mesh center,
        or "point_normal" to align it with the mesh point normal.
        The default is "center".

    return_debug : bool, optional
        If True, return additional diagnostic information.
        The default is False.

    Raises
    ------
    ValueError
        Raised when the mesh has fewer than three points or when fewer
        than three neighboring points are available for PCA.

    Returns
    -------
    normal : np.ndarray
        Estimated unit surface normal vector.

    anchor_point : np.ndarray
        Closest mesh point to the query point.

    debug : dict, optional
        Diagnostic information returned only when return_debug is True.
    """

    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()

    if mesh.n_points < 3:
        raise ValueError("The mesh contains too few points.")

    query_point = np.asarray(query_point, dtype=float).reshape(3)

    mesh_tri = mesh.triangulate()

    if orient_mode == "point_normal":
        if "Normals" not in mesh_tri.point_data:
            mesh_tri = mesh_tri.compute_normals(
                point_normals=True,
                cell_normals=False,
                inplace=False
            )

    # Find the closest mesh vertex to the query point
    anchor_id = mesh_tri.find_closest_point(query_point)
    anchor_point = mesh_tri.points[anchor_id]

    # Build a KDTree for fast nearest-neighbor search
    tree = cKDTree(mesh_tri.points)

    # Search neighboring vertices within the specified radius
    ids = tree.query_ball_point(anchor_point, radius)
    ids = np.asarray(ids)

    # Fall back to k-nearest neighbors if too few points are found
    if len(ids) < min_points:

        d, idx = tree.query(anchor_point, k=fallback_k)
        ids = idx

    neigh_pts = mesh_tri.points[ids]

    if len(neigh_pts) < 3:
        raise ValueError("Fewer than three neighboring points were found.")

    # Estimate the local surface normal using PCA
    centroid = neigh_pts.mean(axis=0)
    X = neigh_pts - centroid
    cov = X.T @ X

    eigvals, eigvecs = np.linalg.eigh(cov)

    normal = eigvecs[:, np.argmin(eigvals)]
    normal = normal / np.linalg.norm(normal)

    # Correct the normal orientation
    if orient_mode == "center":

        mesh_center = mesh_tri.center
        ref_vec = anchor_point - np.asarray(mesh_center)

        if np.dot(normal, ref_vec) < 0:
            normal = -normal

    elif orient_mode == "point_normal":

        ref_normal = mesh_tri.point_data["Normals"][anchor_id]

        if np.dot(normal, ref_normal) < 0:
            normal = -normal

    if return_debug:

        debug = dict(
            anchor_id=anchor_id,
            neighbor_ids=ids,
            centroid=centroid,
            eigenvalues=eigvals
        )

        return normal, anchor_point, debug

    return normal, anchor_point

def make_oriented_plane(
    anchor_point: np.ndarray,
    normal: np.ndarray,
    size: float = 20.0
) -> tuple[pv.PolyData, np.ndarray, np.ndarray]:
    """
    Create a square plane centered at an anchor point with a specified
    normal vector.

    The plane is explicitly constructed from four vertices.

    Parameters
    ----------
    anchor_point : np.ndarray
        3D coordinate of the plane center.

    normal : np.ndarray
        Normal vector of the plane.

    size : float, optional
        Side length of the square plane.
        The default is 20.0.

    Returns
    -------
    plane : pv.PolyData
        Square plane represented as PyVista PolyData.

    axis1 : np.ndarray
        First in-plane unit axis.

    axis2 : np.ndarray
        Second in-plane unit axis.
    """

    anchor_point = np.asarray(anchor_point, dtype=float).reshape(3)
    normal = np.asarray(normal, dtype=float).reshape(3)
    normal = normal / np.linalg.norm(normal)

    # Select a reference vector that is not nearly parallel to the normal
    ref = np.array([0.0, 0.0, 1.0])

    if abs(np.dot(normal, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    # Compute two orthonormal axes lying in the plane
    axis1 = np.cross(normal, ref)
    axis1 = axis1 / np.linalg.norm(axis1)

    axis2 = np.cross(normal, axis1)
    axis2 = axis2 / np.linalg.norm(axis2)

    h = size / 2.0

    p0 = anchor_point - h * axis1 - h * axis2
    p1 = anchor_point + h * axis1 - h * axis2
    p2 = anchor_point + h * axis1 + h * axis2
    p3 = anchor_point - h * axis1 + h * axis2

    points = np.vstack([p0, p1, p2, p3])

    # Define a quadrilateral face
    faces = np.array([4, 0, 1, 2, 3])
    plane = pv.PolyData(points, faces)

    return plane, axis1, axis2

def estimate_axis_scaled_transform_from_dicts(
    mni_landmark: dict,
    montage: dict,
    keys: list | None = None,
    return_keys: bool = False,
    allow_reflection: bool = False,
    use_full_affine: bool = False
):
    """
    Estimate a transformation from MNI landmark coordinates to montage coordinates.

    Parameters
    ----------
    mni_landmark : dict
        Source dictionary containing landmark coordinates as {key: [x, y, z]}.

    montage : dict
        Target dictionary containing landmark coordinates as {key: [x, y, z]}.

    keys : list or None, optional
        Ordered list of corresponding keys to use for estimation.
        If None, common keys are sorted and used.
        The default is None.

    return_keys : bool, optional
        If True, also return the keys used for transformation estimation.
        The default is False.

    allow_reflection : bool, optional
        Only used when use_full_affine is False.
        If True, allow rigid rotation estimation with det(R) < 0.
        The default is False.

    use_full_affine : bool, optional
        If False, estimate a transform composed of rotation,
        axis-wise scaling, and translation without shear:

            y = D @ R @ x + t

        If True, estimate a full affine transform:

            y = A @ x + t

        The full affine transform may include rotation, anisotropic scaling,
        shear, and reflection.
        The default is False.

    Returns
    -------
    T_mni_to_montage : np.ndarray
        4 x 4 homogeneous transformation matrix from MNI coordinates
        to montage coordinates.

    T_montage_to_mni : np.ndarray
        4 x 4 homogeneous transformation matrix from montage coordinates
        to MNI coordinates.

    used_keys : list
        Keys used for transformation estimation.
        Returned only when return_keys is True.
    """

    # -------------------------
    # Determine keys to use
    # -------------------------
    if keys is None:
        used_keys = sorted(set(mni_landmark.keys()) & set(montage.keys()))
    else:
        used_keys = [
            k for k in keys
            if (k in mni_landmark) and (k in montage)
        ]

    if len(used_keys) < 3:
        raise ValueError("At least three corresponding points are required.")

    X = np.array([mni_landmark[k] for k in used_keys], dtype=float)   # source
    Y = np.array([montage[k] for k in used_keys], dtype=float)        # target

    valid = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X = X[valid]
    Y = Y[valid]
    used_keys = [k for k, v in zip(used_keys, valid) if v]

    if len(used_keys) < 3:
        raise ValueError("Fewer than three valid corresponding points remain.")

    # =========================================================
    # Full affine mode
    # =========================================================
    if use_full_affine:

        # Estimate Y ≈ A @ X + t.
        # Because the least-squares problem is solved using row vectors:
        #
        #     [x y z 1] @ M ≈ [X' Y' Z']
        #
        # where M has shape (4, 3).
        Xh = np.hstack([X, np.ones((X.shape[0], 1))])   # (N, 4)
        M, residuals, rank, s = np.linalg.lstsq(
            Xh,
            Y,
            rcond=None
        )   # M: (4, 3)

        # Convert the estimated parameters to a 4 x 4 homogeneous matrix
        T_mni_to_montage = np.eye(4)
        T_mni_to_montage[:3, :3] = M[:3, :].T
        T_mni_to_montage[:3, 3] = M[3, :]

        T_montage_to_mni = np.linalg.inv(T_mni_to_montage)

        if return_keys:
            return T_mni_to_montage, T_montage_to_mni, used_keys
        else:
            return T_mni_to_montage, T_montage_to_mni

    # =========================================================
    # Standard mode: rotation + axis-wise scaling + translation
    # =========================================================

    # -------------------------
    # Center the source and target points
    # -------------------------
    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)

    Xc = X - muX
    Yc = Y - muY

    # -------------------------
    # First estimate the rigid rotation R
    # Y ≈ R X
    # -------------------------
    H = Yc.T @ Xc
    U, _, Vt = np.linalg.svd(H)

    R = U @ Vt

    if (not allow_reflection) and (np.linalg.det(R) < 0):
        Vt[-1, :] *= -1
        R = U @ Vt

    # -------------------------
    # Then estimate axis-wise scaling after rotation
    # Yc ≈ D @ (R @ Xc)
    # -------------------------
    XR = (R @ Xc.T).T   # (N, 3)

    scales = np.zeros(3, dtype=float)

    for j in range(3):

        denom = np.sum(XR[:, j] ** 2)

        if denom < 1e-12:
            scales[j] = 1.0
        else:
            scales[j] = np.sum(XR[:, j] * Yc[:, j]) / denom

    D = np.diag(scales)

    # -------------------------
    # Estimate translation
    # y = D R x + t
    # -------------------------
    t = muY - D @ R @ muX

    # -------------------------
    # Convert to a 4 x 4 homogeneous transformation matrix
    # -------------------------
    T_mni_to_montage = np.eye(4)
    T_mni_to_montage[:3, :3] = D @ R
    T_mni_to_montage[:3, 3] = t

    T_montage_to_mni = np.linalg.inv(T_mni_to_montage)

    if return_keys:
        return T_mni_to_montage, T_montage_to_mni, used_keys
    else:
        return T_mni_to_montage, T_montage_to_mni

def apply_affine(
    points: np.ndarray,
    A: np.ndarray
) -> np.ndarray:
    """
    Apply a 4 x 4 affine transformation matrix to 3D points.

    Parameters
    ----------
    points : array-like
        Input point or point cloud.
        The shape should be either (3,) or (N, 3).

    A : np.ndarray
        4 x 4 affine transformation matrix.

    Returns
    -------
    transformed : np.ndarray
        Transformed 3D point or point cloud.
        If the input shape is (3,), the output shape is (3,).
        If the input shape is (N, 3), the output shape is (N, 3).
    """

    pts = np.asarray(points, dtype=float)

    one_point = False

    if pts.ndim == 1:
        pts = pts[None, :]
        one_point = True

    # Convert points to homogeneous coordinates
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])   # (N, 4)

    # Apply the affine transformation
    out = (A @ pts_h.T).T[:, :3]

    if one_point:
        return out[0]

    return out

def progress_bar(i, total, start_time, bar_len=40, prefix="Progress", eta_start=50):
    frac = i / total if total > 0 else 1.0
    filled = int(bar_len * frac)
    bar = "#" * filled + "-" * (bar_len - filled)

    elapsed = time.time() - start_time
    rate = i / elapsed if elapsed > 0 else 0.0

    if i < eta_start or rate <= 0:
        eta_str = "ETA calculating..."
    else:
        eta = (total - i) / rate
        eta_str = f"ETA {eta:8.1f}s"

    sys.stdout.write(
        f"\r{prefix} [{bar}] {i}/{total} "
        f"{frac*100:5.1f}%  {rate:7.2f} it/s  {eta_str}"
    )
    sys.stdout.flush()

    if i == total:
        print()

def normalize_rows(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize each row vector of a 2D array.

    Parameters
    ----------
    v : np.ndarray
        Input array of shape (N, M), where each row represents a vector.

    Returns
    -------
    out : np.ndarray
        Row-wise normalized vectors.
        Rows with zero norm remain zero.

    ok : np.ndarray
        Boolean array indicating which rows had nonzero norms
        and were successfully normalized.
    """

    v = np.asarray(v, dtype=float)

    n = np.linalg.norm(v, axis=1, keepdims=True)

    out = np.zeros_like(v)

    ok = n[:, 0] > 0

    out[ok] = v[ok] / n[ok]

    return out, ok

def compute_smoothed_cell_normals(
    skin: pv.PolyData,
    radius: float | None = None,
    k: int | None = None,
    sigma: float | None = None,
    use_gaussian_weight: bool = True,
    show_progress: bool = True
) -> tuple[pv.PolyData, np.ndarray, np.ndarray]:
    """
    Smooth cell normals by averaging neighboring face normals without
    modifying the mesh geometry.

    Parameters
    ----------
    skin : pv.PolyData
        Input scalp mesh.

    radius : float or None, optional
        Neighborhood radius around each face center.
        The default is None.

    k : int or None, optional
        Number of nearest neighboring face centers to use.
        The default is None.

    sigma : float or None, optional
        Standard deviation used for Gaussian weighting.
        If None, radius / 2.0 is used when radius is specified;
        otherwise, 1.0 is used.
        The default is None.

    use_gaussian_weight : bool, optional
        Whether to apply Gaussian distance-based weighting.
        The default is True.

    show_progress : bool, optional
        Whether to display progress during normal smoothing.
        The default is True.

    Returns
    -------
    skin2 : pv.PolyData
        Triangulated surface mesh with computed cell normals.

    face_centers : np.ndarray
        Face center coordinates with shape (N, 3).

    smoothed_normals : np.ndarray
        Smoothed unit cell normals with shape (N, 3).
    """

    skin2 = skin.extract_surface().triangulate().compute_normals(
        cell_normals=True,
        point_normals=False,
        consistent_normals=True,
        auto_orient_normals=True,
        inplace=False
    )

    face_centers = skin2.cell_centers().points
    raw_normals = np.asarray(skin2.cell_data["Normals"], dtype=float)
    raw_normals, valid_mask = normalize_rows(raw_normals)

    tree = cKDTree(face_centers)
    n = len(face_centers)
    smoothed = np.zeros((n, 3), dtype=float)

    if sigma is None:
        if radius is not None:
            sigma = radius / 2.0
        else:
            sigma = 1.0

    start_time = time.time()

    for i in range(n):
        p = face_centers[i]
        n0 = raw_normals[i]

        if np.linalg.norm(n0) == 0:
            continue

        if radius is not None:
            idx = tree.query_ball_point(p, r=radius)
            idx = np.asarray(idx, dtype=int)

        elif k is not None:
            _, idx = tree.query(p, k=k)
            idx = np.atleast_1d(idx)

        else:
            raise ValueError("Either radius or k must be specified.")

        neigh_normals = raw_normals[idx].copy()
        neigh_centers = face_centers[idx]

        # Align neighboring normals to the reference normal direction
        dots = neigh_normals @ n0
        flip_mask = dots < 0
        neigh_normals[flip_mask] *= -1.0

        # Compute neighbor weights
        if use_gaussian_weight:
            dist = np.linalg.norm(neigh_centers - p, axis=1)
            w = np.exp(-(dist ** 2) / (2 * sigma ** 2 + 1e-12))
        else:
            w = np.ones(len(idx), dtype=float)

        v = (neigh_normals * w[:, None]).sum(axis=0)
        nv = np.linalg.norm(v)

        if nv > 0:
            smoothed[i] = v / nv
        else:
            smoothed[i] = n0

        if show_progress and ((i + 1) % 200 == 0 or (i + 1) == n):
            progress_bar(i + 1, n, start_time, prefix="SmoothNormals")

    return skin2, face_centers, smoothed

def _multi_ray_trace_first_hits(
    mesh: pv.PolyData,
    origins: np.ndarray,
    directions: np.ndarray,
    retry: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Perform multi-ray tracing and return the first hit point for each ray.

    Parameters
    ----------
    mesh : pv.PolyData
        Input mesh used for ray tracing.

    origins : np.ndarray
        Ray origin coordinates with shape (N, 3).

    directions : np.ndarray
        Ray direction vectors with shape (N, 3).

    retry : bool, optional
        Whether to retry rays that fail during the initial ray-tracing pass.
        The default is True.

    Raises
    ------
    RuntimeError
        Raised when the return format of multi_ray_trace cannot be interpreted.

    Returns
    -------
    hit_points_all : np.ndarray
        Coordinates of the first hit points.

    ray_ids_all : np.ndarray
        Indices of rays that produced valid hit points.

    cell_ids_all : np.ndarray or None
        Indices of the intersected mesh cells, if returned by multi_ray_trace.
    """

    result = mesh.multi_ray_trace(
        origins=origins,
        directions=directions,
        first_point=True,
        retry=retry
    )

    if isinstance(result, tuple):
        if len(result) == 3:
            hit_points_all, ray_ids_all, cell_ids_all = result
        elif len(result) == 2:
            hit_points_all, ray_ids_all = result
            cell_ids_all = None
        else:
            raise RuntimeError(
                "Could not interpret the return format of multi_ray_trace."
            )
    else:
        raise RuntimeError(
            "Could not interpret the return format of multi_ray_trace."
        )

    hit_points_all = np.asarray(hit_points_all, dtype=float)
    ray_ids_all = np.asarray(ray_ids_all, dtype=int)

    return hit_points_all, ray_ids_all, cell_ids_all

def compute_first_hit_matrix_from_normals_fast(
    anchor_points: np.ndarray,
    normals: np.ndarray,
    mesh: pv.PolyData,
    ray_length: float = 300.0,
    eps: float = 1e-3,
    batch_size: int = 5000,
    retry: bool = True,
    show_progress: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the closest ray-mesh intersection points along both positive
    and negative normal directions.

    For each anchor point, rays are cast in both directions along the
    corresponding normal vector. The closest valid hit is retained.

    Parameters
    ----------
    anchor_points : np.ndarray
        Anchor point coordinates with shape (N, 3).

    normals : np.ndarray
        Normal vectors associated with the anchor points, with shape (N, 3).

    mesh : pv.PolyData
        Mesh used for ray tracing.

    ray_length : float, optional
        Length of each ray.
        The default is 300.0.

    eps : float, optional
        Small offset applied to ray origins to avoid self-intersection.
        The default is 1e-3.

    batch_size : int, optional
        Number of anchor points processed per batch.
        The default is 5000.

    retry : bool, optional
        Whether to retry rays that fail during the initial ray-tracing pass.
        The default is True.

    show_progress : bool, optional
        Whether to display progress during ray tracing.
        The default is True.

    Returns
    -------
    hit_points : np.ndarray
        Closest hit point coordinates with shape (N, 3).
        Rows remain NaN when no valid hit is found.

    hit_distances : np.ndarray
        Distances from anchor points to the closest hit points.
        Values remain NaN when no valid hit is found.

    hit_directions : np.ndarray
        Direction of the selected hit for each anchor point.
        A value of 1 indicates the positive normal direction,
        -1 indicates the negative normal direction, and 0 indicates no hit.
    """

    # Convert the mesh to a triangulated surface mesh for ray tracing
    mesh = mesh.extract_surface().triangulate()

    anchor_points = np.asarray(anchor_points, dtype=float)
    normals = np.asarray(normals, dtype=float)

    # Normalize input normal vectors
    normals_unit, valid_mask = normalize_rows(normals)

    n = len(anchor_points)

    hit_points = np.full((n, 3), np.nan, dtype=float)
    hit_distances = np.full(n, np.nan, dtype=float)
    hit_directions = np.zeros(n, dtype=int)

    valid_ids = np.where(valid_mask)[0]
    total_rays = 2 * len(valid_ids)
    processed_rays = 0
    start_time = time.time()

    for s in range(0, len(valid_ids), batch_size):
        e = min(s + batch_size, len(valid_ids))
        batch_ids = valid_ids[s:e]

        A = anchor_points[batch_ids]
        N = normals_unit[batch_ids]

        # Create rays in the positive normal direction
        origins_pos = A + N * eps
        directions_pos = N * ray_length

        # Create rays in the negative normal direction
        origins_neg = A - N * eps
        directions_neg = -N * ray_length

        origins = np.vstack([origins_pos, origins_neg])
        directions = np.vstack([directions_pos, directions_neg])

        hit_pts_all, ray_ids_all, _ = _multi_ray_trace_first_hits(
            mesh=mesh,
            origins=origins,
            directions=directions,
            retry=retry
        )

        best_dist = np.full(len(batch_ids), np.inf, dtype=float)
        best_pt = np.full((len(batch_ids), 3), np.nan, dtype=float)
        best_sign = np.zeros(len(batch_ids), dtype=int)

        for hp, rid in zip(hit_pts_all, ray_ids_all):

            if rid < len(batch_ids):
                local_idx = rid
                sign = 1
            else:
                local_idx = rid - len(batch_ids)
                sign = -1

            dist = np.linalg.norm(hp - A[local_idx])

            if dist < best_dist[local_idx]:
                best_dist[local_idx] = dist
                best_pt[local_idx] = hp
                best_sign[local_idx] = sign

        ok = np.isfinite(best_dist)
        global_ids = batch_ids[ok]

        hit_points[global_ids] = best_pt[ok]
        hit_distances[global_ids] = best_dist[ok]
        hit_directions[global_ids] = best_sign[ok]

        processed_rays += 2 * len(batch_ids)

        if show_progress:
            progress_bar(
                processed_rays,
                total_rays,
                start_time,
                prefix="RayTrace"
            )

    return hit_points, hit_distances, hit_directions

def compute_hits_with_smoothed_normals(
    skin: pv.PolyData,
    mesh: pv.PolyData,
    radius: float = 5.0,
    k: int | None = None,
    sigma: float | None = None,
    ray_length: float = 300.0,
    eps: float = 1e-3,
    batch_size: int = 5000,
    retry: bool = True,
    show_progress: bool = True
) -> tuple[pv.PolyData, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Smooth cell normals without modifying the mesh geometry, and perform
    bidirectional ray tracing using the smoothed normals.

    Parameters
    ----------
    skin : pv.PolyData
        Input scalp mesh used to compute smoothed cell normals.

    mesh : pv.PolyData
        Target mesh used for ray tracing.

    radius : float, optional
        Neighborhood radius used for normal smoothing.
        The default is 5.0.

    k : int or None, optional
        Number of nearest neighboring face centers used for smoothing.
        The default is None.

    sigma : float or None, optional
        Standard deviation used for Gaussian weighting during normal smoothing.
        The default is None.

    ray_length : float, optional
        Length of each ray.
        The default is 300.0.

    eps : float, optional
        Small offset applied to ray origins to avoid self-intersection.
        The default is 1e-3.

    batch_size : int, optional
        Number of anchor points processed per ray-tracing batch.
        The default is 5000.

    retry : bool, optional
        Whether to retry rays that fail during the initial ray-tracing pass.
        The default is True.

    show_progress : bool, optional
        Whether to display progress during normal smoothing and ray tracing.
        The default is True.

    Returns
    -------
    skin2 : pv.PolyData
        Triangulated scalp mesh with computed cell normals.

    face_centers : np.ndarray
        Face center coordinates with shape (N, 3).

    smoothed_normals : np.ndarray
        Smoothed unit cell normals with shape (N, 3).

    hit_points : np.ndarray
        Closest hit point coordinates with shape (N, 3).

    hit_distances : np.ndarray
        Distances from face centers to the closest hit points.

    hit_directions : np.ndarray
        Direction of the selected hit for each face center.
        A value of 1 indicates the positive normal direction,
        -1 indicates the negative normal direction, and 0 indicates no hit.
    """

    t0 = time.time()

    skin2, face_centers, smoothed_normals = compute_smoothed_cell_normals(
        skin=skin,
        radius=radius,
        k=k,
        sigma=sigma,
        use_gaussian_weight=True,
        show_progress=show_progress
    )

    t1 = time.time()
    print(f"Normal smoothing time: {t1 - t0:.1f} s")

    hit_points, hit_distances, hit_directions = (
        compute_first_hit_matrix_from_normals_fast(
            anchor_points=face_centers,
            normals=smoothed_normals,
            mesh=mesh,
            ray_length=ray_length,
            eps=eps,
            batch_size=batch_size,
            retry=retry,
            show_progress=show_progress
        )
    )

    t2 = time.time()
    print(f"Ray-tracing time: {t2 - t1:.1f} s")
    print(f"Total processing time: {t2 - t0:.1f} s")

    return (
        skin2,
        face_centers,
        smoothed_normals,
        hit_points,
        hit_distances,
        hit_directions
    )

def project_to_nearest_brain_points(
    hit_mni: np.ndarray,
    brain_model: pv.PolyData
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project input coordinates onto the nearest points of a brain surface mesh.

    Parameters
    ----------
    hit_mni : np.ndarray
        Input coordinates in MNI space with shape (N, 3).

    brain_model : pv.PolyData
        Brain surface mesh used for nearest-point projection.

    Returns
    -------
    projected_points : np.ndarray
        Nearest brain surface coordinates corresponding to the input points.

    distances : np.ndarray
        Euclidean distances between the input points and the projected points.

    nearest_ids : np.ndarray
        Indices of the nearest mesh vertices in the brain model.
    """

    hit_mni = np.asarray(hit_mni, dtype=float)
    brain_points = np.asarray(brain_model.points, dtype=float)

    projected_points = np.full_like(hit_mni, np.nan, dtype=float)
    distances = np.full(hit_mni.shape[0], np.nan, dtype=float)
    nearest_ids = np.full(hit_mni.shape[0], -1, dtype=int)

    valid = ~np.isnan(hit_mni).any(axis=1)

    if not np.any(valid):
        return projected_points, distances, nearest_ids

    tree = cKDTree(brain_points)

    dist, idx = tree.query(hit_mni[valid], k=1)

    projected_points[valid] = brain_points[idx]
    distances[valid] = dist
    nearest_ids[valid] = idx

    return projected_points, distances, nearest_ids

# Class
class Scanner_Param:
    def __init__(self, obj_data, rank_int=1, target=[4,6]):
        """
        Initialize scanner-space and MNI-space parameters for neuronavigation.

        Parameters
        ----------
        obj_data : dict
            Dictionary containing paths to the scalp mesh and landmark files.

        rank_int : int, optional
            Rank of the shortest path used during montage estimation.
            The default is 1.

        target : list, optional
            Target Brodmann areas used for ROI extraction.
            The default is [4, 6].

        Returns
        -------
        None.
        """
        root = Path(os.path.dirname(obj_data["ply"]))
        files = list(root.rglob("*.npz"))

        if len(files) > 0:
            self.isLoad = True
            print("Cached data found.")
        else:
            self.isLoad = False

        if self.isLoad:

            # Load precomputed data
            npzdata = np.load(
                obj_data["ply"][0:-4] + "_np.npz",
                allow_pickle=True
            )

            self.skin = pv.read(obj_data["ply"]).triangulate()

            self.BAs = npzdata["BAs"]
            self.arc_pts = npzdata["arc_pts"]
            self.atlas_xyz = npzdata["atlas_xyz"]
            self.brain_distances = npzdata["brain_distances"]

            # Load brain surface model in MNI space
            self.brain_model = pv.read(
                BRAIN_MODEL_PATH
            )

            self.brain_nearest_ids = npzdata["brain_nearest_ids"]
            self.brain_nearest_points = npzdata["brain_nearest_points"]

            self.brain_points_mni = npzdata["brain_points_mni"]
            self.brain_points_scanner = npzdata["brain_points_scanner"]

            self.brain_surface_mni_from_anchor = \
                npzdata["brain_surface_mni_from_anchor"]

            self.brain_surface_scanner_from_anchor = \
                npzdata["brain_surface_scanner_from_anchor"]

            # KDTree for nearest-neighbor search
            self.brain_tree_scanner = cKDTree(self.brain_points_scanner)
            self.brain_tree_mni = cKDTree(self.brain_points_mni)

            self.centroid_head = npzdata["centroid_head"]
            self.centroid_mni = npzdata["centroid_mni"]
            self.centroid_skin = npzdata["centroid_skin"]

            self.control_points = npzdata["control_points"].tolist()

            self.hit_directions = npzdata["hit_directions"]
            self.hit_distances = npzdata["hit_distances"]
            self.hit_mni = npzdata["hit_mni"]
            self.hit_points_scanner = npzdata["hit_points_scanner"]

            self.landmarks = npzdata["landmarks"].tolist()

            self.mask_target = npzdata["mask_target"]

            self.mat_3dscan_to_mni = npzdata["mat_3dscan_to_mni"]
            self.mat_mni_to_3dscan = npzdata["mat_mni_to_3dscan"]

            # Brain model transformed into scanner space
            self.mesh_scanner = self.brain_model.copy()
            self.mesh_scanner.transform(self.mat_mni_to_3dscan)

            # Alias for compatibility with plot_calculation_result()
            self.mesh = self.mesh_scanner

            self.mniControl = npzdata["mniControl"]
            self.montage = npzdata["montage"].tolist()

            self.nearest_BAs = npzdata["nearest_BAs"]
            self.nearest_atlas_points = npzdata["nearest_atlas_points"]

            self.normals = npzdata["normals"]

            self.scanControl = npzdata["scanControl"]

            self.skin_anchor_points = npzdata["skin_anchor_points"]
            self.skin_anchor_tree = cKDTree(self.skin_anchor_points)

            self.target_pts_head = npzdata["target_pts_head"]
            self.target_pts_head_surface = npzdata["target_pts_head_surface"]
            self.target_pts_mni = npzdata["target_pts_mni"]

            # Construct ROI meshes in scanner space
            if len(self.target_pts_head) >= 4:

                cloud_head_brain = pv.PolyData(self.target_pts_head)
                cloud_head_skin = pv.PolyData(self.target_pts_head_surface)

                self.mesh_ROI_head_brain = (
                    cloud_head_brain.delaunay_3d().extract_surface()
                )

                self.mesh_ROI_head_skin = (
                    cloud_head_skin.delaunay_3d().extract_surface()
                )

            else:
                self.mesh_ROI_head_brain = pv.PolyData(self.target_pts_head)
                self.mesh_ROI_head_skin = pv.PolyData(
                    self.target_pts_head_surface
                )

            # Construct ROI meshes in MNI space
            if len(self.target_pts_mni) >= 4:

                cloud_mni = pv.PolyData(self.target_pts_mni)

                self.mesh_ROI_mni = (
                    cloud_mni.delaunay_3d().extract_surface()
                )

            else:
                self.mesh_ROI_mni = pv.PolyData(self.target_pts_mni)

        else:

            # Compute montage from the scalp mesh
            self.skin, self.montage, self.arc_pts, \
            self.landmarks, self.control_points = \
                auto_calculate_montage(
                    obj_data,
                    rank_int=rank_int
                )

            self.plot_mesh_landmarks()

            print("Accept the estimated montage? (y/N)")
            isAccept_montage = input(' ')

            if isAccept_montage == "y":

                # -------------------------
                # Estimate coordinate transformation matrices
                # -------------------------
                self.mat_mni_to_3dscan, self.mat_3dscan_to_mni = \
                    estimate_axis_scaled_transform_from_dicts(
                        mni_landmark,
                        self.montage,
                        keys=["Nz", "Iz", "Cz", "A1", "A2"]
                    )

                # -------------------------
                # Load brain model in MNI space
                # -------------------------
                self.brain_model = pv.read(
                    BRAIN_MODEL_PATH
                )

                # Transform the brain model into scanner space
                self.mesh_scanner = self.brain_model.copy()
                self.mesh_scanner.transform(self.mat_mni_to_3dscan)

                # Alias for compatibility with plot_calculation_result()
                self.mesh = self.mesh_scanner

                # Create KDTree structures for fast nearest-neighbor search
                self.brain_points_mni = np.asarray(
                    self.brain_model.points,
                    dtype=float
                )

                self.brain_points_scanner = np.asarray(
                    self.mesh_scanner.points,
                    dtype=float
                )

                self.brain_tree_scanner = cKDTree(
                    self.brain_points_scanner
                )

                self.brain_tree_mni = cKDTree(
                    self.brain_points_mni
                )

                # -------------------------
                # Perform ray tracing from scalp normals to brain surface
                # -------------------------
                skin2, anchor_points, normals, hit_points, \
                hit_distances, hit_directions = \
                    compute_hits_with_smoothed_normals(
                        skin=self.skin,
                        mesh=self.mesh_scanner,
                        radius=6.0,
                        k=None,
                        sigma=3.0,
                        ray_length=300.0,
                        eps=1e-3,
                        batch_size=5000,
                        retry=True,
                        show_progress=True
                    )

                self.anchor_points = np.asarray(
                    anchor_points,
                    dtype=float
                )

                self.normals = np.asarray(
                    normals,
                    dtype=float
                )

                self.hit_points_scanner = np.asarray(
                    hit_points,
                    dtype=float
                )

                self.hit_distances = np.asarray(
                    hit_distances,
                    dtype=float
                )

                self.hit_directions = np.asarray(hit_directions)

                # Transform hit points back into MNI space
                self.hit_mni = apply_affine(
                    self.hit_points_scanner,
                    self.mat_3dscan_to_mni
                )

                # Snap hit points to nearest brain surface vertices
                brain_nearest_points, brain_distances, \
                brain_nearest_ids = \
                    project_to_nearest_brain_points(
                        self.hit_mni,
                        self.brain_model
                    )

                self.brain_nearest_points = np.asarray(
                    brain_nearest_points,
                    dtype=float
                )

                self.brain_distances = np.asarray(
                    brain_distances,
                    dtype=float
                )

                self.brain_nearest_ids = np.asarray(
                    brain_nearest_ids,
                    dtype=int
                )

                # Build anchor-to-brain mapping structures
                self.skin_anchor_points = self.anchor_points.copy()

                self.skin_anchor_tree = cKDTree(
                    self.skin_anchor_points
                )

                self.brain_surface_mni_from_anchor = \
                    self.brain_nearest_points.copy()

                self.brain_surface_scanner_from_anchor = \
                    self.hit_points_scanner.copy()

                # -------------------------
                # Load Brodmann atlas
                # -------------------------
                fname_atlas = BRODMANN_ATLAS_PATH

                nii = nib.load(fname_atlas)

                atlas_data = nii.get_fdata()
                affine = nii.affine

                atlas_xyz = []
                BAs = []

                for x in range(atlas_data.shape[0]):
                    for y in range(atlas_data.shape[1]):
                        for z in range(atlas_data.shape[2]):

                            atlas_xyz.append([x, y, z])

                            BAs.append(
                                int(atlas_data[x, y, z])
                            )

                self.atlas_xyz = np.array(atlas_xyz)
                self.BAs = np.array(BAs)

                atlas_mni = nib.affines.apply_affine(
                    affine,
                    self.atlas_xyz
                )

                nearest_points = np.full(
                    (self.hit_mni.shape[0], 3),
                    np.nan
                )

                nearest_BAs = np.full(
                    self.hit_mni.shape[0],
                    np.nan
                )

                dist_atlas = np.full(
                    self.hit_mni.shape[0],
                    np.nan
                )

                idx_atlas = np.full(
                    self.hit_mni.shape[0],
                    -1,
                    dtype=int
                )

                valid_mask = np.isfinite(
                    self.hit_mni
                ).all(axis=1)

                tree = cKDTree(atlas_mni)

                dist_valid, idx_valid = tree.query(
                    self.hit_mni[valid_mask],
                    k=1
                )

                dist_atlas[valid_mask] = dist_valid
                idx_atlas[valid_mask] = idx_valid

                nearest_points[valid_mask] = atlas_mni[idx_valid]
                nearest_BAs[valid_mask] = self.BAs[idx_valid]

                # Compute scanner-space distance between scalp and brain
                dist_scanner = np.linalg.norm(
                    self.hit_points_scanner - self.anchor_points,
                    axis=1
                )

                # Select target ROI
                mask = (
                    np.isin(nearest_BAs, target) &
                    (self.brain_nearest_points[:, 0] < 0) &
                    (dist_scanner < 20)
                )

                self.mask_target = mask
                self.nearest_BAs = nearest_BAs
                self.nearest_atlas_points = nearest_points

                # -------------------------
                # Construct ROI in scanner space
                # -------------------------
                self.target_pts_head = self.hit_points_scanner[mask]

                self.target_pts_head_surface = \
                    self.anchor_points[mask]

                if len(self.target_pts_head) >= 4:

                    cloud_head_brain = pv.PolyData(
                        self.target_pts_head
                    )

                    cloud_head_skin = pv.PolyData(
                        self.target_pts_head_surface
                    )

                    self.mesh_ROI_head_brain = (
                        cloud_head_brain
                        .delaunay_3d()
                        .extract_surface()
                    )

                    self.mesh_ROI_head_skin = (
                        cloud_head_skin
                        .delaunay_3d()
                        .extract_surface()
                    )

                else:
                    self.mesh_ROI_head_brain = pv.PolyData(
                        self.target_pts_head
                    )

                    self.mesh_ROI_head_skin = pv.PolyData(
                        self.target_pts_head_surface
                    )

                # -------------------------
                # Landmark coordinates
                # -------------------------
                self.scanControl = np.array([
                    self.montage[k]
                    for k in mni_landmark.keys()
                ])

                self.mniControl = np.array([
                    apply_affine(
                        mni_landmark[k],
                        self.mat_mni_to_3dscan
                    )
                    for k in mni_landmark.keys()
                ])

                # -------------------------
                # Construct ROI in MNI space
                # -------------------------
                self.target_pts_mni = \
                    self.brain_nearest_points[mask]

                if len(self.target_pts_mni) >= 4:

                    cloud_mni = pv.PolyData(
                        self.target_pts_mni
                    )

                    self.mesh_ROI_mni = (
                        cloud_mni
                        .delaunay_3d()
                        .extract_surface()
                    )

                else:
                    self.mesh_ROI_mni = pv.PolyData(
                        self.target_pts_mni
                    )

                # -------------------------
                # Compute ROI centroids
                # -------------------------
                if len(self.target_pts_head) > 0:

                    self.centroid_head = np.mean(
                        self.target_pts_head,
                        axis=0
                    )

                    self.centroid_skin = np.mean(
                        self.target_pts_head_surface,
                        axis=0
                    )

                else:
                    self.centroid_head = np.array(
                        [np.nan, np.nan, np.nan]
                    )

                    self.centroid_skin = np.array(
                        [np.nan, np.nan, np.nan]
                    )

                if len(self.target_pts_mni) > 0:

                    self.centroid_mni = np.mean(
                        self.target_pts_mni,
                        axis=0
                    )

                else:
                    self.centroid_mni = np.array(
                        [np.nan, np.nan, np.nan]
                    )

                self.plot_calculation_result()
                
    def plot_mesh_landmarks(self):
        line = pv.lines_from_points(self.arc_pts)
        p = pv.Plotter()

        p.add_mesh(self.skin, show_edges=False)

        for l in self.landmarks.keys():
            temp = np.array([self.landmarks[l]])
            p.add_points(temp, color="red", point_size=10, render_points_as_spheres=True)

        for m in self.montage.keys():
            p.add_points(self.montage[m], color="green", point_size=10, render_points_as_spheres=True)

        for c in self.control_points.keys():
            temp = np.array(self.control_points[c])
            p.add_points(temp, color="blue", point_size=10, render_points_as_spheres=True)

        p.add_mesh(line, color="cyan", line_width=5)
        p.add_legend(
            [
                ["Landmark(original)", "red"],
                ["Landmark(modified)", "green"],
                ["Control points", "blue"],
                ["Midline", "cyan"]
            ],
            bcolor="white"
        )
        p.show()

    def plot_calculation_result(self):
        p = pv.Plotter(shape=(1, 2))

        # Scanner/head coordinate system
        p.subplot(0, 0)
        p.add_mesh(self.skin, color="lightgray", opacity=0.7)
        p.add_mesh(self.mesh, color="pink", opacity=0.4)

        if self.mesh_ROI_head_brain.n_points > 0:
            p.add_mesh(self.mesh_ROI_head_brain, color="blue")
        if self.mesh_ROI_head_skin.n_points > 0:
            p.add_mesh(self.mesh_ROI_head_skin, color="green", opacity=0.4)

        if np.isfinite(self.centroid_head).all():
            p.add_points(self.centroid_head[None, :], color="red", point_size=20, render_points_as_spheres=True)
        if np.isfinite(self.centroid_skin).all():
            p.add_points(self.centroid_skin[None, :], color="red", point_size=20, render_points_as_spheres=True)

        p.add_points(self.scanControl, color="black", point_size=10, render_points_as_spheres=True)
        p.add_points(self.mniControl, color="green", point_size=10, render_points_as_spheres=True)

        # MNI coordinate system
        p.subplot(0, 1)
        p.add_mesh(self.brain_model, color="gray", opacity=0.5)

        if self.mesh_ROI_mni.n_points > 0:
            p.add_mesh(self.mesh_ROI_mni, color="blue")

        if np.isfinite(self.centroid_mni).all():
            p.add_points(self.centroid_mni[None, :], color="red", point_size=20, render_points_as_spheres=True)

        p.show()

    def convert_scanner_coord_to_MNI(self, xyz: np.ndarray, thr: float = 30):
        """
        Map scanner-space coordinates to the corresponding brain-surface
        coordinates in MNI space.

        For each scanner-space coordinate, the nearest skin anchor point is
        found first. The coordinate is then mapped to the MNI brain-surface
        point associated with that anchor.

        Parameters
        ----------
        xyz : (3,) or (N,3)
            Input coordinate or coordinates in scanner space.

        thr : float
            Distance threshold to the nearest skin anchor in scanner space [mm].

        Returns
        -------
        dict
            Dictionary containing the input coordinate, nearest skin anchor,
            mapped brain coordinate in scanner space, mapped brain coordinate
            in MNI space, distance to skin, anchor ID, and validity flag.
        """
        pts = np.asarray(xyz, dtype=float)
        one_point = False

        if pts.ndim == 1:
            pts = pts[None, :]
            one_point = True

        dist, idx = self.skin_anchor_tree.query(pts, k=1)

        nearest_skin_scanner = self.skin_anchor_points[idx]
        mapped_brain_scanner = self.brain_surface_scanner_from_anchor[idx]
        mapped_brain_mni = self.brain_surface_mni_from_anchor[idx]
        is_valid = dist <= thr

        out = {
            "input_scanner": pts.tolist() if not one_point else pts[0],
            "nearest_skin_scanner": nearest_skin_scanner.tolist() if not one_point else nearest_skin_scanner[0],
            "mapped_brain_scanner": mapped_brain_scanner.tolist() if not one_point else mapped_brain_scanner[0],
            "mapped_brain_mni": mapped_brain_mni.tolist() if not one_point else mapped_brain_mni[0],
            "distance_to_skin": dist.tolist() if not one_point else float(dist[0]),
            "anchor_id": idx.tolist() if not one_point else int(idx[0]),
            "is_valid": is_valid.tolist() if not one_point else bool(is_valid[0]),
        }
        return out

def save_geometrical_data(scan: Scanner_Param, obj_data: dict):
    save_path = obj_data["ply"][0:-4]+"_np.npz"
    np.savez(save_path,
             obj_data = obj_data,
             BAs = scan.BAs, anchor_points = scan.anchor_points,
             arc_pts = np.array(scan.arc_pts),atlas_xyz = scan.atlas_xyz,
             brain_distances = scan.brain_distances,
             brain_nearest_ids = scan.brain_nearest_ids,
             brain_nearest_points = scan.brain_nearest_points,
             brain_points_mni = scan.brain_points_mni,
             brain_points_scanner = scan.brain_points_scanner,
             brain_surface_mni_from_anchor = scan.brain_surface_mni_from_anchor,
             brain_surface_scanner_from_anchor = scan.brain_surface_scanner_from_anchor,
             centroid_head = scan.centroid_head,
             centroid_mni = scan.centroid_mni,
             centroid_skin = scan.centroid_skin,
             control_points = scan.control_points,
             hit_directions = scan.hit_directions,
             hit_distances = scan.hit_distances,
             hit_mni = scan.hit_mni,
             hit_points_scanner = scan.hit_points_scanner,
             landmarks = scan.landmarks,
             mask_target = scan.mask_target,
             mat_3dscan_to_mni = scan.mat_3dscan_to_mni,
             mat_mni_to_3dscan = scan.mat_mni_to_3dscan,
             mniControl = scan.mniControl,
             montage = scan.montage,
             nearest_BAs = scan.nearest_BAs,
             nearest_atlas_points = scan.nearest_atlas_points,
             normals = scan.normals,
             scanControl = scan.scanControl,
             skin_anchor_points = scan.skin_anchor_points,
             target_pts_head = scan.target_pts_head,
             target_pts_head_surface = scan.target_pts_head_surface,
             target_pts_mni = scan.target_pts_mni
             )

#%% Processes
if __name__ == "__main__":
    obj_data = load_obj_data()

    if obj_data is not None:
        scan = Scanner_Param(obj_data)

        if not scan.isLoad:
            save_geometrical_data(scan, obj_data)
            print("Geometrical data saved.")
        else:
            print("Cached geometrical data loaded. No recalculation/save needed.")
