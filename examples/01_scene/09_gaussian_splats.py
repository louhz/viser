"""Gaussian splats

Viser includes a WebGL-based Gaussian splat renderer.

**Features:**

* :meth:`viser.SceneApi.add_gaussian_splats` to add a Gaussian splat object
* Correct sorting when multiple splat objects are present
* Compositing with other scene objects

.. note::
    This example requires external assets. To download them, run:

    .. code-block:: bash

        git clone https://github.com/nerfstudio-project/viser.git
        cd viser/examples
        ./assets/download_assets.sh
        python 01_scene/09_gaussian_splats.py  # With viser installed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TypedDict

import os
import mujoco as mj
import mujoco.viewer as mjv
import sys
import mujoco.mjx as mjx

import numpy as np
import numpy.typing as npt
import tyro
from plyfile import PlyData

import viser
from viser import transforms as tf

from typing import Dict

SEMANTIC_HANDLE_REGISTRY: Dict[int, object] = {}
# add transform with mujoco
# from engine.mdh import create_transformation_matrix_mdh, reflect_axis
# from engine.mjx import apply_action_and_step



# --- small math helpers -------------------------------------------------------
def quat_wxyz_to_rotmat(q):
    """[w,x,y,z] -> 3x3 rotation matrix."""
    w, x, y, z = q
    # normalize to be safe
    n = np.linalg.norm(q)
    if n == 0:
        return np.eye(3)
    w, x, y, z = q / n
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)],
    ], dtype=float)
    return R

def rotmat_to_quat_wxyz(R):
    """3x3 rotation matrix -> [w,x,y,z]."""
    m00, m01, m02 = R[0,0], R[0,1], R[0,2]
    m10, m11, m12 = R[1,0], R[1,1], R[1,2]
    m20, m21, m22 = R[2,0], R[2,1], R[2,2]
    t = m00 + m11 + m22
    if t > 0.0:
        S = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)

def yz_swap_on_pose(R, p):
    """
    Change of basis: (x, y, z) -> (x, z, y) for BOTH rotation and translation.
    R_new = P * R * P^T, p_new = P * p, where P swaps Y/Z.
    """
    P = np.array([[1,0,0],[0,0,1],[0,1,0]], dtype=float)  # yz swap
    return P @ R @ P.T, (P @ p)

def build_transform(R, p, scale=1.0):
    """
    Compose a 4x4 transform. Scale is uniform and only applied to the rotation
    block; translation is not scaled.
    """
    T = np.eye(4, dtype=float)
    T[:3, :3] = scale * R
    T[:3, 3] = p
    return T

def _try_set_splat_transform(splat, T, scale=1.0):
    """
    Best-effort apply transform to a Viser object handle across common APIs.
    """
    # If the handle exposes a dedicated scale attribute, set it first.
    if hasattr(splat, "scale"):
        try:
            splat.scale = float(scale)
        except Exception:
            pass

    # 1) set_transform(4x4)
    if hasattr(splat, "set_transform"):
        try:
            splat.set_transform(T)
            return
        except Exception:
            pass

    # 2) direct transform property
    if hasattr(splat, "transform"):
        try:
            splat.transform = T
            return
        except Exception:
            pass

    # 3) set_pose(position=..., wxyz=...)
    R = T[:3, :3]
    p = T[:3, 3]
    # If we embedded scale in R, strip it to get a pure rotation for the quat.
    s_est = np.cbrt(np.linalg.det(R)) if np.linalg.det(R) > 0 else 1.0
    if s_est != 0:
        R_no_scale = R / s_est
    else:
        R_no_scale = R
    q = rotmat_to_quat_wxyz(R_no_scale)

    if hasattr(splat, "set_pose"):
        try:
            splat.set_pose(position=p, wxyz=q)
            return
        except Exception:
            pass

    # 4) last-resort: set position / wxyz directly
    if hasattr(splat, "position"):
        try:
            splat.position = p
        except Exception:
            pass
    if hasattr(splat, "wxyz"):
        try:
            splat.wxyz = q
        except Exception:
            pass

def register_semantic_handle(semantic_id_value: int, handle) -> None:
    """Remember which handle corresponds to a given semantic id."""
    sid = int(semantic_id_value)
    SEMANTIC_HANDLE_REGISTRY[sid] = handle
    # Tag the handle for debugging / fallback searches
    try:
        setattr(handle, "semantic_id_value", sid)
    except Exception:
        pass

def _find_splat_handle_by_semantic(server, target_semantic_id: int):
    """
    Locate the splat handle created by your semantic filter UI.
    Priority:
      1) the registry (set when user clicks 'Keep id = ...'),
      2) fallback: scan common containers for a handle whose name ends with '_<id>'.
    """
    sid = int(target_semantic_id)

    # 1) Try the registry (fast path)
    h = SEMANTIC_HANDLE_REGISTRY.get(sid, None)
    if h is not None:
        return h

    # 2) Fallback: best-effort scan for a node with name like ".../filtered_<sid>"
    suffix = f"_{sid}"
    for attr_name in ("objects", "nodes", "drawables", "scene_nodes", "renderables"):
        if hasattr(server, attr_name):
            container = getattr(server, attr_name)
            try:
                it = container.values() if hasattr(container, "values") else container
                for obj in it:
                    nm = getattr(obj, "name", "")
                    if isinstance(nm, str) and "filtered" in nm and nm.endswith(suffix):
                        return obj
            except Exception:
                pass

    # Optional: some builds expose scene.find/get by name
    try:
        scene = getattr(server, "scene", None)
        fetch = getattr(scene, "find", None) or getattr(scene, "get", None)
        if callable(fetch):
            cand = fetch(f"/gaussians/filtered{suffix}")
            if cand is not None:
                return cand
    except Exception:
        pass

    raise RuntimeError(
        f"Could not find a splat handle for semantic id {sid}. "
        f"First filter the model via 'Semantic filtering' → click 'Keep id = {sid}', "
        "then press Play again."
    )


def load_mujoco():
    """
    Load MuJoCo-logged world poses and play them on the splat selected by semantic id.
    Coordinate fix: xyz -> xzy (swap Y/Z). Scale: 1.0 -> 1.3.
    """
    # ---- config ----
    path = "ketchup_pose.txt"
    semantic_target_id = 12        # <-- your ketchup semantic id
    uniform_scale = 1.3
    basis_fix = "xyz->xzy"

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Pose file not found: {path}")

    raw = np.loadtxt(path, comments="#")
    if raw.ndim == 1:
        raw = raw[None, :]

    t = raw[:, 0]
    p_mj = raw[:, 1:4]
    q_mj = raw[:, 4:8]

    R_mj = np.stack([quat_wxyz_to_rotmat(q) for q in q_mj], axis=0)

    if basis_fix == "xyz->xzy":
        R_view = np.empty_like(R_mj)
        p_view = np.empty_like(p_mj)
        for i in range(len(t)):
            R_view[i], p_view[i] = yz_swap_on_pose(R_mj[i], p_mj[i])
    else:
        R_view, p_view = R_mj, p_mj

    Ts = [build_transform(R_view[i], p_view[i], scale=uniform_scale) for i in range(len(t))]

    server = (globals().get("server")
              or globals().get("srv")
              or globals().get("viewer")
              or globals().get("client"))
    if server is None:
        raise RuntimeError("No Viser server found in globals (server/srv/viewer/client).")

    btn_label = f"▶ Play MuJoCo Trajectory (semantic id = {semantic_target_id})"
    try:
        play_btn = server.add_gui_button(btn_label)
    except Exception:
        play_btn = server.add_gui_button(btn_label + " (root)")

    @play_btn.on_click
    def _(_evt):
        # Look up the filtered handle on click (handles may be created later)
        try:
            splat = _find_splat_handle_by_semantic(server, semantic_target_id)
        except Exception as e:
            print(f"[load_mujoco] {e}")
            print("Tip: In 'Semantic filtering' UI, choose the id and click 'Keep id = ...' first.")
            return

        # Apply trajectory
        # if hasattr(splat, "scale"):
        #     try:
        #         splat.scale = float(uniform_scale)
        #     except Exception:
        #         pass

        for i, T in enumerate(Ts):
            _try_set_splat_transform(splat, T, scale=uniform_scale)
            if i + 1 < len(t):
                dt = max(1e-3, float(t[i + 1] - t[i]))
            else:
                dt = 0.0
            try:
                server.sleep(dt)
            except Exception:
                time.sleep(dt)

    print(
        f"[load_mujoco] Ready. Loaded {len(t)} poses from '{path}', "
        f"mapped xyz→xzy, scale={uniform_scale}. "
        f"Click '{btn_label}' after you 'Keep id = {semantic_target_id}' in the Semantic UI."
    )

class SplatFile(TypedDict):
    """Data loaded from an antimatter15-style splat file."""

    centers: npt.NDArray[np.floating]
    """(N, 3)."""
    rgbs: npt.NDArray[np.floating]
    """(N, 3). Range [0, 1]."""
    opacities: npt.NDArray[np.floating]
    """(N, 1). Range [0, 1]."""
    covariances: npt.NDArray[np.floating]
    """(N, 3, 3)."""


class SplatFileSam(TypedDict):
    """Data loaded from an antimatter15-style splat file."""

    centers: npt.NDArray[np.floating]
    """(N, 3)."""
    rgbs: npt.NDArray[np.floating]
    """(N, 3). Range [0, 1]."""
    opacities: npt.NDArray[np.floating]
    """(N, 1). Range [0, 1]."""
    covariances: npt.NDArray[np.floating]
    """(N, 3, 3)."""
    semantic_id: npt.NDArray[np.floating]
    """(N, 1). Semantic ID for each Gaussian."""

def load_splat_file(splat_path: Path, center: bool = False) -> SplatFile:
    """Load an antimatter15-style splat file."""
    start_time = time.time()
    splat_buffer = splat_path.read_bytes()
    bytes_per_gaussian = (
        # Each Gaussian is serialized as:
        # - position (vec3, float32)
        3 * 4
        # - xyz (vec3, float32)
        + 3 * 4
        # - rgba (vec4, uint8)
        + 4
        # - ijkl (vec4, uint8), where 0 => -1, 255 => 1.
        + 4
    )
    assert len(splat_buffer) % bytes_per_gaussian == 0
    num_gaussians = len(splat_buffer) // bytes_per_gaussian

    # Reinterpret cast to dtypes that we want to extract.
    splat_uint8 = np.frombuffer(splat_buffer, dtype=np.uint8).reshape(
        (num_gaussians, bytes_per_gaussian)
    )
    scales = splat_uint8[:, 12:24].copy().view(np.float32)
    wxyzs = splat_uint8[:, 28:32] / 255.0 * 2.0 - 1.0
    Rs = tf.SO3(wxyzs).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    centers = splat_uint8[:, 0:12].copy().view(np.float32)
    if center:
        centers -= np.mean(centers, axis=0, keepdims=True)
    print(
        f"Splat file with {num_gaussians=} loaded in {time.time() - start_time} seconds"
    )
    return {
        "centers": centers,
        # Colors should have shape (N, 3).
        "rgbs": splat_uint8[:, 24:27] / 255.0,
        "opacities": splat_uint8[:, 27:28] / 255.0,
        # Covariances should have shape (N, 3, 3).
        "covariances": covariances,
    }



# max sh degree is only 3 in this case
def load_ply_file(ply_file_path: Path, center: bool = False) -> SplatFile:
    """Load Gaussians stored in a PLY file."""
    start_time = time.time()

    SH_C0 = 0.28209479177387814

    plydata = PlyData.read(ply_file_path)
    v = plydata["vertex"]
    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1)
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1))
    wxyzs = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
    colors = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    opacities = 1.0 / (1.0 + np.exp(-v["opacity"][:, None]))

    Rs = tf.SO3(wxyzs).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    if center:
        positions -= np.mean(positions, axis=0, keepdims=True)

    num_gaussians = len(v)
    print(
        f"PLY file with {num_gaussians=} loaded in {time.time() - start_time} seconds"
    )
    return {
        "centers": positions,
        "rgbs": colors,
        "opacities": opacities,
        "covariances": covariances,
    }

def load_ply_file_sam(ply_file_path: Path, center: bool = False) -> SplatFileSam:
    """Load Gaussians stored in a PLY file."""
    start_time = time.time()

    SH_C0 = 0.28209479177387814

    plydata = PlyData.read(ply_file_path)
    semantic_id = np.array(plydata.elements[0]["semantic_id"])[..., np.newaxis]
    v = plydata["vertex"]
    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1)
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1))
    wxyzs = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
    colors = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    opacities = 1.0 / (1.0 + np.exp(-v["opacity"][:, None]))

    Rs = tf.SO3(wxyzs).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    if center:
        positions -= np.mean(positions, axis=0, keepdims=True)

    num_gaussians = len(v)
    print(
        f"PLY file with {num_gaussians=} loaded in {time.time() - start_time} seconds"
    )
    return {
        "centers": positions,
        "rgbs": colors,
        "opacities": opacities,
        "covariances": covariances,
        'semantic_id': semantic_id,
    }


def load_ply(path, max_sh_degree=3):
    plydata = PlyData.read(path)

    xyz = np.stack(
        (np.array(plydata.elements[0]["x"]), np.array(plydata.elements[0]["y"]), np.array(plydata.elements[0]["z"])),
        axis=1,
    )
    opacities = np.array(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 1, 3), dtype=np.float32)
    features_dc[:, 0, 0] = np.array(plydata.elements[0]["f_dc_0"])
    features_dc[:, 0, 1] = np.array(plydata.elements[0]["f_dc_1"])
    features_dc[:, 0, 2] = np.array(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
    # e.g. 3 * (max_sh_degree + 1) ^ 2 - 3  ->  3*(3+1)^2 - 3 = 3*16 - 3 = 48 - 3 = 45
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)), dtype=np.float32)
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.array(plydata.elements[0][attr_name])
    # reshape to (num_points, 3, (#SHcoeffs except DC))
    
    features_extra = features_extra.reshape((features_extra.shape[0], 3, 15))

    # scale
    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)), dtype=np.float32)
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.array(plydata.elements[0][attr_name])

    # rotation
    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)), dtype=np.float32)
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.array(plydata.elements[0][attr_name])

    return xyz, features_dc, features_extra, opacities, scales, rots


def load_ply_sam(path, max_sh_degree=3):
    plydata = PlyData.read(path)

    xyz = np.stack(
        (np.array(plydata.elements[0]["x"]), np.array(plydata.elements[0]["y"]), np.array(plydata.elements[0]["z"])),
        axis=1,
    )
    opacities = np.array(plydata.elements[0]["opacity"])[..., np.newaxis]
    semantic_id = np.array(plydata.elements[0]["semantic_id"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1), dtype=np.float32)
    features_dc[:, 0, 0] = np.array(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.array(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.array(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)), dtype=np.float32)
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.array(plydata.elements[0][attr_name])
    features_extra = features_extra.reshape((features_extra.shape[0], 3, -1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)), dtype=np.float32)
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.array(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)), dtype=np.float32)
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.array(plydata.elements[0][attr_name])

    return xyz, features_dc, features_extra, opacities, scales, rots, semantic_id



def filter_with_semantic(semantic_id,assigned_ids,xyz,opacities,scales,features_extra,rots,features_dc,index=0):
    semantic_id_ind=(semantic_id==assigned_ids[index]).reshape(-1)
    semantic_id_ind_sam=semantic_id[semantic_id==assigned_ids[index]].reshape(-1,1)
        
    select_xyz= np.array(xyz[semantic_id_ind])
    select_opacities=  np.array(opacities[semantic_id_ind])
    select_scales =  np.array(scales[semantic_id_ind])
    select_features_extra =  np.array(features_extra[semantic_id_ind])
    select_rotation =  np.array(rots[semantic_id_ind])
    select_feature_dc =  np.array(features_dc[semantic_id_ind])


    return select_xyz,select_opacities,select_scales,select_features_extra,select_rotation,select_feature_dc,semantic_id_ind_sam


def filter_with_semantic_sam_first(
    semantic_id: np.ndarray,
    assigned_ids: np.ndarray,
    splat: SplatFileSam,
    index: int = 0,
) -> SplatFileSam:
    """
    Filter a 'first-format' splat dict (centers, rgbs, opacities, covariances) by a target semantic id.

    Parameters
    ----------
    semantic_id : (N,) or (N,1) np.ndarray
        Per-point semantic labels.
    assigned_ids : (K,) np.ndarray
        List/array of semantic ids; we keep the points equal to assigned_ids[index].
    splat : dict
        Dict from load_ply_file: {'centers','rgbs','opacities','covariances'}.
        All arrays must have length N on axis 0.
    index : int
        Which entry in assigned_ids to select.

    Returns
    -------
    filtered_splat : dict
        Same keys as input splat, but only rows where semantic_id == assigned_ids[index].
    semantic_id_ind_sam : (M,1) np.ndarray
        The selected semantic ids (one per kept point), shaped as a column vector.
    """
    sid = np.asarray(semantic_id).reshape(-1)
    aids = np.asarray(assigned_ids).reshape(-1)

    if not (0 <= index < aids.size):
        raise IndexError(f"'index' {index} is out of range for assigned_ids of length {aids.size}.")

    target = aids[index]
    mask = (sid == target).reshape(-1)

    # Optional sanity checks: ensure all splat arrays are length-N along axis 0
    N = sid.shape[0]
    for key in ("centers", "rgbs", "opacities", "covariances"):
        if key not in splat:
            raise KeyError(f"Missing key '{key}' in splat.")
        if splat[key].shape[0] != N:
            raise ValueError(f"splat['{key}'] has length {splat[key].shape[0]}, expected {N} to match semantic_id.")

    filtered_splat: SplatFileSam = {
        "centers":     np.array(splat["centers"][mask], copy=True),
        "rgbs":        np.array(splat["rgbs"][mask], copy=True),
        "opacities":   np.array(splat["opacities"][mask], copy=True),    # stays (M,1)
        "covariances": np.array(splat["covariances"][mask], copy=True),  # stays (M,3,3)
        'semantic_id': np.array(sid[mask], copy=True).reshape(-1, 1),  # add semantic_id key
    }



    return filtered_splat

# def update_with_semantic():
import numpy as np

def setup_editable_keep_ui(
    server,
    handle_ref,              # {"h": <GaussianSplatHandle>}  (mutable dict, see below)
    splat_all,               # dict with keys: centers, rgbs, opacities, covariances
    semantic_id,             # (N,) or (N,1) numpy array
    assigned_ids,            # sequence of unique IDs; index i selects assigned_ids[i]
    default_index: int = 0,
    remove_button=None,      # optional preexisting remove button; will be (re)wired
):
    """
    Creates GUI elements that let the user edit the index i at runtime and apply the filter.

    Requirements:
      - filter_with_semantic_sam_first(semantic_id, assigned_ids, splat, index) -> (filtered_splat, ids)
      - handle_ref is a dict: {"h": <current GaussianSplatHandle>}
      - server.scene.add_gaussian_splats(...) is available (from SceneApi)
    """

    # ------- Small helper to (re)draw a splat node from a splat dict -------
    def _replace_splats(splat_dict, name_suffix=""):
        old = handle_ref.get("h", None)
        name = getattr(old, "name", f"/gaussians/filtered{name_suffix}")
        wxyz = getattr(old, "wxyz", (1.0, 0.0, 0.0, 0.0))
        position = getattr(old, "position", (0.0, 0.0, 0.0))
        visible = getattr(old, "visible", True)

        try:
            if old is not None:
                old.remove()
        except Exception:
            pass

        new_h = server.scene.add_gaussian_splats(
            name=name,
            centers=np.ascontiguousarray(splat_dict["centers"], dtype=np.float32),
            covariances=np.ascontiguousarray(splat_dict["covariances"], dtype=np.float32),
            rgbs=np.ascontiguousarray(np.clip(splat_dict["rgbs"], 0.0, 1.0), dtype=np.float32),
            opacities=np.ascontiguousarray(np.clip(splat_dict["opacities"], 0.0, 1.0), dtype=np.float32),
            wxyz=wxyz,
            position=position,
            visible=visible,
        )
        handle_ref["h"] = new_h
        return new_h  

    # ------- Build the controls -------
    folder = server.gui.add_folder("Semantic filtering", expand_by_default=True)
    with folder:
        # Editable index i
        i_input = server.gui.add_number(
            "i (index into assigned_ids)",
            initial_value=int(np.clip(default_index, 0, len(assigned_ids)-1)),
            min=0, max=max(0, len(assigned_ids)-1), step=1
        )

        # A tiny readout showing the actual semantic ID and estimated count
        init_i = int(i_input.value)
        init_id = assigned_ids[init_i]
        init_count = int((np.asarray(semantic_id).reshape(-1) == init_id).sum())
        id_readout = server.gui.add_markdown(
            f"**Selected:** `i={init_i}` → id=`{init_id}` — **{init_count}** splats"
        )

        # The action buttons
        keep_btn  = server.gui.add_button(f"Keep id = {init_id}")
        reset_btn = server.gui.add_button("Reset (show all)")
        if remove_button is None:
            remove_button = server.gui.add_button("Remove node")

    # Live feedback as the user edits i
    @i_input.on_update
    def _(_e, i_input=i_input, id_readout=id_readout, keep_btn=keep_btn,
           semantic_id=semantic_id, assigned_ids=assigned_ids):
        i_val = int(np.clip(i_input.value, 0, len(assigned_ids)-1))
        sel_id = assigned_ids[i_val]
        count = int((np.asarray(semantic_id).reshape(-1) == sel_id).sum())

        # Update readout
        id_readout.content = f"**Selected:** `i={i_val}` → id=`{sel_id}` — **{count}** splats"

        # Try to update button label (supported in recent viser)
        try:
            keep_btn.label = f"Keep id = {sel_id}"
        except Exception:
            # If label setter is unavailable, it's okay; the readout already shows the choice.
            pass

    @keep_btn.on_click
    def _(
        _,
        i_input=i_input,
        semantic_id=semantic_id,
        assigned_ids=assigned_ids,
        splat_all=splat_all,
    ):
        i_val = int(np.clip(i_input.value, 0, len(assigned_ids)-1))
        sel_id = int(assigned_ids[i_val])

        filtered_splat = filter_with_semantic_sam_first(
            semantic_id=semantic_id,
            assigned_ids=assigned_ids,
            splat=splat_all,
            index=i_val,
        )
        h = _replace_splats(filtered_splat, name_suffix=f"_{sel_id}")

        # <<< NEW: remember that this handle is the 'sel_id' object
        register_semantic_handle(sel_id, h)

    # Restore the full (unfiltered) model
    @reset_btn.on_click
    def _(_):
        _replace_splats(splat_all, name_suffix="_all")

    # Make sure the remove button always acts on the latest handle
    @remove_button.on_click
    def _(_):
        h = handle_ref.get("h", None)
        if h is not None:
            try:
                h.remove()
            except Exception:
                pass
        remove_button.remove()


# change covariances without change of centers



def change_covariances(
    server: viser.ViserServer,
    handle: viser.GaussianSplatHandle,
    covariances: npt.ArrayLike,
    indices: npt.ArrayLike | None = None,
) -> None:
    """Replace per-splat covariances without moving centers.

    Args:
        server: Viser server (batched update).
        handle: GaussianSplatHandle from `add_gaussian_splats(...)`.
        covariances:
            - shape (M, 6) or (6,) in triu order: (xx, xy, xz, yy, yz, zz)
            - shape (M, 3, 3) or (3, 3) full symmetric covariances
            If a single covariance is given ((6,) or (3,3)), it will broadcast to selection.
        indices:
            - None: update all splats
            - 1D int array/list: positions to update
            - 1D bool array: mask of splats to update (len == N)

    Notes:
        - Only the 12-byte covariance block per splat is touched.
        - Values are stored as float16 and clamped to finite f16 range.
    """
    if not isinstance(handle, viser.GaussianSplatHandle):
        raise TypeError("handle must be a GaussianSplatHandle.")
    if not hasattr(handle, "buffer"):
        raise AttributeError("Expected `handle.buffer` on GaussianSplatHandle.")

    buf = handle.buffer  # (N, 8) uint32
    if buf.ndim != 2 or buf.shape[1] != 8 or buf.dtype != np.uint32:
        raise ValueError(f"Unexpected buffer shape/dtype: {buf.shape} {buf.dtype} (expected (N,8) uint32)")

    N = buf.shape[0]

    # Build selection
    if indices is None:
        sel = slice(None)
        M = N
    else:
        idx = np.asarray(indices)
        if idx.dtype == bool:
            if idx.shape[0] != N:
                raise ValueError(f"Boolean mask length {idx.shape[0]} must equal number of splats {N}.")
            sel = idx
            M = int(np.count_nonzero(idx))
        else:
            idx = idx.astype(np.intp, copy=False).ravel()
            sel = idx
            M = idx.shape[0]
        if M == 0:
            return

    cov = np.asarray(covariances)

    # ---- normalize input to shape (M, 6) float32 in triu order ----
    def _full3x3_to_triu6(C: np.ndarray) -> np.ndarray:
        C = 0.5 * (C + np.swapaxes(C, -1, -2))  # symmetrize just in case
        out = np.empty(C.shape[:-2] + (6,), dtype=C.dtype)
        out[..., 0] = C[..., 0, 0]  # xx
        out[..., 1] = C[..., 0, 1]  # xy
        out[..., 2] = C[..., 0, 2]  # xz
        out[..., 3] = C[..., 1, 1]  # yy
        out[..., 4] = C[..., 1, 2]  # yz
        out[..., 5] = C[..., 2, 2]  # zz
        return out

    if cov.ndim == 1 and cov.shape == (6,):
        cov_triu32 = np.broadcast_to(cov[None, :], (M, 6)).astype(np.float32, copy=True)
    elif cov.ndim == 2 and cov.shape == (M, 6):
        cov_triu32 = cov.astype(np.float32, copy=False)
    elif cov.ndim == 2 and cov.shape == (3, 3):
        cov_triu32 = np.broadcast_to(_full3x3_to_triu6(cov)[None, :], (M, 6)).astype(np.float32, copy=True)
    elif cov.ndim == 3 and cov.shape == (M, 3, 3):
        cov_triu32 = _full3x3_to_triu6(cov.astype(np.float32, copy=False))
    else:
        # Allow (N,6) or (N,3,3) when indices is None
        if indices is None and cov.ndim == 2 and cov.shape == (N, 6):
            cov_triu32 = cov.astype(np.float32, copy=False)
            M = N
            sel = slice(None)
        elif indices is None and cov.ndim == 3 and cov.shape == (N, 3, 3):
            cov_triu32 = _full3x3_to_triu6(cov.astype(np.float32, copy=False))
            M = N
            sel = slice(None)
        else:
            raise ValueError(
                "Unsupported `covariances` shape. Provide (M,6), (6,), (M,3,3), or (3,3). "
                f"Got {cov.shape} for selection size {M}."
            )

    # Convert to float16 (clamp to valid finite f16 range) and ensure contiguity.
    cov_triu16 = np.clip(cov_triu32, -65504.0, 65504.0).astype(np.float16, copy=False)
    cov_triu16 = np.ascontiguousarray(cov_triu16)  # <- avoids the 'last axis must be contiguous' error

    # ---- write into the handle's buffer WITHOUT viewing the input ----
    raw = buf.view(np.uint8).reshape(N, 32).copy()
    # View the target slice [16:28) as float16 (N,6) and assign; this side is contiguous.
    cov_block_f16 = raw[:, 16:28].view(np.float16).reshape(N, 6)
    cov_block_f16[sel] = cov_triu16  # centers/rgba untouched

    new_buffer = raw.view(np.uint32).reshape(N, 8)
    with server.atomic():
        handle.buffer = new_buffer

# add camviewer  



# add camera trajectory


# 
    


def rescale_alpha(
    server: viser.ViserServer,
    handle: viser.GaussianSplatHandle,
    alpha_scale: float = 1.0,
) -> None:
    """Multiply only the per-splat alpha (opacity) by `alpha_scale`.

    Args:
        server: Viser server (used for atomic update).
        handle: A GaussianSplatHandle created via `add_gaussian_splats(...)`.
        alpha_scale: Multiplicative factor for opacity; 1.0 is no-op.

    Raises:
        TypeError/ValueError if `handle.buffer` has an unexpected layout.
    """
    if not isinstance(handle, viser.GaussianSplatHandle):
        raise TypeError("handle must be a GaussianSplatHandle.")
    if not hasattr(handle, "buffer"):
        raise AttributeError("Expected `handle.buffer` to exist on GaussianSplatHandle.")

    s = float(alpha_scale)
    if s == 1.0:
        return  # no-op

    buf = handle.buffer  # (N, 8), uint32
    if buf.ndim != 2 or buf.shape[1] != 8 or buf.dtype != np.uint32:
        raise ValueError(
            f"Unexpected buffer shape/dtype: {buf.shape} {buf.dtype} (expected (N,8) uint32)"
        )

    N = buf.shape[0]
    # View as raw bytes for packed access.
    raw = buf.view(np.uint8).reshape(N, 32).copy()

    # RGBA occupies the last 4 bytes [28:32). Alpha is the final (index 3).
    rgba = raw[:, 28:32]                  # shape (N, 4), dtype=uint8
    a = rgba[:, 3].astype(np.float32)     # existing alpha in [0, 255]
    a_new = np.clip(np.round(a * s), 0.0, 255.0).astype(np.uint8)
    rgba[:, 3] = a_new
    raw[:, 28:32] = rgba

    new_buffer = raw.view(np.uint32).reshape(N, 8)
    with server.atomic():
        handle.buffer = new_buffer



def rescale_gaussian_splats(
    server: viser.ViserServer,
    handle: viser.GaussianSplatHandle,
    scale: float = 1.0,
) -> None:
    """Uniformly rescale a Gaussian splat object created via `add_gaussian_splats()`.

    This scales centers by `scale` and covariances (second moments) by `scale**2`
    in the splat's local coordinate frame. Colors/opacity are preserved.
    """
    if not isinstance(handle, viser.GaussianSplatHandle):
        raise TypeError("handle must be a GaussianSplatHandle.")
    if not hasattr(handle, "buffer"):
        raise AttributeError(
            "This viser build exposes Gaussian splats via a `buffer` property; not found."
        )

    s = float(scale)
    if s == 1.0:
        return  # no-op

    buf = handle.buffer  # dtype=uint32, shape (N, 8)
    if buf.ndim != 2 or buf.shape[1] != 8 or buf.dtype != np.uint32:
        raise ValueError(f"Unexpected buffer shape/dtype: {buf.shape} {buf.dtype} (expected (N,8) uint32)")

    N = buf.shape[0]

    # View as raw bytes for precise slicing.
    raw = buf.view(np.uint8).reshape(N, 32)

    # Layout (see viser implementation):
    # - bytes [ 0:12): centers as 3 x float32
    # - bytes [12:16): reserved (keep)
    # - bytes [16:28): 6 x float16 (upper-triangular covariance terms: xx, xy, xz, yy, yz, zz)
    # - bytes [28:32): RGBA as 4 x uint8 (keep)
    #
    # Decode:
    centers_f32 = raw[:, 0:12].view(np.float32).reshape(N, 3)
    cov_triu_f16 = raw[:, 16:28].view(np.float16).reshape(N, 6)

    # Scale:
    centers_scaled = centers_f32 * s
    cov_scaled_f16 = (cov_triu_f16.astype(np.float32) * (s * s)).astype(np.float16)

    # Repack (preserve reserved word and RGBA):
    raw_new = raw.copy()
    raw_new[:, 0:12]  = centers_scaled.view(np.uint8).reshape(N, 12)
    raw_new[:, 16:28] = cov_scaled_f16.view(np.uint8).reshape(N, 12)
    new_buffer = raw_new.view(np.uint32)

    # Apply atomically so clients don't see intermediate frames.
    with server.atomic():
        handle.buffer = new_buffer










def _quaternion_to_matrix(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    """Return a 3x3 rotation matrix from a quaternion (w, x, y, z)."""
    w, x, y, z = map(float, quaternion)
    n = np.sqrt(w*w + x*x + y*y + z*z)
    if n == 0.0:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w/n, x/n, y/n, z/n

    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    R = np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)],
    ], dtype=np.float32)
    return R


def rotate_points(points: npt.NDArray[np.floating],
                  quaternion: tuple[float, float, float, float]
                 ) -> npt.NDArray[np.floating]:
    """Rotate Nx3 (or 3,) points by a quaternion (w, x, y, z)."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts[None, :]  # (1, 3)
    assert pts.shape[-1] == 3, "points must have shape (N, 3) or (3,)"

    R = _quaternion_to_matrix(quaternion)  # (3, 3)
    # For row-vectors, p' = p @ R^T  ⇔  (R p^T)^T
    return (pts @ R.T).astype(np.float32)


# ---------- covariance helpers ----------

def _triu6_to_full3x3(triu: np.ndarray) -> np.ndarray:
    """
    Convert N x 6 upper-triangular (xx, xy, xz, yy, yz, zz) to N x 3 x 3 symmetric matrices.
    """
    triu = np.asarray(triu, dtype=np.float32)
    N = triu.shape[0]
    C = np.zeros((N, 3, 3), dtype=np.float32)
    xx, xy, xz, yy, yz, zz = [triu[:, i] for i in range(6)]
    C[:, 0, 0] = xx
    C[:, 0, 1] = C[:, 1, 0] = xy
    C[:, 0, 2] = C[:, 2, 0] = xz
    C[:, 1, 1] = yy
    C[:, 1, 2] = C[:, 2, 1] = yz
    C[:, 2, 2] = zz
    return C


def _full3x3_to_triu6(C: np.ndarray) -> np.ndarray:
    """
    Convert N x 3 x 3 symmetric matrices to N x 6 upper-triangular ordering
    (xx, xy, xz, yy, yz, zz).
    """
    C = np.asarray(C, dtype=np.float32)
    N = C.shape[0]
    out = np.empty((N, 6), dtype=np.float32)
    out[:, 0] = C[:, 0, 0]
    out[:, 1] = C[:, 0, 1]
    out[:, 2] = C[:, 0, 2]
    out[:, 3] = C[:, 1, 1]
    out[:, 4] = C[:, 1, 2]
    out[:, 5] = C[:, 2, 2]
    return out


# ---------- main transform ----------

def transform_gaussian_splats(
    server: viser.ViserServer,
    handle: viser.GaussianSplatHandle,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),  # (w, x, y, z)
) -> None:
    """
    Transform Gaussian splats by applying translation *then* rotation:
        centers' = R · (centers + t)
        covariances' = R · covariances · R^T
    Colors/opacity are preserved.

    Notes:
      • Translation does not affect covariance; only rotation does.
      • Rotation is specified as a quaternion (w, x, y, z).
    """
    if not isinstance(handle, viser.GaussianSplatHandle):
        raise TypeError("handle must be a GaussianSplatHandle.")
    if not hasattr(handle, "buffer"):
        raise AttributeError("This viser build should expose splats via `handle.buffer`.")

    # View the packed buffer (N, 8) uint32 as raw bytes for slicing.
    buf = handle.buffer
    if buf.ndim != 2 or buf.shape[1] != 8 or buf.dtype != np.uint32:
        raise ValueError(f"Unexpected buffer shape/dtype: {buf.shape} {buf.dtype} (expected (N,8) uint32)")
    N = buf.shape[0]
    raw = buf.view(np.uint8).reshape(N, 32)

    # Decode:
    #  [ 0:12): centers (3 x float32)
    #  [12:16): reserved (keep as-is)
    #  [16:28): cov upper-tri (6 x float16) in order (xx, xy, xz, yy, yz, zz)
    #  [28:32): RGBA (keep)
    centers = raw[:, 0:12].view(np.float32).reshape(N, 3)
    cov_triu = raw[:, 16:28].view(np.float16).reshape(N, 6)

    # Build transform.
    t = np.asarray(translation, dtype=np.float32).reshape(1, 3)  # broadcast
    R = _quaternion_to_matrix(rotation)  # (3, 3)

    # --- centers: translation then rotation ---
    centers_tp = centers + t                 # (N, 3)
    centers_new = (centers_tp @ R.T)         # (N, 3)

    # --- covariances: only rotation (C' = R C R^T) ---
    C = _triu6_to_full3x3(cov_triu.astype(np.float32))  # (N, 3, 3)
    # Vectorized similarity transform: for each n, C'[n] = R @ C[n] @ R^T
    C_rot = np.einsum('ai, nij, bj -> nab', R, C, R).astype(np.float32)
    cov_triu_new = _full3x3_to_triu6(C_rot).astype(np.float16)

    # Repack; preserve reserved and RGBA bytes.
    raw_out = raw.copy()
    raw_out[:, 0:12]  = centers_new.astype(np.float32).view(np.uint8).reshape(N, 12)
    raw_out[:, 16:28] = cov_triu_new.view(np.uint8).reshape(N, 12)
    new_buffer = raw_out.view(np.uint32)

    # Commit atomically to avoid flicker across clients.
    with server.atomic():
        handle.buffer = new_buffer





def insert_gaussian_splats(
    server: viser.ViserServer,
    splat_data: SplatFile,
    name: str = "/gaussian_splats",
) -> viser.GaussianSplatHandle: 
    """Insert a Gaussian splat object into the scene."""
    return server.scene.add_gaussian_splats(
        name=name,
        centers=splat_data["centers"],
        rgbs=splat_data["rgbs"],
        opacities=splat_data["opacities"],
        covariances=splat_data["covariances"],
    )






# ------------------------- basic setters -------------------------

def set_camera(
    client: viser.ClientHandle,
    position: npt.ArrayLike | None = None,
    wxyz: npt.ArrayLike | None = None,
    look_at: npt.ArrayLike | None = None,
    fov: float | None = None,
    near: float | None = None,
    far: float | None = None,
) -> None:
    """Instantly set camera fields atomically."""
    with client.atomic():
        if wxyz is not None:
            client.camera.wxyz = np.asarray(wxyz, dtype=np.float32)
        if position is not None:
            client.camera.position = np.asarray(position, dtype=np.float32)
        if look_at is not None:
            client.camera.look_at = np.asarray(look_at, dtype=np.float32)
        if fov is not None:
            client.camera.fov = float(fov)
        if near is not None:
            client.camera.near = float(near)
        if far is not None:
            client.camera.far = float(far)


# ------------------------- easing & helpers -------------------------

def _ease(t: float, mode: str = "smoothstep") -> float:
    """Simple easing functions on [0,1]."""
    t = np.clip(t, 0.0, 1.0)
    if mode == "linear":
        return t
    if mode == "smoothstep":          # C^1
        return t * t * (3.0 - 2.0 * t)
    if mode == "ease_in_out":         # cosine
        return 0.5 - 0.5 * cos(pi * t)
    if mode == "ease_in":             # quad in
        return t * t
    if mode == "ease_out":            # quad out
        return 1.0 - (1.0 - t) * (1.0 - t)
    return t


def look_at_quaternion(
    eye: npt.ArrayLike,
    target: npt.ArrayLike,
    up: npt.ArrayLike = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Quaternion (wxyz) that orients the camera to look from `eye` to `target`."""
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    f = target - eye
    nf = np.linalg.norm(f)
    if nf < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # no-op
    f /= nf

    # Camera looks down -Z in its local frame; build world basis [x y z]
    z = -f
    x = np.cross(up, z)
    nx = np.linalg.norm(x)
    if nx < 1e-6:
        # up ~ collinear; pick a fallback up
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        x = np.cross(up, z)
        nx = np.linalg.norm(x)
        if nx < 1e-6:
            x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            nx = 1.0
    x /= nx
    y = np.cross(z, x)

    R = np.stack([x, y, z], axis=1).astype(np.float32)  # columns are basis vectors
    return tf.SO3.from_matrix(R).wxyz


# ------------------------- animated moves -------------------------

def animate_camera_to(
    client: viser.ClientHandle,
    target_position: npt.ArrayLike | None = None,
    target_wxyz: npt.ArrayLike | None = None,
    *,
    duration: float = 1.0,
    fps: int = 60,
    ease: str = "smoothstep",
    final_look_at: npt.ArrayLike | None = None,
    also_set_fov: float | None = None,
    also_set_near: float | None = None,
    also_set_far: float | None = None,
) -> None:
    """Animate the camera from its current pose to the target pose.

    Interpolates rigid motion with SE(3) exp/log (no gimbal issues), with optional
    FOV/near/far interpolation. If a target component is None, it keeps current.

    Args:
        duration: seconds for the move.
        fps: steps per second.
        ease: one of {"linear","smoothstep","ease_in_out","ease_in","ease_out"}.
        final_look_at: set at the end (affects mouse orbit center).
    """
    wxyz0 = client.camera.wxyz
    pos0 = client.camera.position

    wxyz1 = np.asarray(target_wxyz, dtype=np.float32) if target_wxyz is not None else wxyz0
    pos1 = np.asarray(target_position, dtype=np.float32) if target_position is not None else pos0

    T0 = tf.SE3.from_rotation_and_translation(tf.SO3(wxyz0), pos0)
    T1 = tf.SE3.from_rotation_and_translation(tf.SO3(wxyz1), pos1)
    Xi = (T0.inverse() @ T1).log()

    steps = max(1, int(round(duration * fps)))
    for k in range(steps + 1):
        t = _ease(k / steps, ease)
        T = T0 @ tf.SE3.exp(Xi * t)

        with client.atomic():
            client.camera.wxyz = T.rotation().wxyz
            client.camera.position = T.translation()
            if also_set_fov is not None:
                f0 = client.camera.fov
                client.camera.fov = float(f0 + (also_set_fov - f0) * t)
            if also_set_near is not None:
                n0 = client.camera.near
                client.camera.near = float(n0 + (also_set_near - n0) * t)
            if also_set_far is not None:
                f1 = client.camera.far
                client.camera.far = float(f1 + (also_set_far - f1) * t)

        client.flush()            # push frame to client
        if k < steps:
            time.sleep(duration / steps)

    if final_look_at is not None:
        client.camera.look_at = np.asarray(final_look_at, dtype=np.float32)


def orbit_camera(
    client: viser.ClientHandle,
    center: npt.ArrayLike,
    *,
    yaw_delta: float = 0.0,    # radians, +CCW around +Z
    pitch_delta: float = 0.0,  # radians, +up/down (clamped)
    radius: float | None = None,
    duration: float = 1.0,
    fps: int = 60,
    up: npt.ArrayLike = (0.0, 0.0, 1.0),
    ease: str = "smoothstep",
) -> None:
    """Orbit camera about `center` by (yaw_delta, pitch_delta). Keeps camera looking at center."""
    c = np.asarray(center, dtype=np.float32)
    eye0 = client.camera.position
    v0 = eye0 - c
    r0 = np.linalg.norm(v0)
    if r0 < 1e-6:
        r0 = 1.0
        v0 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    r = float(radius) if radius is not None else r0

    # Current yaw/pitch (Z-up convention)
    yaw0 = np.arctan2(v0[1], v0[0])
    pitch0 = np.arcsin(np.clip(v0[2] / r0, -1.0, 1.0))
    yaw1 = yaw0 + float(yaw_delta)
    pitch1 = np.clip(pitch0 + float(pitch_delta), -np.pi/2 + 1e-3, np.pi/2 - 1e-3)

    # Target eye
    v1 = np.array([
        np.cos(pitch1) * np.cos(yaw1),
        np.cos(pitch1) * np.sin(yaw1),
        np.sin(pitch1),
    ], dtype=np.float32) * r
    eye1 = c + v1

    # Orientation that looks at center
    wxyz1 = look_at_quaternion(eye1, c, up=np.asarray(up, dtype=np.float32))

    animate_camera_to(
        client,
        target_position=eye1,
        target_wxyz=wxyz1,
        duration=duration,
        fps=fps,
        ease=ease,
        final_look_at=c,
    )


def dolly_camera(
    client: viser.ClientHandle,
    distance: float,
    *,
    duration: float = 0.3,
    fps: int = 60,
    ease: str = "smoothstep",
) -> None:
    """Move the camera along its current forward direction by `distance` (positive = forward)."""
    # Camera forward is -Z in camera frame; get world forward from orientation.
    R = tf.SO3(client.camera.wxyz).as_matrix()        # columns are [x y z]
    forward_world = -R[:, 2]                          # -Z
    target_pos = client.camera.position + forward_world * float(distance)
    animate_camera_to(client, target_position=target_pos, duration=duration, fps=fps, ease=ease)



# reoriginize dataloading and absolute path utilities



# manange the button in a smarter way

def look_at_quaternion(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float32),
) -> np.ndarray:
    """Return (w,x,y,z) that looks from `eye` to `target` with `up`."""
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    f = target - eye
    nf = np.linalg.norm(f)
    if nf < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    f /= nf

    # Camera looks along -Z; build world basis [x y z]
    z = -f
    x = np.cross(up, z)
    nx = np.linalg.norm(x)
    if nx < 1e-6:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        x = np.cross(up, z)
        nx = np.linalg.norm(x)
        if nx < 1e-6:
            x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            nx = 1.0
    x /= nx
    y = np.cross(z, x)

    R = np.stack([x, y, z], axis=1).astype(np.float32)  # columns
    return tf.SO3.from_matrix(R).wxyz


def _ease(t: float) -> float:
    """Smoothstep easing on [0,1]."""
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def animate_camera_to(
    client: viser.ClientHandle,
    target_position: np.ndarray | None = None,
    target_wxyz: np.ndarray | None = None,
    *,
    duration: float = 0.6,
    fps: int = 60,
    final_look_at: np.ndarray | None = None,
) -> None:
    """SE(3) interpolate the camera to target pose."""
    wxyz0 = client.camera.wxyz
    pos0 = client.camera.position

    wxyz1 = np.asarray(target_wxyz, dtype=np.float32) if target_wxyz is not None else wxyz0
    pos1 = np.asarray(target_position, dtype=np.float32) if target_position is not None else pos0

    T0 = tf.SE3.from_rotation_and_translation(tf.SO3(wxyz0), pos0)
    T1 = tf.SE3.from_rotation_and_translation(tf.SO3(wxyz1), pos1)
    Xi = (T0.inverse() @ T1).log()

    steps = max(1, int(round(duration * fps)))
    for k in range(steps + 1):
        t = _ease(k / steps)
        T = T0 @ tf.SE3.exp(Xi * t)
        with client.atomic():
            client.camera.wxyz = T.rotation().wxyz
            client.camera.position = T.translation()
        client.flush()
        if k < steps:
            time.sleep(duration / steps)

    if final_look_at is not None:
        client.camera.look_at = np.asarray(final_look_at, dtype=np.float32)


def _compute_center_radius(centers: np.ndarray) -> tuple[np.ndarray, float]:
    """Mean center and max distance (simple, robust enough for framing)."""
    c = centers.mean(axis=0).astype(np.float32)
    r = float(np.linalg.norm(centers - c, axis=1).max())
    return c, r


def _safe_aspect(client: viser.ClientHandle) -> float:
    asp = getattr(client.camera, "aspect", 0.0) or 0.0
    if asp <= 0.0:
        w = getattr(client.camera, "image_width", 0) or 1280
        h = getattr(client.camera, "image_height", 0) or 720
        asp = max(1e-6, float(w) / float(h))
    return asp


def _fit_distance_for_radius(client: viser.ClientHandle, radius: float, margin: float = 1.1) -> float:
    """Distance so a sphere of `radius` fits in view given vertical FOV and aspect."""
    fov_deg = float(client.camera.fov)  # Viser fov is in degrees
    fov_v = np.deg2rad(fov_deg)
    tan_v = np.tan(fov_v / 2.0)
    tan_h = _safe_aspect(client) * tan_v
    tan_min = max(1e-6, min(tan_v, tan_h))  # constrain by the smaller half-FOV
    return margin * radius / tan_min


def _add_client_frames_for_splat(
    client: viser.ClientHandle,
    i: int,
    center: np.ndarray,
    radius: float,
    *,
    num_yaw: int = 6,
    pitch_deg: float = 20.0,
) -> None:
    """Create a ring of frames around a splat; clicking animates the camera."""
    pitch = np.deg2rad(pitch_deg)
    dist = _fit_distance_for_radius(client, radius)
    for k in range(num_yaw):
        yaw = 2.0 * np.pi * (k / num_yaw)
        # Offset on a sphere of radius `dist`
        dir_vec = np.array([
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
        ], dtype=np.float32)
        eye = center + dist * dir_vec
        wxyz = look_at_quaternion(eye, center)

        # Client-local frame + label
        frame = client.scene.add_frame(f"/splat_{i}/view_{k}", wxyz=wxyz, position=eye)
        client.scene.add_label(f"/splat_{i}/view_{k}/label", text=f"Splat {i} View {k}")

        @frame.on_click
        def _(_evt, eye=eye, wxyz=wxyz, center=center):
            animate_camera_to(
                client,
                target_position=eye,
                target_wxyz=wxyz,
                duration=0.6,
                final_look_at=center,
            )



def add_near_far_gui(client: viser.ClientHandle):
    """Create Near/Far sliders with safe ranges and invariants."""
    # Current camera values (fallbacks just in case)
    near0 = float(getattr(client.camera, "near", 0.01) or 0.01)
    far0  = float(getattr(client.camera, "far", 10.0) or 10.0)

    # Enforce far > near
    eps = 1e-3
    if far0 <= near0 + eps:
        far0 = near0 + 10.0  # bump far safely

    # Choose ranges that include the current values.
    # Near is usually small; Far can be very large. Make the ranges generous.
    near_min = 1e-3
    near_max = max(10.0, near0 * 10.0)

    far_min  = max(near0 + eps, 0.01)
    far_max  = max(1000.0, far0 * 10.0)   # wide to avoid future assertion issues

    # Clamp initial values into the ranges to satisfy add_slider's assertion.
    near_init = float(np.clip(near0, near_min, near_max))
    far_init  = float(np.clip(far0,  far_min,  far_max))

    near_slider = client.gui.add_slider(
        "Near", min=near_min, max=near_max, step=1e-3, initial_value=near_init
    )
    far_slider  = client.gui.add_slider(
        "Far",  min=far_min,  max=far_max,  step=1e-2, initial_value=far_init
    )

    @near_slider.on_update
    def _(_evt) -> None:
        # Set near, and ensure far remains > near.
        client.camera.near = near_slider.value
        if client.camera.far <= client.camera.near + eps:
            client.camera.far = client.camera.near + eps
            # Also move the slider's thumb to the new value.
            far_slider.value = client.camera.far

    @far_slider.on_update
    def _(_evt) -> None:
        # Keep far strictly larger than near.
        if far_slider.value <= client.camera.near + eps:
            far_slider.value = client.camera.near + eps
        client.camera.far = far_slider.value

    return near_slider, far_slider


# --------------- your main, merged with client camera API ---------------

def main(
    splat_paths: tuple[Path, ...] = (
        Path(__file__).absolute().parent.parent / "assets" / "final_scene_with_ids.ply",
    ),
) -> None:
    server = viser.ViserServer()
    server.scene.world_axes.visible = True

    # --- (NEW) make load_mujoco's button call work on this server API ---
    if hasattr(server, "gui") and hasattr(server.gui, "add_button") and not hasattr(server, "add_gui_button"):  # <<< NEW
        server.add_gui_button = server.gui.add_button  # type: ignore[attr-defined]                         # <<< NEW

    splat_metas: list[dict] = []

    # Choose which loaded splat is the ketchup object you want to drive.
    KETCHUP_SPLAT_INDEX = 0  # change if needed                                                                # <<< NEW

    for i, splat_path in enumerate(splat_paths):
        if splat_path.suffix in (".splat", ".ply"):
            splat_data = load_ply_file_sam(splat_path, center=True) if splat_path.suffix == ".splat" \
                         else load_ply_file_sam(splat_path, center=True)
        else:
            raise SystemExit("Please provide a .splat or .ply path.")

        server.scene.add_transform_controls(f"/{i}")
        gs_handle = server.scene.add_gaussian_splats(
            f"/{i}/gaussian_splats",
            centers=splat_data["centers"],
            rgbs=splat_data["rgbs"],
            opacities=splat_data["opacities"],
            covariances=splat_data["covariances"],
        )

        # --- (NEW) tag the ketchup handle so load_mujoco() can find it by id=14 ---
        if i == KETCHUP_SPLAT_INDEX:                                                                         # <<< NEW
            try:                                                                                             # <<< NEW
                setattr(gs_handle, "id", 10)  # tag this handle as 'ketchup'                                 # <<< NEW
            except Exception:                                                                                # <<< NEW
                pass                                                                                         # <<< NEW

        center_i, radius_i = _compute_center_radius(splat_data["centers"])
        splat_metas.append(dict(index=i, handle=gs_handle, centers=splat_data["centers"],
                                center=center_i, radius=radius_i))

        # ---------- your existing server-GUI buttons ----------
        remove_button = server.gui.add_button(f"Remove splat object {i}")
        insert_button = server.gui.add_button(f"Insert splat object {i}")
        rescale_button = server.gui.add_button(f"Rescale splat object {i}")
        transform_button = server.gui.add_button(f"Transform splat object {i}")
        alpha_button = server.gui.add_button(f"Alpha × splat object {i}")
        covariance_button1 = server.gui.add_button(f"Cov: isotropic σ² (splat {i})")
        covariance_button2 = server.gui.add_button(f"Cov: subset diag (splat {i})")
        covariance_button3 = server.gui.add_button(f"Cov: triu6 (splat {i})")

        def insert_gaussian_splats(server: viser.ViserServer, splat_data: dict, name: str):
            return server.scene.add_gaussian_splats(
                name,
                centers=splat_data["centers"],
                rgbs=splat_data["rgbs"],
                opacities=splat_data["opacities"],
                covariances=splat_data["covariances"],
            )

        splat_path2 = Path(__file__).absolute().parent.parent / "assets" / "duckbox.ply"
        splat_data_2 = load_ply_file(splat_path2, center=True)
        handle_ref = {"h": gs_handle}
        ui_setup = globals().get("setup_editable_keep_ui", None)
        if callable(ui_setup):
            ui_setup(
                server=server,
                handle_ref=handle_ref,
                splat_all=splat_data,
                semantic_id=splat_data.get("semantic_id", None),
                assigned_ids=list(range(15)),
                default_index=0,
                remove_button=remove_button,
            )

        @insert_button.on_click
        def _(_evt, sd=splat_data_2, i=i):
            insert_gaussian_splats(server=server, splat_data=sd, name=f"/gaussian_splats_{i}")

        @rescale_button.on_click
        def _(_evt, h=gs_handle):
            rescale_gaussian_splats(server=server, handle=h, scale=0.5)

        @transform_button.on_click
        def _(_evt, h=gs_handle):
            transform_gaussian_splats(
                server=server,
                handle=h,
                translation=(0.1, 0.0, 0.0),
                rotation=(1.0, 0.0, 0.0, 0.0),
            )

        @alpha_button.on_click
        def _(_evt, h=gs_handle):
            rescale_alpha(server, h, alpha_scale=0.1)

        @covariance_button1.on_click
        def _(_evt, h=gs_handle):
            sigma2 = 0.02 ** 2
            iso = np.diag([sigma2, sigma2, sigma2]).astype(np.float32)
            change_covariances(server, h, iso)

        @covariance_button2.on_click
        def _(_evt, h=gs_handle):
            idx = np.array([0, 5, 9, 10, 42], dtype=np.int64)
            covs_subset = np.stack([np.diag([0.001, 0.002, 0.003]) for _ in idx], axis=0).astype(np.float32)
            change_covariances(server, h, covs_subset, indices=idx)

        @covariance_button3.on_click
        def _(_evt, h=gs_handle):
            triu6 = np.array([0.001, 0.0, 0.0, 0.001, 0.0, 0.002], dtype=np.float32)
            change_covariances(server, h, triu6)

    # --- (NEW) expose server globally and register the playback button ---
    globals()["server"] = server   # so load_mujoco() can find it                                              # <<< NEW
    load_mujoco()                  # registers the “Play MuJoCo...” button in the Viser UI                     # <<< NEW

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        print(f"Client {client.client_id} connected.")
        near_slider, far_slider = add_near_far_gui(client)
        @near_slider.on_update
        def _(_evt): client.camera.near = near_slider.value
        @far_slider.on_update
        def _(_evt): client.camera.far  = far_slider.value

        for meta in splat_metas:
            i = meta["index"]; c = meta["center"]; r = meta["radius"]
            _add_client_frames_for_splat(client, i, c, r)
            fit_btn = client.gui.add_button(f"Fit to Splat {i}")
            @fit_btn.on_click
            def _(_evt, center=c, radius=r):
                dist = _fit_distance_for_radius(client, radius)
                R = tf.SO3(client.camera.wxyz).as_matrix()
                eye = center - R[:, 2] * dist
                wxyz = look_at_quaternion(eye, center)
                animate_camera_to(client, target_position=eye, target_wxyz=wxyz, duration=0.4, final_look_at=center)

        @client.camera.on_update
        def _(_cam: viser.CameraHandle) -> None:
            print(f"[client {client.client_id}] camera updated")

    try:
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    tyro.cli(main)




# add new function,  merge gaussian_splats from new scene



# manipulation of gaussian_splats

# apply force on robotic hand or object

# play piano

# camviewer , also need the extrinsic changes, not just near and far


# vr



# add camera trajectory 

# add depth, segmentation(based on assigned id map),normal map        