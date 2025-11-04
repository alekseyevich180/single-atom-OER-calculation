#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

# Import helpers from sibling modules when available; otherwise stay functional
try:
    from .atom_location import read_structure  # type: ignore
except Exception:
    from atom_location import read_structure  # type: ignore

try:
    from .functional_groups import load_par, get_par_path, build_bonds, detect_groups  # type: ignore
except Exception:
    # minimal fallbacks if imported as script without package context
    from functional_groups import load_par, get_par_path, build_bonds, detect_groups  # type: ignore


# ----------------------
# Math helpers
# ----------------------

def _as_np(v):
    return np.asarray(v, dtype=float)


def unit(vec: np.ndarray) -> np.ndarray:
    v = _as_np(vec)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def centroid(coords: List[List[float]]) -> np.ndarray:
    if not coords:
        return np.zeros(3)
    arr = _as_np(coords)
    return arr.mean(axis=0)


def plane_normal(coords: List[List[float]]) -> np.ndarray:
    arr = _as_np(coords)
    c = arr - arr.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(c, full_matrices=False)
    return unit(vh[-1])


def rotate_points(points: np.ndarray,
                  pivot: np.ndarray,
                  axis: np.ndarray,
                  angle_deg: float) -> np.ndarray:
    # Rodrigues rotation around a unit axis through pivot
    P = _as_np(points)
    a = unit(axis)
    theta = math.radians(float(angle_deg))
    kx, ky, kz = a
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]], dtype=float)
    R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
    centered = P - pivot
    return centered @ R.T + pivot


# ----------------------
# Group vector definitions
# ----------------------

def methyl_vectors(entry: Dict[str, Any], atoms: List[Tuple[str, List[float]]],
                   include_c: bool = False) -> Dict[str, Any]:
    c_idx = entry.get('C')
    h_idx = entry.get('H', [])
    x_idx = entry.get('X')  # heavy neighbor if present
    if c_idx is None:
        raise ValueError('methyl entry missing C index')
    c = _as_np(atoms[c_idx][1])
    if x_idx is not None:
        x = _as_np(atoms[x_idx][1])
        axis = unit(x - c)
    else:
        # fallback: normal of H triangle
        h_coords = [_as_np(atoms[i][1]) for i in h_idx[:3]]
        axis = plane_normal(h_coords)
    pivot = c
    move = [c_idx] + h_idx if include_c else list(h_idx)
    return {'pivot': pivot, 'axis': axis, 'move_indices': move, 'anchor': [x_idx] if x_idx is not None else []}


def hydroxyl_vectors(entry: Dict[str, Any], atoms: List[Tuple[str, List[float]]], include_o: bool = False) -> Dict[str, Any]:
    o_idx = entry.get('O')
    h_idx = entry.get('H')
    x_idx = entry.get('X')
    if o_idx is None or x_idx is None:
        raise ValueError('hydroxyl entry requires O and heavy neighbor X')
    o = _as_np(atoms[o_idx][1])
    x = _as_np(atoms[x_idx][1])
    axis = unit(x - o)  # rotate around O–X bond
    pivot = o
    move = [h_idx] + ([o_idx] if include_o else [])
    return {'pivot': pivot, 'axis': axis, 'move_indices': move, 'anchor': [x_idx]}


def ether_vectors(entry: Dict[str, Any], atoms: List[Tuple[str, List[float]]], which: int = 0, rotate_side_only: bool = True) -> Dict[str, Any]:
    o_idx = entry.get('O')
    c_list = entry.get('C', [])
    if o_idx is None or len(c_list) != 2:
        raise ValueError('ether entry requires O and two C neighbors')
    ci = c_list[which % 2]
    o = _as_np(atoms[o_idx][1])
    c = _as_np(atoms[ci][1])
    axis = unit(c - o)
    pivot = o
    # Minimal rotation: rotate O only (flip lone-pair orientation)
    move = [o_idx]
    return {'pivot': pivot, 'axis': axis, 'move_indices': move, 'anchor': [ci]}


def amine_vectors(entry: Dict[str, Any], atoms: List[Tuple[str, List[float]]], include_n: bool = False) -> Dict[str, Any]:
    n_idx = entry.get('N')
    h_list = entry.get('H', [])
    x_idx = entry.get('X')
    if n_idx is None or x_idx is None:
        raise ValueError('amine entry requires N and heavy neighbor X')
    n = _as_np(atoms[n_idx][1])
    x = _as_np(atoms[x_idx][1])
    axis = unit(x - n)
    pivot = n
    move = list(h_list) + ([n_idx] if include_n else [])
    return {'pivot': pivot, 'axis': axis, 'move_indices': move, 'anchor': [x_idx]}


def nitro_vectors(entry: Dict[str, Any], atoms: List[Tuple[str, List[float]]]) -> Dict[str, Any]:
    n_idx = entry.get('N')
    o_list = entry.get('O', [])
    x_idx = entry.get('X')
    if n_idx is None or x_idx is None or len(o_list) < 2:
        raise ValueError('nitro entry requires N, two O, and heavy neighbor X')
    n = _as_np(atoms[n_idx][1])
    x = _as_np(atoms[x_idx][1])
    axis = unit(x - n)  # rotation around C–N bond
    pivot = n
    move = [n_idx] + list(o_list)  # rotate NO2 as rigid
    return {'pivot': pivot, 'axis': axis, 'move_indices': move, 'anchor': [x_idx]}


def carbonyl_vectors(entry: Dict[str, Any], atoms: List[Tuple[str, List[float]]]) -> Dict[str, Any]:
    c_idx = entry.get('C')
    o_idx = entry.get('O')
    if c_idx is None or o_idx is None:
        raise ValueError('carbonyl entry requires C and O')
    # choose heavy neighbor X of carbon that is not O
    # This function expects caller to supply X via entry if available
    x_idx = entry.get('X')
    c = _as_np(atoms[c_idx][1])
    if x_idx is not None:
        x = _as_np(atoms[x_idx][1])
        axis = unit(x - c)  # torsion around C–X single bond
    else:
        # fallback: rotate O around C–O as axis
        o = _as_np(atoms[o_idx][1])
        axis = unit(o - c)
    pivot = c
    move = [o_idx]
    return {'pivot': pivot, 'axis': axis, 'move_indices': move, 'anchor': [x_idx] if x_idx is not None else []}


def benzene_vectors(entry: Dict[str, Any], atoms: List[Tuple[str, List[float]]]) -> Dict[str, Any]:
    ring = entry.get('ring', [])
    coords = [_as_np(atoms[i][1]) for i in ring]
    ax = plane_normal(coords)  # normal to ring plane
    pv = centroid(coords)
    return {'pivot': pv, 'axis': ax, 'move_indices': list(ring), 'anchor': []}


def chain_vectors(entry: Dict[str, Any], atoms: List[Tuple[str, List[float]]]) -> Dict[str, Any]:
    chain = entry.get('indices', [])
    L = len(chain)
    if L < 2:
        raise ValueError('chain too short')
    coords = [_as_np(atoms[i][1]) for i in chain]
    if L % 2 == 1:
        mid = chain[L // 2]
        # pick axis along local tangent (mid to mid+1 if exists)
        if L // 2 + 1 < L:
            a = _as_np(atoms[mid][1])
            b = _as_np(atoms[chain[L // 2 + 1]][1])
            ax = unit(b - a)
        else:
            a = _as_np(atoms[chain[L // 2 - 1]][1])
            b = _as_np(atoms[mid][1])
            ax = unit(b - a)
        pv = _as_np(atoms[mid][1])
    else:
        i1 = chain[L // 2 - 1]
        i2 = chain[L // 2]
        a = _as_np(atoms[i1][1])
        b = _as_np(atoms[i2][1])
        ax = unit(b - a)
        pv = 0.5 * (a + b)
    return {'pivot': pv, 'axis': ax, 'move_indices': list(chain), 'anchor': []}


GROUP_VECTOR_BUILDERS = {
    'methyl': methyl_vectors,
    'hydroxyl': hydroxyl_vectors,
    'ether': ether_vectors,
    'amine': amine_vectors,
    'nitro': nitro_vectors,
    'carbonyl': carbonyl_vectors,
    'benzene': benzene_vectors,
    'carbon_chain': chain_vectors,
}


# ----------------------
# High-level helpers
# ----------------------

def define_group_vectors(group_type: str,
                         entry: Dict[str, Any],
                         atoms: List[Tuple[str, List[float]]],
                         **kwargs) -> Dict[str, Any]:
    b = GROUP_VECTOR_BUILDERS.get(group_type)
    if b is None:
        raise ValueError(f'Unsupported group type: {group_type}')
    return b(entry, atoms, **kwargs)


def rotate_group(atoms: List[Tuple[str, List[float]]],
                 vector_def: Dict[str, Any],
                 angle_deg: float) -> List[Tuple[str, List[float]]]:
    move_idx: List[int] = list(vector_def['move_indices'])
    pivot = _as_np(vector_def['pivot'])
    axis = _as_np(vector_def['axis'])
    # prepare arrays
    coords = np.array([a[1] for a in atoms], dtype=float)
    moved = rotate_points(coords[move_idx], pivot, axis, angle_deg)
    new_atoms: List[Tuple[str, List[float]]] = []
    mset = {i for i in move_idx}
    it = iter(moved)
    for i, (sym, coord) in enumerate(atoms):
        if i in mset:
            new_atoms.append((sym, list(next(it))))
        else:
            new_atoms.append((sym, list(coord)))
    return new_atoms


def load_groups_from_par(par_path: Optional[str] = None,
                         structure_path: Optional[str] = None):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    par = par_path or get_par_path(base_dir)
    cfg = load_par(par)
    path = structure_path or cfg.get('STRUCTURE_FILE') or cfg.get('STRUCT_FILE')
    if not path or not os.path.isfile(path):
        raise FileNotFoundError('STRUCTURE_FILE not set in par or file not found')
    atoms, meta = read_structure(path)
    adj = build_bonds(atoms, bond_tol=float(cfg.get('BOND_TOL', 0.45)))
    groups = detect_groups(atoms, adj)
    return atoms, groups


# The module provides vector definitions and rotation utilities.
# Example usage in a driver script:
#   atoms, groups = load_groups_from_par('carbon_chain_rot/par')
#   entry = groups['methyl'][0]
#   vdef = define_group_vectors('methyl', entry, atoms)
#   atoms_rot = rotate_group(atoms, vdef, angle_deg=30)

