#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from typing import Dict, List, Tuple, Any

import math
import numpy as np

try:
    import json
except Exception:
    json = None

# support both package and script execution
try:
    from .atom_location import read_structure  # type: ignore
except Exception:
    sys.path.append(os.path.dirname(__file__))
    from atom_location import read_structure  # type: ignore


# ----------------------
# Utilities
# ----------------------

def load_par(path: str) -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    if not path or not os.path.isfile(path):
        return cfg
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            k, v = line.split('=', 1)
            cfg[k.strip()] = v.strip()
    return cfg


def get_par_path(base_dir: str) -> str:
    env = os.environ.get('PAR_FILE')
    if env and os.path.isfile(env):
        return env
    cand = os.path.join(base_dir, 'par')
    if os.path.isfile(cand):
        return cand
    cand2 = os.path.join(base_dir, 'par_all')
    if os.path.isfile(cand2):
        return cand2
    return ''


# Covalent radii (Å) – minimal set with sensible defaults
COV_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39,
    'B': 0.85, 'Si': 1.11,
}


def cov_radius(sym: str, default: float = 0.77) -> float:
    return COV_RADII.get(sym, COV_RADII.get(sym.capitalize(), default))


def distance(a: List[float], b: List[float]) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def build_bonds(atoms: List[Tuple[str, List[float]]], bond_tol: float = 0.45) -> List[List[int]]:
    n = len(atoms)
    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        si, ci = atoms[i]
        ri = cov_radius(si)
        for j in range(i+1, n):
            sj, cj = atoms[j]
            rj = cov_radius(sj)
            cutoff = ri + rj + bond_tol
            if distance(ci, cj) <= cutoff:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def connected_components(adj: List[List[int]]) -> List[List[int]]:
    n = len(adj)
    seen = [False]*n
    comps: List[List[int]] = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        comp = []
        seen[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps


# ----------------------
# Functional group detection (heuristic)
# ----------------------

def detect_groups(atoms: List[Tuple[str, List[float]]], adj: List[List[int]]):
    groups: Dict[str, List[Dict[str, Any]]] = {
        'methyl': [],
        'hydroxyl': [],
        'carbonyl': [],
        'carboxyl': [],
        'amine': [],
        'nitro': [],
        'ether': [],
        'halogen': [],
        'benzene': [],
    }

    n = len(atoms)

    # index helpers
    def neighbors(i):
        return adj[i]

    # map element counts for amine, etc.
    for i in range(n):
        sym_i, _ = atoms[i]

        # methyl: C bonded to exactly three H and at least one heavy (or none for methane -> still CH3 pattern if 3 H)
        if sym_i == 'C':
            neigh = neighbors(i)
            h_neigh = [j for j in neigh if atoms[j][0] == 'H']
            heavy = [j for j in neigh if atoms[j][0] != 'H']
            if len(h_neigh) == 3:
                groups['methyl'].append({'C': i, 'H': h_neigh, 'X': heavy[0] if heavy else None})

        # hydroxyl / ether on oxygen
        if sym_i == 'O':
            neigh = neighbors(i)
            h_neigh = [j for j in neigh if atoms[j][0] == 'H']
            heavy = [j for j in neigh if atoms[j][0] != 'H']
            if len(h_neigh) == 1 and len(heavy) >= 1:
                groups['hydroxyl'].append({'O': i, 'H': h_neigh[0], 'X': heavy[0]})
            # ether: O connected to two carbons and no hydrogens
            c_neigh = [j for j in neigh if atoms[j][0] == 'C']
            if len(c_neigh) == 2 and len(h_neigh) == 0:
                groups['ether'].append({'O': i, 'C': c_neigh})

        # amine: N with two or three H neighbors and >=1 heavy neighbor
        if sym_i == 'N':
            neigh = neighbors(i)
            h_neigh = [j for j in neigh if atoms[j][0] == 'H']
            heavy = [j for j in neigh if atoms[j][0] != 'H']
            if len(h_neigh) in (2, 3) and len(heavy) >= 1:
                groups['amine'].append({'N': i, 'H': h_neigh, 'X': heavy[0]})

    # carbonyl and carboxyl require C-O distance check and context
    def is_short_co(ci, oi) -> bool:
        # carbonyl-like distance threshold
        d = distance(atoms[ci][1], atoms[oi][1])
        return d <= 1.30  # Å, rough cutoff

    for i in range(n):
        sym_i, _ = atoms[i]
        if sym_i != 'C':
            continue
        onbrs = [j for j in adj[i] if atoms[j][0] == 'O']
        if len(onbrs) == 1:
            o = onbrs[0]
            # carbonyl if O is terminal (no H) and short C=O
            o_neigh = [k for k in adj[o] if k != i]
            has_H = any(atoms[k][0] == 'H' for k in o_neigh)
            if not has_H and is_short_co(i, o):
                groups['carbonyl'].append({'C': i, 'O': o})
        elif len(onbrs) >= 2:
            # carboxyl: one O part of hydroxyl (O-H), another O double-bond-like
            oh_list = []
            odbl_list = []
            for o in onbrs:
                o_neigh = [k for k in adj[o] if k != i]
                has_H = any(atoms[k][0] == 'H' for k in o_neigh)
                if has_H:
                    oh_list.append(o)
                elif is_short_co(i, o):
                    odbl_list.append(o)
            if oh_list and odbl_list:
                groups['carboxyl'].append({'C': i, 'O_H': oh_list[0], 'O_dbl': odbl_list[0]})

    # nitro: N attached to two O (no H on O) and at least one heavy (e.g., a carbon)
    for i in range(n):
        sym_i, _ = atoms[i]
        if sym_i != 'N':
            continue
        o_neigh = [j for j in adj[i] if atoms[j][0] == 'O']
        heavy = [j for j in adj[i] if atoms[j][0] not in ('O', 'H')]
        good_o = []
        for o in o_neigh:
            o_neigh2 = [k for k in adj[o] if k != i]
            if not any(atoms[k][0] == 'H' for k in o_neigh2):
                good_o.append(o)
        if len(good_o) >= 2 and heavy:
            groups['nitro'].append({'N': i, 'O': good_o[:2], 'X': heavy[0]})

    # halogen substituents: F/Cl/Br/I attached to a carbon
    halogens = { 'F', 'Cl', 'Br', 'I' }
    for i in range(n):
        sym_i, _ = atoms[i]
        if sym_i not in halogens:
            continue
        neigh = neighbors(i)
        c_neigh = [j for j in neigh if atoms[j][0] == 'C']
        if c_neigh:
            groups['halogen'].append({'X': i, 'C': c_neigh[0], 'symbol': sym_i})

    # benzene rings: detect 6-membered carbon cycles with planarity
    groups['benzene'] = [ {'ring': ring} for ring in find_benzene_rings(atoms, adj) ]

    return groups


def classify_structure(atoms: List[Tuple[str, List[float]]], adj: List[List[int]]):
    # Heuristic classification: molecular vs extended (non-molecular)
    n = len(atoms)
    h_count = sum(1 for s, _ in atoms if s == 'H')
    comps = connected_components(adj)
    largest = max((len(c) for c in comps), default=0)
    avg_deg = sum(len(adj[i]) for i in range(n)) / n if n else 0.0

    # rules of thumb
    if n >= 20 and largest >= 0.8*n and h_count/n < 0.05 and avg_deg >= 2.5:
        return 'non-molecular'
    if h_count == 0 and avg_deg >= 2.5:
        return 'non-molecular'
    return 'molecular'


def _plane_rms(coords: np.ndarray) -> float:
    c = coords - coords.mean(axis=0, keepdims=True)
    # PCA: smallest singular value corresponds to normal direction
    _, s, vh = np.linalg.svd(c, full_matrices=False)
    normal = vh[-1]
    dists = np.abs(c.dot(normal))
    return float(np.sqrt((dists**2).mean()))


def find_benzene_rings(atoms: List[Tuple[str, List[float]]], adj: List[List[int]]) -> List[List[int]]:
    # Build carbon-only adjacency
    n = len(atoms)
    carbon = [i for i,(s,_) in enumerate(atoms) if s == 'C']
    cset = set(carbon)
    cadj = {i: [j for j in adj[i] if j in cset] for i in carbon}

    rings = set()

    def canonical_cycle(cycle: List[int]) -> Tuple[int, ...]:
        # rotate to minimal index start and canonical direction
        m = min(cycle)
        idx = cycle.index(m)
        rot1 = cycle[idx:] + cycle[:idx]
        rot2 = list(reversed(cycle))
        idx2 = rot2.index(m)
        rot2 = rot2[idx2:] + rot2[:idx2]
        t1 = tuple(rot1)
        t2 = tuple(rot2)
        return t1 if t1 < t2 else t2

    # DFS limited to length 6
    def dfs(path: List[int], target: int):
        if len(path) > 6:
            return
        u = path[-1]
        for v in cadj.get(u, []):
            if len(path) == 6:
                if v == target:
                    cyc = canonical_cycle(path)
                    rings.add(cyc)
                continue
            if v in path:
                continue
            # pruning: ensure we don't branch to nodes with too many non-carbon neighbors? keep simple
            dfs(path + [v], target)

    for start in carbon:
        for nb in cadj[start]:
            # enforce ordering to limit duplicates
            if nb <= start:
                continue
            dfs([start, nb], start)

    results: List[List[int]] = []
    for cyc in rings:
        ring_idx = list(cyc)
        # planarity check
        coords = np.array([atoms[i][1] for i in ring_idx])
        rms = _plane_rms(coords)
        # bond length sanity between consecutive atoms (~1.2 - 1.6 Å)
        ok_bonds = True
        for i in range(6):
            a = ring_idx[i]
            b = ring_idx[(i+1) % 6]
            d = distance(atoms[a][1], atoms[b][1])
            if d < 1.2 or d > 1.6:
                ok_bonds = False
                break
        if rms <= 0.25 and ok_bonds:
            results.append(ring_idx)
    return results


def find_carbon_chains(atoms: List[Tuple[str, List[float]]],
                       adj: List[List[int]],
                       min_len: int = 3) -> List[List[int]]:
    """Find carbon chain segments as paths in C-only graph with internal degree<=2.
    Returns list of index paths (0-based) with length >= min_len.
    """
    carbon = [i for i, (s, _) in enumerate(atoms) if s == 'C']
    cset = set(carbon)
    cadj = {i: [j for j in adj[i] if j in cset] for i in carbon}

    # Subgraph of nodes with degree<=2 (chain-like)
    nodes = [i for i in carbon if len(cadj[i]) <= 2]
    nset = set(nodes)
    sub_adj = {i: [j for j in cadj[i] if j in nset] for i in nodes}

    # Find connected components in subgraph
    seen = set()
    comps: List[List[int]] = []
    for u in nodes:
        if u in seen:
            continue
        stack = [u]
        comp = []
        seen.add(u)
        while stack:
            x = stack.pop()
            comp.append(x)
            for v in sub_adj[x]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(comp)

    from collections import deque

    def bfs_far(start: int):
        q = deque([start])
        dist = {start: 0}
        parent = {start: -1}
        while q:
            u = q.popleft()
            for v in sub_adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
        far = max(dist, key=dist.get)
        return far, dist, parent

    chains: List[List[int]] = []
    for comp in comps:
        if not comp:
            continue
        # find path by double-BFS
        s0 = comp[0]
        s, _, _ = bfs_far(s0)
        t, _, par = bfs_far(s)
        # reconstruct s..t path
        path = []
        cur = t
        while cur != -1:
            path.append(cur)
            cur = par.get(cur, -1)
        path.reverse()
        if len(path) >= min_len:
            chains.append(path)
    chains.sort(key=len, reverse=True)
    return chains


def run(par_path: str = None, out_path: str = None, structure_path: str = None):
    base_dir = os.path.dirname(os.path.dirname(__file__))  # carbon_chain_rot
    par = par_path or get_par_path(base_dir)
    cfg = load_par(par)

    struct_file = structure_path or cfg.get('STRUCTURE_FILE') or cfg.get('STRUCT_FILE')
    if not struct_file or not os.path.isfile(struct_file):
        raise FileNotFoundError('STRUCTURE_FILE not set in par or file not found')

    bond_tol = float(cfg.get('BOND_TOL', 0.45))

    atoms, meta = read_structure(struct_file)
    adj = build_bonds(atoms, bond_tol=bond_tol)

    # Optional molecule selection by z-axis range: SELECT_Z_RANGE=zmin,zmax
    sel = cfg.get('SELECT_Z_RANGE')
    selected_component = None
    if sel:
        try:
            zmin_str, zmax_str = [x.strip() for x in sel.split(',')]
            zmin = float(zmin_str); zmax = float(zmax_str)
            comps = connected_components(adj)
            best_comp = None
            best_count = 0
            zs = [coord[2] for _, coord in atoms]
            for comp in comps:
                cnt = sum(1 for i in comp if zmin <= zs[i] <= zmax)
                if cnt > best_count:
                    best_count = cnt
                    best_comp = comp
            if best_comp and best_count > 0:
                selected_component = sorted(best_comp)
        except Exception:
            selected_component = None

    if selected_component is not None:
        # remap atoms/adj to the selected component
        index_map = {old: new for new, old in enumerate(selected_component)}
        atoms = [atoms[i] for i in selected_component]
        new_adj = [[] for _ in range(len(selected_component))]
        for old_i in selected_component:
            for old_j in adj[old_i]:
                if old_j in index_map:
                    new_adj[index_map[old_i]].append(index_map[old_j])
        adj = new_adj

    groups = detect_groups(atoms, adj)

    # carbon chains detection
    chain_min_len = int(cfg.get('CHAIN_MIN_LEN', 3))
    chains = find_carbon_chains(atoms, adj, min_len=chain_min_len)
    if chains:
        groups['carbon_chain'] = [{'indices': ch, 'length': len(ch)} for ch in chains]
    else:
        groups['carbon_chain'] = []

    stype = classify_structure(atoms, adj)

    result = {
        'structure_file': struct_file,
        'structure_type': stype,
        'counts': {k: len(v) for k, v in groups.items()},
        'groups': groups,
        'selected_component_size': len(atoms),
    }

    text = None
    if json is not None:
        text = json.dumps(result, indent=2)
    else:
        # fallback simple text
        lines = [f"structure_file: {struct_file}", f"structure_type: {stype}"]
        for k, v in result['counts'].items():
            lines.append(f"{k}: {v}")
        text = "\n".join(lines)

    # longest chain summary with middle section indices
    if chains:
        longest = chains[0]
        L = len(longest)
        middle = [longest[L//2]] if L % 2 == 1 else [longest[L//2 - 1], longest[L//2]]
        result['longest_chain'] = {'indices': longest, 'length': L, 'middle_indices': middle}

    if out_path:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(text + '\n')
    return result


def main():
    ap = argparse.ArgumentParser(description='Detect functional groups and methyl from structure file via par')
    ap.add_argument('structure', nargs='?', default=None,
                    help='Path to structure file (override par STRUCTURE_FILE)')
    ap.add_argument('--par', default=None, help='Path to par (default: auto-detect)')
    ap.add_argument('--out', default=None, help='Write JSON/text output to file')
    args = ap.parse_args()

    try:
        res = run(par_path=args.par, out_path=args.out, structure_path=args.structure)
    except Exception as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(2)
    if not args.out and json is not None:
        print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()


"""
结果说明

result.groups 包含各官能团命中的原子索引（0 基）。例如：
groups.methyl: {C: c_idx, H: [h1,h2,h3], X: 邻接重原子或 None}
groups.hydroxyl: {O: o, H: h, X: 邻接重原子}
groups.ether: {O: o, C: [c1,c2]}
groups.halogen: {X: 卤素原子, C: 相连碳, symbol: 元素符号}
groups.benzene: {ring: [c1,...,c6]}
groups.carbon_chain: [{indices: [c...], length: L}, ...]
result.longest_chain 给出最长碳链的 indices、length，以及 middle_indices（偶数长度取中间两个，奇数长度取中间一个）。
如果需要 1 基索引给后续脚本用，可在读取后将列表里的索引 +1。
按 z 轴选择分子

par 中设置 SELECT_Z_RANGE=zmin,zmax 时，程序先按全体原子判键和分子连通分量，再选择在该 z 窗内原子数最多的那一个分子进行识别，便于只分析薄层或目标区域。
"""