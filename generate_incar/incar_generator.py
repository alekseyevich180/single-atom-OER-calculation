#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
incar_generator.py
------------------
Generate a VASP INCAR using:
 - Composition (from POSCAR or manual counts)
 - Phase type (oxide / metal)
 - Global switches from the "par" file

Typical usage:
  python incar_generator.py --poscar POSCAR > INCAR
  python incar_generator.py --poscar POSCAR --out INCAR
  python incar_generator.py --poscar POSCAR --phase oxide

Notes:
- This script is designed to be the single source of truth for INCAR format.
- All global behaviors (DFT+U, spin, vdW, dipole correction, SOC...) are
  controlled via the external file "par".

The other script (make_incar_from_par.py) is a wrapper that:
  - infers NUPDOWN from the folder name (e.g. Fe_3 --> NUPDOWN=3)
  - calls this core logic

Author: single-atom workflow
"""

import argparse
import os
import re
from collections import defaultdict

# ======================
# 1. Helpers: read `par`
# ======================

def load_par(path="par"):
    """
    Read a simple KEY=VALUE config file.
    Lines starting with # or blank are skipped.
    true/false -> bool
    everything else -> string
    """
    cfg = {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Global control file "{path}" not found.')
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            low = val.lower()
            if low in ["true", "false"]:
                cfg[key] = (low == "true")
            else:
                cfg[key] = val
    return cfg


# =================================================
# 2. Parse POSCAR to get {element: count} as dict
# =================================================

def parse_poscar_counts(poscar_path):
    """
    Read a VASP POSCAR/CONTCAR-like file and extract element symbols and counts.
    Supports both POSCAR with "element symbols line" + "counts line"
    or POTCAR-style lines in the 6th line (but we assume standard POSCAR).
    Returns OrderedDict-like: {"Fe":4,"O":6} preserving order.
    """
    if not os.path.isfile(poscar_path):
        raise FileNotFoundError(f'POSCAR file "{poscar_path}" not found.')

    with open(poscar_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines()]

    if len(lines) < 7:
        raise ValueError("POSCAR too short, cannot parse atom counts.")

    # VASP POSCAR formats:
    # line 5: element symbols, e.g. "Fe O"
    # line 6: element counts, e.g. "4 6"
    # Then Selective dynamics? / Direct or Cartesian...
    #
    # BUT some POSCARs omit the line of symbols, and line5 is counts.
    # We'll try to detect which is which.

    # Attempt parse style A:
    elems_line = lines[5 - 1]  # index 4
    cnts_line  = lines[6 - 1]  # index 5
    tokens_e = elems_line.split()
    tokens_c = cnts_line.split()

    # Heuristic: are all tokens_c integers?
    def all_int(seq):
        for x in seq:
            if not re.match(r"^\d+$", x):
                return False
        return True

    if all_int(tokens_c):
        # Great, we are in standard form
        elems = tokens_e
        counts = list(map(int, tokens_c))
    else:
        # Fallback guess: sometimes POSCAR doesn't list elems separately,
        # so line4 are "scaling matrix row", line5 are counts etc.
        # We'll just do a best-effort parse.
        raise ValueError(
            "Unable to parse element symbols/counts from POSCAR automatically. "
            "Please provide --counts manually like Fe=4,O=6."
        )

    if len(elems) != len(counts):
        raise ValueError("Mismatch between number of element symbols and counts in POSCAR.")

    out = {}
    for e, n in zip(elems, counts):
        out[e] = n
    return out


def parse_counts_string(counts_str):
    """
    Turn something like "Fe=4,O=6" into dict {"Fe":4,"O":6}
    """
    out = {}
    for chunk in counts_str.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Bad counts chunk: {chunk}")
        el, num = chunk.split("=", 1)
        el = el.strip()
        num = int(num.strip())
        out[el] = num
    return out


# =======================================================
# 3. Internal data tables: MAGMOM guesses and +U values
# =======================================================

# Initial magnetic moments (μB) per atom for common elements.
# These are only initial guesses; VASP will relax.
MAGMOM_TABLE = {
    # 3d
    "Sc": 1.0, "Ti": 1.0, "V": 3.0, "Cr": 3.5, "Mn": 5.0,
    "Fe": 3.0, "Co": 2.0, "Ni": 1.0, "Cu": 0.0, "Zn": 0.0,
    # p-block / etc. default 0
    "B": 0.0, "C": 0.0, "N": 0.0, "O": 0.0, "F": 0.0,
    "Al": 0.0, "Si": 0.0, "P": 0.0, "S": 0.0, "Cl": 0.0,
    "Ga": 0.0, "Ge": 0.0, "As": 0.0, "Se": 0.0, "Br": 0.0,
    # 4d
    "Y": 0.5, "Zr": 0.5, "Nb": 1.0, "Mo": 0.5, "Tc": 1.0,
    "Ru": 0.5, "Rh": 0.5, "Pd": 0.0, "Ag": 0.0, "Cd": 0.0,
    # 5d
    "Hf": 0.5, "Ta": 1.0, "W": 0.5, "Re": 1.0, "Os": 0.5,
    "Ir": 0.5, "Pt": 0.0, "Au": 0.0, "Hg": 0.0,
}

# Typical DFT+U parameters in Dudarev format (LDAUTYPE=2).
# You should change to your group's reference values if needed.
# Values are U_eff = U - J in eV.
LDAU_U_TABLE = {
    "Ti": 4.5,
    "V": 3.3,
    "Cr": 3.5,
    "Mn": 4.0,
    "Fe": 4.0,
    "Co": 3.3,
    "Ni": 6.0,
    "Cu": 4.5,
    "Ce": 5.0,
    "Ir": 2.0,
}


# ==================================================
# 4. Logic for SOC, spin, +U, and the INCAR blocks
# ==================================================

def decide_soc_blocks(cfg, elements):
    """
    Decide SOC settings.
    FORCE_SOC in par can be:
      - "on"  -> always SOC
      - "off" -> never SOC
      - "auto"-> heavy elements trigger SOC
    We'll consider 5d elements (Hf -> Au) as "heavy".
    """
    force_soc = cfg.get("FORCE_SOC", "auto").lower()
    heavy_5d = {"Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg"}

    use_soc = False
    if force_soc == "on":
        use_soc = True
    elif force_soc == "off":
        use_soc = False
    else:
        # auto
        if any(el in heavy_5d for el in elements):
            use_soc = True
        else:
            use_soc = False

    if use_soc:
        lines = [
            "ISYM    = 0",
            "LSORBIT = .TRUE.",
            "LNONCOLLINEAR = .TRUE.",
            "GGA_COMPAT = .FALSE.",
            "SAXIS = 0 0 1"
        ]
    else:
        lines = [
            "ISYM    = -1",
            "LSORBIT = .FALSE.",
            "LNONCOLLINEAR = .FALSE.",
            "GGA_COMPAT = .TRUE.",
        ]
    return lines, use_soc


def build_magmom_block(cfg, counts_dict):
    """
    Build ISPIN / MAGMOM / (maybe NUPDOWN) lines based on par.

    - USE_SPIN=false:
        ISPIN=1
        no MAGMOM
        no NUPDOWN (unless FIX_NUPDOWN=true, but normally you'd keep false)
    - USE_SPIN=true:
        ISPIN=2, MAGMOM based on table
        if FIX_NUPDOWN=true -> NUPDOWN = NUPDOWN_VALUE
    """
    use_spin = bool(cfg.get("USE_SPIN", True))
    fix_nup = bool(cfg.get("FIX_NUPDOWN", False))

    lines = []
    mag_list = []

    if use_spin:
        lines.append("ISPIN = 2")

        # MAGMOM line: repeat per-element moment * count
        for el, num in counts_dict.items():
            m = MAGMOM_TABLE.get(el, 0.0)
            mag_list.extend([m] * num)

        # Build MAGMOM in VASP style: MAGMOM = n1*m1 n2*m2 ...
        # For readability we actually print compressed form "elcount*m"
        # but VASP also allows "MAGMOM = 4*5.0 6*0.0 ..." etc.
        # We'll use compressed form.
        chunks = []
        for el, num in counts_dict.items():
            m = MAGMOM_TABLE.get(el, 0.0)
            chunks.append(f"{num}*{m}")
        lines.append("MAGMOM = " + " ".join(chunks))

        if fix_nup:
            nup_val = cfg.get("NUPDOWN_VALUE", "0")
            lines.append(f"NUPDOWN = {nup_val}")
    else:
        lines.append("ISPIN = 1")
        # normally don't write MAGMOM
        # If user *forces* NUPDOWN with ISPIN=1 that's nonsense,
        # so we just ignore FIX_NUPDOWN in non-spin mode.

    return lines


def build_ldau_block(cfg, elements):
    """
    Build +U block if USE_LDAU=true in par; else return commented block.
    We'll assume LDAUTYPE=2 (Dudarev). We generate arrays per unique element.
    """
    use_ldau = bool(cfg.get("USE_LDAU", False))

    # We will output arrays following the order of 'elements'
    U_list = []
    L_list = []
    for el in elements:
        if el in LDAU_U_TABLE:
            U_list.append(str(LDAU_U_TABLE[el]))
            # assume d orbitals for TMs: L=2
            L_list.append("2")
        else:
            U_list.append("0.0")
            L_list.append("-1")

    if use_ldau:
        lines = [
            "LDAU      = .TRUE.",
            "LDAUTYPE  = 2",
            "LDAUL     = " + " ".join(L_list),
            "LDAUU     = " + " ".join(U_list),
            "LDAUJ     = " + " ".join("0.0" for _ in elements),
            "LMAXMIX   = 4"
        ]
    else:
        lines = [
            "LDAU      = .FALSE.",
            "# LDAUTYPE  = 2",
            "# LDAUL     = " + " ".join(L_list),
            "# LDAUU     = " + " ".join(U_list),
            "# LDAUJ     = " + " ".join("0.0" for _ in elements),
            "# LMAXMIX   = 4"
        ]
    return lines


def build_common_block(cfg):
    """
    Tail block with ENCUT, relax controls, IVDW, DIPOL, IO flags, etc.
    These are heavily group-dependent defaults.

    USE_VDW / VDW_FLAG
    USE_DIPOL / IDIPOL_DIR
    """
    use_vdw = bool(cfg.get("USE_VDW", True))
    vdw_flag = cfg.get("VDW_FLAG", "12")

    use_dipol = bool(cfg.get("USE_DIPOL", False))
    idipol_dir = cfg.get("IDIPOL_DIR", "4")

    blk = []
    blk.append("# ===== Electronic loop controls =====")
    blk.append("ENCUT  = 400")
    blk.append("ALGO   = ALL")
    blk.append("EDIFF  = 1E-8")
    blk.append("NELM   = 500")
    blk.append("ISMEAR = 0")
    blk.append("SIGMA  = 0.050000")
    blk.append("")
    blk.append("# ===== Relaxation control =====")
    blk.append("IBRION = 2")
    blk.append("NSW    = 5000")
    blk.append("POTIM  = 0.100000")
    blk.append("ISIF   = 0")
    blk.append("EDIFFG = -0.020000")
    blk.append("")
    blk.append("# ===== Parallelization =====")
    blk.append("NPAR   = 5")
    blk.append("LREAL  = AUTO")
    blk.append("NSIM   = 1")
    blk.append("LPLANE = .TRUE.")
    blk.append("")
    blk.append("# ===== vdW dispersion / D3 / etc. =====")
    if use_vdw:
        blk.append(f"IVDW   = {vdw_flag}")
    else:
        blk.append("# IVDW   = (disabled)")
    blk.append("")
    blk.append("# ===== I/O flags =====")
    blk.append("LAECHG = .FALSE.")
    blk.append("LCHARG = .FALSE.")
    blk.append("LWAVE  = .FALSE.")
    blk.append("LELF   = .FALSE.")
    blk.append("LVTOT  = .FALSE.")
    blk.append("LVHAR  = .FALSE.")
    blk.append("LREAL  = .FALSE.")
    blk.append("")
    blk.append("# ===== Dipole correction =====")
    if use_dipol:
        blk.append(f"IDIPOL = {idipol_dir}")
        blk.append("LDIPOL = .TRUE.")
    else:
        blk.append("# IDIPOL = 4")
        blk.append("# LDIPOL = .FALSE.")

    return blk


def generate_incar(cfg, counts_dict, phase_mode=None):
    """
    Compose the full INCAR as a list of lines (strings).
    cfg         : dict from par
    counts_dict : {"Fe":4,"O":6}
    phase_mode  : override PHASE_MODE in par if not None

    PHASE_MODE currently not deeply used (oxide vs metal),
    but we keep it for future e.g. change ISMEAR, SIGMA etc.
    """

    # which phase?
    ph = (phase_mode if phase_mode is not None
          else cfg.get("PHASE_MODE", "oxide")).lower()

    # sort elements in deterministic order (as in POSCAR reading)
    elements = list(counts_dict.keys())

    # Decide SOC lines
    soc_lines, soc_enabled = decide_soc_blocks(cfg, elements)

    # Spin / MAGMOM / NUPDOWN
    spin_lines = build_magmom_block(cfg, counts_dict)

    # +U lines
    ldau_lines = build_ldau_block(cfg, elements)

    # Tail block: ENCUT, relax, IVDW, dipole, etc.
    tail_lines = build_common_block(cfg)

    # Headers
    header = [
        "SYSTEM = auto-generated INCAR",
        f"# Phase mode guess: {ph}",
        "",
        "### --- SCF / electronic setup ---",
    ]

    # If phase is "metal", maybe tweak ISMEAR/SIGMA slightly
    # (simple heuristic)
    # oxide: ISMEAR=0 SIGMA=0.05 (already in tail)
    # metal: we like ISMEAR=1 SIGMA=0.2 for relaxation
    # We'll post-edit the tail_lines for metal
    if ph == "metal":
        tail_lines_mod = []
        for line in tail_lines:
            if line.startswith("ISMEAR"):
                tail_lines_mod.append("ISMEAR = 1")
            elif line.startswith("SIGMA"):
                tail_lines_mod.append("SIGMA  = 0.2")
            else:
                tail_lines_mod.append(line)
        tail_lines = tail_lines_mod

    # SOC and spin interplay:
    # If SOC is enabled, VASP needs noncollinear mode -> MAGMOM handling changes
    # We're already setting LNONCOLLINEAR and SAXIS etc. above.
    # For simplicity we still print spin_lines as-is; for production you might
    # want to adjust MAGMOM format for noncollinear runs.

    lines = []
    lines.extend(header)
    lines.append("### --- Spin / magnetism ---")
    lines.extend(spin_lines)
    lines.append("")
    lines.append("### --- SOC control ---")
    lines.extend(soc_lines)
    lines.append("")
    lines.append("### --- Hubbard U block ---")
    lines.extend(ldau_lines)
    lines.append("")
    lines.append("### --- Common tail block ---")
    lines.extend(tail_lines)
    lines.append("")

    return lines


# =======================================
# 5. CLI main
# =======================================

def main():
    ap = argparse.ArgumentParser(
        description="Generate INCAR from POSCAR and global par switches."
    )
    ap.add_argument("--poscar", default="POSCAR",
                    help="Path to POSCAR-like file (default: POSCAR)")
    ap.add_argument("--counts", default=None,
                    help='Manual counts override like "Fe=4,O=6"')
    ap.add_argument("--phase", default=None,
                    help="Override PHASE_MODE from par (oxide / metal)")
    ap.add_argument("--parfile", default="par",
                    help="Global control file path (default: par)")
    ap.add_argument("--out", default=None,
                    help="If given, write INCAR there instead of stdout")
    args = ap.parse_args()

    cfg = load_par(args.parfile)

    if args.counts:
        counts_dict = parse_counts_string(args.counts)
    else:
        counts_dict = parse_poscar_counts(args.poscar)

    incar_lines = generate_incar(cfg, counts_dict, phase_mode=args.phase)

    text = "\n".join(incar_lines) + "\n"

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text, end="")

if __name__ == "__main__":
    main()
