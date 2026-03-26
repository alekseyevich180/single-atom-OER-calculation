#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_incar_from_par.py
----------------------

Goal:
    uv run make_incar_from_par.py --Fe_3
or
    uv run make_incar_from_par.py --dir Fe_pv_3

What it will do:
  - Interpret "Fe_pv_3" (or "Fe_3", "Ti_sv_2", etc.) as:
        base element, POTCAR tag, spin hint
  - Optionally look inside that directory for subfolders (e.g. OH, OOH, O, ...)
    controlled by par: SCAN_SUBDIRS=true/false. For each target folder that contains POSCAR:
        * read global par
        * (optionally) ensure USE_SPIN=True if name had _3 etc.
        * generate INCAR in that subfolder
        * generate POTCAR_TAG.txt in that subfolder

So:
    Fe_3/
      OH/POSCAR   -> write Fe_3/OH/INCAR etc.
      OOH/POSCAR  -> write Fe_3/OOH/INCAR etc.

We assume:
  - global "par" file lives either in CWD or script dir
  - children share the same element / spin preference from the parent folder name.
"""

import os, re, sys, argparse

########################
# Helpers
########################

def load_par(path="par"):
    cfg={}
    with open(path,"r",encoding="utf-8") as f:
        for l in f:
            l=l.strip()
            if not l or l.startswith("#") or "=" not in l:
                continue
            k,v=l.split("=",1); k=k.strip(); v=v.strip()
            if v.lower() in ["true","false"]:
                cfg[k]=(v.lower()=="true")
            else:
                cfg[k]=v
    return cfg

def parse_poscar_counts(poscar):
    with open(poscar,"r",encoding="utf-8") as f:
        lines=[x.strip() for x in f]
    # Here we assume POSCAR format with element symbols on line[5],
    # counts on line[6]. Example:
    # line[5]: "Fe O H"
    # line[6]: "1 2 1"
    elems_line = lines[5].split()
    nums_line  = lines[6].split()
    counts = {el:int(n) for el,n in zip(elems_line, nums_line)}
    return counts

MAGMOM = {
    "Sc":1.0,"Ti":1.0,"V":3.0,"Cr":3.5,"Mn":5.0,
    "Fe":3.0,"Co":2.0,"Ni":1.0,"Cu":0.0,"Zn":0.0,
    "Ir":0.5,"W":0.5,"Os":0.5,"Pt":0.0,"Au":0.0,
    "O":0.0,"H":0.0,"C":0.0,"N":0.0,"S":0.0,
}
LDAU_U = {
    "Ti":4.5,"V":3.3,"Cr":3.5,"Mn":4.0,
    "Fe":4.0,"Co":3.3,"Ni":6.0,"Cu":4.5,
    "Ir":2.0,
}

def decide_soc(cfg, elems):
    heavy5d = {"Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg"}
    mode = cfg.get("FORCE_SOC","auto").lower()
    use_soc = (mode=="on") or (mode=="auto" and any(e in heavy5d for e in elems))
    if use_soc:
        return [
            "ISYM=0",
            "LSORBIT=.TRUE.",
            "LNONCOLLINEAR=.TRUE.",
            "GGA_COMPAT=.FALSE.",
            "SAXIS=0 0 1"
        ]
    else:
        return [
            "ISYM=2",
            "LSORBIT=.FALSE.",
            "LNONCOLLINEAR=.FALSE.",
            "GGA_COMPAT=.TRUE."
        ]

def build_incar_text(cfg, counts):
    """
    Build a nicely formatted INCAR with separated logical blocks:
    1. General (spin, SOC, smearing)
    2. GGA+U
    3. SCF / relax / parallel / vdW / IO / dipole
    """
    elems = list(counts.keys())

    # ----- 1. Spin / MAGMOM / NUPDOWN -----
    spin_lines = []
    if cfg.get("USE_SPIN", True):
        spin_lines.append("ISPIN = 2")

        mag_chunks = []
        for el, num in counts.items():
            m = MAGMOM.get(el, 0.0)
            mag_chunks.append(f"{num}*{m}")
        spin_lines.append("MAGMOM = " + " ".join(mag_chunks))

        # ✅ 关键逻辑：只有 par 里 FIX_NUPDOWN=true 时才写 NUPDOWN
        if cfg.get("FIX_NUPDOWN", False):
            spin_lines.append(f"NUPDOWN = {cfg.get('NUPDOWN_VALUE','0')}")
    else:
        spin_lines.append("ISPIN = 1")

    # smearing block (electronic occupation)
    phase_mode = cfg.get("PHASE_MODE","oxide").lower()
    if phase_mode != "metal":
        spin_lines.append("ISMEAR = 0")
        spin_lines.append("SIGMA  = 0.05")
    else:
        spin_lines.append("ISMEAR = 1")
        spin_lines.append("SIGMA  = 0.2")

    # ----- SOC block -----
    soc_lines = decide_soc(cfg, elems)
    pretty_soc = []
    for line in soc_lines:
        if "=" in line and " " not in line.split("=",1)[0]:
            k,v = line.split("=",1)
            k = k.strip()
            v = v.strip()
            pretty_soc.append(f"{k} = {v}")
        else:
            m = line.split("=",1)
            if len(m)==2:
                pretty_soc.append(f"{m[0].strip()} = {m[1].strip()}")
            else:
                pretty_soc.append(line)

    # ----- 2. GGA+U block -----
    if cfg.get("USE_LDAU", False):
        ldaul_list=[]
        ldauu_list=[]
        for el in elems:
            if el in LDAU_U:
                ldaul_list.append("2")
                ldauu_list.append(str(LDAU_U[el]))
            else:
                ldaul_list.append("-1")
                ldauu_list.append("0.0")

        u_lines = [
            "LDAU      = .TRUE.",
            "LDAUTYPE  = 2",
            "LDAUL     = " + " ".join(ldaul_list),
            "LDAUU     = " + " ".join(ldauu_list),
            "LDAUJ     = " + " ".join("0.0" for _ in elems),
            "LASPH     = .TRUE.",
            "LMAXMIX   = 4",
        ]
    else:
        u_lines = [
            "LDAU      = .FALSE.",
            "LASPH     = .TRUE.",
            "LMAXMIX   = 4",
        ]

    # ----- 3. Tail block (SCF / relax / parallel / vdW / IO / dipole) -----
    tail_lines = []

    tail_lines.append("# Electronic loop controls:")
    tail_lines.append("ENCUT  = 400")
    tail_lines.append("ALGO   = ALL")
    tail_lines.append("EDIFF  = 1E-8")
    tail_lines.append("NELM   = 500")

    tail_lines.append("")
    tail_lines.append("# Relaxation control:")
    tail_lines.append("IBRION = 2")
    tail_lines.append("NSW    = 5000")
    tail_lines.append("POTIM  = 0.100000")
    tail_lines.append("ISIF   = 0")
    tail_lines.append("EDIFFG = -0.020000")

    tail_lines.append("")
    tail_lines.append("# Parallelization:")
    tail_lines.append("NPAR   = 5")
    tail_lines.append("LREAL  = AUTO")
    tail_lines.append("NSIM   = 1")
    tail_lines.append("LPLANE = .TRUE.")

    tail_lines.append("")
    tail_lines.append("# vdW:")
    if cfg.get("USE_VDW", True):
        tail_lines.append(f"IVDW   = {cfg.get('VDW_FLAG','12')}")
    else:
        tail_lines.append("# IVDW   = (disabled)")

    tail_lines.append("")
    tail_lines.append("# I/O:")
    tail_lines.append("LAECHG = .FALSE.")
    tail_lines.append("LCHARG = .FALSE.")
    tail_lines.append("LWAVE  = .FALSE.")
    tail_lines.append("LELF   = .FALSE.")
    tail_lines.append("LVTOT  = .FALSE.")
    tail_lines.append("LVHAR  = .FALSE.")
    tail_lines.append("LREAL  = .FALSE.")

    tail_lines.append("")
    tail_lines.append("# Dipole correction:")
    if cfg.get("USE_DIPOL", False):
        tail_lines.append(f"IDIPOL = {cfg.get('IDIPOL_DIR','4')}")
        tail_lines.append("LDIPOL = .TRUE.")
    else:
        tail_lines.append("# IDIPOL = 4")
        tail_lines.append("# LDIPOL = .FALSE.")

    # ----- final join -----
    dump = []
    dump.append("# ==== General (spin / SOC / smearing) ====")
    dump.extend(spin_lines)
    dump.extend(pretty_soc)
    dump.append("")
    dump.append("# ==== GGA+U (DFT+U) ====")
    dump.extend(u_lines)
    dump.append("")
    dump.append("# ==== SCF / relax / parallel / vdW / IO ====")
    dump.extend(tail_lines)
    dump.append("")

    return "\n".join(dump) + "\n"


########################
# folder name interpreter
########################

def parse_parent_folder(name):
    """
    Interpret names like:
      Fe_3        -> element=Fe, potcar_tag=Fe, spin=3
      Fe_pv_3     -> element=Fe, potcar_tag=Fe_pv, spin=3
      Ti_sv_2     -> element=Ti, potcar_tag=Ti_sv, spin=2
      Fe_pv       -> element=Fe, potcar_tag=Fe_pv, spin=None
      Ir_0        -> element=Ir, potcar_tag=Ir, spin=0
    """
    toks = name.split("_")
    elem = toks[0]

    suffix = None
    spin = None

    if len(toks) == 1:
        pass
    elif len(toks) == 2:
        if re.fullmatch(r"\d+", toks[1]):
            spin = toks[1]
        else:
            suffix = toks[1]
    else:
        last = toks[-1]
        mid = toks[1:-1]
        if re.fullmatch(r"\d+", last):
            spin = last
            suffix = "_".join(mid) if mid else None
        else:
            suffix = "_".join(toks[1:])
            spin = None

    potcar_tag = f"{elem}_{suffix}" if suffix else elem
    return elem, potcar_tag, spin


########################
# main workflow
########################

def main():
    # 支持两种调用方式：
    #   uv run make_incar_from_par.py --Fe_3
    #   uv run make_incar_from_par.py --dir Fe_3
    if len(sys.argv) == 2 and sys.argv[1].startswith("--") and "=" not in sys.argv[1]:
        parent_name = sys.argv[1][2:]
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("--dir", required=True,
                        help="Parent folder like Fe_3 or Fe_pv_3")
        parent_name = ap.parse_args().dir

    parent_abs = os.path.abspath(parent_name)
    if not os.path.isdir(parent_abs):
        raise NotADirectoryError(f"{parent_name} is not a directory")

    # 1. 解析父目录名 -> 元素/势能/自旋提示
    elem, potcar_tag, spin_val = parse_parent_folder(os.path.basename(parent_abs))

    # 2. 找 par:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.isfile("par"):
        par_path = "par"
    elif os.path.isfile(os.path.join(script_dir,"par")):
        par_path = os.path.join(script_dir,"par")
    else:
        raise FileNotFoundError("Cannot find 'par' file (looked in CWD and script dir).")

    base_cfg = load_par(par_path)

    # 如果父目录名字里指定了spin，比如 Fe_3：
    # 我们只确保它是自旋极化 (USE_SPIN=True)
    # 但是不强制 FIX_NUPDOWN / NUPDOWN_VALUE
    if spin_val is not None:
        base_cfg = dict(base_cfg)  # shallow copy
        base_cfg["USE_SPIN"] = True
        # do NOT touch FIX_NUPDOWN or NUPDOWN_VALUE here

    # 3. 遍历父目录（可选：包含第一层子目录）
    scan_subdirs = base_cfg.get("SCAN_SUBDIRS", True)
    if scan_subdirs:
        subdirs = [os.path.join(parent_abs, d)
                   for d in os.listdir(parent_abs)
                   if os.path.isdir(os.path.join(parent_abs, d))]
        subdirs.append(parent_abs)
    else:
        subdirs = [parent_abs]

    any_done = False
    for d in subdirs:
        poscar_path = os.path.join(d, "POSCAR")
        if not os.path.isfile(poscar_path):
            continue
        any_done = True

        # 4. 读取 POSCAR 原子计数
        counts = parse_poscar_counts(poscar_path)

        # 5. 生成 INCAR
        incar_txt = build_incar_text(base_cfg, counts)

        # 6. 写 INCAR 和 POTCAR_TAG.txt
        with open(os.path.join(d, "INCAR"), "w", encoding="utf-8") as f:
            f.write(incar_txt)
        with open(os.path.join(d, "POTCAR_TAG.txt"), "w", encoding="utf-8") as f:
            f.write(potcar_tag + "\n")

        print(f"[OK] {d}")
        print(f"     POTCAR -> {potcar_tag}")
        if base_cfg.get("FIX_NUPDOWN", False):
            print(f"     NUPDOWN locked -> {base_cfg.get('NUPDOWN_VALUE','0')}")
        else:
            print("     NUPDOWN not fixed")

    if not any_done:
        if scan_subdirs:
            raise FileNotFoundError(
                f"No POSCAR found under {parent_abs} or its first-level subfolders."
            )
        else:
            raise FileNotFoundError(
                f"No POSCAR found in {parent_abs}."
            )

if __name__ == "__main__":
    main()
