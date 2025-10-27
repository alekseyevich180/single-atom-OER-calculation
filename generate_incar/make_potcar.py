#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_potcar_from_tag.py  (upgraded)

Usage:
    uv run make_potcar_from_tag.py --Fe_sv_3 --lib /path/to/POTCAR_LIB
or:
    uv run make_potcar_from_tag.py --dir Fe_sv_3 --lib /path/to/POTCAR_LIB

What it does (new behavior):
  - 解析父目录名，得到主金属的 POTCAR tag，例如:
        Fe_3        -> Fe
        Fe_sv_3     -> Fe_sv
        Ir_0        -> Ir
    我们称它为 main_tag。
  - 对父目录本身以及它的一层子目录（OH, OOH, O, ...）：
        * 如果该目录包含 POSCAR：
            1. 解析 POSCAR 的元素顺序，比如 ["Fe","O","H"]
            2. 对每个元素决定要用哪个赝势子目录：
                - 如果元素等于 main element (Fe)：
                      用 main_tag (例如 "Fe_sv")
                - 否则：
                      用元素本名作为 tag（"O", "H", ...）
            3. 从势能库里按顺序把这些子势能的 POTCAR 拼接起来
               写成 calc_dir/POTCAR
        * 如果该目录没有 POSCAR，就跳过

Assumptions / Requirements:
  - 赝势库结构如下：
        /path/to/POTCAR_LIB/Fe_sv/POTCAR
        /path/to/POTCAR_LIB/Fe/POTCAR
        /path/to/POTCAR_LIB/O/POTCAR
        /path/to/POTCAR_LIB/H/POTCAR
    即：每种 tag 都是一个文件夹，里面有一个叫 POTCAR 的文件。
  - POSCAR 的第6行是原子个数，第5行是元素符号列表 (标准VASP格式)。
"""

import os, sys, argparse, re

############################
# helpers
############################

def parse_parent_folder(name):
    """
    同你旧版逻辑，用来决定主金属的赝势tag。

    Examples:
        "Fe_3"        -> main_elem="Fe", main_tag="Fe"
        "Fe_sv_3"     -> main_elem="Fe", main_tag="Fe_sv"
        "Fe_pv"       -> main_elem="Fe", main_tag="Fe_pv"
        "Ir_0"        -> main_elem="Ir", main_tag="Ir"
    """
    toks = name.split("_")
    elem = toks[0]

    suffix = None
    spin = None
    if len(toks) == 1:
        pass
    elif len(toks) == 2:
        # could be Fe_3 or Fe_sv
        if re.fullmatch(r"\d+", toks[1]):
            spin = toks[1]
        else:
            suffix = toks[1]
    else:
        # Fe_sv_3 etc.
        maybe_spin = toks[-1]
        mid = toks[1:-1]
        if re.fullmatch(r"\d+", maybe_spin):
            spin = maybe_spin
            suffix = "_".join(mid) if mid else None
        else:
            suffix = "_".join(toks[1:])
            spin = None

    potcar_tag = f"{elem}_{suffix}" if suffix else elem
    return elem, potcar_tag


def read_poscar_elements(poscar_path):
    """
    读取 POSCAR 前6行，拿到元素顺序列表。
    假设标准格式：
        line 5: "Fe O H"
        line 6: "1 2 1"
    返回 ["Fe","O","H"].
    """
    with open(poscar_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]

    if len(lines) < 7:
        raise ValueError(f"POSCAR too short: {poscar_path}")

    elems_line = lines[5].split()
    counts_line = lines[6].split()

    if len(elems_line) != len(counts_line):
        raise ValueError(f"POSCAR format not recognized in {poscar_path}")

    return elems_line  # preserve order


def build_potcar_for_dir(calc_dir, main_elem, main_tag, potcar_lib):
    """
    根据 calc_dir/POSCAR 里的元素顺序拼接 POTCAR.
    规则：
      - 如果元素 == main_elem (例如 Fe)，使用 main_tag (e.g. "Fe_sv")
      - 其他元素直接用元素名当tag (e.g. "O","H","C",...)
    输出文件: calc_dir/POTCAR
    """
    poscar_path = os.path.join(calc_dir, "POSCAR")
    if not os.path.isfile(poscar_path):
        # 没有 POSCAR 就不做
        return False

    elems = read_poscar_elements(poscar_path)

    # 准备拼接
    potcar_chunks = []
    for el in elems:
        if el == main_elem:
            tag = main_tag
        else:
            tag = el  # assume H -> H, O -> O, etc.

        src_potcar = os.path.join(potcar_lib, tag, "POTCAR")
        if not os.path.isfile(src_potcar):
            raise FileNotFoundError(
                f"Cannot find POTCAR for tag '{tag}' under {potcar_lib}"
            )

        with open(src_potcar, "r", encoding="utf-8") as fsrc:
            potcar_chunks.append(fsrc.read())

    final_potcar_path = os.path.join(calc_dir, "POTCAR")
    with open(final_potcar_path, "w", encoding="utf-8") as fdst:
        for chunk in potcar_chunks:
            fdst.write(chunk)
            # VASP doesn't care about an extra newline, but keep one anyway
            if not chunk.endswith("\n"):
                fdst.write("\n")

    print(f"[OK] {calc_dir}")
    print(f"     POTCAR built from: {', '.join(elems)} "
          f"(main {main_elem} -> {main_tag})")
    return True


############################
# main
############################
def main():
    # 允许两种调用:
    #   --Fe_sv_3 --lib /.../potpaw_PBE.54
    #   --dir Fe_sv_3 --lib /.../potpaw_PBE.54
    if (
        len(sys.argv) >= 2
        and sys.argv[1].startswith("--")
        and "=" not in sys.argv[1]
        and sys.argv[1] not in ("--dir", "--lib")
    ):
        parent_name = sys.argv[1][2:]
        ap = argparse.ArgumentParser()
        ap.add_argument("--lib", required=True,
                        help="Path to POTCAR library root (contains tag subdirs like Fe_sv/, O/, H/...)")
        known, unknown = ap.parse_known_args(sys.argv[2:])
        potcar_lib = known.lib
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("--dir", required=True,
                        help="Parent folder like Fe_3 or Fe_sv_3")
        ap.add_argument("--lib", required=True,
                        help="Path to POTCAR library root (contains tag subdirs like Fe_sv/, O/, H/...)")
        args = ap.parse_args()
        parent_name = args.dir
        potcar_lib  = args.lib

    parent_abs = os.path.abspath(parent_name)
    if not os.path.isdir(parent_abs):
        raise NotADirectoryError(f"{parent_name} is not a directory")

    # 解析父目录名，拿主元素和主tag
    main_elem, main_tag = parse_parent_folder(os.path.basename(parent_abs))

    # 遍历：父目录自己 + 一层子目录
    targets = [parent_abs] + [
        os.path.join(parent_abs, d)
        for d in os.listdir(parent_abs)
        if os.path.isdir(os.path.join(parent_abs, d))
    ]

    any_done = False
    for d in targets:
        ok = build_potcar_for_dir(d, main_elem, main_tag, potcar_lib)
        if ok:
            any_done = True

    if not any_done:
        raise FileNotFoundError(
            f"No POSCAR found under {parent_abs} or its first-level subfolders."
        )

if __name__ == "__main__":
    main()
