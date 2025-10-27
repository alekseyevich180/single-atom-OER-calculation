#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_potcar_from_tag.py

Usage:
    uv run make_potcar_from_tag.py --Fe_sv_3 --lib /path/to/POTCAR_LIB
or:
    uv run make_potcar_from_tag.py --dir Fe_sv_3 --lib /path/to/POTCAR_LIB

What it does:
  - Interpret parent folder name: Fe_3, Fe_sv_3, Fe_pv_2, Ni_0, Ir_0, ...
    (same naming convention as make_incar_from_par.py)
    -> parent directory = that folder
  - For each immediate subdirectory of that parent, plus the parent itself:
      if POTCAR_TAG.txt exists:
          read tag string (e.g. "Fe_sv")
          copy {lib}/{tag}/POTCAR  -->  that subdir/POTCAR
  - If POTCAR already exists in that subdir, it will be overwritten.

Assumptions:
  - Your POTCAR library layout is like:
        /path/to/POTCAR_LIB/Fe_sv/POTCAR
        /path/to/POTCAR_LIB/Fe_pv/POTCAR
        /path/to/POTCAR_LIB/Fe/POTCAR
        /path/to/POTCAR_LIB/Ni/POTCAR
    i.e. each tag has its own subdirectory with a ready POTCAR file.

This script does NOT touch INCAR, par, spin, or NUPDOWN.
It only populates POTCAR based on POTCAR_TAG.txt.
"""

import os, sys, argparse, shutil, re

############################
# helper: parse parent folder
############################
def parse_parent_folder(name):
    """
    Re-use the same rule as make_incar_from_par.py:

    Fe_3        -> potcar_tag="Fe",     spin_hint="3"
    Fe_sv_3     -> potcar_tag="Fe_sv",  spin_hint="3"
    Fe_pv       -> potcar_tag="Fe_pv",  spin_hint=None
    Ni_0        -> potcar_tag="Ni",     spin_hint="0"
    Ir          -> potcar_tag="Ir",     spin_hint=None
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

    potcar = f"{elem}_{suffix}" if suffix else elem
    return potcar

############################
# main
############################
def main():
    # allow:
    #   --Fe_sv_3 --lib /path/to/POTCAR_LIB
    #   --dir Fe_sv_3 --lib /path/to/POTCAR_LIB
    # we REQUIRE --lib because we need to know where POTCAR templates live
    if len(sys.argv) >= 2 and sys.argv[1].startswith("--") and "=" not in sys.argv[1] and sys.argv[1] not in ("--dir","--lib"):
        # pattern: --Fe_sv_3 --lib /lib/path
        parent_name = sys.argv[1][2:]
        # parse rest using argparse just for --lib
        ap = argparse.ArgumentParser()
        ap.add_argument("--lib", required=True,
                        help="Path to POTCAR library root (contains tag subdirs)")
        # ignore parse errors from leftover args not starting with --
        known, unknown = ap.parse_known_args(sys.argv[2:])
        potcar_lib = known.lib
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("--dir", required=True,
                        help="Parent folder like Fe_3 or Fe_sv_3")
        ap.add_argument("--lib", required=True,
                        help="Path to POTCAR library root (contains tag subdirs)")
        args = ap.parse_args()
        parent_name = args.dir
        potcar_lib  = args.lib

    parent_abs = os.path.abspath(parent_name)
    if not os.path.isdir(parent_abs):
        raise NotADirectoryError(f"{parent_name} is not a directory")

    # We will NOT trust POTCAR_TAG.txt blindly if it doesn't exist;
    # but we also don't try to guess which tag each subdir should use
    # from scratch. Instead:
    # - First priority: POTCAR_TAG.txt (if present)
    # - Otherwise: fall back to parent folder name's potcar_tag
    #   (This matches your shell logic expectation.)
    parent_tag = parse_parent_folder(os.path.basename(parent_abs))

    # Collect first-level subdirectories + parent itself
    subdirs = [os.path.join(parent_abs, d)
               for d in os.listdir(parent_abs)
               if os.path.isdir(os.path.join(parent_abs, d))]
    subdirs.append(parent_abs)

    for d in subdirs:
        # decide which tag to use for this subdir
        tag_file = os.path.join(d, "POTCAR_TAG.txt")
        if os.path.isfile(tag_file):
            with open(tag_file, "r", encoding="utf-8") as f:
                tag = f.read().strip()
                if not tag:
                    tag = parent_tag
        else:
            # if we don't see a tag file yet (maybe INCAR step not run),
            # fallback to the parent's tag
            tag = parent_tag

        # find source POTCAR in library
        src_potcar = os.path.join(potcar_lib, tag, "POTCAR")
        if not os.path.isfile(src_potcar):
            print(f"[WARN] {d}: No POTCAR found for tag '{tag}' in {potcar_lib}", file=sys.stderr)
            continue

        # copy to this directory
        dst_potcar = os.path.join(d, "POTCAR")
        shutil.copyfile(src_potcar, dst_potcar)

        print(f"[OK] {d}")
        print(f"     tag   -> {tag}")
        print(f"     POTCAR copied from {src_potcar}")

if __name__ == "__main__":
    main()
