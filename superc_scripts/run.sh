#!/bin/bash
#PJM -L rscgrp=a-pj24001724
#PJM -L node=1
#PJM --mpi proc=120
#PJM -L elapse=128:00:00
#PJM -j

set -euo pipefail

module load intel
module load impi
module load vasp

source /home/pj24001724/ku40000345/wu/python_venv/ase_env/bin/activate

POTCAR_LIB="/home/pj24001724/ku40000345/vasp_potential/potpaw_PBE.54"
summary_file="$(pwd)/energy_summary.tsv"

if [[ ! -f "$summary_file" ]]; then
  printf "element_dir\tcalc_dir\tEdisp(eV)\tTOTEN(eV)\tE0_noS(eV)\tfe_last_col3\tfe_last_line\tlog_last_line\n" > "$summary_file"
fi

run_calc_dir () {
  local parent_dir="$1"
  local calc_dir="$2"

  cd "$calc_dir"

  mpiexec ~/vasp_6.4.3_vtst_genkai_0725/bin/vasp_std >& log
  command -v vef.pl >/dev/null 2>&1 && vef.pl || true

  local LOG_LAST
  LOG_LAST=$(tail -n 1 log 2>/dev/null || echo "NA")

  local FE_COL3_LAST="NA"
  local FE_LINE_LAST="NA"
  if [[ -f fe.dat ]]; then
    FE_COL3_LAST=$(awk 'END {print $3}' fe.dat 2>/dev/null || echo "NA")
    FE_LINE_LAST=$(tail -n 1 fe.dat 2>/dev/null || echo "NA")
  elif [[ -f OUTCAR ]]; then
    FE_COL3_LAST="NA"
    FE_LINE_LAST="NA"
  fi

  local Edisp="NA"
  if grep -q "Edisp" OUTCAR 2>/dev/null; then
    Edisp=$(grep Edisp OUTCAR | tail -n 1 | awk '{print $3}')
  fi

  local TOTEN="NA"
  if grep -q "free  energy   TOTEN" OUTCAR 2>/dev/null; then
    TOTEN=$(grep "free  energy   TOTEN" OUTCAR | tail -n 1 | awk '{print $5}')
  fi

  local E0="NA"
  if grep -q "energy  without entropy" OUTCAR 2>/dev/null; then
    E0=$(grep "energy  without entropy" OUTCAR | tail -n 1 | awk '{print $7}')
  fi

  cd - >/dev/null

  local calc_name
  calc_name=$(basename "$calc_dir")
  cp -f "$calc_dir/log"        "./$parent_dir/log_${calc_name}"        2>/dev/null || true
  cp -f "$calc_dir/fe.dat"     "./$parent_dir/fe_${calc_name}.dat"     2>/dev/null || true
  cp -f "$calc_dir/OUTCAR"     "./$parent_dir/OUTCAR_${calc_name}"     2>/dev/null || true

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$parent_dir" \
    "$calc_name" \
    "$Edisp" \
    "$TOTEN" \
    "$E0" \
    "$FE_COL3_LAST" \
    "$FE_LINE_LAST" \
    "$LOG_LAST" \
    >> "$summary_file"
}

for parent_dir in */ ; do
  parent_dir="${parent_dir%/}"
  [[ -d "$parent_dir" ]] || continue

  echo "=== [prep INCAR] $parent_dir ==="
  uv run make_incar_from_par.py --"$parent_dir"

  echo "=== [prep POTCAR] $parent_dir ==="
  uv run make_potcar_from_tag.py --"$parent_dir" --lib "$POTCAR_LIB"

  for calc_dir in "$parent_dir"/* ; do
    [[ -d "$calc_dir" ]] || continue
    if [[ -f "$calc_dir/POSCAR" && -f "$calc_dir/INCAR" && -f "$calc_dir/POTCAR" ]]; then
      echo "=== [run VASP] $calc_dir ==="
      run_calc_dir "$parent_dir" "$calc_dir"
    else
      echo ">>> skip $calc_dir (missing POSCAR/INCAR/POTCAR)"
    fi
  done

done

echo "✅ 任务完成：能量结果在  $summary_file"
echo "   每个父目录下也留了 log_*, fe_*.dat, OUTCAR_* 快速检查用"
