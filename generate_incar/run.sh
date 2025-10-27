#!/bin/bash
#PJM -L rscgrp=a-pj24001724
#PJM -L node=1
#PJM --mpi proc=120
#PJM -L elapse=128:00:00
#PJM -j

set -euo pipefail

#####################################
# 1. 环境准备
#####################################

module load intel
module load impi
module load vasp

# 激活 Python 环境 (uv会用到)
source /home/pj24001724/ku40000345/wu/python_venv/ase_env/bin/activate

# 赝势库根目录：make_potcar.py 会从这里拷贝
POTCAR_LIB="/home/pj24001724/ku40000345/vasp_potential/potpaw_PBE.54"

# 汇总能量输出文件（全局）
summary_file="$(pwd)/energy_summary.tsv"

# 如果 summary_file 还没建，就写表头
if [[ ! -f "$summary_file" ]]; then
  printf "element_dir\tcalc_dir\tEdisp(eV)\tTOTEN(eV)\tE0_noS(eV)\tfe_last_col3\tfe_last_line\tlog_last_line\n" > "$summary_file"
fi


#####################################
# 2. 跑单个计算目录的 VASP，并且收集能量
#####################################
run_calc_dir () {
  local parent_dir="$1"   # e.g. Fe_3
  local calc_dir="$2"     # e.g. Fe_3/OH

  # 进入具体计算子目录
  cd "$calc_dir"

  # 运行 VASP
  mpiexec ~/vasp_6.4.3_vtst_genkai_0725/bin/vasp_std >& log
  # 如果有 vef.pl 就跑一下（没有也不报错）
  command -v vef.pl >/dev/null 2>&1 && vef.pl || true

  # 从输出文件中提取信息
  # log 最后一行
  local LOG_LAST
  LOG_LAST=$(tail -n 1 log 2>/dev/null || echo "NA")

  # fe.dat (vtst 输出), 第三列末行 & 最后一整行
  local FE_COL3_LAST="NA"
  local FE_LINE_LAST="NA"
  if [[ -f fe.dat ]]; then
    FE_COL3_LAST=$(awk 'END {print $3}' fe.dat 2>/dev/null || echo "NA")
    FE_LINE_LAST=$(tail -n 1 fe.dat 2>/dev/null || echo "NA")
  elif [[ -f OUTCAR ]]; then
    # 一些情况下 fe.dat 还没生成，我们就不强求
    FE_COL3_LAST="NA"
    FE_LINE_LAST="NA"
  fi

  # Edisp (vdW校正能)
  local Edisp="NA"
  if grep -q "Edisp" OUTCAR 2>/dev/null; then
    Edisp=$(grep Edisp OUTCAR | tail -n 1 | awk '{print $3}')
  fi

  # TOTEN (自由能)
  local TOTEN="NA"
  if grep -q "free  energy   TOTEN" OUTCAR 2>/dev/null; then
    TOTEN=$(grep "free  energy   TOTEN" OUTCAR | tail -n 1 | awk '{print $5}')
  fi

  # E0_noS (能量 without entropy)
  local E0="NA"
  if grep -q "energy  without entropy" OUTCAR 2>/dev/null; then
    E0=$(grep "energy  without entropy" OUTCAR | tail -n 1 | awk '{print $7}')
  fi

  # 回到作业提交根目录之前，顺手把 log / fe.dat / OUTCAR 复制到父目录上一级方便查看
  # 比如 Fe_3/OH/log -> Fe_3/log_OH
  cd - >/dev/null

  local calc_name
  calc_name=$(basename "$calc_dir")
  cp -f "$calc_dir/log"        "./$parent_dir/log_${calc_name}"        2>/dev/null || true
  cp -f "$calc_dir/fe.dat"     "./$parent_dir/fe_${calc_name}.dat"     2>/dev/null || true
  cp -f "$calc_dir/OUTCAR"     "./$parent_dir/OUTCAR_${calc_name}"     2>/dev/null || true

  # 记录到全局 summary
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


#####################################
# 3. 主循环：对每个父目录做准备+运行
#####################################
# 假设当前目录下有很多父目录，比如 Fe_3, Fe_sv_3, Ni_0, Ir_0 ...
# 我们逐个处理
for parent_dir in */ ; do
  parent_dir="${parent_dir%/}"
  [[ -d "$parent_dir" ]] || continue

  echo "=== [prep INCAR] $parent_dir ==="
  uv run make_incar_from_par.py --"$parent_dir"

  echo "=== [prep POTCAR] $parent_dir ==="
  uv run make_potcar_from_tag.py --"$parent_dir" --lib "$POTCAR_LIB"

  # 现在 parent_dir 下应该有若干实际计算子目录，比如 OH / OOH / O / ...
  # 每个子目录都应该已经有：
  #   POSCAR
  #   INCAR   (刚生成)
  #   POTCAR  (刚拷贝)
  #   KPOINTS (你应该事先放好，脚本不会生成)
  #
  # 我们对这些子目录逐个跑 vasp
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


#####################################
# 4. 结束信息
#####################################
echo "✅ 任务完成：能量结果在  $summary_file"
echo "   每个父目录下也留了 log_*, fe_*.dat, OUTCAR_* 快速检查用"
