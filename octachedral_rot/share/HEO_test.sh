#!/bin/bash
#PJM -L rscgrp=a-pj24001724
#PJM -L node=1
##PJM -L vnode-core=60
#PJM --mpi proc=120
#PJM -L elapse=6:00:00
#PJM -j

module load intel
module load impi
module load vasp

dir_name=$(basename $(pwd))
end_time=$(date -d '+6 hours' '+%Y-%m-%d %H:%M:%S')

echo -e "${dir_name}\n${end_time}" >> job.txt

mpiexec ~/vasp_6.4.3_vtst_genkai_0725/bin/vasp_std >& log
if [ $? -ne 0 ]; then
  echo "vasp error" > error.txt
fi

# cp POSCAR fe.dat
vef.pl
