!/bin/bash

#SBATCH --job-name=[your job name]    # 你的任务名，可以随意改
#SBATCH --partition=64c512g           # 目标分区，思源1号就用 64c512g
#SBATCH -n 40                         # 需求的核数。需要按需取自定，重要
#SBATCH --ntasks-per-node=64          # 单个节点的核数。如果用思源1号就用64
#SBATCH --output=hpclog/%j.out        # 输出记录查询地址，重要
#SBATCH --error=hpclog/%j.err         # 报错记录查询地址，重要

source activate template       # 一定要用 source 不要用conda, 重要

python m1_fit.py -d='exp1data' -n='RL' -s=420 -f=40 -c=40 -m='map' -a='BFGS'
