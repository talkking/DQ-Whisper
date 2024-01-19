#!/bin/bash
#SBATCH --job-name=jp_quan    #定义一个任务名
#SBATCH --partition=2080ti,gpu   #指定队列名，cpu任务队列名是cpu；GPU任务可以指定gpu或者2080ti的队列名
#SBATCH -n 8   #需要运行10个任务
#SBATCH --cpus-per-task=2         # 单任务使用的 CPU 核心数为 2，不指定的话默认是1
#SBATCH --gres=gpu:1              # 单个节点使用 1 块 GPU 卡
#SBATCH --output=%j.out  #标准输出
#SBATCH --error=%j.err #标准错误输出

. ./path.sh
. cmd.sh
conf=
expdir=

nj=16

 . utils/parse_options.sh
$cuda_cmd JOB=1:$nj $expdir/log/train.JOB.log python -u -m asr.launch \
    -c $conf \
    $@  || exit 1

#$cuda_cmd JOB=1:$nj slurm/train.JOB.log python -u -m asr.launch \
#    -c conf/config.yaml \
#    -c conf/stage/char_ctc.yaml \
#    $@  || exit 1

# you can also train monophone CTC for fun
# python -u -m asr.launch -c conf/config.yaml -c conf/stage/mono_ctc.yaml
