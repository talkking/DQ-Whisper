#!/bin/bash

. ./path.sh
. cmd.sh

$cuda_cmd JOB=1:16 slurm/train.JOB.log python -u -m asr.launch \
    -c conf/config.yaml \
    -c conf/stage/char_mmi.yaml \
    $@
