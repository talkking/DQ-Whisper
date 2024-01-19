#!/usr/bin/bash 
. ./cmd.sh
rootdir=$1 # checkpoint_dir
scale=$2 # model scale
testdir=$3 # testset dir
lang=$4 # lang id
nj=$5 # number of jobs
python myscripts/splitjob.py $nj $nj $testdir test.csv 
if [ ! -e $rootdir/decode ]; then
    #rm -rf $rootdir/decode
    mkdir -p $rootdir/decode
#else
#    mkdir -p $rootdir/decode
fi
if [ -e $rootdir/decode/$lang ]; then
    rm -rf $rootdir/decode/$lang
    mkdir -p $rootdir/decode/$lang
else
    mkdir -p $rootdir/decode/$lang
fi
if [ -e $rootdir/decode/$lang/split$nj ]; then
    rm -rf $rootdir/decode/$lang/split$nj
    mkdir -p $rootdir/decode/$lang/split$nj
else
    mkdir -p $rootdir/decode/$lang/split$nj
fi
bash myscripts/slurm_job.sh $nj $rootdir inference $testdir $lang $scale
bash myscripts/merge.sh $rootdir/decode/$lang ref.txt $nj
bash myscripts/merge.sh $rootdir/decode/$lang hyp.txt $nj
python myscripts/score.py $rootdir/decode/$lang
