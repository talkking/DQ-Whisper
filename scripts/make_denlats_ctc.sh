#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# Create denominator lattices for MMI/MPE training.
# Creates its output in $dir/lat.*.gz

# Begin configuration section.
nj=4
cmd=run.pl
beam=15.0
lattice_beam=8.0
acwt=1.0
use_gpu=yes # yes|no|optional
nnet_forward=
num_threads=15
# Possibly use multi-threaded decoder
max_active=7000
max_mem=20000000 # This will stop the processes getting too large.
# This is in bytes, but not "real" bytes-- you have to multiply
# by something like 5 or 10 to get real bytes (not sure why so large)
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

thread_string=
[ $num_threads -gt 1 ] && thread_string="-mapped-parallel --num-threads=$num_threads"

if [ $# != 3 ]; then
   echo "Usage: steps/make_denlats_nnet.sh [options] <data-dir> <lang-dir> <exp-dir>"
   echo "  e.g.: steps/make_denlats.sh data/train data/lang exp/tri1_denlats"
   echo "Works for (delta|lda) features, and (with --transform-dir option) such features"
   echo " plus transforms."
   echo ""
   echo "Main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --nnet-forward <python decode>                   # nnet forward command"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --sub-split <n-split>                            # e.g. 40; use this for "
   echo "                           # large databases so your jobs will be smaller and"
   echo "                           # will (individually) finish reasonably soon."
   echo "  --transform-dir <transform-dir>   # directory to find fMLLR transforms."
   exit 1;
fi

data=$1
lang=$2
dir=$3

sdata=$data/split$nj
mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh --per-utt $data $nj || exit 1;
echo $nj > $dir/num_jobs

# Check that files exist
for f in $sdata/1/feats.scp $sdata/1/text $lang/TL.fst; do
  [ ! -f $f ] && echo "$0: missing file $f" && exit 1;
done


#1) We don't use reference path here...

echo "Generating the denlats"
#2) Generate the denominator lattices
  $cmd JOB=1:$nj $dir/log/decode_den.JOB.log \
    $nnet_forward \| \
    latgen-faster-ctc$thread_string --beam=$beam --lattice-beam=$lattice_beam --acoustic-scale=$acwt \
      --max-mem=$max_mem --max-active=$max_active --word-symbol-table=$lang/words.txt  \
      $lang/TLG.fst ark:- ark,scp:$dir/lat.JOB.ark,$dir/lat.JOB.scp || exit 1;

#3) Merge the SCPs to create full list of lattices (will use random access)
echo Merging to single list $dir/lat.scp
for ((n=1; n<=nj; n++)); do
  cat $dir/lat.$n.scp
done > $dir/lat.scp


echo "$0: done generating denominator lattices."
