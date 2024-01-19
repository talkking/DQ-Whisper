#!/bin/bash
# Copyright 2012-2013 Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# Aligns 'data' to sequences of transition-ids using Neural Network based acoustic model.
# Optionally produces alignment in lattice format, this is handy to get word alignment.

# Begin configuration section.
nj=4
cmd=run.pl
# Begin configuration.
scale_opts="--acoustic-scale=0.1 --self-loop-scale=0.1"
beam=10
retry_beam=40
nnet_forward=

# End configuration options.

[ $# -gt 0 ] && echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "usage: $0 <data-dir> <lang-dir> <align-dir>"
   echo "e.g.:  $0 data/train data/lang exp/tri1_ali"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --nnet-forward <python decode>                   # nnet forward command"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
lang=$2
dir=$3

mkdir -p $dir/log
echo $nj > $dir/num_jobs
sdata=$data/split$nj
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh --per-utt $data $nj || exit 1;

# Check that files exist
for f in $sdata/1/feats.scp $sdata/1/text $lang/TL.fst; do
  [ ! -f $f ] && echo "$0: missing file $f" && exit 1;
done

echo "$0: aligning data '$data', putting alignments in '$dir'"

#wd007
# Map oovs in reference transcription,
#oov=`cat $lang/oov.int` || exit 1;
#tra="ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata/JOB/text|";
tra="ark:utils/sym2int.pl --map-oov 1 -f 2- $lang/words.txt $sdata/JOB/text|";
# We could just use align-mapped in the next line, but it's less efficient as it compiles the
# training graphs one by one.
train_graphs="ark:compile-train-graphs-ctc $lang/TL.fst '$tra' ark:- |"
$cmd JOB=1:$nj $dir/log/align.JOB.log \
  compile-train-graphs-ctc $lang/TL.fst "$tra" ark:- \| \
  align-compiled-mapped-ctc $scale_opts --beam=$beam --retry-beam=$retry_beam ark:- \
  ark:"$nnet_forward |" "ark,scp:$dir/ali.JOB.ark,$dir/ali.JOB.scp" || exit 1;

  #Merge the SCPs to create full list of alignment (will use random access)
echo Merging to single list $dir/ali.scp
for ((n=1; n<=nj; n++)); do
    cat $dir/ali.$n.scp
done > $dir/ali.scp

echo "$0: done aligning data."
