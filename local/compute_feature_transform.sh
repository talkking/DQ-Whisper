#!/bin/bash

. ./path.sh


online=false
cmvn_opts="--norm-means=true --norm-vars=false"
delta_opts="--delta-order=0"
splice=5
splice_step=1
num=100000

. utils/parse_options.sh || exit 1;

if [ $# != 2 ]; then
   echo "usage: compute_feature_transform.sh [options] <data-tr> <dir>";
   echo "options: "
   echo "  --online <true/false>                    # whether to apply cmvn"
   echo "  --cmvn-opts <cmvn_opts>                  # cmvn options"
   echo "  --delta-opts <delta_opts>                # delta options"
   echo "  --splice <splice_num>                    # number of frames to concatenate with the central frame"
   exit 1;
fi

data_tr=$1
dir=$2

mkdir -p $dir
mkdir -p $dir/log

shuf $data_tr/feats.scp | head -n $num > $dir/feats.scp.part

feats="ark:copy-feats scp:$dir/feats.scp.part ark:- |"
if [ ! -z "$cmvn_opts" -a "$online" == "false" ];then
    feats="$feats apply-cmvn $cmvn_opts --utt2spk=ark:$data_tr/utt2spk scp:$data_tr/cmvn.scp ark:- ark:- |"
elif [ ! -z "$cmvn_opts" -a "$online" == "true" ];then
    feats="$feats apply-cmvn-sliding $cmvn_opts ark:- ark:- |"
else
    echo "apply-cmvn is not used"
fi

feats="$feats add-deltas $delta_opts ark:- ark:- |"

feat_dim=$(feat-to-dim --print-args=false "$feats" -)

# Generate the splice transform
echo "Using splice +/- $splice , step $splice_step"
feature_transform=$dir/tr_splice${splice}-${splice_step}.nnet
utils/nnet/gen_splice.py --fea-dim=$feat_dim --splice=$splice --splice-step=$splice_step > $feature_transform

feature_transform_old=$feature_transform
feature_transform=${feature_transform%.nnet}_cmvn-g.nnet
echo "Renormalizing MLP input features into $feature_transform"
nnet-forward --use-gpu=no \
    $feature_transform_old "$feats" \
    ark:- 2>$dir/log/nnet-forward-cmvn.log |\
  compute-cmvn-stats ark:- - | cmvn-to-nnet - - |\
  nnet-concat --binary=false $feature_transform_old - $feature_transform
  
[ ! -f $feature_transform ] && cat $dir/log/nnet-forward-cmvn.log && echo "Error: Global CMVN failed, was the CUDA GPU okay?" && echo && exit 1

feats="$feats nnet-forward --use-gpu=no $feature_transform ark:- ark:- |"
new_feat_dim=$(feat-to-dim --print-args=false "$feats" -)
echo "$feats" > $dir/feats.rspecifier
echo "$new_feat_dim" > $dir/feats_dim

