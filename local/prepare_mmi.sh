#!/bin/bash

. ./path.sh
. cmd.sh

data=
nj=50
ctc_dir=
prior=
lang=
stage=0

. utils/parse_options.sh || exit 1

# Build mmi related exp directories
data_dir=$data/tr
ali_dir=${ctc_dir}_ali
den_dir=${ctc_dir}_denlats
mmi_data_dir=${ctc_dir}_mmi/data/tr

# Set nnet forward options
nnet=$ctc_dir/checkpoint;

# Check exist
for f in $nnet $prior; do
  [ ! -f $f ] && echo "$0: missing file $f" && exit 1;
done

sdata=$data_dir/split$nj

nnet_forward="python -u -m asr.decode --dir $ctc_dir --testset ${sdata}/JOB ${prior:+--prior $prior}"

# Run

if [ $stage -le 0 ]; then
    align_ctc.sh --nj $nj --cmd "$cuda_cmd" --nnet-forward "$nnet_forward" $data_dir $lang $ali_dir
fi

if [ $stage -le 1 ]; then
    make_denlats_ctc.sh --nj $nj --cmd "$decode_cmd" --num-threads 15 --acwt 1.0 --nnet-forward "$nnet_forward" $data_dir $lang $den_dir
fi

echo "Post-processing ali and genlat"
rm -rf $mmi_data_dir
mkdir -p $mmi_data_dir
# Get the utterances with alignment
awk '{++count[$1]} END { for (uid in count) if (count[uid] == 3) print uid }' ${data_dir}/feats.scp ${ali_dir}/ali.scp ${den_dir}/lat.scp > /tmp/good.list
awk 'NR==FNR{a[$1]=1}NR>FNR{if ($1 in a) print $0}' /tmp/good.list $data_dir/feats.scp > $mmi_data_dir/feats.scp
awk 'NR==FNR{a[$1]=1}NR>FNR{if ($1 in a) print $0}' /tmp/good.list $ali_dir/ali.scp > $mmi_data_dir/ali.scp
awk 'NR==FNR{a[$1]=1}NR>FNR{if ($1 in a) print $0}' /tmp/good.list $den_dir/lat.scp > $mmi_data_dir/lat.scp
