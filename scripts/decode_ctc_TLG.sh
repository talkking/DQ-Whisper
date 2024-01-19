#!/bin/bash

. ./path.sh
. cmd.sh

dir=
data=/mnt/lustre/aifs/home/tt321/asr/work-2020/baseline_chn_3K/data/test
transform=
delta_order=0
context=5
skip_frame=3
target_delay=0
stream=160
frame_limit=4096

prior=
prior_scale=1

nj=20
graphdir=data/chn_3K_200319/lang_decode_ctc_2966
acwt=2.0
num_threads=4
suffix=
sets=

. utils/parse_options.sh

#sets="acar accent afar aitrans aitv arbt child common noise oppo"
for setname in $sets; do

    test_set=${data}/${setname} 
    sdata="${test_set}/split${nj}"
    [[ ! -d $sdata && ${data}/${setname}/feats.scp -ot $sdata ]] || split_data.sh --per-utt $data/${setname} $nj || exit 1;
    echo $nj > $dir/num_jobs
    [ ! -d $sdata ] && echo "No such dir $sdata" && exit 1;

    nnet_forward="python -u -m asr.decode --dir $dir --testset $sdata/JOB ${prior:+--prior $prior} --prior-scale $prior_scale "

    decode_dir=${dir}/decode$suffix/$( basename $setname )_${acwt}
    decode.sh --nnet_forward "${nnet_forward}" \
            --nj ${nj} \
            --acwt ${acwt} \
            --num_threads ${num_threads} \
            $graphdir $test_set $decode_dir &
done
