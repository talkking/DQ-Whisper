#!/bin/bash

. ./path.sh
. cmd.sh

dir=
data=
beam=8
lang=data/Japan/decode_lang/lang_decode_e2e_4303  

nj=10
suffix=
sets=""

. utils/parse_options.sh

for setname in $sets; do

    test_set=${data}/${setname}
    sdata="${test_set}/split${nj}"
    [[ ! -d $sdata && ${data}/${setname}/feats.scp -ot $sdata ]] || split_data.sh --per-utt $data/${setname} $nj || exit 1;
    echo $nj > $dir/num_jobs

    nnet_forward="python -u -m asr.decode_e2e --dir $dir --testset $sdata/JOB --beam $beam"

    decode_dir=${dir}/decode$suffix/$( basename $setname )
    decode_e2e.sh --nnet_forward "${nnet_forward}" \
            --nj ${nj} \
            $lang $test_set $decode_dir &
done
wait
