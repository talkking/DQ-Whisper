#!/bin/bash
. ./path.sh
. ./cmd.sh

nj=100
compress=true

. utils/parse_options.sh || exit 1;

data_ori=$1
data=$2

mkdir -p $data
mkdir -p $data/tmp
mkdir -p $data/dump

sort -nk2 $data_ori/featlen > $data/featlen

for f in cmvn.scp spk2utt text utt2spk ; do
    awk 'NR==FNR{a[$1]=$0;}NR>FNR{if(a[$1]!="") print a[$1];}' \
        $data_ori/$f $data/featlen > $data/$f
done

awk 'NR==FNR{a[$1]=$0;}NR>FNR{if(a[$1]!="") print a[$1];}' \
    $data_ori/feats.scp $data/featlen > $data/feats.scp.tmp

split_scps=""
for n in $(seq $nj); do
    split_scps="$split_scps $data/tmp/feats.$n.scp"
done

utils/split_scp.pl $data/feats.scp.tmp $split_scps || exit 1;

dumpdir=`pwd`/$data/dump
$cpu_cmd JOB=1:$nj $data/tmp/dump_feat.JOB.log \
    copy-feats --compress=$compress scp:$data/tmp/feats.JOB.scp \
      ark,scp:${dumpdir}/feats.JOB.ark,${dumpdir}/feats.JOB.scp \
      || exit 1;

for n in $(seq $nj); do
    cat ${dumpdir}/feats.$n.scp || exit 1;
done > $data/feats.scp
