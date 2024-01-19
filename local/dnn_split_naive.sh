#!/bin/bash

if [ $# -ne 3 ] && [ $# -ne 4 ]; then
  echo "usage: $0 numjobs data_dir split_data_dir block_size=1000(optional)"
  exit
fi

# set -x

numjobs=$1
data_dir=$2
split_data_dir=$3
block_size=$4
block_size=${block_size:=1024}  # contiguous utterences in a block

master_bn=feats.scp
master_file=$2/$master_bn
dir="$split_data_dir/split${numjobs}"

# In case some splits contain no utterences
total_lines=`wc -l $master_file | awk '{print $1;}'`
minimal_block_size=`expr $total_lines / $numjobs`
block_size=$(( block_size < minimal_block_size ? block_size : minimal_block_size ))
echo "Using block size: $block_size"

rm -rf $dir
mkdir -p $dir/log
for split_id in $(seq 1 $numjobs);
do
mkdir -p $dir/$(expr $split_id - 1)
done

awk -v BS=$block_size -v JB=$numjobs -v FN=$master_bn -v DR=$dir \
  'BEGIN{j=0}
   {
    bid=int(j/BS);
    print $0 >> DR"/"bid"/"FN;
    ++j; if (j==JB*BS) j=0;
   }' $master_file || exit 1

pids=()
for data_file in $data_dir/*; do
base_name=$(basename $data_file)
# Don't split feats.scp again
if [ $base_name != $master_bn -a ! -d $data_file ]; then
  $cpu_cmd JOB=1:$numjobs $dir/log/split.JOB.${base_name}.log \
    awk 'NR==FNR{a[$1]=$0;}NR>FNR{if(a[$1]!="") print a[$1];}' \
    $data_file $dir/\$[JOB-1]/$master_bn \> $dir/\$[JOB-1]/$base_name &
  pids+=($!);
fi
done

# Check splitting tasks work as expected
for pid in ${pids[*]}; do
    wait $pid || exit 100
done
