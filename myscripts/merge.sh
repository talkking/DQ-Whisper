nj=$3
root=$1/split$nj
merge_file=$2
#$cpu_cmd JOB=1:$nj $dir/log/split.JOB.log bash process.sh JOB
touch $merge_file
for i in `seq 1 $nj`; do
  cat ${merge_file} $root/$i/${merge_file} > tmp 
  mv tmp ${merge_file} 
done
mv ${merge_file} $1
