nj=$1
root=$2/split$nj
task=$3
if [ $task = "feats" ]; then
$cpu_cmd JOB=1:$nj $root/log/split.JOB.log python myscripts/make_mfcc.py $nj JOB ${root}
elif [ $task = "utt2dur" ]; then
$cpu_cmd JOB=1:$nj $root/log/split.JOB.log bash utils/get_utt2dur.sh $root/JOB
elif [ $task = "mp3towav" ]; then
$cpu_cmd JOB=1:$nj $root/log/split.JOB.log python myscripts/mp3towav.py $root/JOB/wav.scp 
elif [ $task = "inference" ]; then
testdir=$4
lang=$5
scale=$6
$cuda_cmd JOB=1:$nj $testdir/split$nj/log/split.JOB.log python inference.py $scale $2 $testdir/split$nj/JOB/test.csv $lang $nj JOB
fi
