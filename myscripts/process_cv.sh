lang=$1

rootdir=data/cv_5.1/cv-corpus-5.1-2020-06-22/$lang

find ~/low_resource/data/cv_5.1/cv-corpus-5.1-2020-06-22/$lang/clips -name "*.mp3" > $rootdir/wav.scp

awk '{split($0,s,".mp3"); split(s[1],s1,"/"); print(s1[14]" "$0);}' $rootdir/wav.scp > $rootdir/wav.scp.tmp

mv $rootdir/wav.scp.tmp $rootdir/wav.scp

python myscripts/read_tsv.py $rootdir/train.tsv > $rootdir/train

python myscripts/read_tsv.py $rootdir/dev.tsv > $rootdir/dev

python myscripts/read_tsv.py $rootdir/test.tsv > $rootdir/tt


if [ -e "$rootdir/tr" ]; then
   rm -rf $rootdir/{tr,cv,test}
   mkdir $rootdir/{tr,cv,test}
else
   mkdir $rootdir/{tr,cv,test}
fi

awk 'NR==FNR{a[$1]=1}NR>FNR{if($1 in a) print($0)}' $rootdir/train $rootdir/wav.scp > $rootdir/tr/wav.scp

awk 'NR==FNR{a[$1]=1}NR>FNR{if($1 in a) print($0)}' $rootdir/dev $rootdir/wav.scp > $rootdir/cv/wav.scp

awk 'NR==FNR{a[$1]=1}NR>FNR{if($1 in a) print($0)}' $rootdir/tt $rootdir/wav.scp > $rootdir/test/wav.scp

# convert mp3 to wav

. ./cmd.sh
python myscripts/splitjob.py 64 64 $rootdir/tr wav.scp
python myscripts/splitjob.py 64 64 $rootdir/cv wav.scp
python myscripts/splitjob.py 64 64 $rootdir/test wav.scp

bash myscripts/slurm_job.sh 64 $rootdir/tr mp3towav
bash myscripts/slurm_job.sh 64 $rootdir/cv mp3towav
bash myscripts/slurm_job.sh 64 $rootdir/test mp3towav

sed -i 's/.mp3/.wav/g' $rootdir/tr/wav.scp
sed -i 's/.mp3/.wav/g' $rootdir/cv/wav.scp
sed -i 's/.mp3/.wav/g' $rootdir/test/wav.scp


mv $rootdir/train $rootdir/tr/text

mv $rootdir/dev $rootdir/cv/text

mv $rootdir/tt $rootdir/test/text

python myscripts/tokenizer.py $rootdir/tr/{text,dec_input_ids.ark,labels.ark} $lang

python myscripts/tokenizer.py $rootdir/cv/{text,dec_input_ids.ark,labels.ark} $lang

awk 'NR==FNR{a[$1]=$2}NR>FNR{name=$1;$1="";if(name in a) print(name"|"a[name]"|"$0);}' $rootdir/test/wav.scp $rootdir/test/text > $rootdir/test/test.csv

#feat

python myscripts/splitjob.py 64 64 $rootdir/tr wav.scp
python myscripts/splitjob.py 64 64 $rootdir/cv wav.scp
python myscripts/splitjob.py 64 64 $rootdir/test wav.scp

bash myscripts/slurm_job.sh 64 $rootdir/tr feats
bash myscripts/merge.sh $rootdir/tr feats.scp 64
bash myscripts/slurm_job.sh 64 $rootdir/tr utt2dur
bash myscripts/merge.sh $rootdir/tr utt2dur 64

bash myscripts/slurm_job.sh 64 $rootdir/cv feats
bash myscripts/merge.sh $rootdir/cv feats.scp 64
bash myscripts/slurm_job.sh 64 $rootdir/cv utt2dur
bash myscripts/merge.sh $rootdir/cv utt2dur 64

bash myscripts/slurm_job.sh 64 $rootdir/test utt2dur
bash myscripts/merge.sh $rootdir/test utt2dur 64






