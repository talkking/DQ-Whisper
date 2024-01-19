## wav
find ~/low_resource/data/cv-corpus-12.0-delta-2022-12-07/de/clips -name "*.mp3" > wav.scp

awk '{split($0,s,".mp3"); split(s[1],s1,"/"); print(s1[14]" "$0);}' wav.scp > tmp

mv tmp wav.scp

##从wav.scp中选取出文本存在的utterance
awk 'NR==FNR{a[$1]=1}NR>FNR{if($1 in a) print($0)}' train wav.scp > train_wav.scp

mv wav.scp data/cv-corpus-12.0-delta-2022-12-07/de

python myscripts/splitjob.py 64 64 data/cv-corpus-12.0-delta-2022-12-07/de 

. ./cmd.sh

bash myscripts/slurm_job.sh data/cv-corpus-12.0-delta-2022-12-07/de

1;$s/.mp3/.wav/g


## text
python myscripts/read_tsv.py data/cv-corpus-12.0-delta-2022-12-07/de/other.tsv > tmp

cat tmp tmp1 tmp2 > data/cv-corpus-12.0-delta-2022-12-07/de/text

mkdir data/cv-corpus-12.0-delta-2022-12-07/de/{tr,cv,test}
mv tmp data/cv-corpus-12.0-delta-2022-12-07/de/tr/text

awk 'NR==FNR{a[$1]=$2;}NR>FNR{if($1 in a) print($1" "a[$1])}' data/cv-corpus-12.0-delta-2022-12-07/de/wav.scp data/cv-corpus-12.0-delta-2022-12-07/de/tr/text > data/cv-corpus-12.0-delta-2022-12-07/de/tr/wav.scp


#python myscripts/tokenizer.py data/cv-corpus-12.0-delta-2022-12-07/de/tr/text dec_input label de
python myscripts/tokenizer.py data/cv_5.1/cv-corpus-5.1-2020-06-22/pt/tr/text data/cv_5.1/cv-corpus-5.1-2020-06-22/pt/tr/dec_input_ids.ark data/cv_5.1/cv-corpus-5.1-2020-06-22/pt/tr/labels.ark pt

## feats
python myscripts/splitjob.py 64 64 data/cv-corpus-12.0-delta-2022-12-07/de/tr

. ./cmd.sh
bash myscripts/slurm_job.sh data/cv-corpus-12.0-delta-2022-12-07/de/tr

bash myscripts/merge.sh data/cv-corpus-12.0-delta-2022-12-07/de/tr feats.scp

## test set
awk 'NR==FNR{a[$1]=$2}NR>FNR{name=$1;$1="";if(name in a) print(name"|"a[name]"|"$0);}' data/cv-corpus-12.0-delta-2022-12-07/de/test/wav.scp data/cv-corpus-12.0-delta-2022-12-07/de/test/text > csv


