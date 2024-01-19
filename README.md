train stage:
bash scripts/train_ce_ctc.sh --nj 1 --expdir exp/jp_CSJ_KD_logits_v1_alpha0 --conf conf/jap_ts.yaml checkpoint_dir=exp/jp_CSJ_KD_logits_v1_alpha0 data.data_dir=data/csj_whisper optim.lr=3e-5 data.collector.minibatch_size=20 loss.alpha=0

inference stage:
python inference.py base exp/trans20L-lstm2L_jp_CSJ_KD_logits_v1_alpha0.3


tr cv
数据准备
1.先准备好wav.scp,text,utt2spk,spk2utt
先用wav.scp和text生成audio-text paired csv
awk 'NR==FNR{a[$1]=$2}NR>FNR{name=$1;$1="";if(name in a) print(name","a[name]","$0)}' data/cv-corpus-12.0-delta-2022-12-07/it/wav.scp data/cv-corpus-12.0-delta-2022-12-07/it/text > csv
得到csv之后用`python myscripts/tokenizer.py csv dec_input_ids.ark labels.ark it`得到dec_input_ids.ark和labels.ark

然后进行split
把wav.scp dec_inputs_id.ark和labels.ark进行split
python myscripts/splitjob.py 64 n data/cv-corpus-12.0-delta-2022-12-07/it/cv

提取特征
分布式提取特征
myscripts/make_mfcc.py提取特征
用myscripts/slurm_job.py提交任务
bash myscripts/slurm_job.sh cv-corpus-12.0-delta-2022-12-07/it/tr
合并特征
bash myscripts/merge.sh cv-corpus-12.0-delta-2022-12-07/it/tr feats.scp

test数据集准备只用csv文件即可
awk 'NR==FNR{a[$1]=$2}NR>FNR{name=$1;$1="";if(name in a) print(name","a[name]","$0)}' data/cv-corpus-12.0-delta-2022-12-07/it/test/wav.scp data/cv-corpus-12.0-delta-2022-12-07/it/test/text > csv


两个测试集：
data/csj.csv
data/test.csv

