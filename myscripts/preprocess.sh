##cu01 no env
awk '{len=split($2,ss,"");s=ss[1]; for(i=2;i<=len;i++) s = s " " ss[i]; print($1 " " s)}' text > tmp
awk 'NR==FNR{a[$1]=$2}NR>FNR{for(i=2;i<=NF;i++) if($i in a) $i=a[$i]; else $i=""; print($0)}' data/Japan/decode_lang/lang_decode_e2e_4303/units.txt data/csj_large/tr/text > textid
awk '{for(i=1;i<=NF;i++) if(i==1) printf("%s 0 ", $i); else if(i==NF) printf("%s 0\n", $i); else printf("%s ", $i);}' textid > tmp
copy-int-vector ark,t:textid ark,scp:/mnt/lustre02/jiangsu/aispeech/home/hs418/low_resource/data/csj_large/tr/ali_4303.e2e_ctc.ark,/mnt/lustre02/jiangsu/aispeech/home/hs418/low_resource/data/csj_large/tr/ali_4303.e2e_ctc.scp
