#!/bin/bash
# Apache 2.0


# begin configuration section.
cmd=run.pl
stage=0
min_acwt=5
max_acwt=15
acwt_factor=0.1   # the scaling factor for the acoustic scale. The scaling factor for acoustic likelihoods
                 # needs to be 0.5 ~1.0. However, the job submission script can only take integers as the
                 # job marker. That's why we set the acwt to be integers (5 ~ 10), but scale them with 0.1'
                 # when they are actually used.
#end configuration section.

[ -f ./path.sh ] && . ./path.sh

. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: local/score.sh [--cmd (run.pl|queue.pl...)] <data-dir> <lang-dir|graph-dir> <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --min_acwt <int>                # minumum LM-weight for lattice rescoring "
  echo "    --max_acwt <int>                # maximum LM-weight for lattice rescoring "
  exit 1;
fi

data=$1
lang_or_graph=$2
dir=$3

symtab=$lang_or_graph/words.txt

for f in $symtab $dir/lat.1.gz $data/text; do
  [ ! -f $f ] && echo "score.sh: no such file $f" && exit 1;
done

mkdir -p $dir/scoring/log

function filter_text {
  perl -e 'foreach $w (@ARGV) { $bad{$w} = 1; } 
   while(<STDIN>) { @A  = split(" ", $_); $id = shift @A; print "$id ";
     foreach $a (@A) { if (!defined $bad{$a}) { print "$a "; }} print "\n"; }' \
   '[NOISE]' '[LAUGHTER]' '[VOCALIZED-NOISE]' '<UNK>' '%HESITATION'
}
filter_text <$data/text >$dir/scoring/text.filt

#--ascale-factor=$acwt_factor
$cmd ACWT=$min_acwt:$max_acwt $dir/scoring/log/best_path.ACWT.log \
	local/best_path.sh ACWT $dir $symtab
  #lattice-scale --acoustic-scale=`awk -v aa=ACWT 'BEGIN{print aa*0.1}'` "ark:gunzip -c $dir/lat.*.gz|" ark:- \| \
  #lattice-best-path --word-symbol-table=$symtab ark:- ark,t:$dir/scoring/ACWT.tra || exit 1;

for acwt in `seq $min_acwt $max_acwt`; do
  cat $dir/scoring/${acwt}.tra | utils/int2sym.pl -f 2- $symtab | \
    filter_text > $dir/scoring/$acwt.txt || exit 1;
done

unset LC_ALL
#for character error rate
awk '{printf("%s", $1); for(j=2;j<=NF;j++){num=split($j,ss,"");for(i=1;i<=num;i++){c=ss[i];if(c~/[\000-\177]/){s=s""c; if(i==num){printf(" %s",s);s="";}} else if(c!~/[\000-\177]/){if(s!=""){printf(" %s",s);s="";}printf(" %s",c);}} } printf("\n");}' $dir/scoring/text.filt > $dir/scoring/char.filt

for acwt in `seq $min_acwt $max_acwt`; do
    awk '{printf("%s", $1); for(j=2;j<=NF;j++){num=split($j,ss,"");for(i=1;i<=num;i++){c=ss[i];if(c~/[\000-\177]/){s=s""c; if(i==num){printf(" %s",s);s="";}} else if(c!~/[\000-\177]/){if(s!=""){printf(" %s",s);s="";}printf(" %s",c);}} } printf("\n");}' $dir/scoring/$acwt.txt > $dir/scoring/${acwt}.char
done

export LC_ALL=C

#$cmd ACWT=$min_acwt:$max_acwt $dir/scoring/log/score.ACWT.log \
#  compute-wer --text --mode=present \
#   ark:$dir/scoring/text.filt ark:$dir/scoring/ACWT.txt ">&" $dir/wer_ACWT || exit 1;

$cmd ACWT=$min_acwt:$max_acwt $dir/scoring/log/score.ACWT.cer.log \
  compute-wer --text --mode=present \
   ark:$dir/scoring/char.filt ark:$dir/scoring/ACWT.char ">&" $dir/cer_ACWT || exit 1;



if [ $stage -le 1 ]; then

  grep WER $dir/cer_* | utils/best_wer.sh >& $dir/scoring/best_cer || exit 1

  best_wer_file=$(awk '{print $NF}' $dir/scoring/best_cer)
  best_lmwt=$(echo $best_wer_file | awk -F_ '{print $NF}')

  if [ -z "$best_lmwt" ]; then
    echo "$0: we could not get the details of the best WER from the file $dir/wer_*.  Probably something went wrong."
    exit 1;
  fi  

  if $stats; then
    mkdir -p $dir/scoring/wer_details
    echo $best_lmwt > $dir/scoring/wer_details/lmwt # record best language model weight

    $cmd $dir/scoring/log/stats1.log \
      cat $dir/scoring/$best_lmwt.char \| \
      align-text --special-symbol="'***'" ark:$dir/scoring/char.filt ark:- ark,t:- \|  \
      utils/scoring/wer_per_utt_details.pl --special-symbol "'***'" \| tee $dir/scoring/wer_details/per_utt \|\
       utils/scoring/wer_per_spk_details.pl $data/utt2spk \> $dir/scoring/wer_details/per_spk || exit 1;

    $cmd $dir/scoring/log/stats2.log \
      cat $dir/scoring/wer_details/per_utt \| \
      utils/scoring/wer_ops_details.pl --special-symbol "'***'" \| \
      sort -b -i -k 1,1 -k 4,4rn -k 2,2 -k 3,3 \> $dir/scoring/wer_details/ops || exit 1;
  fi  
fi

exit 0;
