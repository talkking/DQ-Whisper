#!/bin/bash
# Apache 2.0


# begin configuration section.
cmd=run.pl
stage=0

[ -f ./path.sh ] && . ./path.sh

. parse_options.sh || exit 1;

if [ $# -ne 2 ]; then
  echo "Usage: local/score.sh [--cmd (run.pl|queue.pl...)] <data-dir> <lang-dir|graph-dir> <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  exit 1;
fi

data=$1
dir=$2

for f in $dir/1.char $data/text; do
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

unset LC_ALL
#for character error rate
awk '{printf("%s", $1); for(j=2;j<=NF;j++){num=split($j,ss,"");for(i=1;i<=num;i++){c=ss[i];if(c~/[\000-\177]/){s=s""c; if(i==num){printf(" %s",s);s="";}} else if(c!~/[\000-\177]/){if(s!=""){printf(" %s",s);s="";}printf(" %s",c);}} } printf("\n");}' $dir/scoring/text.filt > $dir/scoring/char.filt

export LC_ALL=C

cat $dir/*.char | awk '{printf $1; for (i=2;i<=NF;++i) if ($i~"[A-Z]"){ if ( $i~"▁" ) printf" "; printf $i; } else printf" "$i; printf"\n"}' | sed 's=▁==g'> $dir/scoring/hyp.char.txt
compute-wer --text --mode=present \
   ark:$dir/scoring/char.filt ark:$dir/scoring/hyp.char.txt > $dir/cer || exit 1;

  if $stats; then
    mkdir -p $dir/scoring/wer_details

    $cmd $dir/scoring/log/stats1.log \
      cat $dir/scoring/hyp.char.txt \| \
      align-text --special-symbol="'***'" ark:$dir/scoring/char.filt ark:- ark,t:- \|  \
      utils/scoring/wer_per_utt_details.pl --special-symbol "'***'" \| tee $dir/scoring/wer_details/per_utt \|\
       utils/scoring/wer_per_spk_details.pl $data/utt2spk \> $dir/scoring/wer_details/per_spk || exit 1;

    $cmd $dir/scoring/log/stats2.log \
      cat $dir/scoring/wer_details/per_utt \| \
      utils/scoring/wer_ops_details.pl --special-symbol "'***'" \| \
      sort -b -i -k 1,1 -k 4,4rn -k 2,2 -k 3,3 \> $dir/scoring/wer_details/ops || exit 1;
  fi  

exit 0;
