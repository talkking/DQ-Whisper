rootdir=data/cv_5.1/cv-corpus-5.1-2020-06-22
similar_lang=fr
target_lang=pt
mkdir $rootdir/multilingual_$target_lang
#only tr
for i in tr 
do
  mkdir $rootdir/multilingual_$target_lang/$i
  for j in feats.scp labels.ark dec_input_ids.ark
  do
    echo $rootdir/multilingual/$i/$j
    for lang in $similar_lang
    do
      cat $rootdir/$target_lang/$i/$j $rootdir/$lang/$i/$j > $rootdir/multilingual_$target_lang/$i/$j
    done
  done
done

for i in cv
do
  mkdir $rootdir/multilingual_$target_lang/$i
  for j in feats.scp labels.ark dec_input_ids.ark
  do  
    echo $rootdir/multilingual/$i/$j
    cp $rootdir/$target_lang/$i/$j $rootdir/multilingual_$target_lang/$i
  done
done

