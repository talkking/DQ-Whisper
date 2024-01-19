path=$1
des=$2
for i in `seq 0 6840`; do
 if [ $i -lt 10 ]; then 
  id=000$i
 elif [ $i -lt 100 ]; then
  id=00$i
 elif [ $i -lt 1000 ]; then
  id=0$i
 else
  id=$i
 fi
 name=meian_${id}.wav
 sox $path/$name -r 16000 -b 16 $des/$name
done
