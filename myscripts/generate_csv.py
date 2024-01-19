#awk 'NR==FNR{a[$1]=$2}NR>FNR{name=$1; $1=""; if(name in a) print(name","a[name]","$0)}' data/cv-corpus-12.0-delta-2022-12-07/it/tr/wav.scp data/cv-corpus-12.0-delta-2022-12-07/it/text > csv


import csv
import torchaudio
import sys


src=sys.argv[1]
des=sys.argv[2]

with open(des,'w') as file, open(src, "r") as f:
  writer = csv.writer(file) 
  writer.writerow(["name","audio","text"])
  for line in f:
    line = line.split()
    name = line[0]
    text = line[2]
    audio = line[1]
    #label = line[3]
    #audio = torchaudio.load(audio)
    writer.writerow([name, audio, text])


