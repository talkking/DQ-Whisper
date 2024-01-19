import pandas as pd
import csv
import sys
import whisper


src=sys.argv[1] #text file
dec_input_ids=sys.argv[2] #dec_input_ids path
labels=sys.argv[3] #dec_input_ids labels
lid=sys.argv[4]

woptions = whisper.DecodingOptions(language=lid, without_timestamps=True)
wtokenizer = whisper.tokenizer.get_tokenizer(True, language=lid, task=woptions.task)
i=0

with open(src, "r") as f1, open(dec_input_ids, "w") as f2, open(labels, "w") as f3:
    #reader = csv.reader(f1)
    #df = pd.read_csv(f1)
    #rows = [row for row in reader]
    #for i in range(0, len(rows)):
    for line in f1: 
        #line = rows[i]
        line = line.split()
        name = line[0]
        text = " ".join(line[1:])
        text = [*wtokenizer.sot_sequence_including_notimestamps] + wtokenizer.encode(text)
        labels = text[1:] + [wtokenizer.eot]
        strlist = [str(s) for s in text]
        f2.write(name + " " + " ".join(strlist) + "\n")
        strlist = [str(s) for s in labels]
        f3.write(name + " " + " ".join(strlist) + "\n")
        #if i % 100 == 0:
        #   print(i)
        i += 1 
    # for line in reader:
    #     print(line) 
    

