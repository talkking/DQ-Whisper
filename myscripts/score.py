import sys
import evaluate
import opencc
import os
decode_dir = sys.argv[1]
ref_path= os.path.join(decode_dir, "ref.txt") 
hyp_path= os.path.join(decode_dir, "hyp.txt")
#hyp_path_convert=hyp_path+".cvr"

#converter = opencc.OpenCC('t2s')

refs = []
res = []
#if not os.path.exists(hyp_path_convert): 
#   os.mknod(hyp_path_convert)
#else:
#   os.remove(hyp_path_convert)
#   os.mknod(hyp_path_convert)
with open(ref_path, "r", encoding="utf-8") as f1, open(hyp_path, "r", encoding="utf-8") as f2:
    for line in f1:
        line = line.split()
        refs.append(" ".join(line))
    for line in f2:
        line = line.split()
        # import pdb
        # pdb.set_trace()
        #simple_text = converter.convert("".join(line))
        simple_text = " ".join(line)
        #f3.write(simple_text + "\n")
        res.append(simple_text)
        #print(line, " " ,simple_text)
cer_metrics = evaluate.load("cer")
cer = cer_metrics.compute(references=refs, predictions=res)

wer_metrics = evaluate.load("wer")
wer = wer_metrics.compute(references=refs, predictions=res)

result_file = os.path.join(decode_dir, "result.txt")
if not os.path.exists(result_file):
   os.mknod(result_file)
else:
   os.remove(result_file)
   os.mknod(result_file)
rf = open(result_file, "a")
rf.write(f"cer={cer}, wer={wer}" + "\n\n")
for i in range(len(refs)):
    #print(refs[i], "|", res[i])
    rf.write(refs[i] + "|" + res[i] + "\n")
#print(f"cer={cer}, wer={wer}")
