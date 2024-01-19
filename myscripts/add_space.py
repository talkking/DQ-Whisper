#encoding=utf-8
import os
import sys
path=sys.argv[1]
des = sys.argv[2]
def is_en(w):
    if 'a'<=w<='z' or 'A'<=w<='Z':
        return True
def is_mark(w):
   if w == "..." or w == "?" or w == "。" or w == "!" or w == "、":
     return True

if os.path.exists(des):
    os.remove(des)
    os.mknod(des)
with open(path, "r") as f, open (des, "a") as f1:
    for line in f:
        str = []
        #line = line.splitlines()[0]
        name = line.split()[0]
        str.append(name)
        line = line.split()[1]
        i = 0
        while i < len(line):
           if not is_en(line[i]) and not is_mark(line[i]): 
             str.append(line[i])
           i += 1
        #print(str)
        str = " ".join(str)
        str += '\n'
        f1.write(str)

