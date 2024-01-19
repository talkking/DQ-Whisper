from whisper import *
import sys

text = sys.argv[0]

des = sys.argv[1]

with open(text, "r") as f1, open(des, "a") as f2:
   for line in f1:
     name = line.split()[0]
     txt = line.split()[1]
     txt = remove_symbols_and_diacritics(txt)
     f2.write(name + " " + txt + "\n")

