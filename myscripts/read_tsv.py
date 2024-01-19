import pandas as pd

import sys

path=sys.argv[1]

with open(path, "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        line = line.strip("\n").split("\t")
        name = line[1].split(".")[0]
        text = line[2]
        print(name + " " + text)
