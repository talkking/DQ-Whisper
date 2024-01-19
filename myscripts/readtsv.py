#encoding=utf-8
import csv
import sys

tsv_file_path = sys.argv[1]

mp = {}
with open(tsv_file_path, 'r', ) as tsv_file:
    reader = csv.reader(tsv_file, delimiter='\t')
    header = next(reader)
    for row in reader:
        # 处理每一行数据
        #print(row)
	#break
        name = row[0]
        mp[name] = 1
        #text = row[2]
        #print(name + " " + text)
   
print(len(mp))

