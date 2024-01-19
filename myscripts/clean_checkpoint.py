#!/usr/bin/env python3

import os
import sys  
import shutil


def get_all_files(directory):
    file_list = []
    for root, directories, files in os.walk(directory):
        #print(directories)
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_list.append(file_path)
    return file_list

# 指定目录路径
directory_path = sys.argv[1]

# 获取目录下的所有文件
files = get_all_files(directory_path)

# 打印所有文件路径
maxidx = 0
for file in files:
    #print(file)   
    # if file.split('/')[-1].startswith("checkpoint.dump"):
    #     os.remove(file)
    if file.split('/')[-1].startswith("checkpoint"):
        #print(file)
        checkpoint = file.split('/')[-1]
        idx = checkpoint.split('-')
        if len(idx) > 1:
            idx = int(idx[-1])
            maxidx = max(maxidx, idx)
for file in files:
    if file.split('/')[-1].startswith("checkpoint"):
        #print(file)
        checkpoint = file.split('/')[-1]
        idx = checkpoint.split('-')
        if len(idx) > 1:
            idx = int(idx[-1])
            if idx < maxidx:
            #shutil.rmtree(file)
                os.remove(file)



for file in files:
    if file.split('/')[-1].startswith("checkpoint"):
        print(file)


