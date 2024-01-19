import sys

nj=sys.argv[1]
n=sys.argv[2]
root=sys.argv[3]
split_file= sys.argv[4]

nj=int(nj)
n=int(n)
import os
def splitfile(filename):
    filepath = os.path.join(root, filename) 
    with open(filepath, "r") as f:
        i = 0
        ff = []
        for j in range(1, nj+1): #从1到nj编号
            
            path = os.path.join(root, "split" + str(nj), str(j))
            if not os.path.exists(path):
                os.mkdir(path)
            
            if not os.path.exists(os.path.join(path, filename)):
                os.mknod(os.path.join(path, filename))
            ff.append(open(os.path.join(path, filename), "a"))
        for line in f:
            ff[i % nj].write(line)
            i += 1

if __name__ =="__main__":
    path = os.path.join(root, "split" + str(nj))
    if not os.path.exists(path):
       os.mkdir(path)
    else:
       import shutil
       shutil.rmtree(path)
       os.mkdir(path)
    splitfile(split_file)
    
