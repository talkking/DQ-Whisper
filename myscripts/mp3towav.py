from pydub import AudioSegment
import sys
import os

wavscp = sys.argv[1]

i = 0
with open(wavscp, "r") as f:
    for line in f:
        if i % 100 == 0:
            print(i)
        i += 1
        name = line.split()[0]
        path = line.split()[1]
        pathdir = "/".join(path.split("/")[:-1])
        sound = AudioSegment.from_mp3(path)
        sound = sound.set_frame_rate(16000)
        sound.export(os.path.join(pathdir, name + ".wav"), format="wav", bitrate="16k")
