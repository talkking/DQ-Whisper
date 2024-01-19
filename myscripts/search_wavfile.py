
from pathlib import Path
from tqdm import tqdm

DATASET_DIR="/mnt/lustre02/jiangsu/aispeech/home/hs418/low_resource/data/personal/jvs_ver1"

dataset_dir = Path(DATASET_DIR)
transcripts_path_list = list(dataset_dir.glob("*/*/transcripts_utf8.txt"))
print(len(transcripts_path_list))

def get_audio_file_list(transcripts_path_list, text_max_length=120, audio_max_sample_length=480000, sample_rate=16000):
    audio_transcript_pair_list = open("text", "w")
    for transcripts_path in tqdm(transcripts_path_list):
        audio_dir = transcripts_path.parent / "wav24kHz16bit"
        subset = str(transcripts_path.parent.parent).split('/')[-1]
        subsubset = str(transcripts_path.parent).split('/')[-1]
        # import pdb
        # pdb.set_trace()

        if not audio_dir.exists():
            print(f"{audio_dir}は存在しません。")
            continue
        # 从翻译的文本中获取AudioId和文本。
        with open(transcripts_path, "r") as f:
            text_list = f.readlines()
        for text in text_list:
            audio_id, text = text.replace("\n", "").split(":")
            #print(audio_id, text)

            audio_path = audio_dir / f"{audio_id}.wav"
            if audio_path.exists():
                # 数据检查
                # audio = load_wave(audio_path, sample_rate=sample_rate)[0]
                # if len(text) > text_max_length or len(audio) > audio_max_sample_length:
                #     print(len(text), len(audio))
                #     continue
                audio_transcript_pair_list.write(("_".join([subset, subsubset, audio_id]) + " " + text + "\n"))
    #return audio_transcript_pair_list

get_audio_file_list(transcripts_path_list)

