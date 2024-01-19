import whisper
import sys
import os

scale = sys.argv[1]
model = whisper.load_model(os.path.join('Pretrained_model', scale + ".pt"))

#audio = whisper.load_audio('/mnt/lustre02/jiangsu/aispeech/home/xqy30/work21/model-pack-decode/aijp/xiaomi/small_wavs/cut_audio_dir/非自然死亡session2_part2/非自然死亡session2_part2.2535600000-2540000000.wav')


audio_path = "/mnt/lustre02/jiangsu/aispeech/home/hs418/low_resource/data/test_jp/aijp_test_csj_common/wav.scp"
#audio_path = "/mnt/lustre02/jiangsu/aispeech/home/hs418/low_resource/data/test_jp/aijp_test_xiaomi_common/wav.scp"
#audio = whisper.load_audio('/mnt/lustre02/jiangsu/aispeech/home/xqy30/work21/model-pack-decode/aijp/xiaomi/small_wavs/cut_audio_dir/非自然死亡session2_part2/非自然死亡session2_part2.4794000000-4802800000.wav')


res_path = sys.argv[2]
if os.path.exists(res_path):
  os.remove(res_path)

i = 0
with open(audio_path, "r") as f, open(res_path, "a") as f1: 
  for line in f:
    name = line.split()[0]
    audio = whisper.load_audio(line.split()[1])
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(language="japanese", beam_size=8, length_penalty=0)
    result = whisper.decode(model, mel, options)
    #result = model.transcribe(audio)
    i += 1
    if i % 50 == 0:
      print(i)
    f1.write(name + " " + result.text + "\n")

    


#print(result)
