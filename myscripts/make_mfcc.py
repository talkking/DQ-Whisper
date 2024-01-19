import whisper
import torch
import torchaudio
import sys
from kaldiio import WriteHelper
import logging
logger = logging.getLogger(__name__)


def load_wave(wave_path, sample_rate:int=16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform

nj = sys.argv[1]
i = sys.argv[2]
#setname=sys.argv[3]
dir = sys.argv[3]
padding_length=15


j=0
path= f"{dir}/" + i + "/wav.scp" 
ark = f"{dir}/" + i + '/feats.ark'
scp = f"{dir}/" + i + '/feats.scp'
with WriteHelper(f'ark,scp:{ark},{scp}') as writer:
    with open(path, "r") as f:
        for line in f:
            wav = line.split()[1]
            name = line.split()[0]
            wav = load_wave(wav, sample_rate=16000)
            wav = whisper.pad_or_trim(wav.flatten(), 16000*padding_length)
            mel = whisper.log_mel_spectrogram(wav)
            #print(mel.numpy().shape)
            writer(name, mel.numpy())
            j += 1
            if j % 100 == 0:
                logging.info(f"{j}")
            #break


