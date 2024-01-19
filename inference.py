#!/usr/bin/env python3
#encoding=utf-8
import whisper
import numpy as np
import torch
from torch import nn
import pandas as pd
import torchaudio
import torchaudio.transforms as at
from pathlib import Path
import os

from pytorch_lightning import LightningModule
from pytorch_lightning import seed_everything

from tqdm import tqdm

import evaluate
import sys

SAMPLE_RATE = 16000
BATCH_SIZE = 20 #2
TRAIN_RATE = 0.8

AUDIO_MAX_LENGTH = 480000
TEXT_MAX_LENGTH = 120
SEED = 3407
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
seed_everything(SEED, workers=True)
padding_length = 15
import logging

scale = sys.argv[1]
checkpoint_dir = sys.argv[2] # "exp/trans20L-lstm2L_jp_CSJ_KD_logits_v1_alpha0.9" #解码模型存放目录
testset = sys.argv[3] # test csv文件存放位置
lang = sys.argv[4] # 语言种类
nj = sys.argv[5]
job = sys.argv[6] # 任务编号

def get_audio_file_list_from_csv(csv, sample_rate=16000):
    audio_transcript_pair_list = []
    dd = pd.read_csv(csv, sep="|", header=None)
    for i in range(len(dd)):
        name = dd[0][i]
        audio = dd[1][i]
        text = dd[2][i]
        audio_transcript_pair_list.append((name, audio, text))

    return audio_transcript_pair_list

def load_wave(wave_path, sample_rate:int=16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform



class JvsSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, audio_info_list, tokenizer, sample_rate) -> None:
        super().__init__()

        self.audio_info_list = audio_info_list
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.audio_info_list)
    
    def __getitem__(self, id):
        audio_id, audio_path, text = self.audio_info_list[id]

        # audio
        audio = load_wave(audio_path, sample_rate=self.sample_rate)
        audio = whisper.pad_or_trim(audio.flatten(), 16000*padding_length)
        mel = whisper.log_mel_spectrogram(audio)
        
        #text = text_kana_convert(text)
        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
        labels = text[1:] + [self.tokenizer.eot]

        #print(f"inputs_ids:{mel}, labels={labels}, dec_input_ids={text}")
        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": text
        }


class WhisperDataCollatorWhithPadding:
    def __call__(self, features):
        input_ids, labels, dec_input_ids = [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])

        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])
        
        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths+dec_input_ids_length)

        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_ids, dec_input_ids_length)] # 50257 is eot token id

        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch["input_ids"] = input_ids

        return batch


eval_audio_transcript_pair_list = get_audio_file_list_from_csv(testset)

print("EVAL AUDIO DATASET NUM: ", len(eval_audio_transcript_pair_list))
print("data load finished!!")

class Config:
    learning_rate = 5e-6 #0.0005
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_steps = 2
    batch_size = 4 #4 #8 #8 #16
    num_worker = 2 #2
    num_train_epochs = 10
    gradient_accumulation_steps = 4 #4 #2 #4 #2 #1
    sample_rate = SAMPLE_RATE
    mode = "finetune"

    def __init__(self, mode="finetune") -> None:
        self.mode = mode


class WhisperModelModule(LightningModule):
    def __init__(self, cfg:Config, model_name="base", lang="ja") -> None:
        super().__init__()
        self.options = whisper.DecodingOptions(language=lang, without_timestamps=True)
        self.model = whisper.load_model(model_name) # model 添加了weight_alpha怎么办？
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language=lang, task=self.options.task)

        #only decoder training
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")
        self.cfg = cfg
    
    def forward(self, batch):
        return self.model(batch)


def normalize(text):
    #「」『』！
    text = text.replace("、","")
    text = text.replace("。","")
    text = text.replace("？","")
    text = text.replace("?","")
    text = text.replace("—","")
    text = text.replace("「","")
    text = text.replace("」","")
    text = text.replace("！","")
    text = text.replace("!","")
    text = text.replace(",","")
    text = text.replace("，","")
    return text

def inference(checkpoint_path): 

    global checkpoint_dir
    cfg = Config()
    model_name = os.path.join("Pretrained_model", scale + ".pt")
    whisper_model = WhisperModelModule(cfg, model_name=model_name, lang=lang)
    #import pdb
    #pdb.set_trace()
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path)
        state_dict = state_dict['model']
        for k,v in list(state_dict.items()):
            state_dict.pop(k)
            #if k == "encoder.positional_embedding":
                     
            k = "model." + k
            state_dict[k] = v
        whisper_model.load_state_dict(state_dict, strict=False)



    woptions = whisper.DecodingOptions(language=lang, beam_size=8, without_timestamps=True) #

    wtokenizer = whisper.tokenizer.get_tokenizer(True, language=lang, task=woptions.task)
    dataset = JvsSpeechDataset(eval_audio_transcript_pair_list, wtokenizer, SAMPLE_RATE)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=WhisperDataCollatorWhithPadding())

    refs = []
    res = []

    n_res = []
    n_refs = []
    
    # import pdb
    # pdb.set_trace()
    # if checkpoint_dir == "whisper-" + scale:
    #     checkpoint_dir = "exp/" + "whisper-" + scale
    #     if not os.path.exists(checkpoint_dir):
    #         os.mkdir(checkpoint_dir)

    hyp_path = os.path.join(checkpoint_dir, "decode")


    # if not os.path.exists(hyp_path):
    #     os.mkdir(hyp_path)
    # import pdb
    # pdb.set_trace()
    #if job == 1:
    # if not os.path.exists(os.path.join(hyp_path, lang)):
    #     os.mkdir(os.path.join(hyp_path, lang))
        # else:
        #     import shutil
        #     shutil.rmtree(os.path.join(hyp_path, lang))
        #     os.mkdir(os.path.join(hyp_path, lang))

    hyp_path = os.path.join(hyp_path, lang)

    # if not os.path.exists(os.path.join(hyp_path, "split"+nj)):
    #     os.mkdir(os.path.join(hyp_path, "split"+nj))

    


    hyp_path = os.path.join(hyp_path, "split"+nj)
    if not os.path.exists(os.path.join(hyp_path, job)):
        os.mkdir(os.path.join(hyp_path, job))
    hyp_path = os.path.join(hyp_path, job)
    if not os.path.exists(os.path.join(hyp_path, "hyp.txt")):
        os.mknod(os.path.join(hyp_path, "hyp.txt"))
    else:
        os.remove(os.path.join(hyp_path, "hyp.txt"))
        os.mknod(os.path.join(hyp_path, "hyp.txt"))
    if not os.path.exists(os.path.join(hyp_path, "ref.txt")):
        os.mknod(os.path.join(hyp_path, "ref.txt"))
    else:
        os.remove(os.path.join(hyp_path, "ref.txt"))
        os.mknod(os.path.join(hyp_path, "ref.txt")) 
        #os.mknod(os.path.join(hyp_path, "results.txt"))

    hyp = open(os.path.join(hyp_path, "hyp.txt"), "w")
    refer = open(os.path.join(hyp_path, "ref.txt"), "w")
    #result = open(os.path.join(hyp_path, "results.txt"), "w")

    # from fvcore.nn import FlopCountAnalysis, flop_count_table
    import time
    total_time = 0
    for batch in tqdm(loader):
        input_ids = batch["input_ids"].half().cuda()
        labels = batch["labels"].long().cuda()
        batch = {k:v.cuda() for k,v in batch.items()}
        with torch.no_grad():
            # import pdb; pdb.set_trace()
            # flops = FlopCountAnalysis(whisper_model.model, batch)
            # print(flop_count_table(flops))
            t1 = time.time()
            results = whisper_model.model.decode(input_ids, woptions)
            t2 = time.time()
            total_time += t2 - t1
            for r in results:
                res.append(normalize(r.text))
                hyp.write(normalize(r.text)+"\n")
                n_res.append(r.text)
            
            for l in labels:
                l[l == -100] = wtokenizer.eot
                ref = wtokenizer.decode(l, skip_special_tokens=True)
                refs.append(normalize(ref))
                refer.write(normalize(ref)+"\n")
                n_refs.append(ref)
    
    hyp.close()
    refer.close()
    

    #scoring
    # cer_metrics = evaluate.load("cer")
    # cer = cer_metrics.compute(references=refs, predictions=res)
    # n_cer = cer_metrics.compute(references=n_refs, predictions=n_res)

    # wer_metrics = evaluate.load("wer")
    # wer = wer_metrics.compute(references=refs, predictions=res)
    # n_wer = wer_metrics.compute(references=n_refs, predictions=n_res)
    
    # print(f"cer={cer}, wer={wer}")
    # print(f"w/o normalize: cer={n_cer}, wer={n_wer}")
    # print(f"total_time=", total_time / len(loader))
    # result.write(f"cer={cer}, wer={wer}")
    # result.close()
    
def main():
    checkpoint = os.path.join(checkpoint_dir, "checkpoint")
    if checkpoint_dir == "exp/"+ "whisper-" + scale:
        inference(None)
    else:
        inference(checkpoint)

if __name__ == "__main__":
    main()




