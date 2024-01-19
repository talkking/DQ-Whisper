#!/usr/bin/env python3

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
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from tqdm import tqdm

import pyopenjtalk
import evaluate

import sys

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

from extend_codes.transformer_kd import Transformer 

from asr.data.field import Field

#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
DATASET_DIR = "data/personal/jvs_ver1"


SAMPLE_RATE = 16000
BATCH_SIZE = 2
TRAIN_RATE = 0.8

AUDIO_MAX_LENGTH = 480000
TEXT_MAX_LENGTH = 120
SEED = 3407
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
seed_everything(SEED, workers=True)


# model = Transformer(ninp=40, nproj=512, nhid=2048, nctc=4303, natt=4303, nlayer=20, ndecode=2, nhid_dec=1024)
# print(model)
# scale = "base"
# wmodel = whisper.load_model(os.path.join("/mnt/lustre02/jiangsu/aispeech/home/hs418/low_resource/Pretrained_model", scale + ".pt"))
# print(wmodel)

# sys.exit()

def load_wave(wave_path, sample_rate:int=16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform


dataset_dir = Path(DATASET_DIR)
transcripts_path_list = list(dataset_dir.glob("*/*/transcripts_utf8.txt"))
print(len(transcripts_path_list))


import pandas as pd
import sys


def get_audio_file_list_from_csv(csv, sample_rate=16000):
    audio_transcript_pair_list = []
    dd = pd.read_csv(csv)
    for i in range(len(dd)):
        name = dd["name"][i]
        audio = dd["audio"][i]
        text = dd["text"][i]
        audio_transcript_pair_list.append((name, audio, text))

    return audio_transcript_pair_list


def get_audio_file_list(transcripts_path_list, text_max_length=120, audio_max_sample_length=480000, sample_rate=16000):
    audio_transcript_pair_list = []
    for transcripts_path in tqdm(transcripts_path_list):
        # 检查音频文件的目录。
        audio_dir = transcripts_path.parent / "wav24kHz16bit"
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
                audio = load_wave(audio_path, sample_rate=sample_rate)[0]
                if len(text) > text_max_length or len(audio) > audio_max_sample_length:
                    print(len(text), len(audio))
                    continue
                audio_transcript_pair_list.append((audio_id, str(audio_path), text))
    return audio_transcript_pair_list

# train_num = int(len(transcripts_path_list) * TRAIN_RATE)
# train_transcripts_path_list, eval_transcripts_path_list = transcripts_path_list[:train_num], transcripts_path_list[train_num:]

train_transcripts_path_list = transcripts_path_list


#eval_audio_transcript_pair_list = get_audio_file_list(eval_transcripts_path_list, TEXT_MAX_LENGTH, AUDIO_MAX_LENGTH, SAMPLE_RATE)

import numpy as np
dict="data/Japan/decode_lang/lang_decode_e2e_4303/units.txt"
mp, mp1 = {}, {}

with open(dict, "r") as f:
    for line in f:
        name = line.split()[0]
        id = line.split()[1]
        mp[name] = int(id)
        mp1[int(id)] = name

def is_en(c):
    return 'a'<=c<='z' or 'A'<=c<='Z'

def is_mark(w):
   if w == "..." or w == "?" or w == "。" or w == "!" or w == "、":
     return True

def encode_text(text):
    i = 0
    str = [0]
    while i < len(text):
        if not is_en(text[i]) and not is_mark(text[i]) and text[i] in mp:
            str.append(mp[text[i]])
        i += 1
    str.append(0)
    return np.array(str, dtype=np.int64)

def decode_text(text):
    ans = []
    for i in range(len(text)):
        if torch.is_tensor(text):
            elem = text[i].item()
        else:
            elem = text[i]
        ans.append(mp1[elem])
    return "".join(ans)

#print(encode_text("がえー高いところ"))
#sys.exit()
##

## g2p???
def text_kana_convert(text):
    text = pyopenjtalk.g2p(text, kana=True)
    return text
print(text_kana_convert("こんにちは、私の名前は、田中一郎です。"))


print("Model load before!")
woptions = whisper.DecodingOptions(language="ja", without_timestamps=True)
scale = sys.argv[1]
wmodel = whisper.load_model(os.path.join("Pretrained_model", scale + ".pt"))
wtokenizer = whisper.tokenizer.get_tokenizer(True, language="ja", task=woptions.task)

print("Model load finished!!")

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
        
        mel = whisper.log_mel_spectrogram(audio.flatten())
        f_len = mel.shape[-1]
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio)
        
        label = encode_text(text)
        
        #text = text_kana_convert(text)
        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
        labels = text[1:] + [self.tokenizer.eot]
        

        #print(f"inputs_ids:{mel}, labels={labels}, dec_input_ids={text}")
        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": text,
            "label": label,
            "uid": audio_id,
            "nframes": f_len, 
        }


class WhisperDataCollatorWhithPadding:
    def __call__(self, features):
        input_ids, labels, dec_input_ids = [], [], []
        label, length = [], [] 
        uid = []
        f_len = []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            label.append(f["label"])
            uid.append(f["uid"])
            f_len.append(f["nframes"])
        

        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])
        #max_frame = max(f_len)
        ### 固定3000帧太长了
        # uid = torch.concat([id[None, :] for id in uid])
        f_len = torch.as_tensor(f_len)

        label_lengths = [len(lab) for lab in labels]

        length = [len(lb) for lb in label]
        max_length = max(length)

        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths+dec_input_ids_length)

        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_ids, dec_input_ids_length)] # 50257 is eot token id

        label = [np.pad(lb, (0, max_length - lb_len), 'constant', constant_values=-1) for lb, lb_len in zip(label, length)]

        label = torch.tensor(np.array(label), requires_grad=False)
        length = torch.tensor(np.array(length), requires_grad=False)
        label = Field(label, length)
        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids,
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch["input_ids"] = input_ids
        batch['label'] = label
        batch['uid'] = uid
        batch['nframes'] = f_len
 
        return batch


# dataset = JvsSpeechDataset(eval_audio_transcript_pair_list, wtokenizer, SAMPLE_RATE)
# loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=WhisperDataCollatorWhithPadding())

eval_audio_transcript_pair_list = get_audio_file_list_from_csv("data/test.csv")

print("EVAL AUDIO DATASET NUM: ", len(eval_audio_transcript_pair_list))
print("data load finished!!")

# for b in loader:
#     print(b["labels"].shape)
#     print(b["input_ids"].shape)
#     print(b["dec_input_ids"].shape)

#     for token, dec in zip(b["labels"], b["dec_input_ids"]):
#         token[token == -100] = wtokenizer.eot
#         text = wtokenizer.decode(token, skip_special_tokens=False)
#         print(text)

#         dec[dec == -100] = wtokenizer.eot
#         text = wtokenizer.decode(dec, skip_special_tokens=False)
#         print(text)


#     #break

#     with torch.no_grad():
#         audio_features = wmodel.encoder(b["input_ids"].cuda())
#         input_ids = b["input_ids"]
#         labels = b["labels"].long()
#         dec_input_ids = b["dec_input_ids"].long()

            
#         audio_features = wmodel.encoder(input_ids.cuda())
#         print(dec_input_ids)
#         print(input_ids.shape, dec_input_ids.shape, audio_features.shape)
#         print(audio_features.shape)
#     out = wmodel.decoder(dec_input_ids.cuda(), audio_features)





class Config:
    learning_rate = 5e-6 #0.0005
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_steps = 2
    batch_size = 4 #4 #8 #8 #16
    num_worker = 4 #2
    num_train_epochs = 20 #10
    gradient_accumulation_steps = 4#4 #2 #4 #2 #1
    sample_rate = SAMPLE_RATE
    lambda1 = 1
    mode = "distill"


class WhisperModelModule(LightningModule):
    def __init__(self, cfg:Config, model_name="base", lang="ja", train_dataset=[], eval_dataset=[]) -> None:
        super().__init__()
        self.options = whisper.DecodingOptions(language=lang, without_timestamps=True)
        self.teacher = whisper.load_model(model_name)
        self.student = Transformer(ninp=80, nproj=512, nhid=2048, nctc=4303, natt=4303, nlayer=20, ndecode=2, nhid_dec=1024)
        
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language="ja", task=self.options.task)

        #fixed teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        self.ce_loss1 = nn.CrossEntropyLoss(ignore_index=-100)
        self.ce_loss2 = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")
        self.mse_loss = nn.MSELoss()
        self.ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        self.lam1 = cfg.lambda1
        self.lam2 = 1
        self.alpha = 0.9
        self.beta = 0.1
        self.mode = cfg.mode

        if scale == "small":
            self.linear_adapter = nn.Linear(2048, 3072)
        elif scale == "medium":
            self.linear_adapter = nn.Linear(2048, 4096)

        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")

        self.cfg = cfg
        self.__train_dataset = train_dataset
        self.__eval_dataset = eval_dataset

        
    
    def forward(self, x):
        return self.student(x)

    def hidden_mse_loss(self, hs, ht):
        loss = self.mse_loss(hs, ht)
        return loss

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]  # [B, D, L]
        labels = batch["labels"].long() # [B, L]
        dec_input_ids = batch["dec_input_ids"].long() # [B, L]
        label = batch["label"]
        ### dec_input_ids：50258, 50266, 50359, 50363 label padding=50257
        ### labels: 50266, 50359, 50363 label 50257 padding=-100
        # import pdb
        # pdb.set_trace()


        #只finetune decoder
        with torch.no_grad():
            audio_features, ht = self.teacher.encoder(input_ids)

        out_t = self.teacher.decoder(dec_input_ids, audio_features) # out_t: [B, L, V=52786]  audio_features: [B, T=1500 ,D=768] base:512 * 2048 small:768 * 2048

        ctc_out, e2e_out, hs = self.student(batch)
        ctcout = ctc_out.tensor.transpose(0, 1).log_softmax(dim=2)
        s_ctc_loss = self.ctc_loss(ctcout, label.tensor, ctc_out.length, label.length)
        s_ce_loss = self.ce_loss2(e2e_out.tensor.reshape(-1, e2e_out.tensor.size(-1)), batch['label'].tensor.reshape(-1))

        # import pdb
        # pdb.set_trace()

        ts_loss = self.hidden_mse_loss(self.linear_adapter(hs.transpose(0,1)), ht)  # hs: [T, B, D=2048] 
        
        #t_ce_loss = self.ce_loss1(out_t.view(-1, out_t.size(-1)), labels.view(-1))

        if self.mode == "distill":
            loss = (1 - self.alpha) * s_ce_loss + self.alpha * ts_loss + self.beta * ((1 - self.alpha) * s_ctc_loss + self.alpha * ts_loss)
        else:
            loss = (1 - self.alpha) * s_ce_loss + self.alpha * s_ctc_loss

        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True, batch_size=self.cfg.batch_size)
        return loss
    
    def validation_step(self, batch, batch_id):
        # input_ids = batch["input_ids"]
        # labels = batch["labels"].long()
        # dec_input_ids = batch["dec_input_ids"].long()

        ctc_out, e2e_out, _ = self.student(batch)
        # import pdb
        # pdb.set_trace()
        loss = self.ce_loss2(e2e_out.tensor.reshape(-1, e2e_out.tensor.size(-1)), batch['label'].tensor.reshape(-1))
        label = batch['label'].tensor
        e2e_out = e2e_out.tensor
        
        label[label == -1] = 0
        e2e_out[e2e_out == -1] = 0


        o_list, l_list = [], []
        for o, l in zip(e2e_out, label):
            l = l[l!=0]
            o = torch.argmax(o, dim=1)
            o = o[o!=0]
            o_list.append(decode_text(o))
            l_list.append(decode_text(l))



        # audio_features = self.model.encoder(input_ids)
        # out = self.model.decoder(dec_input_ids, audio_features)

        # loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        # out[out == -100] = self.tokenizer.eot
        # labels[labels == -100] = self.tokenizer.eot

        # o_list, l_list = [], []
        # for o, l in zip(out, labels):
        #     o = torch.argmax(o, dim=1)
        #     o_list.append(self.tokenizer.decode(o, skip_special_tokens=True))
        #     l_list.append(self.tokenizer.decode(l, skip_special_tokens=True))


        cer = self.metrics_cer.compute(references=l_list, predictions=o_list)
        wer = self.metrics_wer.compute(references=l_list, predictions=o_list)




        self.log("val/loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("val/cer", cer, on_step=True, prog_bar=True, logger=True)
        self.log("val/wer", wer, on_step=True, prog_bar=True, logger=True)

        return {
            "cer": cer,
            "wer": wer,
            "loss": loss
        }

    def configure_optimizers(self):
        #创建优化器和调度器
        student = self.student
        teacher = self.teacher
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in student.named_parameters() 
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [p for n, p in teacher.named_parameters() 
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [p for n, p in student.named_parameters() 
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in teacher.named_parameters() 
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, 
                          lr=self.cfg.learning_rate, 
                          eps=self.cfg.adam_epsilon)
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.cfg.warmup_steps, 
            num_training_steps=self.t_total
        )
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
    
    def setup(self, stage=None):
        #初始设置（加载数据集）。
        if stage == 'fit' or stage is None:
            self.t_total = (
                (len(self.__train_dataset) // (self.cfg.batch_size))
                // self.cfg.gradient_accumulation_steps
                * float(self.cfg.num_train_epochs)
            )
    
    def train_dataloader(self):
        #创建一个训练数据加载器。
        dataset = JvsSpeechDataset(self.__train_dataset, self.tokenizer, self.cfg.sample_rate)
        return torch.utils.data.DataLoader(dataset, 
                          batch_size=self.cfg.batch_size, 
                          drop_last=True, shuffle=True, num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )

    def val_dataloader(self):
        #创建一个验证数据加载器。
        dataset = JvsSpeechDataset(self.__eval_dataset, self.tokenizer, self.cfg.sample_rate)
        return torch.utils.data.DataLoader(dataset, 
                          batch_size=self.cfg.batch_size, 
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )
    
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

    return text

def train():
    

    train_name = "whisper"
    train_id = "00001"

    model_name = os.path.join("/mnt/lustre02/jiangsu/aispeech/home/hs418/low_resource/Pretrained_model", scale + ".pt")
    lang = "ja"
    cfg = Config()

    log_output_dir = os.path.join("whisper-" + scale, cfg.mode, "logs", f"lambda{cfg.lambda1}")
    check_output_dir = os.path.join("whisper-" + scale, cfg.mode, "artifacts", f"lambda{cfg.lambda1}")

    Path(log_output_dir).mkdir(exist_ok=True)
    Path(check_output_dir).mkdir(exist_ok=True)

    tflogger = TensorBoardLogger(
        save_dir=log_output_dir,
        name=train_name,
        version=train_id
    )

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        dirpath=f"{check_output_dir}/checkpoint",
        filename="checkpoint-{epoch:04d}",
        save_top_k=-1 #-1 # all model save
    )

    callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]

    #train_audio_transcript_pair_list = get_audio_file_list(train_transcripts_path_list, TEXT_MAX_LENGTH, AUDIO_MAX_LENGTH, SAMPLE_RATE)
    train_audio_transcript_pair_list = get_audio_file_list_from_csv("data/train_medium.csv")
    print("TRAIN AUDIO DATASET NUM: ", len(train_audio_transcript_pair_list))

    model = WhisperModelModule(cfg, model_name, lang, train_audio_transcript_pair_list, eval_audio_transcript_pair_list)

    trainer = Trainer(
        precision=32,
        accelerator=DEVICE,
        #gpus=4,
        devices=1,
        num_nodes=1,
        max_epochs=cfg.num_train_epochs,
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        logger=tflogger,
        callbacks=callback_list,
        auto_select_gpus=True,
        strategy="ddp",
    )

    trainer.fit(model)


def inference(checkpoint_path): 

    state_dict = torch.load(checkpoint_path)
    #print(state_dict.keys())
    state_dict = state_dict['state_dict']

    cfg = Config()
    model_name = os.path.join("/mnt/lustre02/jiangsu/aispeech/home/hs418/low_resource/Pretrained_model", scale + ".pt")
    whisper_model = WhisperModelModule(cfg, model_name=model_name)
    whisper_model.load_state_dict(state_dict)
    # import pdb
    # pdb.set_trace()
    whisper_model = whisper_model.to(device="cuda")
    

    woptions = whisper.DecodingOptions(language="japanese", beam_size=8, without_timestamps=True)

    dataset = JvsSpeechDataset(eval_audio_transcript_pair_list, wtokenizer, SAMPLE_RATE)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=WhisperDataCollatorWhithPadding())

    refs = []
    res = []

    hyp_path = os.path.join("whisper-" + scale, "decode")
    if not os.path.exists(hyp_path):
        os.mkdir(hyp_path)
        os.mknod(os.path.join(hyp_path, "hyp.txt"))
        os.mknod(os.path.join(hyp_path, "ref.txt"))
        os.mknod(os.path.join(hyp_path, "results.txt"))

    hyp = open(os.path.join("whisper-" + scale, "decode", "hyp.txt"), "w")
    refer = open(os.path.join("whisper-" + scale, "decode", "ref.txt"), "w")
    result = open(os.path.join("whisper-" + scale, "decode", "results.txt"), "w")

    
    #teacher
    # for b in tqdm(loader):
    #     input_ids = b["input_ids"].half().cuda()
    #     labels = b["labels"].long().cuda()
    #     with torch.no_grad():
    #         #audio_features = whisper_model.model.encoder(input_ids)
    #         #out = whisper_model.model.decoder(enc_input_ids, audio_features)
    #         # results = whisper_model.model.decode(input_ids, woptions)
    #         results = whisper_model.teacher.decode(input_ids, woptions)
    #         for r in results:
    #             res.append(normalize(r.text))
    #             hyp.write(normalize(r.text)+"\n")
            
    #         for l in labels:
    #             l[l == -100] = wtokenizer.eot
    #             ref = wtokenizer.decode(l, skip_special_tokens=True)
    #             refs.append(normalize(ref))
    #             refer.write(normalize(ref)+"\n")

    #student
    beamsize = 8
    for b in tqdm(loader):
        b["input_ids"] = b["input_ids"].half().cuda()
        b["labels"] = b["labels"].cuda()
        b["dec_input_ids"] = b["dec_input_ids"].cuda()
        label = b["label"].tensor.long().cuda()
        lb_len = b['label'].length.long().cuda()
        b["label"] = Field(label, lb_len)
        key = b['uid']
        with torch.no_grad():
            # import pdb
            # pdb.set_trace()
            output = whisper_model.student.decode_e2e(b, beamsize)
            ys = output['hyps'].tensor
            xlen = output['hyps'].length


            for uid, out, length, lb, len in zip(key, ys, xlen, label, lb_len):
                out = out[0, 1:int(length)].cpu().tolist()
                #s_hyp.write(uid + (*out))
                #print("-"*10)
                #res.append()
                #print(uid, *out)
                s1 = decode_text(out)
                res.append(s1)
                lb = lb[1:int(len)-1].cpu().tolist()
                s2 = decode_text(lb)
                #print(uid, *lb)
                refs.append(decode_text(lb))
                refer.write(s2)




    
    hyp.close()
    refer.close()
    
    cer_metrics = evaluate.load("cer")
    cer = cer_metrics.compute(references=refs, predictions=res)

    wer_metrics = evaluate.load("wer")
    wer = wer_metrics.compute(references=refs, predictions=res)

    # # for k, v in zip(refs, res):
    # #     print("-"*10)
    # #     print(k)
    # #     print(v)
    
    print(f"cer={cer}, wer={wer}")
    result.write(f"cer={cer}, wer={wer}")
    result.close()
    
def main():
    checkpoint = "whisper-" + scale + "/artifacts/lambda0.3/checkpoint/last-v1.ckpt"
    train()
    inference(checkpoint)

if __name__ == "__main__":
    main()


