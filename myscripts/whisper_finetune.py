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

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

#os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3,4,5,6,7"
DATASET_DIR = "data/personal/jvs_ver1"


SAMPLE_RATE = 16000
BATCH_SIZE = 2
TRAIN_RATE = 0.8

AUDIO_MAX_LENGTH = 480000
TEXT_MAX_LENGTH = 120
SEED = 3407
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
seed_everything(SEED, workers=True)

print("begin")
def load_wave(wave_path, sample_rate:int=16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform


dataset_dir = Path(DATASET_DIR)
transcripts_path_list = list(dataset_dir.glob("*/*/transcripts_utf8.txt"))
print(len(transcripts_path_list))


import pandas as pd

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
        # audioファイルのディレクトリ確認
        # 检查音频文件的目录。
        audio_dir = transcripts_path.parent / "wav24kHz16bit"
        if not audio_dir.exists():
            print(f"{audio_dir}は存在しません。")
            continue

        # 翻訳テキストからAudioIdとテキストを取得
        # 从翻译的文本中获取AudioId和文本。
        with open(transcripts_path, "r") as f:
            text_list = f.readlines()

        for text in text_list:
            audio_id, text = text.replace("\n", "").split(":")
            #print(audio_id, text)

            audio_path = audio_dir / f"{audio_id}.wav"
            if audio_path.exists():
                # データのチェック
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






## g2p???
# def text_kana_convert(text):
#     text = pyopenjtalk.g2p(text, kana=True)
#     return text
# print(text_kana_convert("こんにちは、私の名前は、田中一郎です。"))


print("Model load before!")
woptions = whisper.DecodingOptions(language="ja", without_timestamps=True)
import sys
scale = sys.argv[1]
wmodel = whisper.load_model(os.path.join("/mnt/lustre02/jiangsu/aispeech/home/hs418/low_resource/Pretrained_model", scale + ".pt"))
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
        audio = whisper.pad_or_trim(audio.flatten())
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


# dataset = JvsSpeechDataset(eval_audio_transcript_pair_list, wtokenizer, SAMPLE_RATE)
# loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=WhisperDataCollatorWhithPadding())

eval_audio_transcript_pair_list = get_audio_file_list_from_csv("data/test.csv")

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
    def __init__(self, cfg:Config, model_name="base", lang="ja", train_dataset=[], eval_dataset=[]) -> None:
        super().__init__()
        self.options = whisper.DecodingOptions(language=lang, without_timestamps=True)
        self.model = whisper.load_model(model_name)
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language="ja", task=self.options.task)

        #only decoder training
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")

        self.cfg = cfg
        self.__train_dataset = train_dataset
        self.__eval_dataset = eval_dataset
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        with torch.no_grad():
            audio_features = self.model.encoder(input_ids)

        out = self.model.decoder(dec_input_ids, audio_features)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()


        audio_features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        out[out == -100] = self.tokenizer.eot
        labels[labels == -100] = self.tokenizer.eot

        o_list, l_list = [], []
        for o, l in zip(out, labels):
            o = torch.argmax(o, dim=1)
            o_list.append(self.tokenizer.decode(o, skip_special_tokens=True))
            l_list.append(self.tokenizer.decode(l, skip_special_tokens=True))
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
        """オプティマイザーとスケジューラーを作成する"""
        #创建优化器和调度器
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
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
        """初期設定（データセットの読み込み）"""
        #初始设置（加载数据集）。

        if stage == 'fit' or stage is None:
            self.t_total = (
                (len(self.__train_dataset) // (self.cfg.batch_size))
                // self.cfg.gradient_accumulation_steps
                * float(self.cfg.num_train_epochs)
            )
    
    def train_dataloader(self):
        """訓練データローダーを作成する"""
        #创建一个训练数据加载器。
        dataset = JvsSpeechDataset(self.__train_dataset, self.tokenizer, self.cfg.sample_rate)
        return torch.utils.data.DataLoader(dataset, 
                          batch_size=self.cfg.batch_size, 
                          drop_last=True, shuffle=True, num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )

    def val_dataloader(self):
        """バリデーションデータローダーを作成する"""
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
    cfg = Config(mode="finetune")
    log_output_dir = os.path.join("whisper-" + scale, cfg.mode, "logs")
    check_output_dir = os.path.join("whisper-" + scale, cfg.mode, "artifacts")

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
        save_top_k=1 #-1 # all model save
    )

    callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]

    #train_audio_transcript_pair_list = get_audio_file_list(train_transcripts_path_list, TEXT_MAX_LENGTH, AUDIO_MAX_LENGTH, SAMPLE_RATE)
    train_audio_transcript_pair_list = get_audio_file_list_from_csv("data/train_medium.csv")
    print("TRAIN AUDIO DATASET NUM: ", len(train_audio_transcript_pair_list))

    model = WhisperModelModule(cfg, model_name, lang, train_audio_transcript_pair_list, eval_audio_transcript_pair_list)

    trainer = Trainer(
        precision=16,
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
    #state_dict = state_dict['state_dict']
    state_dict = state_dict['model']
    for k,v in list(state_dict.items()):
        state_dict.pop(k)
        #if k != "encoder.positional_embedding":
        k = "model." + k
        state_dict[k] = v
        

    cfg = Config()
    model_name = os.path.join("Pretrained_model", scale + ".pt")
    whisper_model = WhisperModelModule(cfg, model_name=model_name)

    whisper_model.load_state_dict(state_dict,strict=False)

    woptions = whisper.DecodingOptions(language="japanese", beam_size=8, without_timestamps=True) #

    dataset = JvsSpeechDataset(eval_audio_transcript_pair_list, wtokenizer, SAMPLE_RATE)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=WhisperDataCollatorWhithPadding())

    refs = []
    res = []

    n_res = []
    n_refs = []
    
    tag = sys.argv[2]
    hyp_path = os.path.join("whisper-" + scale, "decode_" + tag)


    if not os.path.exists(hyp_path):
        os.mkdir(hyp_path)
        os.mknod(os.path.join(hyp_path, "hyp.txt"))
        os.mknod(os.path.join(hyp_path, "ref.txt"))
        os.mknod(os.path.join(hyp_path, "results.txt"))

    hyp = open(os.path.join("whisper-" + scale, "decode", "hyp.txt"), "w")
    refer = open(os.path.join("whisper-" + scale, "decode", "ref.txt"), "w")
    result = open(os.path.join("whisper-" + scale, "decode", "results.txt"), "w")

    for b in tqdm(loader):
        input_ids = b["input_ids"].half().cuda()
        labels = b["labels"].long().cuda()
        with torch.no_grad():
            #audio_features = whisper_model.model.encoder(input_ids)
            #out = whisper_model.model.decoder(enc_input_ids, audio_features)
            # import pdb
            # pdb.set_trace()
            results = whisper_model.model.decode(input_ids, woptions)
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
    
    cer_metrics = evaluate.load("cer")
    cer = cer_metrics.compute(references=refs, predictions=res)
    n_cer = cer_metrics.compute(references=n_refs, predictions=n_res)

    wer_metrics = evaluate.load("wer")
    wer = wer_metrics.compute(references=refs, predictions=res)
    n_wer = wer_metrics.compute(references=n_refs, predictions=n_res)

    # for k, v in zip(refs, res):
    #     print("-"*10)
    #     print(k)
    #     print(v)
    
    print(f"cer={cer}, wer={wer}")
    print(f"w/o normalize: cer={n_cer}, wer={n_wer}")
    result.write(f"cer={cer}, wer={wer}")
    result.close()
    
def main():
    #checkpoint = "whisper-" + scale + "/finetune/artifacts/checkpoint/last.ckpt"
    #checkpoint = "exp/trans20L-lstm2L_jp_CSJ_KD_logits_v1_alpha0.5/checkpoint"
    checkpoint = "exp/trans20L-lstm2L_jp_CSJ_KD_logits_v1_alpha0/checkpoint.dump"
    #train()
    inference(checkpoint)

if __name__ == "__main__":
    main()




