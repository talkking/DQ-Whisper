 
<div align="center">
  <h1>DQ-Whisper: Joint Distillation and Quantization for Efficient Multilingual Speech Recognition</h1>
</div>

## Table of Contents

1. [Introduction](#1-introduction)
2. [Usage](#2-usage)
3. [Experiment Results](#3-experiment-results)
4. [Citation](#4-citation)


## 1. Introduction
As a popular multilingual and multitask pre-trained speech model, Whisper has the problem of curse of multilinguality. To enhance multilingual capabilities in small Whisper models, we propose DQ-Whisper, a novel joint distillation and quantization framework to compress Whisper for efficient inference. Firstly, we propose a novel dynamic matching distillation strategy. Then, a quantization-aware distillation framework is introduced to integrate quantization with distillation. Experimental results on various multilingual datasets show that our suggested distillation approach can effectively enhance the multilingual capabilities of small Whisper models without increasing computational costs. Up to 5.18x reduction in model size is achieved with marginal performance degradation. In addition, quantization is compatible with distillation, which can result in a higher compression rate.

## 2. Usage

### Training

```
bash scripts/train_ce_ctc.sh --nj 1 --expdir exp/jp_CSJ_KD_logits_v1_alpha0 --conf conf/jap_ts.yaml checkpoint_dir=exp/jp_CSJ_KD_logits_v1_alpha0 data.data_dir=data/csj_whisper optim.lr=3e-5 data.collector.minibatch_size=20 loss.alpha=0
```

### Inference
```
python inference.py base exp/trans20L-lstm2L_jp_CSJ_KD_logits_v1_alpha0.3
```
## 3. Experiment Results
<img width="663" alt="Clipboard_Screenshot_1738983935" src="https://github.com/user-attachments/assets/b501844f-ffeb-4c16-a6d1-c64cd40557a8" />
<br>
<img width="327" alt="Clipboard_Screenshot_1738983985" src="https://github.com/user-attachments/assets/86644eab-fc7c-4bad-b561-f8a93b52bd3b" />
<img width="324" alt="Clipboard_Screenshot_1738984005" src="https://github.com/user-attachments/assets/cbce2ddb-68f3-41a6-9825-d09960bfb704" />


## 4. Citation
```
@INPROCEEDINGS{
    author={Shao, Hang and Liu, Bei and Wang, Wei and Gong, Xun and Qian, Yanmin},
    booktitle={2024 IEEE Spoken Language Technology Workshop (SLT)}, 
    title={DQ-Whisper: Joint Distillation and Quantization for Efficient Multilingual Speech Recognition}, 
    year={2024},
    pages={240-246},
    keywords={Degradation;Quantization (signal);Computational modeling;Conferences;Merging;Speech recognition;Multilingual;Computational efficiency},
    doi={10.1109/SLT61566.2024.10832149}
  }
```
