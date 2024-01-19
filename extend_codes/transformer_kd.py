import logging
import math

import torch
import torch.nn as nn
import random

from asr.data.field import Field
from asr.model import Model, add_model
from asr.model.module import build_activation
from extend_codes.vgg import VGG2L
from asr.model.transformer.positional_embedding import PositionalEncoding
from asr.model.transformer.decoder.lstm_decoder import LSTMDecoder
from asr.model.transformer.decoder.transformer_decoder import TransformerDecoder

import torch.nn.functional as F

logger = logging.getLogger(__name__)

class Attention(nn.Module):
    def __init__(self, nhid=2048, nproj=512, nhead=8, activation=nn.ReLU6(True), dp=0.1, sd=0):
        super().__init__()
        self.ln = nn.LayerNorm(nproj)
        self.dropout = nn.Dropout(dp, inplace=True)
        self.proj = nn.Sequential(nn.LayerNorm(nproj), nn.Linear(nproj, nhid), activation,
                                  nn.Linear(nhid, nproj, bias=False), self.dropout)
        self.att = nn.MultiheadAttention(nproj, nhead)
        self.sd = sd

    def forward(self, xs, m_mask, attn_mask=None):
        if self.training and random.random() < self.sd:
            return xs
        ln = self.ln(xs)
        h = self.dropout(self.att(ln, ln, ln, key_padding_mask=m_mask, attn_mask=attn_mask)[0])
        hh = h + xs
        h = self.proj(hh) + hh
        return h

# @add_model('Transformer')
# @add_model('TRANSFORM')
class Transformer(Model):
    # mode can be ctc, mt, mmi, las
    def __init__(self, ninp, nhid=2048, nproj=512, nctc=2966, natt=6979,
                 nlayer=12, nhead=8, nhid_dec=1024, ndecode=2, max_norm=1000, activation='relu6', dec='lstm', 
                 dropout=0.1, pos_emb=True, mode='mt', online=0, step=20, sd=0, dec_lim=60, zerocontext=False):
        super().__init__()

        self.activation = build_activation(activation, inplace=True)
        self.conv = VGG2L(1, 64, keepbn=True) if mode == 'mmi' else VGG2L(1, 64)
        #self.conv = VGG2L(1, 128, keepbn=True) if mode == 'mmi' else VGG2L(1, 128)
        ninp = self.conv.outdim(ninp)
        if pos_emb:
            self.proj = nn.Sequential(nn.Linear(ninp, nproj), PositionalEncoding(nproj, dropout))
        else:
            self.proj = nn.Linear(ninp, nproj)
        self.att = nn.ModuleList([Attention(nhid, nproj, nhead=nhead, activation=self.activation, dp=dropout, sd=sd)
                                  for i in range(nlayer)])
        self.max_norm = max_norm
        self.mode = mode
        self.online = online
        self.step = step
        self.dec_limit = dec_lim
        self.zerocontext = zerocontext

        self.out_ctc = nn.Linear(nproj, nctc)
        if dec == 'lstm':
            self.decoder = LSTMDecoder(nhid_dec, nproj, natt, ndecode)
        else:
            self.decoder = TransformerDecoder(nhid_dec, nproj, natt, ndecode, dropout)

    def forward(self, batch):
        xs = batch['input_ids'].transpose(1, 2)   # [B, L, D], mel_features
        length = batch['nframes'] #[B]

        ### BLD
        
        # import pdb
        # pdb.set_trace()
        xs, length = self.conv(xs, length)
        # import pdb
        # pdb.set_trace()
        xs = self.proj(xs).transpose(0, 1)
        memory_key_padding_mask = length.cuda().unsqueeze(1) <= torch.arange(0, xs.size(0), device='cuda').unsqueeze(0)
        attn_mask=None


        ### las online
        if self.online > 0:
            attn_mask=torch.ones(xs.size(0),xs.size(0),device='cuda').bool()
            for i in range(0,xs.size(0),self.step):
                st=i-self.step if i>0 else 0
                attn_mask[i:i+self.step,st:i+self.step]=False
        ###

        ### 


        for i in range(len(self.att)):
            if i == len(self.att) - 1:
                ln = self.att[i].ln(xs)
                h = self.att[i].dropout(self.att[i].att(ln, ln, ln, key_padding_mask=memory_key_padding_mask, attn_mask=attn_mask)[0])
                hh = h + xs
                h1 = self.att[i].proj[0](hh)
                h1 = self.att[i].proj[1](h1)
                hs = self.att[i].proj[2](h1) ## [T, B, D]
                # import pdb
                # pdb.set_trace()
           
            xs = self.att[i](xs, None if self.online > 0 else memory_key_padding_mask,attn_mask) 
            

        #hs = xs  #[L, B, D]

        if self.mode != 'mt':
            xs = xs.detach()
        if self.mode != 'las':
            ctc_out = self.out_ctc(xs).transpose(0, 1).contiguous()
        if self.mode == 'ctc' or self.mode == 'mmi':
            return Field(ctc_out, length)



        ### label得传4303_units的label
        label = batch['label'].tensor.cuda()            # (B, L)
        label = label.masked_fill(label == -1, 0)
        if self.online == 2:
            from.e2e_online import get_online_mask
            m_mask, _ = get_online_mask(self,xs,length,memory_key_padding_mask,batch['label'].length)
            self.decoder.set_online()
            output = self.decoder(xs, m_mask, label)[:,:-1,:]
        else:
            output = self.decoder(xs, memory_key_padding_mask, label)[:, :-1, :]
        
        batch['label'].tensor = batch['label'].tensor[:, 1:]
        batch['label'].length -= 1



        if self.mode == 'mt':
            return Field(ctc_out, length), Field(output, batch['label'].length), hs

        return Field(output, batch['label'].length)

    def decode(self, batch):
#        if self.online:
#            xs = batch['feat'].tensor.cuda()
#            length = batch['feat'].length

#            out_length = ( length - 1 ) // 2 // 2 + 1

#            hid = []
#            inp = xs[:, :86]
#            t_len = length.clamp(0,86)
#            inp, _ = self.conv(inp, t_len)
#            inp = self.proj(inp).transpose(0,1)[:20]
#            conv = inp[10:]

#            for i in range(len(self.att)):
#                inp = self.att[i](inp, None)
#                hid.append(inp)
#            out = []
#            out.append(hid[-1])

#            for t in range(80,xs.size(1),80):
#                inp = xs[:,t-8:t+86]
#                t_len = (length-t+8).clamp(0, 94)
#                inp, _ = self.conv(inp, t_len)
#                inp = self.proj(inp).transpose(0,1)[2:22]
#                inp = torch.cat((conv, inp), dim=0)
#                conv = inp[20:]
#                for i in range(len(self.att)):
#                    tmp = self.att[i](inp, None)
#                    inp = torch.cat((hid[i][10:], tmp[10:]), dim=0)
#                    hid[i] = tmp[10:]
#                out.append(hid[-1])

#            xs = torch.cat(out,dim=0)
#            xs = self.out_ctc(xs).transpose(0,1).contiguous()
#            return Field(xs, out_length)
        self.mode = 'ctc'
        return self.forward(batch)

    def decode_e2e(self, batch, beam, off2on=False):
        self.device = next(self.parameters()).device
        # xs = batch['feat'].tensor.to(self.device)
        # length = batch['feat'].length

        xs = batch['input_ids'].transpose(1, 2).to(self.device)   # [B, L, D], mel_features
        length = batch['nframes'] #[B]




        xs, length = self.conv(xs, length)
        xs = self.proj(xs).transpose(0, 1)
        cu_length = length.to(self.device)
        memory_key_padding_mask = cu_length.unsqueeze(1) <= torch.arange(0, xs.size(0), device=self.device).unsqueeze(0)

        if not off2on:
            attn_mask=None
            if self.online > 0:
                attn_mask=torch.ones(xs.size(0),xs.size(0),device=self.device).bool()
                for i in range(0,xs.size(0),self.step):
                    st=i-10 if i>0 else 0
                    attn_mask[i:i+self.step,st:i+self.step]=False
            for i in range(len(self.att)):
                xs = self.att[i](xs, None if self.online else memory_key_padding_mask,attn_mask)
        else:
            out = []
            hid = [None for i in range(len(self.att))]
            for t in range(0,xs.size(0),10):
                st = t-10 if t>0 else 0
                inp = xs[st:t+20]
                cu_length = torch.ones(length.size(0), device=self.device).int() * inp.size(0)
                for i in range(len(self.att)):
                    tmp = self.att[i](inp, None)
                    inp = torch.cat((hid[i],tmp[10:]), dim=0)  if hid[i] is not None else tmp
                    hid[i] = tmp[:10] if hid[i] is None else tmp[10:20]
                out.append(hid[-1])
            xs = torch.cat(out,dim=0)

        if self.online > 0:
            #from .e2e_jointdecode import joint_decode
            #return joint_decode(self, xs, length, memory_key_padding_mask, 8, 1)
            from .e2e_online import decode_online
            return decode_online(self, xs, length, memory_key_padding_mask, beam)
        ctc_out = self.out_ctc(xs).transpose(0,1).detach()
        ctc_pred = ctc_out.max(dim=-1)[1]
        ctc_pred *= ~memory_key_padding_mask
        max_length = (ctc_pred!=0).sum(dim=-1) + 2
        
        #max_length = (cu_length + 5) // 2
        return self.decoder.decode_e2e(xs, memory_key_padding_mask, max_length, beam, 1)

    def grad_post_processing(self):
        """Clip the accumulated norm of all gradients to max_norm"""
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_norm)
        if norm >= self.max_norm:
            logger.debug(f'Norm overflow: {norm}')
        if math.isnan(norm) or math.isinf(norm):
            self.zero_grad()
            logger.debug(f'Norm is abnormal: {norm}')

    def rescore(self, batch,penalty=0.0):
        xs = batch['feat'].tensor.cuda()
        length = batch['feat'].length

        xs, length = self.conv(xs, length)
        xs = self.proj(xs).transpose(0,1)

        memory_key_padding_mask = length.cuda().unsqueeze(1) <= torch.arange(0, xs.size(0), device='cuda').unsqueeze(0)
        attn_mask=None
        if self.online > 0:
            attn_mask=torch.ones(xs.size(0),xs.size(0),device='cuda').bool()
            for i in range(0,xs.size(0),20):
                st=i-10 if i>0 else 0
                attn_mask[i:i+20,st:i+20]=False
        for i in range(len(self.att)):
            xs = self.att[i](xs, None if self.online else memory_key_padding_mask,attn_mask)

        label = batch['label'].tensor.cuda()            # (B, L)
        label = label.masked_fill(label == -1, 0)
        output = self.decoder(xs, memory_key_padding_mask, label)[:, :-1, :]
        output = output.log_softmax(dim=-1)

        label = batch['label'].tensor.cuda()[:,1:]
        B,T,D = output.size()
        label = label.unsqueeze(2)
        label2 = label.masked_fill(label == -1, 0)
        output = torch.gather(output, 2, label2)
        output = output.masked_fill(label == -1, 0).reshape(B,-1).sum(dim=1)
        return output
