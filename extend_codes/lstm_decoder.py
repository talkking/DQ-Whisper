import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from asr.model.transformer.decoder.attention import multi_head_attention_forward

from .decoder import Decoder
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

class LSTMDecoder(Decoder):
    def __init__(self, nhid: int, nproj: int, nvocab: int, nlayers: int, dropout=0.0):
        super().__init__(nvocab)
        self.nhid = nhid
        self.nproj = nproj
        self.nlayers = nlayers
        dec_in = nproj * 2
        self.embed = nn.Embedding(nvocab, nproj)
        self.rnn = nn.ModuleList(
            [nn.LSTMCell(dec_in if i == 0 else nhid, nhid) for i in range(nlayers)]
        )
        self.attend = nn.MultiheadAttention(nproj, 8)
        self.proj = nn.Linear(nhid + nproj, nproj)  # (att-context & LSTM state) projection
        self.proj_lstm = nn.Linear(nhid, nproj)
        self.linear = nn.Linear(nproj, nvocab)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def rnn_forward(self, x, state):
        if state is None:
            h = [torch.zeros(x.size(0), self.nhid, device=device) for _ in range(self.nlayers)]
            state = {'h': h}
            c = [torch.zeros(x.size(0), self.nhid, device=device) for _ in range(self.nlayers)]
            state['c'] = c

        h = [None] * self.nlayers
        c = [None] * self.nlayers
        for i in range(self.nlayers):
            # dropout applied only
            inp = x if i == 0 else self.dropout(h[i - 1])
            h[i], c[i] = self.rnn[i](inp, (state['h'][i], state['c'][i]))
        state = {'h': h, 'c': c}
        return state

    def attend_on_encode( self, query, key, key_mask ):
        if key is not None:
            embed_dim = self.attend.embed_dim
            ### in_proj_bias : q k v
            _b = self.attend.in_proj_bias[embed_dim:] if self.attend.in_proj_bias is not None else None
            _w = self.attend.in_proj_weight[embed_dim:,:]
            self.cache_k, self.cache_v = F.linear(key, _w, _b).chunk(2, dim=-1)
            T, B, _ = key.size()
            if self.attend.bias_k is not None and self.attend.bias_v is not None:
                self.cache_k = torch.cat([self.cache_k, self.attend.bias_k.repeat(1, B, 1)])
                self.cache_v = torch.cat([self.cache_v, self.attend.bias_v.repeat(1, B, 1)])
            self.cache_k = self.cache_k.contiguous().view(T, B * self.attend.num_heads, -1).transpose(0, 1)
            self.cache_v = self.cache_v.contiguous().view(T, B * self.attend.num_heads, -1).transpose(0, 1)
        

        
        return multi_head_attention_forward(
                query, None, None, self.attend.embed_dim, self.attend.num_heads,
                self.attend.in_proj_weight, self.attend.in_proj_bias,
                self.attend.bias_k, self.attend.bias_v, self.attend.add_zero_attn,
                self.attend.dropout, self.attend.out_proj.weight, self.attend.out_proj.bias,
                training=self.training,
                key_padding_mask=key_mask, static_k=self.cache_k, static_v=self.cache_v)

    def set_online(self):
        self.online=True

    ### training
    def forward(self, henc, hen_mask, ys, sampling_prob=0.0, zerocontext=False):
        bs = henc.size(1)  # batch-size
        # pre-computation of embedding
        # eys = F.dropout(self.embed(ys), self.dropout, self.training) # [B, U, nproj]
        eys = self.dropout(self.embed(ys))
        # initialization of different state
        state = None  # decoder state
        c = torch.zeros(bs, self.nproj, device=device)  # attention context, [B, nproj]
        h_hat = []  # projected embedding for output layer prediction
        # loop along label sequence
        for i in range(eys.size(1)):
            # schedule sampling, only apply on training stage & start from the second token (except <sos>)
            if sampling_prob > 0 and i > 0 and random.random() < sampling_prob:
                # compute last step output distribution
                z_out = self.linear(last_h_hat)
                _, z_out = torch.max(z_out.detach(), dim=-1)  # simply pick the most probable token
                # z_out = torch.multinomial(z_out.softmax(-1), 1).view(-1) # alternate approach is sampling from distribution
                #z_out = F.dropout(self.embed(z_out), self.dropout, self.training)
                z_out = self.embed(z_out)
            else:
                z_out = eys[:, i, :]
            # feed [c_{t-1}, y_{t-1}]
            
            dec_in = torch.cat((z_out, c), dim=1)
            # forward decoder rnn
            state = self.rnn_forward(dec_in, state)
            # use last hidden state to calculate attention
            hdec = state['h'][-1]
            proj = self.proj_lstm(hdec).view(1, bs, -1)
            m_mask = hen_mask[i] if self.online else hen_mask
            ###### context vector = 0 
            if not zerocontext:
                c = self.attend_on_encode( proj, henc, m_mask )[0].view(bs,-1) if i == 0 else self.attend_on_encode( proj, None, m_mask )[0].view(bs,-1)
            #logger.warning(f"c={c}")
            ###### end
            # c = self.attend(proj, henc, henc, key_padding_mask=hen_mask)[0].view(bs, -1)
            # concat hdec and att-context
            h_c = self.proj(torch.cat((hdec, c), dim=1))
            # store last state
            last_h_hat = torch.tanh(h_c)
            h_hat.append(last_h_hat.view(1, bs, -1))
        h_hat = torch.cat(h_hat, dim=0)  # [T, B, nproj]
        y_hat = self.linear(h_hat)  # [T, B, nvocab]
    
        return y_hat.transpose(0, 1)

    ### inference
    def forward_one_step(self, state, tgt, enc, en_mask, textonly=False):

        first_step = False


        if state is None:
            state = {'prev_att': torch.zeros(tgt.size(0), self.nproj, device=device), 
            'rnn_state': None}
            first_step = True

        
        attention = state['prev_att']
        rnn_state = state['rnn_state']

        tgt = self.embed( tgt )
        # feed [c_{t-1}, y_{t-1}]
        if not textonly:
            dec_in = torch.cat((tgt, attention), dim=1)
        else:
            dec_in = torch.cat((tgt, torch.zeros_like(attention)), dim=1)

        # forward decoder rnn
        rnn_state = self.rnn_forward(dec_in, rnn_state)
        # use last hidden state to calculate attention
        hdec = rnn_state['h'][-1]
        proj = self.proj_lstm(hdec).unsqueeze(0)

        
        if not textonly:
            attention = self.attend_on_encode(proj, enc, en_mask) if first_step else self.attend_on_encode(proj, None, en_mask)
            self.att_debug = attention
            attention = attention[0].squeeze(0)
        # if self.training:
        #     attention = F.dropout(attention, 0.1, self.training)
        # concat hdec and att-context
        h_c = self.proj(torch.cat((hdec, attention), dim=1))
        # store last state
        out = self.linear(torch.tanh(h_c)).log_softmax(dim=-1)
        return out, {'prev_att': attention, 'rnn_state': rnn_state}

    
    
    def update_state(self, state, vidx, B, beam ):
        attention = state['prev_att']
        rnn_state = state['rnn_state']
        if vidx is None:
            attention = attention.view(B, 1, -1).repeat(1, beam, 1).view(B * beam, -1)
            _,T,HD = self.cache_k.size()
            self.cache_k = self.cache_k.view(B,1,-1,T,HD).repeat(1,beam,1,1,1).view(-1,T,HD)
            self.cache_v = self.cache_v.view(B,1,-1,T,HD).repeat(1,beam,1,1,1).view(-1,T,HD)
            for j in range(self.nlayers):
                rnn_state['h'][j] = rnn_state['h'][j].view(B, 1, -1).repeat(1, beam, 1).view(B * beam, -1)
                rnn_state['c'][j] = rnn_state['c'][j].view(B, 1, -1).repeat(1, beam, 1).view(B * beam, -1)
        else:
            attention = attention[vidx, :]
            for j in range(self.nlayers):
                rnn_state['h'][j] = rnn_state['h'][j][vidx, :]
                rnn_state['c'][j] = rnn_state['c'][j][vidx, :]
        return {'prev_att': attention, 'rnn_state': rnn_state}

    def update_state1(self, state, vidx, B, beam ):
        attention = state['prev_att']
        rnn_state = state['rnn_state']
        if vidx is None:
            attention = attention.view(B, 1, -1).repeat(1, beam, 1).view(B * beam, -1)
            for j in range(self.nlayers):
                rnn_state['h'][j] = rnn_state['h'][j].view(B, 1, -1).repeat(1, beam, 1).view(B * beam, -1)
                rnn_state['c'][j] = rnn_state['c'][j].view(B, 1, -1).repeat(1, beam, 1).view(B * beam, -1)
        else:
            attention = attention[vidx, :]
            for j in range(self.nlayers):
                rnn_state['h'][j] = rnn_state['h'][j][vidx, :]
                rnn_state['c'][j] = rnn_state['c'][j][vidx, :]
        return {'prev_att': attention, 'rnn_state': rnn_state}

    
