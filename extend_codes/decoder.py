import logging

import torch
import torch.nn as nn
import numpy as np
from .beam_search import Beam

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

class Decoder(nn.Module):

    def __init__(self, nvocab: int):
        super().__init__()
        self.nvocab = nvocab
        self.online = False
        

    def forward_one_step( self, state: dict, tgt: torch.Tensor, enc: torch.Tensor, en_mask: torch.Tensor ):
        raise NotImplementedError('Please implement your own forward')
    def update_state( self, state: dict, vidx: torch.Tensor, B: int, beam: int ):
        raise NotImplementedError('Please implement your own update')
    def decode_e2e(self, enc, en_mask, max_length, beam, nbest=1):
        T, B, D = enc.size()
        maxl = max_length.max() + 6 #(T + 5) // 2
        # beam = 2 * beam
        tgt = torch.zeros(B, device=device).long()
        beam_search = Beam(batch=B, beam=beam, nvocab=self.nvocab)
        yseq = torch.zeros(B, beam, 1, device=device).long()
        vscores = torch.zeros(B, beam, device=device)
        from asr.data.field import Field
        output = {'hyps': Field(torch.zeros(B, nbest+beam, maxl, device=device).long(), torch.ones(B, nbest+beam, device=device).long()), 
        'scores': torch.zeros(B, nbest+beam, device=device).fill_(-10000)
        }

        stop_search = torch.zeros(B, device=device).bool()
        state = None


        sum = torch.zeros(B * beam, 1, device=device).long()
        total = torch.zeros(B * beam, 1, device=device).long()
        for i in range(maxl):
            out, state = self.forward_one_step( state, tgt, enc, en_mask)

            vscores, yseq, vidx, ended_hyps = beam_search.recognize_beam_batch(out, vscores, yseq, stop_search)
           

            tgt = yseq[:,:,-1].view(-1)
            

            #vscores.shape = [B, N] yseq.shape = [B, N, L+1] vidx.shape = [B*N] 
            #ended_hyps:{y_prev.shape = [B, N, L] eos_vscores.shape = [B, N]}
            stop_search = beam_search.end_detect(stop_search, ended_hyps, output, max_length, i)
            #if stop_search.min():
            #    break
            output = beam_search.update_hyp(ended_hyps, output)
            #print(output['scores'])
            if i == 0:
                enc = enc.view(T, B, 1, D).repeat(1, 1, beam, 1).view(T, B * beam, D)
                en_mask = en_mask.view(B, 1, T).repeat(1, beam, 1).view(B * beam, T)
                state = self.update_state( state, None, B, beam )
            else:
                state = self.update_state( state, vidx, 0, 0 )
            
        return {'hyps': Field(output['hyps'].tensor[:,:nbest], output['hyps'].length[:,:nbest]), 
        'scores': output['scores'][:, :nbest], 'vscores': vscores, 'sum': sum[:nbest], 'total': total[:nbest]}
        
    
