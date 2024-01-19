from torch.nn import KLDivLoss, MSELoss, L1Loss
import torch.nn.functional as F
from asr.utils.dynload import add_loss
from asr.loss import Loss
from asr.loss.cross_entropy import CrossEntropyLoss as CELoss

import torch.nn as nn
import logging
logger = logging.getLogger(__name__)

__all__ = ['TSCELoss']

from asr.data import Field
import torch

@add_loss('TSCELoss')
class TSCELoss(Loss):
    r"""TS CE phase loss
    We use KL-Divergence to measure student's and teacher's probability distribution.
    The total loss is interpolated by standard ce loss and KLD-loss with a coefficient :math:`\alpha`.

    .. math::
        loss = (1 - \alpha) * CELoss + \alpha * KLDLoss

    Args:
        alpha (float): interpolation coefficient
        temperature (float): hyper-parameter of KLLoss
        length_tolerance (int): hyper-parameter of [:class:`~asr.loss.ce_loss`]
    """

    def __init__(self, alpha=0.9, temperature=1, length_tolerance=0, distill_mode="static", guided_mode="KD", **kwargs):
        super().__init__()
        self.ce_loss = CELoss(length_tolerance=length_tolerance, **kwargs)
        self.ts_loss = KLDLoss(temperature=temperature, **kwargs)
        self.mse_loss = MSEloss(temperature=temperature, **kwargs)
        self.L1loss = L1loss(temperature=temperature, **kwargs)
        self.alpha = alpha
        self.distill_mode = distill_mode
        self.guided_mode = guided_mode
        
    def extra_repr(self):
        return f'(alpha): {self.alpha:.2f}'

    def forward(self, output, data_batch):
        """
        Parameter
            output (tuple): First is the student's output, second is teacher's.
            data_batch:
        Return:
            loss:
            loss_statistics:
        """
        # import pdb
        # pdb.set_trace()
        # output := student_output, teacher_output, student_hidden, teacher_hidden
        s_ce = Field(output[0], data_batch['label'].length)
        t_ce = Field(output[1], data_batch['label'].length)
        student_hidden, teacher_hidden = output[2], output[3]
        student_hidden_weight, teacher_hidden_weight = output[4], output[5]
        ce_loss, ce_loss_statistics = self.ce_loss(s_ce, data_batch)
        
        #ts_loss, ts_loss_statistics = self.ts_loss(s_ce, t_ce, data_batch)
        
        ts_loss = 0
        if self.guided_mode == "KD":
            if self.distill_mode == "static":
                for i in range(len(student_hidden)):
                    hidden_mse, hidden_mse_statistics = self.mse_loss(student_hidden[i], teacher_hidden[2*i+1])
                    #ts_loss_statistics['loss'] += hidden_mse_statistics['loss']
                    ts_loss += hidden_mse

            elif self.distill_mode == "dynamic_match_nolimit":
                for i in range(len(student_hidden)):
                    min_mse_loss = torch.inf
                    for j in range(len(teacher_hidden)):
                        min_mse_loss = min(min_mse_loss, self.mse_loss(student_hidden[i], teacher_hidden[j])[0])   
                    ts_loss += min_mse_loss
                        
            elif self.distill_mode == "dynamic_match_limit":
                pre = 0
                for i in range(len(student_hidden)):
                    min_mse_loss = torch.inf
                    for j in range(pre, len(teacher_hidden)):
                        h_loss, _ = self.mse_loss(student_hidden[i], teacher_hidden[j])
                        if h_loss < min_mse_loss:
                            min_mse_loss = h_loss
                            pre = j
                    ts_loss += min_mse_loss
            elif self.distill_mode == "logits":
                ts_loss, ts_loss_statistics = self.ts_loss(s_ce, t_ce, data_batch)

        elif self.guided_mode == "KDQ":
            if self.distill_mode == "static":
                for i in range(len(student_hidden_weight)):
                    hidden_l1loss, hidden_l1loss_statistics = self.L1loss(student_hidden_weight[i], teacher_hidden_weight[2*i+1])
                    #ts_loss_statistics['loss'] += hidden_mse_statistics['loss']
                    ts_loss += hidden_l1loss

            elif self.distill_mode == "dynamic_match_nolimit":
                for i in range(len(student_hidden_weight)):
                    min_l1loss = torch.inf
                    for j in range(len(teacher_hidden_weight)):
                        min_l1loss = min(min_l1loss, self.L1loss(student_hidden_weight[i], teacher_hidden_weight[j])[0])   
                    ts_loss += min_l1loss
                        
            elif self.distill_mode == "dynamic_match_limit":
                pre = 0
                for i in range(len(student_hidden_weight)):
                    min_l1loss = torch.inf
                    for j in range(pre, len(teacher_hidden_weight)):
                        h_loss, _ = self.L1loss(student_hidden_weight[i], teacher_hidden_weight[j])
                        if h_loss < min_l1loss:
                            min_l1loss = h_loss
                            pre = j
                    ts_loss += min_l1loss
            elif self.distill_mode == "logits":
                ts_loss, ts_loss_statistics = self.ts_loss(s_ce, t_ce, data_batch)
        else:
            ts_loss = 0
        
         

        # elif self.distill_mode == "logits_and_hidden":
        #     # prediction layer kld loss
        if self.distill_mode != "logits":
            kld_loss, ts_loss_statistics  = self.ts_loss(s_ce, t_ce, data_batch)
            ts_loss /= len(student_hidden)
            ts_loss += kld_loss

        loss = ( 1 - self.alpha ) * ce_loss + self.alpha * ts_loss 
        loss_statistics = {
            'ce_loss': ce_loss_statistics['loss'],
            'ts_loss': ts_loss_statistics['loss'],
            'loss': 1 - ce_loss_statistics['correct_frames'] / ts_loss_statistics['total_frames'],
            'correct_frames': ce_loss_statistics['correct_frames'],
            'total_frames': ts_loss_statistics['total_frames'],
            'merge_criterion': 1
        } 

        return loss, loss_statistics

    def log_line(self, reduced_stat):
        frame_accuracy = reduced_stat['correct_frames'] / reduced_stat['total_frames']
        ce_loss_per_frame = reduced_stat['ce_loss'] / reduced_stat['total_frames']
        ts_loss_per_frame = reduced_stat['ts_loss'] / reduced_stat['total_frames']
        return (f'Frame Acc: {frame_accuracy * 100:.2f} ce_loss: {ce_loss_per_frame:.2f} ts_loss: {ts_loss_per_frame:.2f}')

@add_loss('TSMTLoss')
class TSMTLoss(Loss):
    r"""TS multi-task phase loss
    We use KL-Divergence to measure student's and teacher's probability distribution.
    The total loss is interpolated by standard ce loss and KLD-loss with a coefficient :math:`\alpha` and CTC-loss.

    .. math::
        loss = (1 - \alpha) * CELoss + \alpha * KLDLoss + \beta * ( ( 1 - \alpha ) * CTCLoss + \alpha * KLDLoss )

    Args:
        alpha (float): interpolation coefficient
        beta (float): coefficient for CTC
        temperature (float): hyper-parameter of KLLoss
        length_tolerance (int): hyper-parameter of [:class:`~asr.loss.ce_loss`]
    """

    def __init__(self, alpha=0.9, beta=0.1, temperature=1, length_tolerance=0, **kwargs):
        super().__init__()
        self.ce_loss = CELoss(length_tolerance=length_tolerance, **kwargs)
        self.ts_loss = KLDLoss(temperature=temperature, **kwargs)
        from asr.loss.ctc_loss import CTCLoss
        self.ctc_loss = CTCLoss(size_average=False, blank=0)
        self.alpha = alpha
        self.beta = beta

    def extra_repr(self):
        return f'(alpha): {self.alpha:.2f}'

    def forward(self, output, data_batch):
        """
        Parameter
            output (tuple): First is the student's output, second is teacher's.
            data_batch:
        Return:
            loss:
            loss_statistics:
        """

        data_batch_ctc = {
            'feat': data_batch['feat'],
            'label': data_batch['extra']['label'][0],
        }

        student_out, teacher_out = output
        ctc_loss, ctc_stat = self.ctc_loss(student_out[0], data_batch_ctc)
        ce_loss, ce_loss_statistics = self.ce_loss(student_out[1], data_batch)
        ts_loss, ts_loss_statistics = self.ts_loss(student_out[1], teacher_out[1], data_batch)
        
        loss = ( self.beta * ctc_loss + ( 1 - self.alpha ) * ce_loss + self.alpha * ts_loss ) / student_out[0].tensor.size(0)
        loss_statistics = {
            'ce_loss': ce_loss_statistics['loss'],
            'ts_loss': ts_loss_statistics['loss'],
            'loss': 1 - ce_loss_statistics['correct_frames'] / ce_loss_statistics['total_frames'],
            'correct_frames': ce_loss_statistics['correct_frames'],
            'total_tokens': ce_loss_statistics['total_frames'],
            'correct_tokens': ctc_stat['correct_labels'],
            'total_frames': ctc_stat['total_frames'],
            'merge_criterion': 1
        }

        return loss, loss_statistics

    def log_line(self, reduced_stat):
        frame_accuracy = reduced_stat['correct_frames'] / reduced_stat['total_tokens']
        ce_loss_per_frame = reduced_stat['ce_loss'] / reduced_stat['total_tokens']
        ts_loss_per_frame = reduced_stat['ts_loss'] / reduced_stat['total_tokens']
        token_acc = reduced_stat['correct_tokens'] / reduced_stat['total_tokens']
        return (f'Frame Acc: {frame_accuracy * 100:.3f} CTC Acc: {token_acc * 100:.3f} ce_loss: {ce_loss_per_frame:.2f} ts_loss: {ts_loss_per_frame:.2f}')

class KLDLoss(Loss):
    def __init__(self, temperature: float=1, reduction: str='sum'):
        super().__init__()
        self.loss_kernel = KLDivLoss(reduction=reduction)
        self.temperature = temperature

    def extra_repr(self) -> str:
        return f'(temperature): {self.temperature:.2f}'

    def forward(self, output, target, data_batch):
        label = data_batch['label'].tensor.cuda().view(-1)

        output_tensor, target_tensor = output.tensor.reshape(-1,output.tensor.shape[-1])[label!=-1], target.tensor.reshape(-1,target.tensor.shape[-1])[label!=-1]
        loss = (self.temperature**2) * self.loss_kernel(F.log_softmax(output_tensor/self.temperature, dim=-1),
                                F.softmax(target_tensor/self.temperature, dim=-1))
        loss_statistics = {
            'loss': loss.item(),
            'total_frames': sum(output.length).item()
        }
        return loss, loss_statistics

    def log_line(self, reduced_stat):
        kld_loss_per_frame = reduced_stat['loss'] / reduced_stat['total_frames']
        return (f'kld_loss: {kld_loss_per_frame:.2f}')

class MSEloss(Loss):
    def __init__(self, temperature: float=1, reduction: str='sum'):
        super().__init__()
        self.loss_kernel = MSELoss(reduction=reduction)
        self.temperature = temperature

    def extra_repr(self) -> str:
        return f'(temperature): {self.temperature:.2f}'

    def forward(self, output, target):

        output_tensor, target_tensor = output.reshape(-1,output.shape[-1]), target.reshape(-1,target.shape[-1])
        loss = (self.temperature**2) * self.loss_kernel(F.softmax(output_tensor/self.temperature, dim=-1),
                                F.softmax(target_tensor/self.temperature, dim=-1))
        loss_statistics = {
            'loss': loss.item(),
            'total_frames': 2048
        }
        return loss, loss_statistics

    def log_line(self, reduced_stat):
        mse_loss_per_frame = reduced_stat['loss'] / reduced_stat['total_frames']
        return (f'mse_loss: {mse_loss_per_frame:.2f}')

class L1loss(Loss):
    def __init__(self, temperature: float=1, reduction: str='sum'):
        super().__init__()
        self.loss_kernel = L1Loss(reduction=reduction)
        self.temperature = temperature

    def extra_repr(self) -> str:
        return f'(temperature): {self.temperature:.2f}'

    def forward(self, output, target):

        output_tensor, target_tensor = output.reshape(-1,output.shape[-1]), target.reshape(-1,target.shape[-1])
        loss = (self.temperature**2) * self.loss_kernel(F.softmax(output_tensor/self.temperature, dim=-1),
                                F.softmax(target_tensor/self.temperature, dim=-1))
        
        #loss = self.loss_kernel(output_tensor, target_tensor)
        loss_statistics = {
            'loss': loss.item(),
            'total_frames': 2048
        }
        return loss, loss_statistics

    def log_line(self, reduced_stat):
        mse_loss_per_frame = reduced_stat['loss'] / reduced_stat['total_frames']
        return (f'mse_loss: {mse_loss_per_frame:.2f}')
