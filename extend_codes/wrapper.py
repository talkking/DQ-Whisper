from copy import deepcopy
from typing import Dict, OrderedDict, Union
import logging
import torch
from torch import Tensor
from asr.data.field import Field
from asr.utils.dynload import build_model, add_model
from asr.model import Model

import torch.nn as nn

from .whisper_model import Whisper, ModelDimensions
from .whisper_model_quan import Whisper as QWhisper, ModelDimensions as QModelDimensions


logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

import numpy as np

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)  # log(t)/(c/2 -1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2)) #e^{(log(t)/(c/2-1) * [1,2,3,...,c/2])}
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :] # [T, 1] * [1, c/2] #[1,2,3,...,]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1) 

# def build_teacher(checkpoint: str=None):
#     if checkpoint is not None:
#         logger.warning(f'Build teacher model from {checkpoint}')
#         ckpt = torch.load(checkpoint, map_location='cpu')
#         teacher = build_model(ckpt['hparams'])
#         teacher.load_state_dict(ckpt['model'])
#         for parameter in teacher.parameters():
#             parameter.requires_grad = False
#         return teacher 
#     else:
#         raise ValueError('Teacher\'s checkpoint path must be specified.')

def build_teacher(checkpoint: str=None, bit=8):
    if checkpoint is not None:
        logger.warning(f'Build teacher model from {checkpoint}')
        checkpoint = torch.load(checkpoint, map_location=device)
        
        dims = checkpoint["dims"]
        # import pdb
        # pdb.set_trace()

        n_audio_ctx = 750 
        dims["n_audio_ctx"] = n_audio_ctx
        n_state = dims["n_audio_state"]

        dims = ModelDimensions(**checkpoint["dims"])
        model = Whisper(dims)
        #model = Whisper(dims, bit=8)
        state_dict = checkpoint["model_state_dict"]
        state_dict["encoder.positional_embedding"] = sinusoids(n_audio_ctx, n_state) #state_dict["encoder.positional_embedding"][:n_audio_ctx] # PE is right?
        model.load_state_dict(state_dict, strict=False)
        for parameter in model.parameters():
            parameter.requires_grad = False
        return model
    else:
        raise ValueError('Teacher\'s checkpoint path must be specified.')

# def build_teacher(checkpoint: str=None, bit=8):
#     if checkpoint is not None:
#         logger.warning(f'Build teacher model from {checkpoint}')
#         checkpoint = torch.load(checkpoint, map_location=device)
        
#         model1 = torch.load(checkpoint['hparams']['student'])
#         dims = model1["dims"]
#         # import pdb
#         # pdb.set_trace()

#         # n_audio_ctx = 400 #500  
#         # dims["n_audio_ctx"] = n_audio_ctx
#         # n_state = dims["n_audio_state"]

#         dims = ModelDimensions(**dims)
#         model = Whisper(dims)
#         state_dict = checkpoint["model"] 
#         #state_dict["encoder.positional_embedding"] = sinusoids(n_audio_ctx, n_state) #state_dict["encoder.positional_embedding"][:n_audio_ctx] # PE is right?
#         model.load_state_dict(state_dict, strict=False)
#         for parameter in model.parameters():
#             parameter.requires_grad = False
#         return model
#     else:
#         raise ValueError('Teacher\'s checkpoint path must be specified.')
    

def build_student(checkpoint: str=None, bit=8):
    if checkpoint is not None:
        logger.warning(f'Build student model from {checkpoint}')
        checkpoint = torch.load(checkpoint, map_location=device)
        dims = checkpoint["dims"]

        n_audio_ctx = 750 #500  
        dims["n_audio_ctx"] = n_audio_ctx
        n_state = dims["n_audio_state"]

        dims = ModelDimensions(**checkpoint["dims"])
        #model = QWhisper(dims, bit=bit)
        model = Whisper(dims)
        state_dict = checkpoint["model_state_dict"]

        for name, parameter in model.named_parameters(): # weight_alpha must be trained
            if name.endswith("weight_alpha") or name.startswith("decoder"): 
                parameter.requires_grad = True 
            else:
                parameter.requires_grad = False

        state_dict["encoder.positional_embedding"] = sinusoids(n_audio_ctx, n_state) #state_dict["encoder.positional_embedding"][:n_audio_ctx] # PE is right?
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        return model
    else:
        raise ValueError('Student\'s checkpoint path must be specified.')

# def build_student(checkpoint: str=None, bit=8):
#     if checkpoint is not None:
#         logger.warning(f'Build teacher model from {checkpoint}')
#         checkpoint = torch.load(checkpoint, map_location=device)
        
#         model1 = torch.load(checkpoint['hparams']['student'])
#         dims = model1["dims"]
#         # import pdb
#         # pdb.set_trace()

#         # n_audio_ctx = 400 #500  
#         # dims["n_audio_ctx"] = n_audio_ctx
#         # n_state = dims["n_audio_state"]

#         dims = ModelDimensions(**dims)
#         model = Whisper(dims)
#         state_dict = checkpoint["model"] 
#         #state_dict["encoder.positional_embedding"] = sinusoids(n_audio_ctx, n_state) #state_dict["encoder.positional_embedding"][:n_audio_ctx] # PE is right?
#         model.load_state_dict(state_dict, strict=False)
#         for parameter in model.parameters():
#             parameter.requires_grad = False
#         return model
#     else:
#         raise ValueError('Teacher\'s checkpoint path must be specified.')


# def build_student(checkpoint: str=None):
#     if checkpoint is not None:
#         logger.warning(f'Build student model from {checkpoint}')
#         ckpt = torch.load(checkpoint, map_location='cpu')
#         student = build_model(ckpt['hparams'])
#         student.load_state_dict(ckpt['model'])
#         return student
#     else:
#         raise ValueError('Student\'s checkpoint path must be specified.')
# def build_student(model_params: Dict):
#     student = build_model(model_params)
#     return student

# def build_student(checkpoint: str=None):
#     if checkpoint is not None:
#         logger.warning(f'Build student model from {checkpoint}')
#         ckpt = torch.load(checkpoint, map_location=device)
#         model1 = torch.load(ckpt['hparams']['student'])
#         dims = model1["dims"]

#         dims = ModelDimensions(**dims)
#         model = Whisper(dims)
#         state_dict = ckpt["model"] 
#         model.load_state_dict(state_dict, strict=False)
#         return model
#     else:
#         raise ValueError('Student\'s checkpoint path must be specified.')

@add_model('TSWrapper')
class TSWrapper(Model):
    """ TS Wrapper for ordinary model.
    Parameters:
        params (Dict): A dict contains student's model parameters and teacher's checkpoint path.
    """
    def __init__(self, student: Dict, teacher: str=None, teacher_hidden_size=3072, student_hidden_size=2048, bit=8):
        super().__init__()
        self.teacher = build_teacher(teacher, bit)
        self.student = build_student(student, bit)

        self.linear_adapter = nn.Linear(teacher_hidden_size, student_hidden_size).to(device="cuda")
        self.linear_adapter1 = self.linear_adapter #nn.Linear(teacher_hidden_size, student_hidden_size)
        self.linear_adapter2 = nn.Linear(teacher_hidden_size // 4, student_hidden_size // 4).to(device="cuda")

    def forward(self, batch: Dict):
        
        dec_input_ids = batch['extra']['label'][0].tensor.to(device)
        dec_input_ids[dec_input_ids == -1] = 50257
        mel = batch['feat'].tensor.to(device)
        # import pdb
        # pdb.set_trace()
        ## 只能finetune decoder
        with torch.no_grad():
            teacher_output, teacher_hidden, teacher_hidden_weight = self.teacher(mel, dec_input_ids)
        teacher_output = teacher_output.to(device)
        ## teacher forward on cpu
        #student_output = self.student(batch)

        #teacher_output = self.linear_adapter(teacher_output) 
        ### student encoder fixed or not??
        with torch.no_grad():
            audio_features = self.student.encoder(mel)
        student_output, student_hidden, student_hidden_weight = self.student.decoder(dec_input_ids, audio_features)
        #student_output = self.student(mel, dec_input_ids)
        student_output = student_output.to(device)
        # import pdb
        # pdb.set_trace()
        #teacher_output = self.teacher.decode(batch)
        # if not isinstance(teacher_output, Field):
        #     raise TypeError(f'teacher\'s output must be Field, not {type(teacher_output)}')
        teacher_hidden_out = []
        for i in range(len(teacher_hidden)):
            teacher_hidden_out.append(self.linear_adapter(teacher_hidden[i]))
        # import pdb
        # pdb.set_trace()
        teacher_hidden_weight_out = []
        for i in range(len(teacher_hidden_weight)): 
            teacher_hidden_weight_out.append(self.linear_adapter2(self.linear_adapter1(teacher_hidden_weight[i].permute(1, 0)).permute(1, 0)))  #(4h2, h2).permute(1, 0) * (4h2, 4h1) -> (h2, 4h1).permute(1, 0) * (h2, h1) -> (4h1, h1)


        return student_output, teacher_output, student_hidden, teacher_hidden_out, student_hidden_weight, teacher_hidden_weight_out

    def grad_post_processing(self):
        self.student.grad_post_processing()

    def state_dict(self):
        return self.student.state_dict()

    def load_state_dict(self, state_dict: Union[Dict[str, Tensor], OrderedDict[str, Tensor]], strict: bool = False):
        self.student.load_state_dict(state_dict, strict=strict)

    def decode(self, batch: Dict):
        output = self.student(batch)
        return output

    