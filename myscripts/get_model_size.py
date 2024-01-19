#!/usr/bin/env python3

import humanfriendly
import argparse
import torch

def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate model and configuration file for finetuning the mix-precision quantized model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("model_path", type=str)
    parser.add_argument("bit", type=str)
    return parser

parser = get_parser()
args = parser.parse_args()

state_dict = torch.load(args.model_path, map_location="cpu")
state_dict = state_dict['model']
#state_dict = state_dict['model_state_dict']

result = 0
result_quantable = 0
result_unquantable = 0
for key in state_dict:
    if not key.startswith('encoder') and key.endswith('.weight'):
        result_quantable += state_dict[key].numel() * int(args.bit)
        result += state_dict[key].numel() * 16 #32
    else:
        result += state_dict[key].numel() * 16 #32
        result_unquantable += state_dict[key].numel() * 16 #32
    #    print(key,"calculated with no quant",state_dict[key].numel() / 256)

print("Model Total Size: " + str(result) + " bits = " + str(result / 8 /(1024**2)) + "MB")#humanfriendly.format_size((result) / 8))
print("Model Quantable Size: " + str(result_quantable) + " bits = " + str(result_quantable / 8 /(1024**2)) + "MB") #humanfriendly.format_size((result_quantable) / 8))
print("Model Unquantable Size: " + str(result_unquantable) + " bits = " + str(result_unquantable / 8 /(1024**2)) + "MB") #humanfriendly.format_size((result_unquantable) / 8))
print("Quantization Model Total Size: " + str(result_quantable + result_unquantable) + " bits = " + str((result_quantable + result_unquantable) / 8 /(1024**2)) + "MB") #humanfriendly.format_size((result_quantable + result_unquantable) / 8))

