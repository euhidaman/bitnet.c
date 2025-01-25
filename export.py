"""
This script has functions and utilties for model export.
Basically, we have a bunch of versions of the model, and we
want to export them to .bin files to be read from and inferenced in C.

Among the "input" versions of PyTorch files/models:
- Official BitNet weights released by Meta
- Huggingface weights available on the hub
- bitnet.c (this repo) trained models

Among the "output" versions of .bin files:
- v0: Legacy files of the original bitnet.c repo (will eventually be DEPRECATED)
- v1-vN: Improved .bin files with a proper header, cache alignment, etc.

This script aspires to provide all of these conversions.
"""
import os
import gzip
import shutil
import struct
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from modeling_bitnet import BitnetConfig, BitnetModel

# -----------------------------------------------------------------------------
# common utilities

def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

def serialize_int8(file, tensor):
    """ writes one int8 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    b = struct.pack(f'{len(d)}b', *d)
    file.write(b)

def quantize_q80(w, group_size):
    """
    takes a tensor and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127]
    """
    assert w.numel() % group_size == 0
    ori_shape = w.shape
    w = w.float() # convert to float32
    w = w.reshape(-1, group_size)
    # find the max in each group
    wmax = torch.abs(w).max(dim=1).values
    # calculate the scaling factor such that float = quant * scale
    scale = wmax / 127.0
    # scale into range [-127, 127]
    quant = w / scale[:,None]
    # round to nearest integer
    int8val = torch.round(quant).to(torch.int8)
    # dequantize by rescaling
    fp32val = (int8val.float() * scale[:,None]).view(-1)
    fp32valr = fp32val.reshape(-1, group_size)
    # calculate the max error in each group
    err = torch.abs(fp32valr - w).max(dim=1).values
    # find the max error across all groups
    maxerr = err.max().item()
    return int8val, scale, maxerr

# -----------------------------------------------------------------------------
# legacy

def legacy_export(model, filepath):
    """ Original export of bitnet.c bin files, i.e. version v0 """
    out_file = open(filepath, 'wb')

    # first write out the header
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    p = model.params
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    # legacy format uses negative/positive vocab size as a shared classifier flag
    if not shared_classifier:
        p.vocab_size = -p.vocab_size
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                                    n_kv_heads, p.vocab_size, p.max_seq_len)
    out_file.write(header)

    # next write out the embedding weights
    serialize_fp32(out_file, model.tok_embeddings.weight)

    # now all the layers
    # attention weights
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention_norm.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wq.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wk.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wv.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wo.weight)
    # ffn weights
    for layer in model.layers:
        serialize_fp32(out_file, layer.ffn_norm.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.feed_forward.w1.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.feed_forward.w2.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.feed_forward.w3.weight)
    # final rmsnorm
    serialize_fp32(out_file, model.norm.weight)
    # freqs_cis
    serialize_fp32