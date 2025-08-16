Various things used in my models.

``` python
import torch
import os
import pyworld as pw
from einops import rearrange, pack, unpack, repeat
import numpy as np
import torchaudio
import torch.nn.functional as F
from datasets import load_dataset, Audio
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple
import math
from math import sqrt
import matplotlib.pyplot as plt
from torch import nn, einsum
import torch.nn.init as init
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention
from functools import lru_cache
from subprocess import CalledProcessError, run
from math import gcd
from collections import namedtuple
from functools import partial, reduce

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

def valid(default_value, *items):
    for item in items:
        if item is not None:
            return item
    return default_value

def dict_to(d, device, dtype=dtype):
    return {k: v.to(device, dtype) if isinstance(v, torch.Tensor) else v 
            for k, v in d.items()}  

def have(a):
    return a is not None

def AorB(a, b):
    return a if have(a) else b

def shift(t, amount, mask = None):
    if amount == 0:
        return t

    if have(mask):
        t = t.masked_fill(~mask[..., None], 0.)
    return F.pad(t, (0, 0, amount, -amount), value = 0.)

def always(value):
    return lambda *args, **kwargs: value

def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val
        
def safe_div(n, d, eps = 1e-6):
    return n.div_(d + eps)

def lcm(*numbers):
    return int(reduce(lambda x, y: int((x * y) / gcd(x, y)), numbers, 1))

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_dtype():
    return torch.float32 if torch.cuda.is_available() else torch.float64

def tox():
    return {"device": get_device(), "dtype": get_dtype()}

def cos_xy(x: Tensor, y: Tensor) -> Tensor:
    out = F.softmax(torch.matmul(F.normalize(x, dim=-1), F.normalize(y, dim=-1).transpose(0, 1)) / math.sqrt(x.shape[-1]), dim=-1)
    return out

def cos_qkv(q: Tensor, k: Tensor, v: Tensor, mask) -> Tensor:
    out = torch.matmul(F.softmax(torch.matmul(torch.nn.functional.normalize(q, dim=-1), torch.nn.functional.normalize(k, dim=-1).transpose(-1, -2)) + mask, dim=-1), v)
    return out

def sinusoids(ctx, dims, theta=10000):
    tscales = torch.exp(-torch.log(torch.tensor(float(theta))) / (dims // 2 - 1) * torch.arange(dims // 2, device=device, dtype=torch.float32))
    scaled = torch.arange(ctx, device=device, dtype=torch.float32).unsqueeze(1) * tscales.unsqueeze(0)
    positional_embedding = nn.Parameter(torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1) , requires_grad=True)
    return positional_embedding

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dims, ctx):
        super().__init__()
        self.emb = nn.Embedding(ctx, dims)
    def forward(self, x):
        return self.emb(torch.arange(x.shape[1], device=device))[None, :, :]

class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dims, head, ctx, theta=10000):
        super().__init__()
        freq = (theta / 220.0) * 700 * (
            torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 8000/700)), 
                    (dims // head) // 2, device=theta.device, dtype=theta.dtype) / 2595) - 1) / 1000
        position = torch.arange(0, ctx, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer('emb', emb)
    def forward(self, x):
        return self.emb[None, :x.shape[1], :].to(x)

def pitch_bias(f0):
    f0_flat = f0.squeeze().float()
    f0_norm = (f0_flat - f0_flat.mean()) / (f0_flat.std() + 1e-8)
    f0_sim = torch.exp(-torch.cdist(f0_norm.unsqueeze(1), 
                                f0_norm.unsqueeze(1)))
    return f0_sim.unsqueeze(0).unsqueeze(0)

def theta_freqs(dims, head, theta):
    if theta.dim() == 0:
        theta = theta.unsqueeze(0)
    freq = (theta / 220.0) * 700 * (
        torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 8000/700)), 
                (dims // head) // 2, device=theta.device, dtype=theta.dtype) / 2595) - 1) / 1000
    return freq

def apply_radii(freqs, x, ctx):
    F = x.shape[0] / ctx
    idx = torch.arange(ctx, device=device)
    idx = (idx * F).long().clamp(0, x.shape[0] - 1)
    x = x[idx]
    return torch.polar(x.unsqueeze(-1), freqs)

class axial_freqs(nn.Module):
    def __init__(self, dims, head, ctx, theta=10000, spec_shape=[]):

        time_frames, freq_bins = spec_shape
        time_frames = time_frames
        freq_bins = freq_bins
        
        time_theta = 50.0
        time_freqs = 1.0 / (time_theta ** (torch.arange(0, dims, 4)[:(dims // 4)].float() / dims))
        self.register_buffer('time_freqs', time_freqs)
        
        freq_theta = 100.0
        freq_freqs = 1.0 / (freq_theta ** (torch.arange(0, dims, 4)[:(dims // 4)].float() / dims))
        self.register_buffer('freq_freqs', freq_freqs)

        t = torch.arange(ctx, device=device, dtype=dtype)
        t_x = (t % time_frames).float()
        t_y = torch.div(t, time_frames, rounding_mode='floor').float()
        freqs_x = torch.outer(t_x, time_freqs)
        freqs_y = torch.outer(t_y, freq_freqs)
        freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
        freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
        return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

def mel_scale_scalar(freq: float) -> float:
    return 1127.0 * math.log(1.0 + freq / 700.0)

def mel_scale(freq: Tensor) -> Tensor:
    return 1127.0 * (1.0 + freq / 700.0).log()

def trace_x(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        if isinstance(result, torch.Tensor):
            print(f"  {func.__name__} returned shape: {result.shape}")
        return result
    return wrapper

def track_x(new_x, operation=""):  # track_x(x, "x") 
    x_id = [id(new_x)]
    if new_x is None:
        return new_x
    cur_id = id(new_x)
    if cur_id != x_id[0]:
        print(f"x FLOW: {x_id[0]} → {cur_id} in {operation}")
        x_id[0] = cur_id
    else:
        print(f"x REUSE: {cur_id} in {operation}")
    return new_x

def track_xa(new_xa, operation=""): # track_xa(xa, "xa - decoder")
    xa_id = [id(new_xa)] if new_xa is not None else [None]
    if new_xa is None:
        return new_xa
    cur_id = id(new_xa)
    if cur_id != xa_id[0]:
        print(f"xa FLOW: {xa_id[0]} → {cur_id} in {operation}")
        xa_id[0] = cur_id
    else:
        print(f"xa REUSE: {cur_id} in {operation}")
    return new_xa

def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

def apply_rotory_pos_emb(q, k, sinu_pos):
    sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j = 2)
    sin, cos = sinu_pos.unbind(dim = -2)
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j = 2), (sin, cos))
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    return q, k

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class SLSTM(nn.Module):
    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = False, bias = True, batch_first = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers, bias, batch_first)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y
@staticmethod
def apply_rotary(x, freqs):
    x1 = x[..., :freqs.shape[-1]*2]
    x2 = x[..., freqs.shape[-1]*2:]
    orig_shape = x1.shape
    if x1.ndim == 2:
        x1 = x1.unsqueeze(0)
    x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
    x1 = torch.view_as_complex(x1) * freqs
    x1 = torch.view_as_real(x1).flatten(-2)
    x1 = x1.view(orig_shape)
    return torch.cat([x1.type_as(x), x2], dim=-1)

def scaled_relu(x, sequence_length):
    relu_output = torch.relu(x)
    return relu_output / sequence_length

def taylor_softmax(x, order=2):
    tapprox = 1.0
    for i in range(1, order + 1):
        factorial_i = torch.exp(torch.lgamma(torch.tensor(i + 1, dtype=torch.float32)))
        tapprox += x**i / factorial_i
    return tapprox / torch.sum(tapprox, dim=-1, keepdim=True)

def taylor_masked(x, mask, order=2):
    tapprox = torch.zeros_like(x)
    unmasked = x.masked_select(mask) 
    approx_values = 1.0 + unmasked
    for i in range(1, order + 1):
        factorial_i = torch.exp(torch.lgamma(torch.tensor(i + 1, dtype=torch.float32)))
        approx_values += unmasked**i / factorial_i
    tapprox.masked_scatter_(mask, approx_values)
    sum_approx = torch.sum(tapprox, dim=-1, keepdim=True)
    toutput = tapprox / (sum_approx + 1e-9) 
    toutput = toutput * mask 
    return toutput

def taylor_softmax2(x, mask=None, order=2):
    if mask is None:
        tapprox = 1.0 + x
        for i in range(1, order + 1):
            factorial_i = torch.exp(torch.lgamma(torch.tensor(i + 1, dtype=torch.float32)))
            tapprox += x**i / factorial_i
        return tapprox / torch.sum(tapprox, dim=-1, keepdim=True)

    else:
        tapprox = torch.zeros_like(x)
        unmasked = x.masked_select(mask)
        tapprox = 1.0 + unmasked
        for i in range(1, order + 1):
            factorial_i = torch.exp(torch.lgamma(torch.tensor(i + 1, dtype=torch.float32)))
            tapprox += unmasked**i / factorial_i

        tapprox_full = torch.zeros_like(x)
        tapprox_full.masked_scatter_(mask, tapprox)

        sum_approx = torch.sum(tapprox_full, dim=-1, keepdim=True)
        toutput = tapprox_full / (sum_approx + 1e-9)

        toutput = toutput * mask.float()
        return toutput

def taylor_softmax_2nd_order(x):
    exp_approx = 1 + x + (x**2) / 2
    return exp_approx / torch.sum(exp_approx, dim=-1, keepdim=True)

def taylor_softmax_approximation(x, order=2):
    if order == 0:
        return torch.ones_like(x) / x.size(-1) 
    elif order == 1:
        numerator = 1 + x
    elif order == 2:
        numerator = 1 + x + 0.5 * x**2
    else:
        raise NotImplementedError("Higher orders are not implemented yet.")
    denominator = torch.sum(numerator, dim=-1, keepdim=True)
    return numerator / denominator

def taylor_sine(x, order=5):
    result = torch.zeros_like(x)
    for i in range(order + 1):
        if i % 2 == 1:  
            term = x**i / torch.exp(torch.lgamma(torch.tensor(i + 1, dtype=torch.float32)))
            if (i // 2) % 2 == 1: 
                result -= term
            else:
                result += term
    return result

def taylor_cosine(x, order=5):
    result = torch.zeros_like(x)
    for i in range(order + 1):
        if i % 2 == 0:  
            term = x**i / torch.exp(torch.lgamma(torch.tensor(i + 1, dtype=torch.float32)))
            if (i // 2) % 2 == 1: 
                result -= term
            else:
                result += term
    return result
def vectorized_taylor_sine(x, order=5):
    og_shape = x.shape
    x = x.flatten(0, -2)
    exponents = torch.arange(1, order + 1, 2, device=x.device, dtype=torch.float32)
    x_powers = x.unsqueeze(-1) ** exponents
    factorials = torch.exp(torch.lgamma(exponents + 1))
    signs = (-1)**(torch.arange(0, len(exponents), device=x.device, dtype=torch.float32))
    terms = signs * x_powers / factorials
    result = terms.sum(dim=-1)
    return result.view(og_shape)

def vectorized_taylor_cosine(x, order=5):
    og_shape = x.shape
    x = x.flatten(0, -2)
    exponents = torch.arange(0, order + 1, 2, device=x.device, dtype=torch.float32)
    x_powers = x.unsqueeze(-1) ** exponents
    factorials = torch.exp(torch.lgamma(exponents + 1))
    signs = (-1)**(torch.arange(0, len(exponents), device=x.device, dtype=torch.float32))
    terms = signs * x_powers / factorials
    result = terms.sum(dim=-1)
    return result.view(og_shape)

def taylor_softmax(x, order=2):
    tapprox = 1.0
    for i in range(1, order + 1):
        factorial_i = torch.exp(torch.lgamma(torch.tensor(i + 1, dtype=torch.float32)))
        tapprox += x**i / factorial_i
    return tapprox / torch.sum(tapprox, dim=-1, keepdim=True)

def hz_to_midi(hz): # Converts Hz to MIDI note number. Handles 0 Hz as unvoiced.
    if hz == 0:
        return 0  # Special value for unvoiced
    return 12 * np.log2(hz / 440) + 69

def midi_to_pitch_class(midi_note): # Converts MIDI note number to pitch class (0-11).
    if midi_note == 0:
        return -1 # Represents unvoiced
    return int(round(midi_note)) % 12

def pcp(pitch_hz, num_pitch_classes=12):  # Each frame is a vector representing the strength of each pitch class. 
    pcp = torch.zeros(len(pitch_hz), num_pitch_classes)
    for i, hz in enumerate(pitch_hz):
        midi_note = hz_to_midi(hz)
        pitch_class = midi_to_pitch_class(midi_note)
        if pitch_class != -1: # If it's a voiced frame
            pcp[i, pitch_class] = 1 # Simple binary presence. Could be weighted by confidence.
    return pcp

def one_hot_pitch(pitch_hz, min_hz=50, max_hz=1000, num_bins=200): # Each bin represents a specific frequency range. 
    pitch_bins = np.linspace(min_hz, max_hz, num_bins + 1)
    one_hot = torch.zeros(len(pitch_hz), num_bins)
    for i, hz in enumerate(pitch_hz):
        if hz > 0:
            bin_idx = np.digitize(hz, pitch_bins) - 1  # Find the bin for the cur pitch
            if 0 <= bin_idx < num_bins:
                one_hot[i, bin_idx] = 1
    return one_hot

def gaussian_pitch(pitch_hz, min_hz=50, max_hz=1000, num_bins=200, sigma=1.0): # Pitch as a Gaussian distribution across frequency bins.
    pitch_bins_hz = np.linspace(min_hz, max_hz, num_bins)
    gaussian = torch.zeros(len(pitch_hz), num_bins)

    for i, hz in enumerate(pitch_hz):
        if hz > 0:
            midi_note = hz_to_midi(hz)                 # Calculate the bin index for the cur pitch
            midi_min = hz_to_midi(min_hz)                 # Map MIDI notes to the bin scale
            midi_max = hz_to_midi(max_hz)
            bin_idx_float = (midi_note - midi_min) / (midi_max - midi_min) * (num_bins - 1)

            for bin_j in range(num_bins):# Create a Gaussian distribution around the pitch
                bin_center_midi = midi_min + (bin_j / (num_bins - 1)) * (midi_max - midi_min) # Calculate the center of the bin in MIDI
                gaussian[i, bin_j] = torch.exp(-torch.tensor((midi_note - bin_center_midi)**2 / (2 * sigma**2)))
            gaussian[i] /= gaussian[i].sum() # Normalize each row to sum to 1 (optional, depends on your needs)
            
    return gaussian

def crepe_predict(audio, sample_rate, viterbi=False):
    import torchcrepe
    audio = audio.numpy().astype(np.float32)
    time, frequency, confidence, activation = torchcrepe.predict(
        audio, sample_rate=sample_rate, viterbi=viterbi)
    crepe_time = torch.from_numpy(time)
    crepe_frequency = torch.from_numpy(frequency)
    crepe_confidence = torch.from_numpy(confidence)
    crepe_activation = torch.from_numpy(activation)
    return crepe_time, crepe_frequency, crepe_confidence, crepe_activation


def rbf_scores(q, k, rbf_sigma=1.0, rbf_ratio=0.0):
    dot_scores = torch.matmul(q, k.transpose(-1, -2))
    if rbf_ratio <= 0.0:
        return dot_scores
    q_norm = q.pow(2).sum(dim=-1, keepdim=True)
    k_norm = k.pow(2).sum(dim=-1, keepdim=True)
    qk = torch.matmul(q, k.transpose(-1, -2))
    dist_sq = q_norm + k_norm.transpose(-1, -2) - 2 * qk
    rbf_scores = torch.exp(-dist_sq / (2 * rbf_sigma**2))
    return (1 - rbf_ratio) * dot_scores + rbf_ratio * rbf_scores

def sliding_window_mask(q_len, k_len, window, device):
    idxs = torch.arange(q_len, device=device).unsqueeze(1)
    jdxs = torch.arange(k_len, device=device).unsqueeze(0)
    mask = (jdxs >= (idxs - window + 1)) & (jdxs <= idxs)
    return mask.float()

def mask_win(text_ctx, aud_ctx):
    mask = torch.tril(torch.ones(text_ctx, text_ctx, device=device, dtype=dtype), diagonal=0)
    audio_mask = torch.tril(torch.ones(text_ctx, aud_ctx - text_ctx, device=device, dtype=dtype))
    full_mask = torch.cat([mask, audio_mask], dim=-1)
    return full_mask

def maskc(ctx, device):
    return torch.tril(torch.ones(ctx, ctx, device=device, dtype=dtype), diagonal=0)

def attention_mask(batch_size, ctx, is_causal=True, padding_mask=None, device=None):
    if is_causal:
        mask = torch.triu(torch.ones((ctx, ctx), device=device), diagonal=0)
        mask = mask.expand(batch_size, 1, ctx, ctx)
    else:
        mask = torch.zeros((batch_size, 1, ctx, ctx), device=device)
    if padding_mask is not None:
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2).bool()
        mask = (mask.bool() | (~padding_mask)).float()
    return mask

def calculate_attention(q, k, v, mask=None, temp=1.0):
    scaled_q = q
    if temp != 1.0 and temp > 0:
        scaled_q = q * (1.0 / temp)**.5
    out = scaled_dot_product_attention(scaled_q, k, v, is_causal=mask is not None and q.shape[1] > 1)        
    return out

def calculate_attentionb(q_norm, k_norm, v_iter, mask=None, temp=1.0):
    d_k = q_norm.size(-1)
    scores = torch.matmul(q_norm, k_norm.transpose(-2, -1)) / (torch.sqrt(torch.tensor(d_k, dtype=torch.float32)) / temp)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, v_iter)
    return output

class LocalOut(nn.Module):
    def __init__(self, dims: int, head: int):
        super().__init__()
        self.head_dim = dims // head
        self.dims = dims
        self.q_module = nn.Linear(self.head_dim, self.head_dim)
        self.k_module = nn.Linear(self.head_dim, self.head_dim)
        self.v_module = nn.Linear(self.head_dim, self.head_dim)
        self.o_proj = nn.Linear(self.head_dim, self.head_dim)

    def _reshape_to_output(self, attn_output: Tensor) -> Tensor:
        batch, _, ctx, _ = attn_output.shape
        return attn_output.transpose(1, 2).contiguous().view(batch, ctx, self.dims)      

def qkv_init(dims, head):
    head_dim = dims // head
    q = nn.Linear(dims, dims)
    k = nn.Linear(dims, dims)
    v = nn.Linear(dims, dims)
    o = nn.Linear(dims, dims)
    lna = nn.LayerNorm(dims)
    lnb = nn.LayerNorm(dims)
    lnc = nn.LayerNorm(head_dim)
    lnd = nn.LayerNorm(head_dim)
    return q, k, v, o, lna, lnb, lnc, lnd

def shape(dims, head, q, k, v):
    batch_size = q.shape[0]
    ctx_q = q.shape[1]
    ctx_kv = k.shape[1]
    head_dim = dims // head

    q = q.view(batch_size, ctx_q, head, head_dim).transpose(1, 2)
    k = k.view(batch_size, ctx_kv, head, head_dim).transpose(1, 2)
    v = v.view(batch_size, ctx_kv, head, head_dim).transpose(1, 2)
    return q, k, v

def qkv(dims, head, q, k, v, x, xa):
    head_dim = dims // head
    scale = head_dim ** -0.25
    q = q(x) * scale
    k = k(xa) * scale
    v = v(xa)
    batch, ctx, dims = x.shape
    def _shape(tensor):
        return tensor.view(batch, ctx, head, head_dim).transpose(1, 2).contiguous()
    return _shape(q), _shape(k), _shape(v)

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_ctxgth, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_ctxgth, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out

def get_activation(act: str) -> nn.Module:

    act_map = {
        "gelu": nn.GELU(), 
        "relu": nn.ReLU(), 
        "sigmoid": nn.Sigmoid(), 
        "tanh": nn.Tanh(), 
        "swish": nn.SiLU(), 
        "tanhshrink": nn.Tanhshrink(), 
        "softplus": nn.Softplus(), 
        "softshrink": nn.Softshrink(), 
        "leaky_relu": nn.LeakyReLU(), 
        "elu": nn.ELU()
    }
    return act_map.get(act, nn.GELU())

def get_generation_config(param):
    return GenerationConfig(
        max_length=param.text_ctx,
        pad_token_id=getattr(param, "pad_token_id", 0),
        bos_token_id=getattr(param, "bos_token_id", 1),
        eos_token_id=getattr(param, "eos_token_id", 2),
        do_sample=False,
        num_beams=1,
        early_stopping=False,
        length_penalty=1.0,
        no_repeat_ngram_size=0,
        repetition_penalty=1.0,
        temperature=1.0,
        decoder_start_token_id=1,
        is_multilingual=False,
        use_cache=False,
        return_timestamps=False)

class FEncoder(nn.Module):
    def __init__(self, mels, dims, head, act="relu"):
        super().__init__()
        
        self.head = head
        self.head_dim = dims // head  

        self.dims = dims
        self.mels = mels
        act_fn = get_activation(act)

        self.encoder = nn.Sequential(
           nn.Conv1d(mels, dims, kernel_size=3, stride=1, padding=1), act_fn,
           nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1), act_fn,
           nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), act_fn)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)        
        x = self.encoder(x).permute(0, 2, 1)#.contiguous().to(device=device, dtype=dtype)
        return x

class WEncoder(nn.Module):
    def __init__(self, input_dims, dims, head, act="relu", downsample=True, target_length=None):
        super().__init__()
        
        self.head = head
        self.head_dim = dims // head
        self.dims = dims
        act_fn = get_activation(act)
        self.target_length = target_length

        if downsample:
            self.encoder = nn.Sequential(
                nn.Conv1d(1, dims, kernel_size=127, stride=64, bias=False),
                nn.Conv1d(dims, 2 * dims, kernel_size=7, stride=3),
                nn.Conv1d(2 * dims, dims, kernel_size=3, stride=2),
                nn.GroupNorm(num_groups=1, num_channels=dims, eps=1e-5))
        else:
            self.encoder = nn.Sequential(
               nn.Conv1d(input_dims, dims//4, kernel_size=15, stride=4, padding=7), act_fn,
               nn.Conv1d(dims//4, dims//2, kernel_size=7, stride=2, padding=3), act_fn,
               nn.Conv1d(dims//2, dims, kernel_size=5, stride=2, padding=2), act_fn)
                
    def _get_length(self, input_lengths: torch.LongTensor):
        conv1_length = int((input_lengths - 127) / 64 + 1)
        conv2_length = int((conv1_length - 7) / 3 + 1)
        conv3_length = int((conv2_length - 3) / 2 + 1)
        return conv3_length
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = self.encoder(x).permute(0, 2, 1).contiguous()
        if self.target_length and x.shape[1] != self.target_length:
            x = F.adaptive_avg_pool1d(x.transpose(1, 2), self.target_length).transpose(1, 2)
        return x

class PEncoder(nn.Module):
    def __init__(self, input_dims, dims, head, act="relu", attend_pitch=False):
        super().__init__()
        
        self.head = head
        self.head_dim = dims // head
        self.dims = dims
   
        act_fn = get_activation(act)
        self.attend_pitch = attend_pitch

        if self.attend_pitch:
            self.q, self.k, self.v, self.o, self.scale = qkv_init(dims, head)
            self.mlp = nn.Sequential(
                nn.Linear(dims, dims),
                nn.ReLU(),
                nn.Linear(dims, dims),
            )
        else:
            self.q, self.k, self.v, self.o, self.scale = None, None, None, None, None
            self.mlp = None

        self.pitch_encoder = nn.Sequential(
            nn.Conv1d(1, dims, kernel_size=3, stride=1, padding=1), act_fn,
            nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1), act_fn,
            nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), act_fn)
        
    def rope_to(self, x):
        batch, ctx, dims = x.shape
        x = x.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
        freqs = self.rope(ctx)
        x = self.rope.apply_rotary(x, freqs)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, ctx, dims)
        return x.squeeze(0)
        
    def forward(self, x, xa=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = self.pitch_encoder(x).permute(0, 2, 1).contiguous()
    
        if self.mlp is not None:
            x = self.mlp(x)

        if self.attend_pitch:
            if xa is not None:
                q, k, v = qkv(self.q, self.k, self.v, x=xa, xa=x, head=self.head)
                out, _ = calculate_attention(q, k, v, mask=None, temperature=1.0, is_causal=True)
                x = x + out
        return x

class feature_encoder(nn.Module):
    def __init__(self, mels, input_dims, dims, head, layer, act, features, feature=None, use_rope=False, spec_shape=None, debug=[], attend=False, target_length=None):
        """
        Feature encoder for audio processing.
        """
        super().__init__()

        self.dims = dims
        self.head = head
        self.head_dim = dims // head  
        self.dropout = 0.01 
        self.use_rope = use_rope
        self.attend = attend
        self.target_length = target_length
        self.feature = feature

        self.debug = debug
        act_fn = get_activation(act)

        if self.attend:
            self.mlp = nn.Sequential(nn.Linear(dims, dims), nn.ReLU(), nn.Linear(dims, dims))
        else:
            self.q, self.k, self.v, self.o, self.scale = None, None, None, None, None
            self.mlp = None

        self.spectrogram = nn.Sequential(
           nn.Conv1d(mels, dims, kernel_size=3), act_fn,
           nn.Conv1d(dims, dims, kernel_size=3), act_fn,
           nn.Conv1d(dims, dims, kernel_size=3, groups=dims), act_fn)

        self.waveform = nn.Sequential(
           nn.Conv1d(1, dims//4, kernel_size=15, stride=4, padding=7), act_fn,
           nn.Conv1d(dims//4, dims//2, kernel_size=7, stride=2, padding=3), act_fn,
           nn.Conv1d(dims//2, dims, kernel_size=5, stride=2, padding=2), act_fn)

        self.pitch = nn.Sequential(
           nn.Conv1d(1, dims, kernel_size=7, stride=1, padding=3), act_fn,
           nn.Conv1d(dims, dims, kernel_size=5, stride=1, padding=2), act_fn,
           nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), act_fn)

        self.positional = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)
        self.norm = RMSNorm(dims)

    def rope(self, x, xa=None, mask=None, feats=None, feature=None, layer=None):
        if isinstance(x, int):
            ctx = x 
        elif isinstance(x, torch.Tensor):
            ctx = x.shape[1] if x.dim() > 1 else x.shape[0]
            batch, ctx, dims = x.shape[0], ctx, x.shape[-1]

            x = x.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
        freqs = self.rope(ctx, feats=feats, feature=feature, layer=layer)
        x = self.rope.apply_rotary(x, freqs)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, ctx, dims)
        return x

    def mel_scalar(self, freq: float) -> float:
        return 1127.0 * math.log(1.0 + freq / 700.0)

    def forward(self, x, xa=None, mask=None, feats=None, feature=None, layer=None, max_tscale=36000):
        target_length = x.shape[1] if self.target_length is None else self.target_length

        if feature == "pitch":
            xp = x.clone()
            enc_dict = feats if feats is not None else {}
            enc_dict = dict(enc_dict)  
            enc_dict["f0"] = xp
            feats = enc_dict
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = self.pitch(x).permute(0, 2, 1)
  
        if feature == "phase":
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = self.pitch(x).permute(0, 2, 1)

        if feature == "waveform":
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = self.waveform(x).permute(0, 2, 1)
            if target_length and x.shape[1] != self.target_length:
                x = F.adaptive_avg_pool1d(x.transpose(1, 2), target_length).transpose(1, 2)
        
        if feature == "harmonics":
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = self.spectrogram(x).permute(0, 2, 1)

        if feature == "aperiodic":
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = self.spectrogram(x).permute(0, 2, 1)            

        if feature == "spectrogram":
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = self.spectrogram(x).permute(0, 2, 1)

        if self.use_rope:
            x = x + self.positional(x.shape[1], x.shape[-1], max_tscale).to(device, dtype)
            x = self.rope(x=x, xa=None, mask=None, feats=feats, feature=feature, layer=layer)
        else:
            max_tscale = x.shape[1] * 1000 if max_tscale is None else max_tscale
            x = x + self.positional(x.shape[1], x.shape[-1], max_tscale).to(device, dtype)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.norm(x)

        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.norm(x)
        return x

class OneShot(nn.Module):
    def __init__(self, dims: int, head: int, scale: float = 0.3, features: Optional[List[str]] = None):
        super().__init__()
        if features is None:    
            features = ["spectrogram", "waveform", "pitch", "aperiodic", "harmonics"]
        self.head = head
        self.head_dim = dims // head
        self.scale = 1.0 // len(features) if features else scale

        self.q = Linear(dims, dims)
        self.k = Linear(dims, dims)

    def forward(self, x: Tensor, xa: Tensor, feature=None) -> Tensor | None:
        B, L, D = x.shape
        K = xa.size(1)
        q = self.q(x).view(B, L, self.head, self.head_dim).transpose(1,2)
        k = self.k(xa).view(B, K, self.head, self.head_dim).transpose(1,2)
        bias = (q @ k.transpose(-1, -2)) * self.scale / math.sqrt(self.head_dim)
        return bias

class curiosity(nn.Module):
    def __init__(self, d, h, bias=True):
        super().__init__()
        self.h  = h
        self.dh = d // h
        self.qkv = nn.Linear(d, d * 3, bias=bias)
        self.qkv_aux = nn.Linear(d, d * 3, bias=bias)
        self.o  = nn.Linear(d, d, bias=bias)
        self.g  = nn.Parameter(torch.zeros(h))

    def split(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.h, self.dh).transpose(1, 2)

    def merge(self, x):
        b, h, t, dh = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * dh)

    def forward(self, x, xa, mask=None):

        q, k, v   = self.qkv(x).chunk(3, -1)
        qa, ka, va = self.qkv_aux(xa).chunk(3, -1)
        q, k, v   = map(self.split, (q, k, v))
        qa, ka, va = map(self.split, (qa, ka, va))
        dots      = (q @ k.transpose(-2, -1)) / self.dh**0.5
        dots_aux  = (q @ ka.transpose(-2, -1)) / self.dh**0.5
        if mask is not None: dots = dots.masked_fill(mask, -9e15)
        p   = dots.softmax(-1)
        pa  = dots_aux.softmax(-1)
        h_main = p  @ v
        h_aux  = pa @ va
        g = torch.sigmoid(self.g).view(1, -1, 1, 1)
        out = self.merge(h_main * (1 - g) + h_aux * g)
        return self.o(out)

class PositionalEncoding(nn.Module):
    def __init__(self, dims, ctx):
        super(PositionalEncoding, self).__init__()
        self.dims = dims
        self.ctx = ctx
        self.pe = self.get_positional_encoding(max_ctx=ctx)

    def get_positional_encoding(self, max_ctx):
        pe = torch.zeros(max_ctx, self.dims)
        position = torch.arange(0, max_ctx, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dims, 2, dtype=torch.float32)
            * (-math.log(10000.0) / self.dims)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe.to(device)

    def forward(self, x):
        ctx = x.size(1)
        pe = self.pe[:, :ctx, :]
        x = x * math.sqrt(self.dims)
        x = x + pe
        return x

class Conv2d(nn.Conv2d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias) -> Tensor:
        return super()._conv_forward(x, weight.to(x.device, x.dtype), None if bias is None else bias.to(x.device, x.dtype))

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)
    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)
    
class RMSNorm(nn.Module):
    def __init__(self, dims: Union[int, Tensor, List, Tuple], 
                 eps = 1e-8, elementwise_affine = True):
        super(RMSNorm, self).__init__()
        if isinstance(dims, int):
            self.normalized_shape = (dims,)
        else:
            self.normalized_shape = tuple(dims)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape))
            init.ones_(self.weight)  
        else:
            self.register_parameter("weight", None)
    def forward(self, x):
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
    
def LayerNorm(x: Tensor, normalized_shape: Union[int, Tensor, List, Tuple],
               weight: Optional[Tensor] = None, bias: Optional[Tensor] = None,
               eps: float = 1e-5) -> Tensor:
    return F.layer_norm(x, normalized_shape, weight, bias, eps)


class SelfCriticalRL(nn.Module):
    def __init__(self, model, tokenizer, reward_fn):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn

    def forward(self, input_ids, features, labels=None, max_len=128, feature_name="spectrogram"):

        with torch.no_grad():
            greedy_ids = self.model.generate(input_ids=input_ids, **{feature_name: features}, max_length=max_len)
        greedy_text = [self.tokenizer.decode(ids) for ids in greedy_ids]
        sampled_ids = self.model.generate(input_ids=input_ids, **{feature_name: features}, max_length=max_len, do_sample=True, top_k=5)
        sampled_text = [self.tokenizer.decode(ids) for ids in sampled_ids]
        
        rewards = []
        baseline = []
        for s, g, ref in zip(sampled_text, greedy_text, labels):
            ref_text = self.tokenizer.decode(ref)
            rewards.append(self.reward_fn(s, ref_text))
            baseline.append(self.reward_fn(g, ref_text))
        rewards = torch.tensor(rewards, device=device, dtype=torch.float)
        baseline = torch.tensor(baseline, device=device, dtype=torch.float)
        advantage = rewards - baseline
        logits = self.model(input_ids=sampled_ids, **{feature_name: features})["logits"]
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs_seq = torch.gather(log_probs, 2, sampled_ids.unsqueeze(-1)).squeeze(-1)
        log_probs_sum = log_probs_seq.sum(dim=1)
        loss = -(advantage * log_probs_sum).mean()
        return loss

class SelfTrainingModule(nn.Module):
    def __init__(self, model, tokenizer, quality_fn=None, threshold=0.8):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.quality_fn = quality_fn
        self.threshold = threshold

    def generate_pseudo_labels(self, unlabeled_batch, features, max_len=128, feature_name="spectrogram"):
        with torch.no_grad():
            pred_ids = self.model.generate(input_ids=unlabeled_batch, **{feature_name: features}, max_length=max_len)

        if self.quality_fn is not None:
            quality_scores = self.quality_fn(pred_ids, self.model, features)
            mask = quality_scores > self.threshold
            pred_ids = pred_ids[mask]
        return pred_ids

    def forward(self, unlabeled_batch, features, max_len=128, feature_name="spectrogram"):
        pseudo_labels = self.generate_pseudo_labels(unlabeled_batch, features, max_len, feature_name=feature_name)
        logits = self.model(input_ids=unlabeled_batch, **{feature_name: features}, labels=pseudo_labels)["logits"]
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]), pseudo_labels.view(-1), ignore_index=0)
        return loss

def confidence_indicator(pred_ids, model, features):
    with torch.no_grad():
        logits = model(input_ids=pred_ids, **features)["logits"]
    probs = torch.softmax(logits, dim=-1)
    max_probs, _ = probs.max(dim=-1)
    return max_probs.mean(dim=1)

def wer_reward(hyp, ref):
    hyp_words = hyp.split()
    ref_words = ref.split()
    d = [[0] * (len(ref_words)+1) for _ in range(len(hyp_words)+1)]
    for i in range(len(hyp_words)+1):
        d[i][0] = i
    for j in range(len(ref_words)+1):
        d[0][j] = j
    for i in range(1, len(hyp_words)+1):
        for j in range(1, len(ref_words)+1):
            if hyp_words[i-1] == ref_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
    wer = d[-1][-1] / max(1, len(ref_words))
    return -wer

def clean_ids(ids, pad_token_id=0, bos_token_id=1, eos_token_id=2):
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return [int(id) for id in ids if id != -100 and id != pad_token_id and id != bos_token_id and id != eos_token_id]

def clean_batch(batch_ids, pad_token_id=0, bos_token_id=1, eos_token_id=2):
    return [clean_ids(seq, pad_token_id, bos_token_id, eos_token_id) for seq in batch_ids]

def setup_tokenizer(dir: str):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(f"{dir}")
    orig_encode = tokenizer.encode
    orig_decode = tokenizer.decode

    def enc(text, add_special_tokens=True):
        ids = orig_encode(text).ids
        if not add_special_tokens:
            sp_ids = [tokenizer.token_to_id(t) for t in ["<PAD>", "<BOS>", "<EOS>"]]
            ids = [id for id in ids if id not in sp_ids]
        return ids

    def bdec(ids_list, pad_token_id=0, bos_token_id=1, eos_token_id=2, skip_special_tokens=True):
        results = []
        if isinstance(ids_list, torch.Tensor):
            ids_list = ids_list.tolist()
        elif isinstance(ids_list, np.ndarray):
            ids_list = ids_list.tolist()
        for ids in ids_list:
            ids = [int(id) for id in ids if id not in (pad_token_id, bos_token_id, eos_token_id, -100)]
            results.append(orig_decode(ids))
        return results

    def dec(ids, pad_token_id=0, bos_token_id=1, eos_token_id=2):
        ids = [int(id) for id in ids if id not in (pad_token_id, bos_token_id, eos_token_id, -100)]
        return orig_decode(ids)

    def save_pretrained(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        tokenizer.save(f"{save_dir}/tokenizer.json")

    tokenizer.encode = enc
    tokenizer.batch_decode = bdec
    tokenizer.decode = dec
    tokenizer.save_pretrained = save_pretrained
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    return tokenizer

def tokenize_feature(audio, labels):
    if isinstance(audio, torch.Tensor):
        if audio.dim() == 1:
            ctx = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            ctx = audio.unsqueeze(1)
        elif audio.dim() == 3:
            ctx = audio
    target_length = len(labels)
    current_length = ctx.shape[-1]
    if current_length > target_length:
        tokens = F.adaptive_avg_pool1d(ctx, target_length)
    else:
        tokens = F.interpolate(ctx, size=target_length, mode='linear', align_corners=False)
    return tokens

def load_wave(wave_data, sample_rate=16000):
    if isinstance(wave_data, str):
        waveform, sample_rate = torchaudio.load(uri=wave_data, normalize=True, backend="ffmpeg")
    elif isinstance(wave_data, dict):
        waveform = torch.tensor(data=wave_data["array"]).float()
        sample_rate = wave_data["sampling_rate"]
    else:
        raise TypeError("Invalid wave_data format.")
    return waveform

def world_to_mel(sp, ap, sample_rate=16000, n_mels=128):
    import librosa
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=1024, n_mels=n_mels)
    mel_basis = torch.from_numpy(mel_basis).float()
    sp_mel = torch.matmul(sp, mel_basis.T)
    ap_mel = torch.matmul(ap, mel_basis.T)
    return sp_mel, ap_mel

def spectrogram(audio, sample_rate=16000, n_fft=1024, hop_length=256, window_fn=torch.hann_window):
    torch_windows = {
        'hann': torch.hann_window,
        'hamming': torch.hamming_window,
        'blackman': torch.blackman_window,
        'bartlett': torch.bartlett_window,
        'ones': torch.ones,
        None: torch.ones,
    }

    if isinstance(window_fn, str):
        window_fn = torch_windows[window_fn]
    if window_fn is None:
        window_fn = torch.ones(n_fft)
    if isinstance(window_fn, torch.Tensor):
        window_fn = window_fn.to(device)
    return torchaudio.functional.spectrogram(
        audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
        window=window_fn, center=True, pad_mode="reflect", power=1.0)

def exact_div(x, y):
    assert x % y == 0
    return x // y

def load_audio(file: str, sr: int = 16000):

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def pad_or_trim(array, length: int = 30, *, axis: int = -1):  # N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk 
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device))

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)
    return array

@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"
    filters_path = "D:/newmodel/mod6/mel_filters.npz"
    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)

def mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 128,
    n_fft = 400,
    hop_length = 160,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):

    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(n_fft).to(audio.device)
    stft = torch.stft(audio, n_fft, hop_length, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    spectrogram_tensor = (log_spec + 4.0) / 4.0
    return spectrogram_tensor

def audio_token(audio, labels, sample_rate=16000, hop_length=256, strides=1):

    frames_per_second = exact_div(sample_rate, hop_length)  
    # key = F.softmax(torch.matmul(F.normalize(x, p=2, dim=-1), F.normalize(self.mkey, p=2, dim=-1).transpose(0, 1)) / math.sqrt(x.shape[-1]), dim=-1)
    tokens_per_second = exact_div(sample_rate,  hop_length * strides) # audio "tokens"  20ms per audio token (exampe)
    return tokens_per_second

def harmonics_and_aperiodics(audio, f0, t, sample_rate):
    import pyworld as pw
    wav_np = audio.numpy().astype(np.float64)
    sp = pw.cheaptrick(wav_np, f0, t, sample_rate, fft_size=256)
    ap = pw.d4c(wav_np, f0, t, sample_rate, fft_size=256)
    harmonic_tensor = torch.from_numpy(sp)
    aperiodic_tensor = torch.from_numpy(ap)
    harmonic_tensor = harmonic_tensor[:, :128].contiguous().T
    aperiodic_tensor = aperiodic_tensor[:, :128].contiguous().T
    harmonic_tensor = torch.where(harmonic_tensor == 0.0, torch.zeros_like(harmonic_tensor), harmonic_tensor / 1.0)
    aperiodic_tensor = torch.where(aperiodic_tensor == 0.0, torch.zeros_like(aperiodic_tensor), aperiodic_tensor / 1.0)
    return harmonic_tensor, aperiodic_tensor

def mfcc(audio, sample_rate, n_mels, n_fft, hop_length, window_fn=torch.hann_window):
    transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mels,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "window_fn": window_fn,
            "n_mels": n_mels,
            "center": True,
            "pad_mode": "reflect",
            "norm": None,
            "mel_scale": "htk",
        }
    )
    mfcc_tensor = transform(audio)
    return mfcc_tensor

def safe_div(n, d, eps = 1e-6):
    return n.div_(d + eps)

def lcm(*numbers):
    return int(reduce(lambda x, y: int((x * y) / gcd(x, y)), numbers, 1))

def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)

def expand_dim(t, dim, k, unsqueeze=True):
    if unsqueeze:
        t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def pitch_tokens(audio, labels, sample_rate=16000, hop_length=160, mode="mean", audio_bos=None):
    f0, t = pw.dio(audio.numpy().astype(np.float64), sample_rate, frame_period=hop_length / sample_rate * 1000)
    f0 = pw.stonemask(audio.numpy().astype(np.float64), f0, t, sample_rate)
    duration = len(audio) / sample_rate
    T = len(labels)
    tok_dur = duration / T
    starts = torch.arange(T) * tok_dur
    ends = starts + tok_dur
    start = torch.searchsorted(torch.from_numpy(t), starts, side="left")
    end = torch.searchsorted(torch.from_numpy(t), ends, side="right")
    ptok = torch.zeros(T, dtype=torch.float32)
    for i in range(T):
        lo, hi = start[i], max(start[i]+1, end[i])
        seg = torch.from_numpy(f0)[lo:hi]
        if mode == "mean":
            ptok[i] = seg.mean()
        elif mode == "median":
            ptok[i] = torch.median(seg)
        else:
            ptok[i] = seg[-1]
    ptok[ptok < 100.0] = 0.0
    bos_token = audio_bos if audio_bos is not None else (ptok[0] if len(ptok) > 0 else 0.0)
    tensor = torch.cat([torch.tensor([bos_token]), ptok])
    return torch.where(tensor == 0.0, torch.zeros_like(tensor), (tensor - 71.0) / (500.0 - 71.0))

def extract_features(batch, tokenizer, waveform=False, spec=False, pitch_tokens=False, pitch=False, harmonics=False, sample_rate=16000, hop_length=256, mode="mean", debug=False, phase_mod=False, crepe=False, aperiodics=False, dummy=False, mels=128, n_fft= 1024):

    sample_rate = batch["audio"]["sampling_rate"]
    labels = tokenizer.encode(batch["transcription"])
    audio = load_wave(batch["audio"], sample_rate)
    # tokens = tokenize_feature(audio, labels)
        
    # pitch_tensor_hz = torchaudio.functional.detect_pitch_frequency(waveform, sample_rate)
    # if pitch_tensor_hz.dim() > 1 and pitch_tensor_hz.shape[0] == 1:
    #     pitch_tensor_hz = pitch_tensor_hz.squeeze(0) 
    # ctx = pitch_tensor_hz.shape[0]
    # print(f"Original pitch tensor shape from torchaudio: {pitch_tensor_hz.shape}")
    # pitch_hz_np = pitch_tensor_hz.numpy()
    # num_pitch_classes = 128
    # pcp_data = pcp(pitch_hz_np, num_pitch_classes=num_pitch_classes)
    # pcp_data = pcp_data.unsqueeze(0).transpose(1, 2) # (1, num_pitch_classes, ctx)
    # print(f"PCP data shape: {pcp_data.shape}")

    # embedding_dim = 512
    #nn.Conv1d_pcp = nn.Conv1d(in_channels=num_pitch_classes, out_channels=embedding_dim, kernel_size=3, stride=1, padding=1)
    # output_pcp =nn.Conv1d_pcp(pcp_data)
    # print(f"Conv1D output shape for PCP: {output_pcp.shape}")

    # =========================================================================
    # 2. One-Hot Encoding or Gaussian Distribution
    # =========================================================================
    # min_hz, max_hz, num_pitch_bins = 50, 1000, 200 # Example parameters
    # one_hot_data = one_hot_pitch(pitch_hz_np, min_hz=min_hz, max_hz=max_hz, num_bins=num_pitch_bins)
    # one_hot_data = one_hot_data.unsqueeze(0).transpose(1, 2) # (1, num_pitch_bins, ctx)
    # print(f"One-Hot pitch data shape: {one_hot_data.shape}")

    #nn.Conv1d_one_hot = nn.Conv1d(in_channels=num_pitch_bins, out_channels=embedding_dim, kernel_size=3, stride=1, padding=1)
    # output_one_hot =nn.Conv1d_one_hot(one_hot_data)
    # print(f"Conv1D output shape for One-Hot pitch: {output_one_hot.shape}")

    # gaussian_data = gaussian_pitch(pitch_hz_np, min_hz=min_hz, max_hz=max_hz, num_bins=num_pitch_bins, sigma=0.5)
    # gaussian_data = gaussian_data.unsqueeze(0).transpose(1, 2) # (1, num_pitch_bins, ctx)
    # print(f"Gaussian pitch data shape: {gaussian_data.shape}")

    #nn.Conv1d_gaussian = nn.Conv1d(in_channels=num_pitch_bins, out_channels=embedding_dim, kernel_size=3, stride=1, padding=1)
    # output_gaussian =nn.Conv1d_gaussian(gaussian_data)
    # print(f"Conv1D output shape for Gaussian pitch: {output_gaussian.shape}")

    # =========================================================================
    # 3. Learned Feature Extraction (nn.Embedding)
    # =========================================================================

    # min_midi = 21 # A0
    # max_midi = 108 # C8
    # num_pitch_categories = max_midi - min_midi + 1 + 1 # +1 for unvoiced at index 0

    # quantized_pitch_sequence = torch.zeros(ctx, dtype=torch.long)
    # for i, hz in enumerate(pitch_hz_np):
    #     if hz > 0:
    #         midi_note = hz_to_midi(hz)
    #         quantized_midi = int(round(np.clip(midi_note, min_midi, max_midi)))
    #         quantized_pitch_sequence[i] = quantized_midi - min_midi + 1

    # embedding_layer = nn.Embedding(num_embeddings=num_pitch_categories, embedding_dim=embedding_dim)
    # embedded_pitch = embedding_layer(quantized_pitch_sequence)
    # embedded_pitch = embedded_pitch.unsqueeze(0).transpose(1, 2) # (1, embedding_dim, ctx)
    # print(f"Embedded pitch data shape: {embedded_pitch.shape}")

    #nn.Conv1d_embedding = nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, stride=1, padding=1)
    # output_embedding =nn.Conv1d_embedding(embedded_pitch)
    # print(f"Conv1D output shape for embedded pitch: {output_embedding.shape}")
 

    if crepe:
        crepe_time, crepe_frequency, crepe_confidence, crepe_activation = crepe_predict(audio, sample_rate, viterbi=True)

    else:
        crepe_time = None
        crepe_frequency = None
        crepe_confidence = None
        crepe_activation = None

        # spectrogram_config = {
        #     "hop_length": 256,
        #     "f_min": 150,
        #     "f_max": 2000,
        #     "n_mels": 128,
        #     "n_fft": 1024,
        #     "sample_rate": 16000,
        #     "pad_mode": "constant",
        #     "center": True, 
        #     "power": 1.0,
        #     "window_fn": torch.hann_window,
        #     "mel_scale": "htk",
        #     "norm": None,
        #     "normalized": False,
        # }

        # spectrogram_config = {
        #     "hop_length": 1280,  # Increased significantly for very low temporal resolution
        #     "f_min": 500,        # Narrower range, especially eliminating some lower frequencies
        #     "f_max": 1500,       # Narrower range, limiting higher frequencies
        #     "n_mels": 8,         # Significantly reduced for very low frequency resolution
        #     "n_fft": 128,        # Reduced for lower frequency resolution (related to hop_length and sample_rate)
        #     "sample_rate": 8000, # Also reduced to lower the effective frequency range
        #     "pad_mode": "constant",
        #     "center": True,
        #     "power": 1.0,
        #     "window_fn": torch.hann_window,
        #     "mel_scale": "htk",
        #     "norm": None,
        #     "normalized": False,
        # }

        # spectrogram_config = {
        #     "hop_length": 320,
        #     "f_min": 150,
        #     "f_max": 2000,
        #     "n_mels": 64,
        #     "n_fft": 512,
        #     "sample_rate": 16000,
        #     "pad_mode": "constant",
        #     "center": True,
        #     "power": 1.0,
        #     "window_fn": torch.hann_window,
        #     "mel_scale": "htk",
        #     "norm": None,
        #     "normalized": False,
        # }

        # # Option 1: Reduce temporal resolution (increase hop_length)
        # spectrogram_config_v1 = {
        #     "hop_length": 480,  # Increased hop_length
        #     "f_min": 150,
        #     "f_max": 2000,
        #     "n_mels": 128,
        #     "n_fft": 1024,
        #     "sample_rate": 16000,
        #     "pad_mode": "constant",
        #     "center": True,
        #     "power": 1.0,
        #     "window_fn": torch.hann_window,
        #     "mel_scale": "htk",
        #     "norm": None,
        #     "normalized": False,
        # }

        # # Option 2: Reduce frequency resolution (decrease n_mels)
        # spectrogram_config_v2 = {
        #     "hop_length": 256,
        #     "f_min": 150,
        #     "f_max": 2000,
        #     "n_mels": 64,  # Decreased n_mels
        #     "n_fft": 1024,
        #     "sample_rate": 16000,
        #     "pad_mode": "constant",
        #     "center": True,
        #     "power": 1.0,
        #     "window_fn": torch.hann_window,
        #     "mel_scale": "htk",
        #     "norm": None,
        #     "normalized": False,
        # }

        # # Option 3: Reduce frequency resolution (decrease n_fft)
        # spectrogram_config_v3 = {
        #     "hop_length": 256,
        #     "f_min": 150,
        #     "f_max": 2000,
        #     "n_mels": 128,
        #     "n_fft": 512,  # Decreased n_fft
        #     "sample_rate": 16000,
        #     "pad_mode": "constant",
        #     "center": True,
        #     "power": 1.0,
        #     "window_fn": torch.hann_window,
        #     "mel_scale": "htk",
        #     "norm": None,
        #     "normalized": False,
        # }

        # # Option 4: Combined reduction (example)
        # spectrogram_config = {
        #     "hop_length": 320,
        #     "f_min": 150,
        #     "f_max": 2000,
        #     "n_mels": 64,
        #     "n_fft": 512,
        #     "sample_rate": 16000,
        #     "pad_mode": "constant",
        #     "center": True,
        #     "power": 1.0,
        #     "window_fn": torch.hann_window,
        #     "mel_scale": "htk",
        #     "norm": None,
        #     "normalized": False,
        # }

        # transform = torchaudio.transforms.MelSpectrogram(**spectrogram_config)
        # mel_spectrogram = transform(audio)
        # log_mel = torch.clamp(mel_spectrogram, min=1e-10).log10()
        # log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
        # spectrogram_tensor = (log_mel + 4.0) / 4.0
        # return spectrogram_tensor

        # transform = torchaudio.transforms.MelSpectrogram(**spectrogram_config)
        # return transform(audio).log10()

    if spec: 
        # if dummy:
        #     spectrogram_tensor = torch.ones(mels, 1024)
        # else:
        spectrogram_tensor = mel_spectrogram(audio=audio, n_mels=128, n_fft=400, hop_length=160, padding=0, device=device)
            # spectrogram_tensor = FEncode(spectrogram_tensor)
    else:
        spectrogram_tensor = None

    if pitch_tokens or harmonics or aperiodics:
        wavnp = audio.numpy().astype(np.float64)
        f0_np, t = pw.dio(wavnp, sample_rate, frame_period=hop_length / sample_rate * 1000)
        f0_np = pw.stonemask(wavnp, f0_np, t, sample_rate)

    if pitch_tokens:
        audio = torch.from_numpy(wavnp)
        t2 = torch.from_numpy(t)
        audio_duration = len(audio) / sample_rate
        T = len(labels)
        tok_dur_sec = audio_duration / T
        token_starts = torch.arange(T) * tok_dur_sec
        token_ends = token_starts + tok_dur_sec
        start_idx = torch.searchsorted(t2, token_starts, side="left")
        end_idx = torch.searchsorted(t2, token_ends, side="right")
        pitch_tok = torch.zeros(T, dtype=torch.float32)
        for i in range(T):
            lo, hi = start_idx[i], max(start_idx[i]+1, end_idx[i])
            segment = f0_np[lo:hi]
            if mode == "mean":
                pitch_tok[i] = segment.mean()
            elif mode == "median":
                pitch_tok[i] = torch.median(segment)
            else:
                pitch_tok[i] = segment[-1]
        pitch_tok[pitch_tok < 100.0] = 0.0
        bos_pitch = pitch_tok[0] if len(pitch_tok) > 0 else 0.0
        pitch_tokens_tensor = torch.cat([torch.tensor([bos_pitch]), pitch_tok])
        pitch_tokens_tensor = torch.where(pitch_tokens_tensor == 0.0, torch.zeros_like(pitch_tokens_tensor), (pitch_tokens_tensor - 71.0) / (500.0 - 71.0))
    else:
        pitch_tokens_tensor = None

    if phase_mod:
        tframe = torch.mean(t2[1:] - t2[:-1])
        phi0 = 0.0
        omega = 2 * torch.pi * f0_tensor
        dphi = omega * tframe
        phi = torch.cumsum(dphi, dim=0) + phi0
        phase = torch.remainder(phi, 2 * torch.pi)
    else:
        phase = None

    if pitch:
        p_tensor = torchaudio.functional.detect_pitch_frequency(audio, sample_rate).unsqueeze(0)
        # p_tensor = PEncode(p_tensor)

        # pitch_tensor_hz = torchaudio.functional.detect_pitch_frequency(waveform, sample_rate)
        # if pitch_tensor_hz.dim() > 1 and pitch_tensor_hz.shape[0] == 1:
        #     pitch_tensor_hz = pitch_tensor_hz.squeeze(0) 
        # ctx = pitch_tensor_hz.shape[0]
        # # print(f"Original pitch tensor shape from torchaudio: {pitch_tensor_hz.shape}")
        # pitch_hz_np = pitch_tensor_hz.numpy()
        # num_pitch_classes = 128
        # pcp_data = pcp(pitch_hz_np, num_pitch_classes=num_pitch_classes)
        # p_tensor = pcp_data.unsqueeze(0).transpose(1, 2) # (1, num_pitch_classes, ctx)
        # print(f"PCP data shape: {pcp_data.shape}")

    else:
        p_tensor = None

    if harmonics or aperiodics:
        spnp = pw.cheaptrick(wavnp, f0_np, t, sample_rate, fft_size=256)
        apnp = pw.d4c(wavnp, f0_np, t, sample_rate, fft_size=256)
        harmonic_tensor = torch.from_numpy(spnp)
        aperiodic_tensor = torch.from_numpy(apnp)
        harmonic_tensor = harmonic_tensor[:, :128].contiguous().T
        aperiodic_tensor = aperiodic_tensor[:, :128].contiguous().T
        harmonic_tensor = torch.where(harmonic_tensor == 0.0, torch.zeros_like(harmonic_tensor), harmonic_tensor / 1.0)
        aperiodic_tensor = torch.where(aperiodic_tensor == 0.0, torch.zeros_like(aperiodic_tensor), aperiodic_tensor / 1.0)
    else:
        harmonic_tensor = None
        aperiodic_tensor = None

    if waveform:
        wave_tensor = audio
        # wave_tensor = WEncode(wave_tensor)
    else:
        wave_tensor = None

    if dummy:   
        batch_size = 1
        ctx = 1024
        if spectrogram_tensor is not None:
            # spectrogram_tensor = torch.randn(mels, ctx)
            spectrogram_tensor = torch.ones(mels, ctx)
        
        if p_tensor is not None:
            p_tensor = torch.ones_like(p_tensor) 
      
        if pitch_tokens_tensor is not None:
            dummy_tensor = torch.ones_like(pitch_tokens_tensor)
        
        else:
            batch_size = 128
            ctx = 1024
            dummy_tensor = torch.ones(batch_size, ctx)
            dummy_tensor = dummy_tensor.to(device)
    else:
        dummy_tensor = None
        
    if debug:
        print(f"['pitch_tokens']: {pitch_tokens_tensor.shape if pitch_tokens else None}")
        print(f"['harmonic']: {harmonic_tensor.shape if harmonics else None}")
        print(f"['aperiodic']: {aperiodic_tensor.shape if aperiodics else None}")
        print(f"['spectrogram']: {spectrogram_tensor.shape if spec else None}")
        print(f"['waveform']: {wave_tensor.shape if waveform else None}")
        print(f"['labels']: {len(labels) if labels else None}")
        print(f"['phase']: {phase.shape if phase else None}")
        print(f"['pitch']: {p_tensor.shape if pitch else None}")
        print(f"['crepe_time']: {crepe_time.shape if crepe else None}")  
        print(f"['crepe_frequency']: {crepe_frequency.shape if crepe else None}")
        print(f"['crepe_confidence']: {crepe_confidence.shape if crepe else None}")
        print(f"['crepe_activation']: {crepe_activation.shape if crepe else None}")
        # print(f"['dummy']: {dummy_tensor.shape if dummy else None}")

    return {
        "waveform": wave_tensor if waveform else None,
        "spectrogram": spectrogram_tensor if spec else None,
        "pitch_tokens": pitch_tokens_tensor if pitch_tokens else None,
        "pitch": p_tensor if pitch else None,
        "harmonic": harmonic_tensor if harmonics else None,
        "aperiodic": aperiodic_tensor if aperiodics else None,  
        "labels": labels,
        "phase": phase if phase_mod else None,
        "crepe_time": crepe_time if crepe else None,
        "crepe_frequency": crepe_frequency if crepe else None,
        "crepe_confidence": crepe_confidence if crepe else None,
        "crepe_activation": crepe_activation if crepe else None,
        # "dummy": dummy_tensor if dummy else None,
    }

def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)

def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    import librosa
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")

def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")

def plot_pitch(waveform, sr, pitch):
    figure, axis = plt.subplots(1, 1)
    axis.set_title("Pitch Feature")
    axis.grid(True)

    end_time = waveform.shape[1] / sr
    time_axis = torch.linspace(0, end_time, waveform.shape[1])
    axis.plot(time_axis, waveform[0], linewidth=1, color="gray", alpha=0.3)

    axis2 = axis.twinx()
    time_axis = torch.linspace(0, end_time, pitch.shape[1])
    axis2.plot(time_axis, pitch[0], linewidth=2, label="Pitch", color="green")

    axis2.legend(loc=0)

def prepare_datasets(tokenizer, token, sanity_check=False, sample_rate=16000, streaming=False,
        load_saved=False, save_dataset=False, cache_dir=None, extract_args=None, max_ctx=2048):

    if extract_args is None:
        extract_args = {
        "waveform": False,
        "spec": False,
        "f0": False,
        "pitch_tokens": False,
        "pitch": False,
        "harmonic": False,
        "aperiodic": False,
        "sample_rate": 16000,
        "hop_length": 256,
        "mode": "mean",
        "debug": False,
        "phase_mod": False,
        "crepe": False,
        "dummy": False,
        }

    if load_saved:
        if cache_dir is None:
            cache_dir = "./processed_datasets"
        else:
            cache_dir = cache_dir

        os.makedirs(cache_dir, exist_ok=True)
        cache_file_train = os.path.join(cache_dir, "train.arrow")
        cache_file_test = os.path.join(cache_dir, "test.arrow")

        if os.path.exists(cache_file_train) and os.path.exists(cache_file_test):
            from datasets import Dataset
            train_dataset = Dataset.load_from_disk(cache_file_train)
            test_dataset = Dataset.load_from_disk(cache_file_test)
            return train_dataset, test_dataset   

    if sanity_check:
        test = load_dataset(
            "google/fleurs", "en_us", token=token, split="test", trust_remote_code=True, streaming=streaming).cast_column("audio", Audio(sampling_rate=sample_rate)).take(1)
        dataset = test.map(lambda x: extract_features(x, tokenizer, **extract_args), remove_columns=test.column_names)
        train_dataset = dataset
        test_dataset = dataset
        return train_dataset, test_dataset
    else:
        def filter_func(x):
            return (0 < len(x["transcription"]) < max_ctx and
                    len(x["audio"]["array"]) > 0 and
                    len(x["audio"]["array"]) < max_ctx * 160)

        raw_train = load_dataset(
            "google/fleurs", "en_us", token=token, split="train", trust_remote_code=True, streaming=streaming).take(1000)
        raw_test = load_dataset(
            "google/fleurs", "en_us", token=token, split="test", trust_remote_code=True, streaming=streaming).take(100)

        raw_train = raw_train.filter(filter_func).cast_column("audio", Audio(sampling_rate=sample_rate))
        raw_test = raw_test.filter(filter_func).cast_column("audio", Audio(sampling_rate=sample_rate))
        train_dataset = raw_train.map(lambda x: extract_features(x, tokenizer, **extract_args), remove_columns=raw_train.column_names)
        test_dataset = raw_test.map(lambda x: extract_features(x, tokenizer, **extract_args), remove_columns=raw_test.column_names)
        train_dataset.save_to_disk(cache_file_train) if save_dataset is True else None
        test_dataset.save_to_disk(cache_file_test) if save_dataset is True else None
        return train_dataset, test_dataset

class c_gate(nn.Module):
    def __init__(self, dims, enabled=True):
        super().__init__()
        self.enabled = enabled
        if enabled:
            self.s_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
            self.w_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
            self.p_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
            self.e_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
            self.ph_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
            self.integ = Linear(dims*5, dims)
        
    def forward(self, x, enc):
        if not self.enabled:
            return None
        s_feat = enc.get("spectrogram", x)
        w_feat = enc.get("waveform", x)
        p_feat = enc.get("pitch", x)
        e_feat = enc.get("envelope", x)
        ph_feat = enc.get("phase", x)
        s = self.s_gate(x) * s_feat
        w = self.w_gate(x) * w_feat
        p = self.p_gate(x) * p_feat
        e = self.e_gate(x) * e_feat
        ph = self.ph_gate(x) * ph_feat
        comb = torch.cat([s, w, p, e, ph], dim=-1)
        return self.integ(comb)

class mgate(nn.Module):
    def __init__(self, dims, mem=64):
        super().__init__()
        self.mk = nn.Parameter(torch.randn(mem, dims))
        self.mv = nn.Parameter(torch.randn(mem, 1))
        self.mg = nn.Sequential(Linear(dims, dims//2), nn.SiLU(), Linear(dims//2, 1))
    def forward(self, x, cosim=False):
        if cosim:
            key = F.softmax(torch.matmul(F.normalize(x, p=2, dim=-1), F.normalize(self.mk, p=2, dim=-1).transpose(0, 1)) / math.sqrt(x.shape[-1]), dim=-1)
        else:
            key = F.softmax(torch.matmul(x, self.mk.transpose(0, 1)) / math.sqrt(x.shape[-1]), dim=-1)
        return 0.5 * (torch.sigmoid(self.mg(x)) + torch.sigmoid(torch.matmul(key, self.mv)))

class t_gate(nn.Module):
    def __init__(self, dims: int, head: int, memory_size: int = 64, threshold: float = 0.8):
        super().__init__()
        self.dims=dims
        self.mkeys = {}

        self.xa_proj = nn.Linear(dims, dims)
        self.activation = nn.Sigmoid()
        self.pattern = lambda length: sinusoids(length, dims=dims)  
        self.primer = torch.ones(1, 512, dims)

    def forward(self, x) -> torch.Tensor: 
        if x is None:
            cur = self.primer
            self.key = cur
        else:
            cur = self.pattern(x.shape[1]).to(device, dtype)
    
        self.mkeys["last"] = self.key
        cur = self.xa_proj(cur.mean(dim=1)) 

        for b in range(cur.size(0)):
            cur_xa = cur[b]
            score = -1.0
            best = None
            for last in self.mkeys.items():
                last = self.mkeys["last"]
                similarity = F.cosine_similarity(cur_xa, last, dim=0).mean()

                if similarity > score:
                    score = similarity
                    best = best

            gating_value = self.activation(torch.tensor(score))
            if gating_value > self.threshold and best is not None:
                self.key = cur
            else:
                self.key = last
            threshold = apply_ste_threshold(x, self.threshold)
        return threshold, self.key

class StraightThroughThreshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold):
        binary_output = (x > threshold).float()
        ctx.save_for_backward(x, threshold)
        return binary_output

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_threshold = None
        return grad_x, grad_threshold

apply_ste_threshold = StraightThroughThreshold.apply

def sinusoids(length, dims):
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dims, 2).float() * (-math.log(10000.0) / dims))
    pe = torch.zeros(length, dims)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class t_gate(nn.Module):
    def __init__(self, dims: int, head: int, threshold: float = 0.8):
        super().__init__()
        self.dims = dims
        
        self.register_buffer('last_key', torch.zeros(1, dims))
        self.xa_proj = nn.Linear(dims, dims)
        self.activation = nn.Sigmoid()
        self.register_buffer('primer', torch.ones(1, 512, dims))
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32), requires_grad=True)

    def forward(self, x: torch.Tensor): 
        device, dtype = x.device, x.dtype if x is not None else self.primer.device, self.primer.dtype
        if x is None:
            cur_representation = self.primer.to(device, dtype).mean(dim=1)
        else:
            pattern_tensor = self.pattern(x.shape[1]).to(device, dtype)
            cur_representation = pattern_tensor.mean(dim=1)

        cur_proj = self.xa_proj(cur_representation)
        expanded_last_key = self.last_key.expand(cur_proj.shape[0], -1) 
        similarity_scores = F.cosine_similarity(cur_proj, expanded_last_key, dim=-1).unsqueeze(-1)
        gating_value = self.activation(similarity_scores)
        decision = apply_ste_threshold(gating_value, self.threshold)
        self.last_key.copy_(cur_representation.detach()) 
        return decision, gating_value

class m_gate(nn.Module):
    def __init__(self, dims, mem_size=64):
        super().__init__()
        self.enabled = enabled
        if enabled:
            self.m_key = nn.Parameter(torch.randn(mem_size, dims))
            self.m_val = nn.Parameter(torch.randn(mem_size, 1))
            self.gate_proj = nn.Sequential(Linear(dims, dims//2), nn.SiLU(), Linear(dims//2, 1))
            
    def forward(self, x):
        d_gate = torch.sigmoid(self.gate_proj(x))
        attention = torch.matmul(x, self.m_key.transpose(0, 1))
        attention = F.softmax(attention / math.sqrt(x.shape[-1]), dim=-1)
        m_gate = torch.matmul(attention, self.m_val)
        m_gate = torch.sigmoid(m_gate)
        return 0.5 * (d_gate + m_gate)

class lgate(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
    def forward(self, x):
        return self.gate(x)

class tgate(nn.Module):
    def __init__(self, dims, num_types=2):
        super().__init__()
        self.gates = nn.ModuleList([nn.Sequential(Linear(dims, 1), nn.Sigmoid()) for _ in range(num_types)])
        self.classifier = nn.Sequential(Linear(dims, num_types), nn.Softmax(dim=-1))
    def forward(self, x):
        types = self.classifier(x)
        gates = torch.stack([gate(x) for gate in self.gates], dim=-1)
        return  torch.sum(gates * types.unsqueeze(2), dim=-1)

class CausalAttention(nn.Module):
    def __init__(self, dims, head, ctx):
        super().__init__()
        self.q = nn.Linear(dims, dims)
        self.k = nn.Linear(dims, dims * 2, bias=False)
        self.n = nn.LayerNorm(dims) 
        self.register_buffer('mask', torch.triu(torch.ones(ctx, ctx), diagonal=1)) 
    def forward(self, x):
        k, v = self.k(self.n(x)).chunk(2, dim=-1)
        q = self.q(self.n(x))
        qk = q @ k.transpose(1, 2) 
        qk.masked_fill_(self.mask.bool()[:x.shape[1], :x.shape[1]], -torch.inf) 
        w = torch.softmax(qk / k.shape[-1]**0.5, dim=-1)
        return  w @ v

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, dims, head, ctx):
        super().__init__()
        self.heads = nn.ModuleList([CausalAttention(dims, head, ctx) for _ in range(head)])
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)   

class OneShot(nn.Module):
    def __init__(self, dims: int, head: int, scale: float = 0.3):
        super().__init__()
        self.head  = head
        self.hdim  = dims // head
        self.scale = scale                      
        self.q = nn.Linear(dims, dims)
        self.k = nn.Linear(dims, dims)
    def forward(self, x: torch.Tensor, xa: torch.Tensor) -> torch.Tensor | None:
        B, Q, _ = x.shape
        K = xa.size(1)
        q = self.q(x).view(B, Q, self.head, self.hdim).transpose(1,2)  
        k = self.k(xa).view(B, K, self.head, self.hdim).transpose(1,2)
        return (q @ k.transpose(-1, -2)) * self.scale / math.sqrt(self.hdim) 


# class ContextualBiasingGate2(nn.Module):
#     def __init__(self, dims: int, head: int, memory_size: int, threshold: float = 0.8):
#         super(ContextualBiasingGate, self).__init__()
#         self.dims = dims
#         self.head = head
#         self.memory_size = memory_size
#         self.threshold = threshold
#         self.mkeys = {}  # {pattern_embedding_tuple: bias_scalar_weight_tensor}
#         self.xa_projection = nn.Linear(dims, dims)
#         self.tgate_activation = nn.Sigmoid()
#         self.head_bias_weights = nn.Parameter(torch.ones(head))
#         self.pattern = lambda length, dims, max_tscale: sinusoids(length, dims)  
#         self.embedding = nn.Embedding(self.pattern, dims)
#         self.one_shot = OneShot(dims, head)

#         for _ in range(memory_size): # example
#             pattern = lambda length, dims, max_tscale: sinusoids(length, dims) 
#             bias_weight = OneShot(dims, head)
#             self.mkeys[tuple(pattern.tolist())] = bias_weight

#     def forward(self, shot_bias: torch.Tensor, xa: torch.Tensor) -> torch.Tensor:
#         if shot_bias is None:
#             return None

#         xa = self.xa_projection(xa.mean(dim=1)) # (B, D)
#         logits = []

#         for b in range(xa.size(0)): # Iterate through the batch
#             cur_xa = xa[b]
#             score = -1.0
#             best_weights = None

#             for pattern, bias_weights in self.mkeys.items():
#                 pattern_tensor = torch.tensor(pattern).to(cur_xa.device)
#                 similarity = F.cosine_similarity(cur_xa, pattern_tensor, dim=0)

#                 if similarity > score:
#                     score = similarity
#                     best_weights = bias_weights

#             gating_value = self.tgate_activation(score.unsqueeze(0))
#             shot_bias = shot_bias[b] # (head, Q, K)

#             if gating_value > self.threshold and best_weights is not None:
#                 scaled_bias = shot_bias * (best_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) # (H,Q,K) * (H,1,1,1)
#                 logits.append(scaled_bias)
#             else:
#                 logits.append(shot_bias)
#         return torch.stack(logits, dim=0) # (B, head, Q, K)

class ContextualBias(nn.Module):
    def __init__(self, dims: int, head: int, memory_size: int, threshold: float = 0.8):
        super().__init__()
        self.dims = dims
        self.head = head
        self.hdim = dims // head

        self.q = nn.Linear(dims, dims)
        self.k = nn.Linear(dims, dims)
        self.v = nn.Linear(dims, dims)
        self.one_shot = OneShot(dims, head)
        self.biasing_gate = BiasingGate(dims, head)

    def forward(self, x, xa) -> torch.Tensor:
        B, Q, _ = x.shape
        K = xa.size(1)
        q = self.q(x).view(B, Q, self.head, self.hdim).transpose(1,2)
        k = self.k(x if xa is None else xa).view(B, K, self.head, self.hdim).transpose(1,2)
        v = self.v(x if xa is None else xa).view(B, K, self.head, self.hdim).transpose(1,2)
        qk = (q @ k.transpose(-1, -2)) / math.sqrt(self.hdim) # (B, H, Q, K)
        x = self.one_shot(x, xa)
        if x is not None:
            bias = self.biasing_gate(x, xa)
            qk = qk + bias
        w = F.softmax(qk, dim=-1)
        return (w @ v).transpose(1, 2).reshape(B, Q, self.dims)

class BiasingGate(nn.Module):
    def __init__(self, dims: int, head: int, memory_size: int = 64, threshold: float = 0.8):
        super(BiasingGate, self).__init__()
        self.memory_size = memory_size
        self.threshold = threshold
        self.mkeys = {}
        self.xa_projection = nn.Linear(dims, dims)
        self.activation = nn.Sigmoid()
        self.head_bias_weights = nn.Parameter(torch.ones(head))
        self.pattern = lambda length: sinusoids(length, dims=dims)  
        self.one_shot = OneShot(dims, head)

    def forward(self, x, xa) -> torch.Tensor: # x is bias
        if x is None:
            return None

        for _ in range(self.memory_size): # example
            self.mkeys["pattern"] = self.pattern(xa.shape[1]).to(device, dtype)
            bias_weights = x
            self.mkeys["bias"] = bias_weights
   
        xa = self.xa_projection(xa.mean(dim=1)) # (B, D)
        logits = []

        for b in range(xa.size(0)): # Iterate through the batch
            cur_xa = xa[b]
            score = -1.0
            best_weights = None
            for pattern, bias_weights in self.mkeys.items():
                pattern = self.mkeys["pattern"]
                pattern_tensor = torch.tensor(pattern).to(cur_xa.device)
                similarity = F.cosine_similarity(cur_xa, pattern_tensor, dim=0).mean()

                if similarity > score:
                    score = similarity
                    best_weights = bias_weights

            score = torch.tensor(score)
            gating_value = self.activation(score)
            shot_bias = x[b] # (head, Q, K)

            if gating_value > self.threshold and best_weights is not None:
                scaled_bias = shot_bias * (best_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) # (H,Q,K) * (H,1,1,1)
                logits.append(scaled_bias)
            else:
                logits.append(shot_bias)
        return torch.stack(logits, dim=0) # (B, head, Q, K)

class BiasingGateB(nn.Module):
    def __init__(self, dims: int, head: int, memory_size: int = 64, threshold: float = 0.8):
        super(BiasingGate, self).__init__()
        self.dims = dims
        self.head = head
        self.memory_size = memory_size
        self.threshold = threshold
        self.mkeys = {}
        self.p = nn.Linear(dims, dims)
        self.tgate = nn.Sigmoid()

        self.pattern = lambda length, dims, max_tscale: sinusoids(length, dims)  
        self.one_shot = OneShot(dims, head)

        for _ in range(memory_size): # example
            pattern = lambda length, dims, max_tscale: sinusoids(length, dims) 
            bias_weight = OneShot(dims, head)
            self.mkeys[tuple(pattern.tolist())] = bias_weight

    def forward(self, x, xa) -> torch.Tensor:
        B, T, _ = x.shape
        input = self.p(x.mean(dim=1))
        batch_gate_biases = []
        for b in range(B):
            cur_input = input[b]
            score = -1.0
            best_match = None
            for pattern, gate_bias in self.mkeys.items():
                pattern_tensor = torch.tensor(pattern).to(cur_input.device)
                similarity = F.cosine_similarity(cur_input, pattern_tensor, dim=0)
                if similarity > score:
                    score = similarity
                    best_match = gate_bias
            gating_value = self.tgate(score.unsqueeze(0))
            if gating_value > self.threshold and best_match is not None:
                batch_gate_biases.append(best_match.unsqueeze(0))
            else:
                batch_gate_biases.append(torch.zeros(1, self.head).to(x.device))
        return torch.cat(batch_gate_biases, dim=0)

class CuriosityHead(nn.Module):
    def __init__(self, d, h, bias=True, memory_size=10, cb_threshold=0.7):
        super().__init__()
        self.h  = h              # base heads
        self.dh = d // h
        self.qkv = nn.Linear(d, d * 3, bias=bias)
        self.qkv_aux = nn.Linear(d, d * 3, bias=bias)  # curiosity heads
        self.o  = nn.Linear(d, d, bias=bias)
        self.g  = nn.Parameter(torch.zeros(h))         # per-head gate logit
        self.contextual_biasing_gate = BiasingGateB(d, h, memory_size, cb_threshold)

    def split(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.h, self.dh).transpose(1, 2)  # b h t dh

    def merge(self, x):
        b, h, t, dh = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * dh)

    def forward(self, x, xa, mask=None):
        q, k, v   = self.qkv(x).chunk(3, -1)
        qa, ka, va = self.qkv_aux(xa).chunk(3, -1)
        q, k, v   = map(self.split, (q, k, v))
        qa, ka, va = map(self.split, (qa, ka, va))
        dots      = (q @ k.transpose(-2, -1)) / self.dh**0.5      # b h t t
        dots_aux  = (q @ ka.transpose(-2, -1)) / self.dh**0.5     # b h t ta
        if mask is not None: dots = dots.masked_fill(mask, -9e15)
        p   = dots.softmax(-1)
        pa  = dots_aux.softmax(-1)

        h_main = p  @ v                       # b h t dh
        h_aux  = pa @ va                      # b h t dh
        contextual_gate_bias = self.contextual_biasing_gate(x) # (B, H)
        g_biased_logits = self.g + contextual_gate_bias # (B, H)
        g = torch.sigmoid(g_biased_logits).view(x.size(0), -1, 1, 1) # b h 1 1 broadcast
        out = self.merge(h_main * (1 - g) + h_aux * g)
        return self.o(out)

def get_encoder(mels = None, input_dims = None, dims = None, head=None, act="relu", downsample = True, target_length = False, attend_pitch=False, feature = "spectrogram") -> nn.Module:

    if feature == "spectrogram":
        return FEncoder(mels, dims, head, act, feature = "spectrogram")
    elif feature == "waveform":
        return WEncoder(input_dims, dims, head, act, downsample, target_length, feature = "waveform")
    elif feature == "pitch":
        return PEncoder(input_dims, dims, head, act, attend_pitch, feature = "pitch")
    else:
        raise ValueError(f"Unknown feature type: {feature}")

@dataclass
class DataCollator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        all_keys = set()
        for f in features:
            all_keys.update(f.keys())
        batch = {}
        pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)
        bos_token_id = getattr(self.tokenizer, 'bos_token_id', 1)
        eos_token_id = getattr(self.tokenizer, 'eos_token_id', 2)

        for key in all_keys:
            if key == "labels":
                labels_list = [f["labels"] for f in features]
                max_len = max(len(l) for l in labels_list)
                all_ids, all_labels = [], []

                for label in labels_list:
                    label_list = label.tolist() if isinstance(label, torch.Tensor) else label
                    decoder_input = [bos_token_id] + label_list
                    label_eos = label_list + [eos_token_id]
                    input_len = max_len + 1 - len(decoder_input)
                    label_len = max_len + 1 - len(label_eos)
                    padded_input = decoder_input + [pad_token_id] * input_len
                    padded_labels = label_eos + [pad_token_id] * label_len
                    all_ids.append(padded_input)
                    all_labels.append(padded_labels)
                batch["input_ids"] = torch.tensor(all_ids, dtype=torch.long)
                batch["labels"] = torch.tensor(all_labels, dtype=torch.long)

            elif key in ["spectrogram", "waveform", "pitch", "harmonic", "aperiodic", "pitch_tokens", "f0", "phase", "crepe_time", "crepe_frequency", "crepe_confidence", "crepe_activation", "dummy"]:

                items = [f[key] for f in features if key in f]
                items = [item for item in items if item is not None]
                if not items:  
                    continue
                items = [torch.tensor(item) if not isinstance(item, torch.Tensor) else item for item in items]
                max_len = max(item.shape[-1] for item in items)
                padded = []
                for item in items:
                    pad_width = max_len - item.shape[-1]
                    if pad_width > 0:
                        pad_item = F.pad(item, (0, pad_width), mode='constant', value=pad_token_id)
                    else:
                        pad_item = item

                    # if pad_item == "spectrogram":
                    #     pad_item = FEncode(pad_item)

                    padded.append(pad_item)
                batch[key] = torch.stack(padded)

        return batch

def levenshtein(reference_words, hypothesis_words):
    m, n = len(reference_words), len(hypothesis_words)
    dist_matrix = [[0 for _ in range(n+1)] for _ in range(m+1)]
    for i in range(m+1):
        dist_matrix[i][0] = i
    for j in range(n+1):
        dist_matrix[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if reference_words[i-1] == hypothesis_words[j-1]:
                dist_matrix[i][j] = dist_matrix[i-1][j-1]
            else:
                substitution = dist_matrix[i-1][j-1] + 1
                insertion = dist_matrix[i][j-1] + 1
                deletion = dist_matrix[i-1][j] + 1
                dist_matrix[i][j] = min(substitution, insertion, deletion)
    return dist_matrix[m][n]

def wer_batch(references, hypotheses):
    total_errors = 0
    total_words = 0
    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.lower().split()
        errors = levenshtein(ref_words, hyp.lower().split()) 
        total_errors += errors
        total_words += len(ref_words)
    return (total_errors / total_words) * 100 if total_words > 0 else 0.0

def compute_metrics(pred, tokenizer=None, model=None, print_pred=False, num_samples=0, logits=None, compute_result: bool = False):
    
    def clean(ids, pad_token_id=0, bos_token_id=1, eos_token_id=2):
        
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids[0], (list, torch.Tensor, np.ndarray)):
            return [[int(i) for i in seq if i not in (-100, pad_token_id, bos_token_id, eos_token_id)] for seq in ids]
        else:
            return [int(i) for i in ids if i not in (-100, pad_token_id, bos_token_id, eos_token_id)]

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    if not isinstance(pred_ids, torch.Tensor):
        pred_ids = torch.tensor(pred_ids)

    label_ids = clean(label_ids)
    pred_ids = clean(pred_ids)
    pred_str = tokenizer.batch_decode(pred_ids)
    label_str = tokenizer.batch_decode(label_ids)

    if print_pred:
        for i in range(min(num_samples, len(pred_ids))):

            print(f"Pred tokens: {pred_ids[i]}")
            print(f"Label tokens: {label_ids[i]}")
            print(f"Pred: '{pred_str[i]}'")
            print(f"Label: '{label_str[i]}'")
            print("-" * 40)

    wer = wer_batch(label_str, pred_str)
    if model is not None:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000
        efficiency_score = (100 - wer) / trainable_params if trainable_params > 0 else 0.0
        # jump_logs = model.processor.res.dsl.logs
    else:
        trainable_params = 0.0
        efficiency_score = 0.0
        # jump_logs = None

    return {
        "wer": float(wer),
        "efficiency_score": float(efficiency_score),
        # "jump_logs": jump_logs["jumps"][0],
    }

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels

def hilbert_transform(x):
    N = x.shape[-1]
    xf = torch.fft.rfft(x)
    h = torch.zeros(N // 2 + 1, device=x.device, dtype=x.dtype)
    if N % 2 == 0:
        h[0] = h[N//2] = 1
        h[1:N//2] = 2
    else:
        h[0] = 1
        h[1:(N+1)//2] = 2
    return torch.fft.irfft(xf * h, n=N)

def analytic_signal(x):
    return x + 1j * hilbert_transform(x)

def hilbert_transform_2d(x, dim=-1):
    N = x.shape[dim]
    if dim == -1 or dim == len(x.shape) - 1:
        xf = torch.fft.rfft(x)
    else:
        xf = torch.fft.rfft(x, dim=dim)
    h_shape = [1] * len(x.shape)
    h_shape[dim] = N // 2 + 1
    h = torch.zeros(h_shape, device=x.device, dtype=x.dtype)
    if dim == -1 or dim == len(x.shape) - 1:
        if N % 2 == 0:
            h[..., 0] = h[..., -1] = 1
            h[..., 1:-1] = 2
        else:
            h[..., 0] = 1
            h[..., 1:] = 2
    else:
        pass
    return torch.fft.irfft(xf * h, n=N, dim=dim)

def hilbert_transform_true_2d(x):
    xf = torch.fft.rfft2(x)
    h1, h2 = torch.meshgrid(
        torch.fft.rfftfreq(x.shape[-2]) * 2 - 1,
        torch.fft.rfftfreq(x.shape[-1]) * 2 - 1,
        indexing='ij')
    h = -1j / (math.pi * (h1 + 1j*h2))
    h[0, 0] = 0 
    return torch.fft.irfft2(xf * h.to(x.device))

def process_spectrogram_with_hilbert(spec):
    analytic = spec + 1j * hilbert_transform(spec)
    envelope = torch.abs(analytic)
    phase = torch.angle(analytic)
    return envelope, phase

class MyModel(nn.Module):
    def __init__(self, num_layers, dims, head, act_fn):
        super().__init__()
        self.num_layers = num_layers 

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_dict = {
                'ln': nn.LayerNorm(dims),
                'gate': nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()),
                'adapter': nn.Linear(dims, dims) if i % 2 == 0 else None,
                "res1": residual(dims, head, num_layers, act_fn),
                "res2": residual(dims, head, num_layers, act_fn),
                "res3": residual(dims, head, num_layers, act_fn),      
            }
            if i == 5:
                layer_dict['special_attention'] = attentiona(dims, head)
            self.layers.append(nn.ModuleDict(layer_dict))
        
        # self.attention_auxiliary = attentiona(dims, head) 

    def forward(self, x, xa, enc=None, sequential=False, modal=False, blend=False, kv_cache=None) -> torch.Tensor:    
   
        if self.num_layers > 0:
            cur_dict = self.layers[0]
            xa = cur_dict['res1'](xa)

        if self.num_layers > 1:
            cur_dict = self.layers[1]
            a = cur_dict['res2'](x)

        if self.num_layers > 2:
            cur_dict = self.layers[2]
            b = cur_dict['res3'](x, xa, None)

        if self.num_layers > 3:
            cur_dict = self.layers[3]
            xm = cur_dict['res1'](torch.cat([x, xa], dim=1))

        if self.num_layers > 4:
            cur_dict = self.layers[4]
            c = cur_dict['res2'](xm[:, :x.shape], xm[:, x.shape:], mask=None)

        if self.num_layers > 5:
            cur_dict = self.layers[5]
            x = cur_dict['res3'](a + b + c) 
            
            x = cur_dict['res3'](x, xa, None)

            x = cur_dict['lna'](x)

            if 'special_attention' in cur_dict:
                special_attn_output = cur_dict['special_attention'](x, xa=x, mask=None)
                x = x + special_attn_output

            aux_attn_output = self.attention_auxiliary(x, xa=x, mask=None)
            x = x + aux_attn_output

        for i in range(6, self.num_layers):
            cur_dict = self.layers[i]
            x = cur_dict['res1'](x)
            x = cur_dict['lna'](x)

        return x

        # self.layers = nn.ModuleList()
        # for _ in range(layer):
        #     self.layers.append(nn.ModuleList({
        #         'lna': nn.LayerNorm(dims),
        #         'lnb': nn.LayerNorm(dims),
        #         "res1": residual(dims, head, layer, act_fn),
        #         "res2": residual(dims, head, layer, act_fn),
        #         "res3": residual(dims, head, layer, act_fn),      
                    
        #         }))

        # if self.layer > 0:
        #     res = self.layers[0]
        #     xa = res['res1'](xa) 

        # if self.layer > 1:
        #     res = self.layers[1]
        #     a  = res['res2'](x, mask=mask)

        # if self.layer > 4:
        #     res = self.layers[4]
        #     b = res['res3'](x, xa, None)

        # # for i in range(3, self.layer):
        # #     res = self.layers[i]
        #     xm = res['res4'](torch.cat([x, xa], dim=1))
        #     c  = res['res4'](xm[:, :x.shape], xm[:, x.shape:], mask=None)

        # for i in range(4, self.layer):
        #     res = self.layers[i]
        #     x = res['res3'](a + b + c) 
        #     x = res['res3'](x, xa, None)
        #     x = res['lna'](x)

class attention_a(nn.Module):
    def __init__(self, dims: int, head: int, layer):
        super().__init__()
        self.head = head
        self.dims = dims
        self.head_dim = dims // head

        self.q = nn.Linear(dims, dims) 
        self.kv = nn.Linear(dims, dims * 2, bias=False)
        self.out = nn.Linear(dims, dims)
        self.lna = nn.LayerNorm(dims) 
        self.lnb = nn.LayerNorm(dims // head) 

    def forward(self, x, xa = None, mask = None):
        q = self.q(x)
        k, v = self.kv(self.lna(x if xa is None else xa)).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b c (h d) -> b h c d', h = self.head), (q, k, v))
        a = scaled_dot_product_attention(self.lnc(q), self.lnd(k), v, is_causal=mask is not None and q.shape[2] > 1)
        wv = a.permute(0, 2, 1, 3).flatten(start_dim=2)
        out = self.out(wv)
        return out

class attentio0(nn.Module):
    def __init__(self, dims: int, head: int, layer):
        super().__init__()
        self.head = head
        self.dims = dims
        self.head_dim = dims // head

        self.pad_token = 0
        self.zmin = 1e-6
        self.zmax = 1e-5     
        self.zero = nn.Parameter(torch.tensor(1e-4, device=device, dtype=dtype), requires_grad=False)
        self.taylor_expand_fn = partial(second_taylor)
        self.q = nn.Linear(dims, dims) 
        self.kv = nn.Linear(dims, dims * 2, bias=False)
        self.out = nn.Linear(dims, dims)

        self.lna = nn.LayerNorm(dims) 
        self.lnb = nn.LayerNorm(dims // head) 

        # self.rotary_emb = RotaryEmbedding(dims // head)

       # print(f"x, {x.shape}, xa, {xa.shape if xa is not None else None}, mask {mask.shape if mask is not None else None}")
        # zero = self.zero
    def forward(self, x, xa = None, mask = None,  positions = None):

        q = self.q(x)
        k, v = self.kv(self.lna(x if xa is None else xa)).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b c (h d) -> b h c d', h = self.head), (q, k, v))
        scale = q.shape[-1] ** -0.5

        qk = einsum('b h k d, b h q d -> b h k q', self.lnb(q), self.lnb(k)) * scale 

        if mask is not None:
            mask=mask[:q.shape[2], :q.shape[2]]

        token_ids = k[:, :, :, 0]
        zscale = torch.ones_like(token_ids)
        fzero = torch.clamp(F.softplus(self.fzero), self.minz, self.maxz)
        zscale[token_ids.float() == self.pad_token] = fzero
        
        if xa is not None:
            qk = qk + mask * zscale.unsqueeze(-2).expand(qk.shape)

        qk = qk * zscale.unsqueeze(-2)
        qk = taylor_softmax(qk, order=2)
        wv = einsum('b h k q, b h q d -> b h k d', qk, v) 
        wv = rearrange(wv, 'b h c d -> b c (h d)')
        out = self.out(wv)
        return out

class attentio0b(nn.Module):
    def __init__(self, dims: int, head: int, layer):
        super().__init__()
        self.head = head
        self.dims = dims
        self.head_dim = dims // head

        self.pad_token = 0
        self.zmin = 1e-6
        self.zmax = 1e-5     
        self.zero = nn.Parameter(torch.tensor(1e-4, device=device, dtype=dtype), requires_grad=False)
        self.taylor_expand_fn = partial(second_taylor)
        self.q = nn.Linear(dims, dims) 
        self.kv = nn.Linear(dims, dims * 2, bias=False)
        self.out = nn.Linear(dims, dims)

        self.lna = nn.LayerNorm(dims) 
        self.lnb = nn.LayerNorm(dims // head) 

    def forward(self, x, xa = None, mask = None,  positions = None):

        q = self.q(x)
        k, v = self.kv(self.lna(x if xa is None else xa)).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b c (h d) -> b h c d', h = self.head), (q, k, v))
        scale = q.shape[-1] ** -0.5

        qk = einsum('b h k d, b h q d -> b h k q', self.lnb(q), self.lnb(k)) * scale 

        if mask is not None:
            mask=mask[:q.shape[2], :q.shape[2]]

        token_ids = k[:, :, :, 0]
        zscale = torch.ones_like(token_ids)
        fzero = torch.clamp(F.softplus(self.fzero), self.minz, self.maxz)
        zscale[token_ids.float() == self.pad_token] = fzero
        
        if xa is not None:
            qk = qk + mask * zscale.unsqueeze(-2).expand(qk.shape)

        qk = qk * zscale.unsqueeze(-2)
        qk = F.softmax(qk, dim=-1)
        # qk = taylor_softmax(qk, order=2)
        wv = einsum('b h k q, b h q d -> b h k d', qk, v) 
        wv = rearrange(wv, 'b h c d -> b c (h d)')
        out = self.out(wv)
        return out


class attention(nn.Module):
    def __init__(self, dims: int, head: int, layer):
        super().__init__()
        self.head = head
        self.dims = dims
        self.head_dim = dims // head

        self.pad_token = 0
        self.zmin = 1e-6
        self.zmax = 1e-5     
        self.zero = nn.Parameter(torch.tensor(1e-4, device=device, dtype=dtype), requires_grad=False)
        self.taylor_expand_fn = partial(second_taylor)
        self.q = nn.Linear(dims, dims) 
        self.kv = nn.Linear(dims, dims * 2, bias=False)
        self.out = nn.Linear(dims, dims)

        self.lna = nn.LayerNorm(dims) 
        self.lnb = nn.LayerNorm(dims // head) 

        # self.rotary_emb = RotaryEmbedding(dims // head)

       # print(f"x, {x.shape}, xa, {xa.shape if xa is not None else None}, mask {mask.shape if mask is not None else None}")
        # zero = self.zero

def forward_revised(self, x, xa = None, mask = None):
    q = self.q(x)
    k, v = self.kv(self.lna(x if xa is None else xa)).chunk(2, dim=-1)
    q, k, v = map(lambda t: rearrange(t, 'b c (h d) -> b h c d', h = self.head), (q, k, v))

    q_expanded = second_taylor(q)
    k_expanded = second_taylor(k)
    dk_expanded = k_expanded.shape[-1]
    scale_factor = dk_expanded ** -0.5
    qk_logits = torch.einsum('b h k d, b h q d -> b h k q', self.lnb(q_expanded), self.lnb(k_expanded)) * scale_factor

    if mask is not None:
        seq_len_q, seq_len_k = qk_logits.shape[-2:]
        causal_mask = torch.ones(seq_len_q, seq_len_k, device = qk_logits.device, dtype = torch.bool).triu(seq_len_k - seq_len_q + 1)
        qk_logits = qk_logits.masked_fill(causal_mask, -torch.finfo(qk_logits.dtype).max)

    qk_weights = taylor_softmax(qk_logits, order=2)
    wv = torch.einsum('b h k q, b h q d -> b h k d', qk_weights, v)
    wv = rearrange(wv, 'b h c d -> b c (h d)')
    out = self.out(wv)
    return out


```
