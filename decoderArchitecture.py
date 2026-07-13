import torch
import torch.nn as nn
import numpy as np
import math 
import torch.nn.functional as F
import chess.pgn
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()  # registers this module

        # gamma (scale) and beta (shift) — the LEARNABLE parameters of LayerNorm.
        # nn.Parameter tells PyTorch "this tensor needs a gradient and should
        # be included in model.parameters()" — this is the manual version of
        # what nn.Linear does automatically for you under the hood.
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        
        self.eps = eps  # small constant for numerical stability, NOT a parameter
                          # (no gradient needed, it's just a fixed hyperparameter)

    def forward(self, x):
        # x: (B, T, d_model)
        
        # normalize across the LAST dimension (d_model) — i.e. per token,
        # across its own feature vector. NOT across batch, NOT across sequence.
        mean = x.mean(dim=-1, keepdim=True)   # (B, T, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # (B, T, 1)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # scale and shift — learned per-feature, broadcasts across (B, T, d_model)
        out = self.gamma * x_norm + self.beta
        return out
    
class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()  # mandatory — registers this module's params/submodules
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Q, K, V projections. nn.Linear creates weight + bias as nn.Parameter
        # automatically — these get registered the moment you assign
        # self.q_proj = ..., because nn.Module.__setattr__ intercepts it.
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # output projection after concatenating heads back together
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (B, T, d_model)
        B, T, _ = x.shape

        # linear projections — autograd records these matmuls automatically
        Q = self.q_proj(x)   # (B, T, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # split into heads: (B, T, d_model) -> (B, n_heads, T, d_head)
        Q = Q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # scaled dot-product attention scores: (B, n_heads, T, T)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)

        # causal mask: prevents attending to future positions
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # weighted sum of values: (B, n_heads, T, d_head)
        out = attn_weights @ V

        # merge heads back: (B, n_heads, T, d_head) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # final output projection
        out = self.out_proj(out)
        return out
    
def build_causal_mask(T, device):
    # lower-triangular matrix of 1s: position i can attend to positions <= i
    mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)
    # shape: (1, 1, T, T) — broadcastable across batch and heads
    return mask

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        # two linear layers with an expansion in between:
        # d_model -> d_ff -> d_model
        # d_ff is typically 4x d_model (that's the convention from the
        # original transformer paper — worth knowing it's a convention,
        # not a mathematical requirement)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, d_model)
        x = self.linear1(x)      # (B, T, d_ff)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)      # (B, T, d_model)
        return x
    
class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # weight matrix: shape (out_features, in_features)
        # note the shape convention — it's (out, in), not (in, out).
        # this is because of how the forward pass does x @ W.T
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            # register_parameter with None tells PyTorch "there's no bias here,
            # but don't error if something looks for self.bias" — different
            # from just not setting self.bias at all

        self.reset_parameters()

    def reset_parameters(self):
        # weight initialization — NOT arbitrary. This is Kaiming/He-style init,
        # scaled by fan_in, meant to keep activation variance stable across
        # layers at the start of training. Worth thinking about WHY bad init
        # causes vanishing/exploding activations — same underlying concern
        # as the vanishing/exploding gradient asymmetry you already covered
        # with residual connections.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x: (..., in_features)  — any number of leading dims (B, T, in_features) etc.
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out
        # out: (..., out_features)
        
class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MaskedMultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Pre-LN: normalize BEFORE the sublayer, residual adds the RAW sublayer output
        # (this is the ordering you reasoned through earlier for gradient flow stability
        # at depth — Post-LN is more prone to training instability in deep stacks)
        attn_out = self.attn(self.norm1(x), mask)
        x = x + self.dropout(attn_out)

        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)

        return x
    
class ChessGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_seq_len, dropout=0.1):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # learned positional embeddings — a lookup table, same mechanism as
        # token_embedding, just indexed by position instead of token id
        self.positional_embedding = nn.Embedding(max_seq_len, d_model)

        self.dropout = nn.Dropout(dropout)

        # ModuleList — NOT a plain list. this is what makes .parameters()
        # find every block's weights, as we covered.
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        self.final_norm = LayerNorm(d_model)

        # separate output head (not tied to token_embedding here — 
        # tying is a valid alternative, worth trying later as an experiment)
        self.output_head = Linear(d_model, vocab_size)

        # causal mask precomputed once for max_seq_len, sliced per forward call
        # register_buffer: NOT a Parameter (no gradient, not updated by optimizer),
        # but still moves with the model when you call model.to(device)
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer('causal_mask', causal_mask)

    def forward(self, token_ids):
        # token_ids: (B, T)
        B, T = token_ids.shape

        tok_emb = self.token_embedding(token_ids)          # (B, T, d_model)

        positions = torch.arange(T, device=token_ids.device)  # (T,)
        pos_emb = self.positional_embedding(positions)      # (T, d_model)

        x = self.dropout(tok_emb + pos_emb)                  # (B, T, d_model), broadcasts pos_emb across batch

        mask = self.causal_mask[:, :, :T, :T]                # slice to current sequence length

        for block in self.blocks:
            x = block(x, mask)

        x = self.final_norm(x)
        logits = self.output_head(x)                         # (B, T, vocab_size)

        return logits