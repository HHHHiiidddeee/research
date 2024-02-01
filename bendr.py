import torch
import torch.nn as nn
import numpy as np


class BENDRConvolution(nn.Module):
    def __init__(self, input_channel, output_channel, downsample_size, kernel_size, dropout):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel

        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size]
        if not isinstance(downsample_size, (list, tuple)):
            downsample_size = [downsample_size]
        assert len(kernel_size) == len(downsample_size)

        # Centerable convolutions make life simpler
        kernel_size = [e if e % 2 else e + 1 for e in kernel_size]
        self.kernel_size = kernel_size
        self.downsample_size = downsample_size

        self.encoder = nn.Sequential()

        for i, (kernel, downsample) in enumerate(zip(kernel_size, downsample_size)):
            self.encoder.add_module(f"BENDR encoder {i}", nn.Sequential(
                nn.Conv1d(input_channel, output_channel, kernel_size=kernel, stride=downsample, padding=kernel // 2),
                nn.Dropout(dropout),
                nn.GroupNorm(output_channel // 2, output_channel),
                nn.GELU()
            ))
            input_channel = output_channel


    def forward(self, x):
        # out = torch.transpose(x, 1, 2)
        # out = self.encoder(out)
        # out = torch.transpose(out, 1, 2)
        out = self.encoder(x)
        out = torch.transpose(out, 1, 2)
        return out

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (self.head_dim*heads == embed_size), "Embed size needs to be diveded by heads."

        # Define linear projection
        self.queries = []
        self.keys = []
        self.values = []
        for i in range(self.heads):
            self.queries.append(nn.Linear(self.embed_size, self.head_dim, bias=False))
            self.keys.append(nn.Linear(self.embed_size, self.head_dim, bias=False))
            self.values.append(nn.Linear(self.embed_size, self.head_dim, bias=False))

        self.linear = nn.Linear(self.heads*self.head_dim, self.embed_size, bias=False)

    def forward(self, x, mask=None):
        attention_outs = []
        for i in range(self.heads):
            query = self.queries[i](x)
            key = self.keys[i](x)
            value = self.values[i](x)
            dot_prod = torch.einsum("iqd,ikd->iqk", query, key)
            if mask is not None:
                dot_prod = dot_prod.masked_fill(mask == 0, float(-1e20))
            attention = torch.softmax(dot_prod / self.head_dim**(1/2), dim=2)
            attention_out = torch.einsum("iqk,ikd->iqd", attention, value)
            attention_outs.append(attention_out)
        attention_outs = torch.cat(attention_outs, 2)
        out = self.linear(attention_outs)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_neuron):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.linear1 = nn.Linear(embed_size, forward_neuron)
        self.linear2 = nn.Linear(forward_neuron, embed_size)
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        out1 = self.attention(x)
        out1 = self.layer_norm1(out1 + x)
        out2 = self.linear1(out1)
        out2 = self.linear2(out2)
        out = self.layer_norm2(out2 + out1)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, timesteps, embed_size):
        super().__init__()
        self.pe = torch.zeros(timesteps, embed_size)
        for t in range(timesteps):
            for i in range(embed_size):
                if i % 2 == 0:
                    self.pe[t, i] = torch.sin(torch.tensor(t / (10000)**(i / embed_size)))
                else:
                    self.pe[t, i] = torch.cos(torch.tensor(t / (10000)**(i / embed_size)))
        self.pe.requires_grad = False

    def forward(self, x):
        return x + self.pe

def positional_encoding(x, timesteps, embed_size):
    pe = torch.zeros(timesteps, embed_size)
    for t in range(timesteps):
        for i in range(embed_size):
            if i % 2 == 0:
                pe[t, i] = torch.sin(torch.tensor(t / (10000) ** (i / embed_size)))
            else:
                pe[t, i] = torch.cos(torch.tensor(t / (10000) ** (i / embed_size)))
    pe.requires_grad = False
    out = x + pe
    return out



class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, heads, forward_neuron, num_transformers):
        super().__init__()
        # self.positional_encoder = PositionalEncoding(timesteps, embed_size)
        self.embed_size = embed_size
        self.transformer_blocks = nn.Sequential()
        for i in range(num_transformers):
            self.transformer_blocks.add_module(f"Transformer block{i+1}", TransformerBlock(embed_size, heads, forward_neuron))

    def forward(self, x):
        # out = self.positional_encoder(x)
        out = positional_encoding(x, x.shape[1], self.embed_size)
        out = self.transformer_blocks(out)
        return out

class BENDRContext(nn.Module):
    def __init__(self, input_channel, embed_size, downsample_size, kernel_size, dropout,
                 heads, forward_neuron, num_transformers, mask_p, mask_span):
        super().__init__()
        self.bender_encoder = BENDRConvolution(input_channel, embed_size, downsample_size,
                                               kernel_size, dropout)
        self.transformer_encoder = TransformerEncoder(embed_size, heads,
                                                      forward_neuron, num_transformers)
        self.mask_p = mask_p
        self.mask_span = mask_span

    def forward(self, x):
        out = self.bender_encoder(x)
        self.bendr_vector = out.clone()
        batch, len_seq, len_feature = out.shape
        self.mask = make_mask(shape=(batch, len_seq), p=self.mask_p, total=len_seq,
                              span=self.mask_span, allow_no_inds=False)
        mask_replacement = torch.nn.Parameter(torch.zeros(len_feature), requires_grad=True)
        out[self.mask] = mask_replacement
        out = self.transformer_encoder(out)
        return out


class ReconstructionLoss(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def forward(self, input, target):
        batches, masked_t = self.mask.shape
        loss = 0
        loss_fn = nn.MSELoss()
        for batch in range(batches):
            for t in range(masked_t):
                loss += loss_fn(input[batch][t], target[batch][t])
        return loss


def make_span_from_seeds(seeds, span, total=None):
    inds = list()
    # Loop for masked indices
    for seed in seeds:
        for i in range(seed, seed + span):
            if total is not None and i >= total:
                break
            # At least, there is a span between indices so that only the head indices can get masked
            elif i not in inds:
                inds.append(int(i))
    return np.array(inds)


def make_mask(shape, p, total, span, allow_no_inds=False):
    # Initialize mask tensor
    mask = torch.zeros(shape, requires_grad=False, dtype=torch.bool)

    for i in range(shape[0]):
        mask_seeds = list()
        while not allow_no_inds and len(mask_seeds) == 0 and p > 0:
            # Get nonzero indices (get masked indices given a probability)
            mask_seeds = np.nonzero(np.random.rand(total) < p)[0]

        # Get final mask tensor
        mask[i, make_span_from_seeds(mask_seeds, span, total=total)] = True

    return mask

