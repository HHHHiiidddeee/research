import torch
import torch.nn as nn
import numpy as np


class MAEEGConvolution(nn.Module):
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
            self.encoder.add_module(f"MAEEG encoder {i}", nn.Sequential(
                nn.Conv1d(input_channel, output_channel, kernel_size=kernel, stride=downsample, padding=kernel//2),
                nn.Dropout(dropout),
                nn.GroupNorm(output_channel // 2, output_channel),
                nn.GELU()
            ))
            input_channel = output_channel


    def forward(self, x):
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
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(forward_neuron, embed_size)
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        out1 = self.attention(x)
        out1 = self.layer_norm1(out1 + x)
        out2 = self.relu(self.linear1(out1))
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
        encoded_out = positional_encoding(x, x.shape[1], self.embed_size)
        out = self.transformer_blocks(encoded_out)

        block_out = encoded_out
        self.layer_outputs = []
        for block in self.transformer_blocks:
            block_out = block(block_out)
            self.layer_outputs.append(block_out)
        # self.layer_outputs = np.vstack(self.layer_outputs)
        # self.layer_outputs = torch.Tensor(self.layer_outputs)
        return out

    def get_transformer_layer_output(self, k):
        return self.layer_outputs[::-1][:k]

class MAEEGConvDecoder(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, upsample_size, dropout):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel

        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size]
        if not isinstance(upsample_size, (list, tuple)):
            upsample_size = [upsample_size]
        assert len(kernel_size) == len(upsample_size)

        # Centerable convolutions make life simpler
        kernel_size = [e if e % 2 else e + 1 for e in kernel_size]
        self.kernel_size = kernel_size[::-1]
        self.upsample_size = upsample_size[::-1]

        self.decoder = nn.Sequential()

        for i, (kernel, upsample) in enumerate(zip(self.kernel_size, self.upsample_size)):
            if i != len(kernel_size) - 1:
                self.decoder.add_module(f"MAEEG decoder {i}", nn.Sequential(
                    nn.ConvTranspose1d(input_channel, input_channel, kernel_size=kernel, stride=upsample,
                                       padding=kernel//2, output_padding=upsample-1),
                    nn.Dropout(dropout),
                    nn.GroupNorm(input_channel // 2, input_channel),
                    nn.GELU()
                ))
            else:
                self.decoder.add_module(f"MAEEG decoder {i}", nn.Sequential(
                    nn.ConvTranspose1d(input_channel, output_channel, kernel_size=kernel, stride=upsample,
                                       padding=kernel//2, output_padding=upsample-1),
                ))

    def forward(self, x):
        out = torch.transpose(x, 1, 2)
        out = self.decoder(out)
        return out

class data2vecMAEEG(nn.Module):
    def __init__(self, input_channel, embed_size, downsample_size, kernel_size, dropout,
                 heads, forward_neuron, num_transformers, mask_p, mask_span):
        super().__init__()
        self.input_channel = input_channel
        self.embed_size = embed_size
        self.downsample_size = downsample_size
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.heads = heads
        self.forward_neuron = forward_neuron
        self.num_transformers = num_transformers
        self.maeeg_encoder = MAEEGConvolution(input_channel=input_channel, output_channel=embed_size,
                                              downsample_size=downsample_size, kernel_size=kernel_size,
                                              dropout=dropout)
        self.transformer_encoder = TransformerEncoder(embed_size=embed_size, heads=heads, forward_neuron=forward_neuron,
                                                      num_transformers=num_transformers)
        self.maeeg_decoder = MAEEGConvDecoder(input_channel=embed_size, output_channel=input_channel,
                                              upsample_size=downsample_size, kernel_size=kernel_size,
                                              dropout=dropout)

        self.mask_p = mask_p
        self.mask_span = mask_span

    def forward(self, x):
        out = self.maeeg_encoder(x)
        batch, len_seq, len_feature = out.shape
        mask = make_mask(shape=(batch, len_seq), p=self.mask_p, total=len_seq,
                         span=self.mask_span, allow_no_inds=False)
        mask_replacement = torch.nn.Parameter(torch.zeros(len_feature), requires_grad=True)
        out[mask] = mask_replacement
        self.mask = (out == 0)
        out = self.transformer_encoder(out)
        out = self.maeeg_decoder(out)
        return out

    def save_student_convolution(self, path):
        torch.save(self.maeeg_encoder.state_dict(), path)

    def save_student_transformer(self, path):
        torch.save(self.transformer_encoder.state_dict(), path)

    def save_teacher_convolution(self, path):
        torch.save(self.maeeg_encoder_teacher.state_dict(), path)

    def save_teacher_transformer(self, path):
        torch.save(self.transformer_encoder_teacher.state_dict(), path)

    def build_teacher(self):
        self.maeeg_encoder_teacher = MAEEGConvolution(input_channel=self.input_channel, output_channel=self.embed_size,
                                                      downsample_size=self.downsample_size, kernel_size=self.kernel_size,
                                                      dropout=self.dropout)
        self.transformer_encoder_teacher = TransformerEncoder(embed_size=self.embed_size, heads=self.heads,
                                                              forward_neuron=self.forward_neuron, num_transformers=self.num_transformers)

    def freeze_teacher(self):
        for param in self.maeeg_encoder_teacher.parameters():
            param.requires_grad = False

        for param in self.transformer_encoder_teacher.parameters():
            param.requires_grad = False


    def activate_teacher(self):
        for param in self.maeeg_encoder_teacher.parameters():
            param.requires_grad = True

        for param in self.transformer_encoder_teacher.parameters():
            param.requires_grad = True

    def teacher_output(self, x):
        out = self.maeeg_encoder_teacher(x)
        out = self.transformer_encoder_teacher(out)
        return out

    def teacher_update(self, tau):
        for teacher_param, student_param in zip(self.maeeg_encoder_teacher.parameters(),
                                                self.maeeg_encoder.parameters()):
            with torch.no_grad():
                new_param = tau * teacher_param + (1 - tau) * student_param
                teacher_param.copy_(new_param)

        for teacher_param, student_param in zip(self.transformer_encoder_teacher.parameters(),
                                                self.transformer_encoder.parameters()):
            with torch.no_grad():
                new_param = tau * teacher_param + (1 - tau) * student_param
                teacher_param.copy_(new_param)

    def get_transformer_student_outputs(self, k):
        return self.transformer_encoder.get_transformer_layer_output(k)

    def get_transformer_student_masked_outputs(self, k):
        outputs = self.get_transformer_student_outputs(k)
        for output in outputs:
            output[~self.mask] = 0
        return outputs

    def get_transformer_student_average_output(self, k):
        outputs = self.get_transformer_student_outputs(k)
        num_layers = len(outputs)
        ave_output = torch.zeros(outputs[0].shape, requires_grad=True)
        for output in outputs:
            ave_output = ave_output + output
        ave_output = ave_output / num_layers
        return ave_output

    def get_transformer_teacher_outputs(self, k):
        return self.transformer_encoder_teacher.get_transformer_layer_output(k)

    def get_transformer_teacher_masked_outputs(self, k):
        outputs = self.get_transformer_teacher_outputs(k)
        for output in outputs:
            output[~self.mask] = 0
        return outputs

    def get_transformer_teacher_average_output(self, k):
        outputs = self.get_transformer_teacher_outputs(k)
        num_layers = len(outputs)
        ave_output = torch.zeros(outputs[0].shape, requires_grad=True)
        for output in outputs:
            ave_output = ave_output + output
        ave_output = ave_output / num_layers
        return ave_output

    def representation_loss(self, k, loss_fn=nn.MSELoss()):
        teacher_average_representation = self.get_transformer_teacher_average_output(k).detach()
        student_average_representation = self.get_transformer_student_average_output(k)
        loss = loss_fn(teacher_average_representation, student_average_representation)
        return loss


class BinaryClassifier(nn.Module):
    def __init__(self, fc_neuron):
        super().__init__()
        self.fc_neuron = fc_neuron
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=1, out_features=fc_neuron)
        self.linear2 = nn.Linear(in_features=fc_neuron, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.flatten(x)
        batch, in_features = out.shape
        if self.linear1.in_features == 1:
            self.linear1 = nn.Linear(in_features=in_features, out_features=self.fc_neuron)
        out = self.relu(self.linear1(out))
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out


class MAEEGClassification(nn.Module):
    def __init__(self, input_channel, embed_size, downsample_size,
                 kernel_size, dropout, heads, forward_neuron,
                 num_transformers, fc_neuron, use_transformer=False):
        super().__init__()
        self.maeeg_encoder = MAEEGConvolution(input_channel=input_channel, output_channel=embed_size,
                                              downsample_size=downsample_size, kernel_size=kernel_size,
                                              dropout=dropout)
        self.transformer_encoder = TransformerEncoder(embed_size=embed_size, heads=heads, forward_neuron=forward_neuron,
                                                      num_transformers=num_transformers)
        self.classifier = BinaryClassifier(fc_neuron=fc_neuron)
        self.use_transformer = use_transformer

    def load_convolution_encoder(self, path):
        self.maeeg_encoder.load_state_dict(torch.load(path))

    def load_transformer_encoder(self, path):
        self.transformer_encoder.load_state_dict(torch.load(path))

    def freeze_convolution(self):
        for param in self.maeeg_encoder.parameters():
            param.requires_grad = False

    def freeze_transformer(self):
        for param in self.transformer_encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.maeeg_encoder(x)
        if self.use_transformer:
            out = self.transformer_encoder(out)
        out = self.classifier(out)
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

