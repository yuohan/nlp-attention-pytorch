import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * term)
        pe[:, 1::2] = torch.cos(pos * term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x.unsqueeze(2) + self.pe[:x.size(0)]

class PositionWiseFullyConnectedLayer(nn.Module):

    def __init__(self, d_model, d_ff):
        super(PositionWiseFullyConnectedLayer, self).__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, d_model, heads):
        super(MultiHeadAttentionLayer, self).__init__()

        assert d_model % heads == 0
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):

        batch_size = query.size(1)
        # linear
        # (heads, batch_size, seq_len, d_k)
        query, key, value = \
            [l(x).view(-1, batch_size, self.heads, self.d_k).transpose(0, 2)
            for l, x in zip((self.fc_q, self.fc_k, self.fc_v), (query, key, value))]

        # scaled dot-product attention
        # (heads, batch_size, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        align = F.softmax(scores, dim=-1)
        # (heads, batch_size, seq_len, d_k)
        x = torch.matmul(align, value)

        # concat
        # (seq_len, batch_size, d_model)
        x = x.transpose(0, 2).contiguous().view(-1, batch_size, self.d_model)

        # linear
        return self.fc_out(x)

class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, heads, p_drop):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttentionLayer(d_model, heads)
        self.attn_norm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_drop)
        self.fead_forward = PositionWiseFullyConnectedLayer(d_model, d_ff)
        self.fc_norm = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p_drop)

    def forward(self, x, input_mask):

        # 1st sub-layer
        # Multi-Head Self-Attention
        # residual connection
        x = x + self.dropout1(self.self_attn(x, x, x, input_mask))
        # layer norm
        x = self.attn_norm(x)

        # 2nd sub-layer
        # Position-Wise Fully Connected Feed Forward
        x = x + self.dropout2(self.fead_forward(x))
        # layer norm
        x = self.fc_norm(x)

        return x

class Encoder(nn.Module):

    def __init__(self, d_model, d_input, num_layers, d_ff, heads, seq_len, p_drop):
        super(Encoder, self).__init__()

        self.d_model = d_model

        self.embedding = nn.Embedding(d_input, d_model)
        self.encoding = PositionalEncoding(d_model, seq_len)
        self.dropout = nn.Dropout(p_drop)
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, heads, p_drop) 
                                    for _ in range(num_layers)])

    def forward(self, x, input_mask):
        # x (seq_len, batch_size)

        # (seq_len, batch_size, model_dim)
        embed = self.embedding(x) * math.sqrt(self.d_model)
        encod = self.encoding(embed)
        x = self.dropout(encod)

        for layer in self.layers:
            x = layer(x, input_mask)

        return x

class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, heads, p_drop):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttentionLayer(d_model, heads)
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_drop)
        self.enc_attn = MultiHeadAttentionLayer(d_model, heads)
        self.enc_attn_norm = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p_drop)
        self.fead_forward = PositionWiseFullyConnectedLayer(d_model, d_ff)
        self.fc_norm = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p_drop)

    def forward(self, x, encoder_outputs, input_mask, target_mask):

        # 1st sub-layer
        # Multi-Head Self-Attention
        x = x + self.dropout1(self.self_attn(x, x, x, target_mask))
        x = self.self_attn_norm(x)

        # 2nd sub-layer
        # Encoder Multi-Head Attention
        x = x + self.dropout2(self.enc_attn(x, encoder_outputs, encoder_outputs, input_mask))
        x = self.enc_attn_norm(x)

        # 3rd sub-layer
        # Position-Wise Fully Connected Feed Forward
        x = x + self.dropout3(self.fead_forward(x))
        x = self.fc_norm(x)

        return x

class Decoder(nn.Module):

    def __init__(self, d_model, d_output, num_layers, d_ff, heads, seq_len, p_drop):
        super(Decoder, self).__init__()

        self.d_model = d_model

        self.embedding = nn.Embedding(d_output, d_model)
        self.encoding = PositionalEncoding(d_model, seq_len)
        self.dropout = nn.Dropout(p_drop)
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ff, heads, p_drop) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, d_output)

    def forward(self, x, encoder_outputs, input_mask, target_mask):

        # (seq_len, batch_size, d_model)
        embed = self.embedding(x) * math.sqrt(self.d_model)
        encod = self.encoding(embed)
        x = self.dropout(encod)

        for layer in self.layers:
            x = layer(x, encoder_outputs, input_mask, target_mask)
            
        return F.log_softmax(self.fc_out(x), dim=2)

class Transformer(nn.Module):

    def __init__(self, d_input, d_output, d_model, d_ff, seq_len, num_layers, num_heads, p_drop,
                src_pad_token, trg_pad_token, device):
        super(Transformer, self).__init__()

        self.encoder = Encoder(d_model, d_input, num_layers, d_ff, num_heads, seq_len, p_drop)
        self.decoder = Decoder(d_model, d_output, num_layers, d_ff, num_heads, seq_len, p_drop)

        self.src_pad_token = src_pad_token
        self.trg_pad_token = trg_pad_token
        self.device = device

    def forward(self, input, target):

        # input mask
        # (1, batch_size, 1, input_len)
        input_mask = (input == self.src_pad_token).transpose(0, 1).unsqueeze(1).unsqueeze(0)

        # target mask
        target_len = target.size(0)
        # (target_len, target_len)
        sub_mask = torch.tril(torch.ones((target_len, target_len), device=self.device)) == 0
        # (1, batch_size, 1, target_len)
        pad_mask = (target == self.trg_pad_token).transpose(0, 1).unsqueeze(1).unsqueeze(0)
        # (1, batch_size, target_len, target_len)
        target_mask = sub_mask | pad_mask

        return self.decoder(target, self.encoder(input, input_mask), input_mask, target_mask)