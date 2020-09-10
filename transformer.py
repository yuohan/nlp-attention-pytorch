import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):

    def __init__(self, model_dim, max_len):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, model_dim)
        pos = torch.arange(0, max_len).unsqueeze(1)
        term = torch.exp(torch.arange(0, model_dim, 2) * -(math.log(10000.0) / model_dim))

        pe[:, 0::2] = torch.sin(pos * term)
        pe[:, 1::2] = torch.cos(pos * term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class PositionWiseFullyConnectedLayer(nn.Module):

    def __init__(self, model_dim, ff_dim):
        super(PositionWiseFullyConnectedLayer, self).__init__()

        self.fc1 = nn.Linear(model_dim, ff_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ff_dim, model_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, model_dim, num_heads):
        super(MultiHeadAttentionLayer, self).__init__()

        assert model_dim % num_heads == 0
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.d_k = model_dim // num_heads

        self.fc_q = nn.Linear(model_dim, model_dim)
        self.fc_k = nn.Linear(model_dim, model_dim)
        self.fc_v = nn.Linear(model_dim, model_dim)
        self.fc_out = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value, mask=None):

        batch_size = query.size(1)
        # linear
        # (num_heads, batch_size, seq_len, d_k)
        query, key, value = \
            [l(x).view(-1, batch_size, self.num_heads, self.d_k).transpose(0, 2)
            for l, x in zip((self.fc_q, self.fc_k, self.fc_v), (query, key, value))]

        # scaled dot-product attention
        # (num_heads, batch_size, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        align = F.softmax(scores, dim=-1)
        # (heads, batch_size, seq_len, d_k)
        x = torch.matmul(align, value)

        # concat
        # (seq_len, batch_size, model_dim)
        x = x.transpose(0, 2).contiguous().view(-1, batch_size, self.model_dim)

        # linear
        return self.fc_out(x)

class EncoderLayer(nn.Module):

    def __init__(self, model_dim, ff_dim, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttentionLayer(model_dim, num_heads)
        self.attn_norm = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(drop_prob)
        self.fead_forward = PositionWiseFullyConnectedLayer(model_dim, ff_dim)
        self.fc_norm = nn.LayerNorm(model_dim)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x, src_mask):

        # 1st sub-layer
        # Multi-Head Self-Attention
        # residual connection
        x = x + self.dropout1(self.self_attn(x, x, x, src_mask))
        # layer norm
        x = self.attn_norm(x)

        # 2nd sub-layer
        # Position-Wise Fully Connected Feed Forward
        x = x + self.dropout2(self.fead_forward(x))
        # layer norm
        x = self.fc_norm(x)

        return x

class Encoder(nn.Module):

    def __init__(self, input_dim, model_dim, ff_dim, num_layers, num_heads, max_len, drop_prob):
        super(Encoder, self).__init__()

        self.model_dim = model_dim

        self.embedding = nn.Embedding(input_dim, model_dim)
        self.encoding = PositionalEncoding(model_dim, max_len)
        self.dropout = nn.Dropout(drop_prob)
        self.layers = nn.ModuleList([EncoderLayer(model_dim, ff_dim, num_heads, drop_prob) 
                                    for _ in range(num_layers)])

    def forward(self, x, src_mask):
        # x (src_len, batch_size)

        # (src_len, batch_size, model_dim)
        embed = self.embedding(x) * math.sqrt(self.model_dim)
        encod = self.encoding(embed)
        x = self.dropout(encod)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x

class DecoderLayer(nn.Module):

    def __init__(self, model_dim, ff_dim, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttentionLayer(model_dim, num_heads)
        self.self_attn_norm = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(drop_prob)
        self.enc_attn = MultiHeadAttentionLayer(model_dim, num_heads)
        self.enc_attn_norm = nn.LayerNorm(model_dim)
        self.dropout2 = nn.Dropout(drop_prob)
        self.fead_forward = PositionWiseFullyConnectedLayer(model_dim, ff_dim)
        self.fc_norm = nn.LayerNorm(model_dim)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self, x, encoder_outputs, src_mask, tgt_mask):

        # 1st sub-layer
        # Multi-Head Self-Attention
        x = x + self.dropout1(self.self_attn(x, x, x, tgt_mask))
        x = self.self_attn_norm(x)

        # 2nd sub-layer
        # Encoder Multi-Head Attention
        x = x + self.dropout2(self.enc_attn(x, encoder_outputs, encoder_outputs, src_mask))
        x = self.enc_attn_norm(x)

        # 3rd sub-layer
        # Position-Wise Fully Connected Feed Forward
        x = x + self.dropout3(self.fead_forward(x))
        x = self.fc_norm(x)

        return x

class Decoder(nn.Module):

    def __init__(self, output_dim, model_dim, ff_dim, num_layers, num_heads, max_len, drop_prob):
        super(Decoder, self).__init__()

        self.model_dim = model_dim

        self.embedding = nn.Embedding(output_dim, model_dim)
        self.encoding = PositionalEncoding(model_dim, max_len)
        self.dropout = nn.Dropout(drop_prob)
        self.layers = nn.ModuleList([DecoderLayer(model_dim, ff_dim, num_heads, drop_prob)
                                    for _ in range(num_layers)])
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, x, encoder_outputs, src_mask, tgt_mask):

        # (tgt_len, batch_size, model_dim)
        embed = self.embedding(x) * math.sqrt(self.model_dim)
        encod = self.encoding(embed)
        x = self.dropout(encod)

        for layer in self.layers:
            x = layer(x, encoder_outputs, src_mask, tgt_mask)
            
        return F.log_softmax(self.fc_out(x), dim=2), None

class Transformer(nn.Module):

    def __init__(self, src_pad, tgt_pad, max_len, input_dim, output_dim,
                model_dim, ff_dim, num_layers, num_heads, drop_prob):
        super(Transformer, self).__init__()

        self.src_pad = src_pad
        self.tgt_pad = tgt_pad
        self.max_len = max_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_dim = model_dim
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.drop_prob = drop_prob

        self.encoder = Encoder(input_dim, model_dim, ff_dim, num_layers, num_heads, max_len, drop_prob)
        self.decoder = Decoder(output_dim, model_dim, ff_dim, num_layers, num_heads, max_len, drop_prob)

    def forward(self, src, tgt):

        tgt = tgt[:-1]
        src_mask = self.get_src_mask(src)
        tgt_mask = self.get_tgt_mask(tgt)
        return self.decoder(tgt, self.encoder(src, src_mask), src_mask, tgt_mask)

    def get_src_mask(self, src):
        # source mask
        # (1, batch_size, 1, input_len)
        src_mask = (src == self.src_pad).transpose(0, 1).unsqueeze(1).unsqueeze(0)
        return src_mask

    def get_tgt_mask(self, tgt):
        # target mask
        tgt_len = tgt.size(0)
        # (target_len, target_len)
        sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)) == 0
        # (1, batch_size, 1, target_len)
        pad_mask = (tgt == self.tgt_pad).transpose(0, 1).unsqueeze(1).unsqueeze(0)
        # (1, batch_size, target_len, target_len)
        tgt_mask = sub_mask | pad_mask
        return tgt_mask