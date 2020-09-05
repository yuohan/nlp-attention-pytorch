import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauEncoder(nn.Module):
    """ Bahdanau-style decoder
    """
    def __init__(self, input_dim, embed_dim, hidden_dim):
        super(BahdanauEncoder, self).__init__()

        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input):
        """
        Parameters
        ----------
        input: tensor (input_length, batch_size)

        Return
        ------
        output: tensor (input_length, batch_size, hidden_size)
        hidden: tensor (num_layers, batch_size, hidden_size)
        """
        # embed (input_length, batch_size, embed_dim)
        embed = self.embedding(input)
        # output (input_length, batch_size, hidden_dim*2)
        output, hidden = self.rnn(embed)

        # s0 = tanh(Wh)
        # hidden (1, batch_size, hidden_dim)
        hidden = torch.tanh(self.fc(hidden[-1:,:,:]))

        return output, hidden

class LuongEncoder(nn.Module):
    """ Luong-style encoder
    """
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers):
        super(LuongEncoder, self).__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers)

    def forward(self, input):
        """
        Parameters
        ----------
        input: tensor (input_length, batch_size)

        Return
        ------
        output: tensor (input_length, batch_size, hidden_dim)
        hidden: tensor (num_layers, batch_size, hidden_dim)
        """
        # embed (input_length, batch_size, embed_dim)
        embed = self.embedding(input)
        output, hidden = self.rnn(embed)

        return output, hidden

class AttentionLayer(nn.Module):

    def __init__(self, name, query_dim, value_dim, attn_dim):
        super(AttentionLayer, self).__init__()

        if name == 'add':
            self.w1 = nn.Linear(query_dim, attn_dim)
            self.w2 = nn.Linear(value_dim, attn_dim)
            self.v = nn.Linear(attn_dim, 1)
            self.score = self.additive_score

        elif name == 'dot':
            self.score = self.dot_score

        elif name in ['general', 'gen']:
            self.w = nn.Linear(value_dim, query_dim)
            self.score = self.general_score

        elif name in ['concat', 'cat']:
            self.w = nn.Linear(query_dim+value_dim, attn_dim)
            self.v = nn.Linear(attn_dim, 1)
            self.score = self.concat_score

        else:
            raise NotImplemented()

    # query: tensor (batch_size, hidden_dim)
    # hidden state of decoder (previous or current)
    # values: tensor (input_length, batch_size, hidden_dim)
    # hidden state of input sequences from encoder
    def additive_score(self, query, values):
        return self.v(torch.tanh(self.w1(query.unsqueeze(0)) + self.w2(values))).squeeze(2)
    
    def dot_score(self, query, values):
        return torch.sum(torch.mul(query, values), dim=2)

    def general_score(self, query, values):
        return torch.sum(torch.mul(query, self.w(values)), dim=2)

    def concat_score(self, query, values):
        query = query.expand(values.size(0), -1, -1)
        return self.v(torch.tanh(self.w(torch.cat((query, values),dim=2)))).squeeze(2)

    def forward(self, dec_hidden, enc_outputs):
        """
        Parameters
        ----------
        dec_hidden: tensor (batch_size, hidden_dim)
        enc_outputs: tensor (input_length, batch_size, hidden_dim)

        Return
        ------
        conext: tensor (batch_size, hidden_dim)
        score: tensor (input_length, batch_size)
        """
        # calculate attention score
        # score (input_length, batch_size)
        score = self.score(dec_hidden, enc_outputs)
        # align (batch_size, input_length)
        align = F.softmax(score.t(), dim=1)
        # context vector (batch_size, 1, hidden_dim)
        context = torch.bmm(align.unsqueeze(1), enc_outputs.transpose(0, 1))
        return context, align

class BahdanauDecoder(nn.Module):
    """ Bahdanau-style decoder
    """
    def __init__(self, attn_name, output_dim, embed_dim, hidden_dim, attn_dim):
        super(BahdanauDecoder, self).__init__()

        self.attn_name = attn_name
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim

        self.attn = AttentionLayer(attn_name, hidden_dim, hidden_dim*2, attn_dim)
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.rnn = nn.GRU((embed_dim+hidden_dim*2), hidden_dim)
        self.out = nn.Linear((hidden_dim*2+hidden_dim+embed_dim), output_dim)

    def forward(self, input, hidden, enc_outputs):
        """
        Parameters
        ----------
        input: tensor (1, batch_size)
        hidden: tensor (num_layers, batch_size, hidden_dim)
        enc_outputs: tensor (input_length, batch_size, hidden_dim)

        Return
        ------
        output: tensor (batch_size, hidden_dim)
        hidden: tensor (num_layers, batch_size, hidden_dim)
        score: tensor (input_length, batch_size)
        """
        context, score = self.attn(hidden[-1], enc_outputs)

        # embed (1, batch_size, embed_dim)
        embed = self.embedding(input)

        # concat context and embedded
        # (1, batch_size, hidden_dim+embed_dim)
        concat_output = torch.cat((context.transpose(0,1), embed), dim=2)

        # rnn (1, batch_size, hidden_dim)
        # hidden (num_layers*num_directions, batch_size, hidden_dim)
        rnn_output, hidden = self.rnn(concat_output, hidden)

        # output (1, batch_size, output_dim)
        output = F.log_softmax(self.out(torch.cat([embed, rnn_output, context.transpose(0,1)], dim=2)), dim=2)

        return output.squeeze(0), hidden, score

class LuongDecoder(nn.Module):
    """ Luong-style decoder
    """
    def __init__(self, attn_name, output_dim, embed_dim, hidden_dim, attn_dim, num_layers):
        super(LuongDecoder, self).__init__()

        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers)
        self.attention = AttentionLayer(attn_name, hidden_dim, hidden_dim, attn_dim)
        self.concat = nn.Linear(hidden_dim*2, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, enc_outputs):
        """
        Parameters
        ----------
        input: tensor (1, batch_size)
        hidden: tensor (num_layers*num_directions, batch_size, hidden_size)
        enc_outputs: tensor (input_length, batch_size, hidden_size)

        Return
        ------
        output: tensor (batch_size, hidden_size)
        hidden: tensor (num_layers*num_directions, batch_size, hidden_size)
        score: tensor (input_length, batch_size)
        """
        # embed (1, batch_size, embed_size)
        embed = self.embedding(input)

        # rnn (1, batch_size, hidden_size)
        rnn_output, hidden = self.rnn(embed, hidden)

        # get context vector
        context, score = self.attention(rnn_output[0], enc_outputs)

        # concat (batch_size, hidden_size)
        attn_hidden = torch.tanh(self.concat(torch.cat((rnn_output[0], context.squeeze(1)), 1)))

        # output (batch_size, hidden_size)
        output = F.log_softmax(self.out(attn_hidden), dim=1)

        return output, hidden, score

class Seq2seq(nn.Module):

    def __init__(self, name, attn_name, tgt_sos, input_dim, output_dim, 
                embed_dim, hidden_dim, attn_dim, num_layers):
        super(Seq2seq, self).__init__()

        self.name = name
        self.attn_name = attn_name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim
        self.num_layers = num_layers
        self.tgt_sos = tgt_sos

        if name == 'Bahdanau':
            self.encoder = BahdanauEncoder(input_dim, embed_dim, hidden_dim)
            self.decoder = BahdanauDecoder(attn_name, output_dim, embed_dim, hidden_dim, attn_dim)
        elif name == 'Luong':
            self.encoder = LuongEncoder(input_dim, embed_dim, hidden_dim, num_layers)
            self.decoder = LuongDecoder(attn_name, output_dim, embed_dim, hidden_dim, attn_dim, num_layers)
        else:
            raise ValueError()

    def forward(self, src, tgt):

        target_length = tgt.size(0)
        batch_size = src.size(1)

        # encoder
        enc_outputs, enc_hidden = self.encoder(src)

        # decoder
        dec_outputs = []
        attentions = []

        dec_input = torch.tensor(self.tgt_sos, dtype=torch.long, device=tgt.device).expand(1, batch_size)
        dec_hidden = enc_hidden

        for i in range(1, target_length):
            dec_output, dec_hidden, attention = \
                self.decoder(dec_input, dec_hidden, enc_outputs)

            topv, topi = dec_output.topk(1, dim=1)
            dec_input = topi.detach().t()

            dec_outputs.append(dec_output)
            attentions.append(attention.squeeze())

        #(target_length, batch_size, output_size), (target_length, batch_size, input_length)
        return torch.stack(deco_outputs, dim=0), torch.stack(attentions, dim=0)

    @classmethod
    def load(cls, state_path, device):

        state = torch.load(state_path, map_location=device)

        params = state['parameter']
        model = cls(**params).to(device)
        model.load_state_dict(state['state_dict'])

        return model, state['input_lang'], state['target_lang']