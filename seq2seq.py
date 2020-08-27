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

    def forward(self, input_tensor):
        """
        Parameters
        ----------
        input: tensor (input_length, batch_size)

        Return
        ------
        output: tensor (input_length, batch_size, hidden_dim)
        hidden: tensor (num_layers, batch_size, hidden_dim)
        """
        # embedded (input_length, batch_size, embed_dim)
        embedded = self.embedding(input_tensor)
        output, hidden = self.rnn(embedded)

        return output, hidden

class AttentionLayer(nn.Module):

    def __init__(self, name, hidden_size, attn_size):
        super(AttentionLayer, self).__init__()

        if name in ['add', 'additive']:
            self.w1 = nn.Linear(hidden_size, attn_size)
            self.w2 = nn.Linear(hidden_size, attn_size)
            self.v = nn.Linear(attn_size, 1)
            self.score = self.additive_score

        elif name == 'dot':
            self.score = self.dot_score

        elif name in ['general', 'mul', 'multiplicative']:
            self.w = nn.Linear(hidden_size, hidden_size)
            self.score = self.general_score

        elif name == 'concat':
            self.w = nn.Linear(hidden_size*2, attn_size)
            self.v = nn.Linear(attn_size, 1)
            self.score = self.concat_score

        else:
            raise NotImplemented()

    # query: tensor (batch_size, hidden_size)
    # hidden state of decoder (previous or current)
    # values: tensor (input_length, batch_size, hidden_size)
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

    def forward(self, decoder_hidden, encoder_outputs):
        """
        Parameters
        ----------
        decoder_hidden: tensor (batch_size, hidden_size)
        encoder_outputs: tensor (input_length, batch_size, hidden_size)

        Return
        ------
        conext: tensor (batch_size, hidden_size)
        score: tensor (input_length, batch_size)
        """
        # calculate attention score
        # score (input_length, batch_size)
        score = self.score(decoder_hidden, encoder_outputs)
        # align (batch_size, input_length)
        align = F.softmax(score.t(), dim=1)
        # context vector (batch_size, 1, hidden_size)
        context = torch.bmm(align.unsqueeze(1), encoder_outputs.transpose(0, 1))
        return context, align

class BahdanauDecoder(nn.Module):
    """ Bahdanau-style decoder
    """
    def __init__(self, attn_name, output_size, embed_size, hidden_size, attn_dim):
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

    def forward(self, input, hidden, encoder_outputs):
        """
        Parameters
        ----------
        input: tensor (1, batch_size)
        hidden: tensor (num_layers, batch_size, hidden_size)
        encoder_outputs: tensor (input_length, batch_size, hidden_size)

        Return
        ------
        output: tensor (batch_size, hidden_size)
        hidden: tensor (num_layers, batch_size, hidden_size)
        score: tensor (input_length, batch_size)
        """
        context, score = self.attn(hidden[-1], encoder_outputs)

        # embed (1, batch_size, embed_dim)
        embed = self.embedding(input)

        # concat context and embedded
        # (1, batch_size, hidden_size+embed_size)
        concat_output = torch.cat((context.transpose(0,1), embed), dim=2)

        # rnn (1, batch_size, hidden_size)
        # hidden (num_layers*num_directions, batch_size, hidden_size)
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

    def forward(self, input, hidden, encoder_outputs):
        """
        Parameters
        ----------
        input: tensor (1, batch_size)
        hidden: tensor (num_layers*num_directions, batch_size, hidden_size)
        encoder_outputs: tensor (input_length, batch_size, hidden_size)

        Return
        ------
        output: tensor (batch_size, hidden_size)
        hidden: tensor (num_layers*num_directions, batch_size, hidden_size)
        score: tensor (input_length, batch_size)
        """
        # embedded (1, batch_size, embed_size)
        embedded = self.embedding(input)

        # rnn (1, batch_size, hidden_size)
        rnn_output, hidden = self.rnn(embedded, hidden)

        # get context vector
        context, score = self.attention(rnn_output[0], encoder_outputs)

        # concat (batch_size, hidden_size)
        attn_hidden = torch.tanh(self.concat(torch.cat((rnn_output[0], context.squeeze(1)), 1)))

        # output (batch_size, hidden_size)
        output = F.log_softmax(self.out(attn_hidden), dim=1)

        return output, hidden, score

class Seq2seq(nn.Module):

    def __init__(self, style, attn_name, input_dim, output_dim, 
                embed_dim, hidden_dim, attn_dim, num_layers, sos_token, device):
        super(Seq2seq, self).__init__()

        self.style = style
        self.attn_name = attn_name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim
        self.num_layers = num_layers
        self.sos_token = sos_token

        self.device = device

        if style == 'Bahdanau':
            self.encoder = BahdanauEncoder(input_dim, embed_dim, hidden_dim)
            self.decoder = BahdanauDecoder(attn_name, output_dim, embed_dim, hidden_dim, attn_dim)
        elif style == 'Luong':
            self.encoder = LuongEncoder(input_dim, embed_dim, hidden_dim, num_layers)
            self.decoder = LuongDecoder(attn_name, output_dim, embed_dim, hidden_dim, attn_dim, num_layers)
        else:
            raise ValueError()

    def forward(self, x, target_length):

        input_length = x.size(0)
        batch_size = x.size(1)

        # encoder
        encoder_outputs, encoder_hidden = self.encoder(x)

        # decoder
        decoder_outputs = []
        attentions = []

        decoder_input = torch.tensor(self.sos_token, dtype=torch.long, device=self.device).expand(1, batch_size)
        decoder_hidden = encoder_hidden

        for i in range(target_length):
            decoder_output, decoder_hidden, attention = \
                self.decoder(decoder_input, decoder_hidden, encoder_outputs)

            topv, topi = decoder_output.topk(1, dim=1)
            decoder_input = topi.detach().t()

            decoder_outputs.append(decoder_output)
            attentions.append(attention.squeeze())

        #(target_length, batch_size, output_size), (target_length, batch_size, input_length)
        return torch.stack(decoder_outputs, dim=0), torch.stack(attentions, dim=0)

    def save(self, path, input_lang, target_lang, info=None):

        state = {
            'state_dict': self.state_dict(),
            'style': self.style,
            'attention_name': self.attn_name,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'embedding_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim,
            'attn_dim': self.attn_dim,
            'num_layers': self.num_layers,
            'input_lang': input_lang,
            'target_lang': target_lang
        }
        if info != None:
            state = {**info, **state}
        torch.save(state, path)

    @classmethod
    def load(cls, state_path, device):

        state = torch.load(state_path, map_location=device)

        dec_name = state['style']
        attn_name = state['attention_name']
        attn_size = state['attn_dim']
        embed_size = state['embedding_dim']
        hidden_size = state['hidden_dim']
        input_size = state['input_dim']
        output_size = state['output_dim']
        num_layrs = state['num_layers']

        model = cls(dec_name, attn_name, input_size, output_size, embed_size, hidden_size, attn_size, num_layers, device).to(device)
        model.load_state_dict(state['state_dict'])

        return model, state['input_lang'], state['target_lang']