import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, input_size, embed_size, hidden_size, num_layers, bidirectional=True):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(input_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, bidirectional=bidirectional)

    def forward(self, input_tensor):
        """
        Parameters
        ----------
        input: tensor (input_length, batch_size)

        Return
        ------
        output: tensor (input_length, batch_size, hidden_size)
        hidden: tensor (num_layers, batch_size, hidden_size)
        """
        # embedded (input_length, batch_size, embed_size)
        embedded = self.embedding(input_tensor)
        output, hidden = self.rnn(embedded)

        if self.bidirectional:
            output = output[:, :, :self.hidden_size] + output[:, : ,self.hidden_size:]
            hidden = hidden[:self.num_layers]

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
    def __init__(self, attn_name, output_size, embed_size, hidden_size, attn_size, num_layers):
        super(BahdanauDecoder, self).__init__()

        self.attn_name = attn_name
        self.output_size = output_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.attn_size = attn_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_size, embed_size)
        self.rnn = nn.GRU(embed_size+hidden_size, hidden_size)
        self.attention = AttentionLayer(attn_name, hidden_size, attn_size)
        self.out = nn.Linear(hidden_size, output_size)

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

        # get context vector
        context, score = self.attention(hidden[-1], encoder_outputs)

        # concat context and embedded
        # (1, batch_size, hidden_size+embed_size)
        concat_output = torch.cat((context.transpose(0,1), embedded), dim=2)

        # rnn (1, batch_size, hidden_size)
        # hidden (num_layers*num_directions, batch_size, hidden_size)
        rnn_output, hidden = self.rnn(concat_output)

        #attentional hidden state
        output = F.log_softmax(self.out(rnn_output.squeeze(0)), dim=1)

        return output, hidden, score

class LuongDecoder(nn.Module):
    """ Luong-style decoder
    """
    def __init__(self, attn_name, output_size, embed_size, hidden_size, attn_size, num_layers):
        super(LuongDecoder, self).__init__()

        self.attn_name = attn_name
        self.output_size = output_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.attn_size = attn_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers)
        self.attention = AttentionLayer(attn_name, hidden_size, attn_size)
        self.concat = nn.Linear(hidden_size*2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

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
        context, score = self.attention(hidden[-1], encoder_outputs)

        # concat (batch_size, hidden_size)
        concat_output = torch.tanh(self.concat(torch.cat((hidden[-1], context.squeeze(1)), 1)))

        # output (batch_size, hidden_size)
        output = F.log_softmax(self.out(concat_output), dim=1)

        return output, hidden, score 

class Seq2seq(nn.Module):

    def __init__(self, dec_name, attn_name, input_size, output_size, 
                embed_size, hidden_size, attn_size, num_layers, sos_token, device):
        super(Seq2seq, self).__init__()

        self.dec_name = dec_name
        self.attn_name = attn_name
        self.input_size = input_size
        self.output_size = output_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.attn_size = attn_size
        self.num_layers = num_layers
        self.sos_token = sos_token

        self.device = device

        self.encoder = Encoder(input_size, embed_size, hidden_size, num_layers)
        if dec_name == 'Bahdanau':
            self.decoder = BahdanauDecoder(attn_name, output_size, embed_size, hidden_size, attn_size, num_layers)
        elif dec_name == 'Luong':
            self.decoder = LuongDecoder(attn_name, output_size, embed_size, hidden_size, attn_size, num_layers)
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
            'decoder_name': self.dec_name,
            'attention_name': self.attn_name,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'embedding_size': self.embed_size,
            'hidden_size': self.hidden_size,
            'attention_size': self.attn_size,
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

        dec_name = state['decoder_name']
        attn_name = state['attention_name']
        attn_size = state['attention_size']
        embed_size = state['embedding_size']
        hidden_size = state['hidden_size']
        input_size = state['input_size']
        output_size = state['output_size']
        num_layrs = state['num_layers']

        model = cls(dec_name, attn_name, input_size, output_size, embed_size, hidden_size, attn_size, num_layers, device).to(device)
        model.load_state_dict(state['state_dict'])

        return model, state['input_lang'], state['target_lang']