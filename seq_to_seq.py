import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)

    def forward(self, input_tensor, hidden):
        """
        Parameters
        ----------
        input: tensor (batch_size, input_length)
        hidden: tensor (num_layers*num_directions, batch_size, hidden_size)

        Return
        ------
        output: tensor (batch_size, input_length, hidden_size)
        hidden: tensor (num_layers*num_directions, batch_size, hidden_size)
        """
        # (batch_size, input_length)
        mask = input_tensor.eq(PAD_TOKEN)
        input_lengths = input_tensor.size(1) - torch.sum(mask, dim=1)

        # embedded (batch_size, input_length, embedding_size)
        embedded = self.embedding(input_tensor)
        packed = pack_padded_sequence(embedded, input_lengths, batch_first=True)
        packed_output, hidden = self.gru(packed)
        output, input_lengths = pad_packed_sequence(packed_output, batch_first=True)

        return output, hidden, mask

class AttentionLayer(nn.Module):

    def __init__(self, score_func, hidden_size, attention_size):
        super(AttentionLayer, self).__init__()

        if score_func in ['add', 'additive']:
            self.w1 = nn.Linear(hidden_size, attention_size)
            self.w2 = nn.Linear(hidden_size, attention_size)
            self.v = nn.Linear(attention_size, 1)
            self.score = self.additive_score

        elif score_func == 'dot':
            self.score = self.dot_score

        elif score_func in ['general', 'mul', 'multiplicative']:
            self.w = nn.Linear(decoder_size, encoder_size)
            self.score = self.general_scoe

        elif score_func == 'concat':
            self.w = nn.Linear(hidden_size*2, attention_size)
            self.v = nn.Linear(attention_size, 1)
            self.score = self.concat_score

    # query: tensor (batch_size, hidden_size)
    # hidden state of decoder (previous or current)
    # value: tensor (batch_size, input_length, hidden_size)
    # hidden state of input sequences from encoder
    def additive_score(self, query, value):
        return self.v(torch.tanh(self.w1(query.unsqueeze(1)) + self.w2(value)))
    
    def dot_score(self, query, value):
        return torch.bmm(value, query.unsqueeze(2))

    def general_score(self, query, value):
        return torch.bmm(value, self.w(query).unsqueeze(2))

    def concat_score(self, query, value)
        query = query.unsqueeze(1).expand(-1, value.size(1), -1)
        return self.v(torch.tanh(self.w(torch.cat((query, value),dim=2))))

    def forward(self, decoder_hidden, encoder_outputs, encoder_mask=None):
        """
        Parameters
        ----------
        decoder_hidden: tensor (batch_size, hidden_size)
        encoder_outputs: tensor (batch_size, input_length, hidden_size)

        Return
        ------
        conext: tensor (batch_size, hidden_size)
        score aligned: tensor (batch_size, input_length)
        """
        # calculate attention score
        score = self.score(decoder_hidden, encoder_outputs)
        # don't attend over padding
        if encoder_mask is not None:
            score = score.masked_fill_(encoder_mask.unsqueeze(2), float('-inf'))
        # align (batch_size, input_length, 1)
        align = F.softmax(score, dim=1)
        # context vector (batch_size, hidden_size)
        context = torch.mul(align, encoder_outputs)
        context = torch.sum(context, dim=1)
        return context, align

class Decoder(nn.Module):

    def __init__(self, embedding_size, hidden_size, output_size, attention):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(output_size, embedding_size)
        self.gru = nn.GRU(embedding_size+hidden_size, hidden_size, batch_first=True)
        self.attention = attention
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        """
        Parameters
        ----------
        input: tensor (batch_size, 1)
        hidden: tensor (num_layers*num_directions, batch_size, hidden_size)
        encoder_outputs: tensor (batch_size, input_length, hidden_size)

        Return
        ------
        output: tensor (batch_size, hidden_size)
        hidden: tensor (num_layers*num_directions, batch_size, hidden_size)
        score: tensor (batch_size, input_length)
        """
        # get context vector
        context, score = self.attention(hidden[-1], encoder_outputs)

        # embedded (batch_size, 1, embedding_size)
        embedded = self.embedding(input)

        # concat context and embedded
        # (batch_size, 1, hidden_size+embedding_size)
        concat = torch.cat((context.unsqueeze(1), embedded), dim=2)

        # output (batch_size, 1, hidden_size)
        # hidden (num_layers*num_directions, batch_size, hidden_size)
        output, hidden = self.gru(concat)

        #attentional hidden state
        output = self.out(output)
        output = F.log_softmax(output.squeeze(1), dim=1)

        return output, hidden, score

class Seq2seq(nn.Module):

    def __init__(self, input_size, output_size, embedding_size, hidden_size, 
                 attention_size, attention, pad_token, device):

        super(Seq2seq, self).__init__()

        if attention == 'additive':
            self.attention = AdditiveAttention(hidden_size, attention_size)
        elif attention == 'dot':
            self.attention = DotAttention()
        elif attention == 'multiplicative' or attention == 'general':
            self.attention = MultiplicativeAttention(hidden_size, hidden_size)
        elif attention == 'concat':
            self.attention = ConcatAttention(hidden_size, attention_size)
        else:
            raise NotImplemented()

        self.encoder = Encoder(input_size, embedding_size, hidden_size)
        self.decoder = Decoder(embedding_size, hidden_size, output_size, self.attention)

        self.input_size = input_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.attention_name = attention

        self.pad_token = pad_token
        self.device = device

    def forward(self, x, target_length):

        batch_size = x.size(0)
        input_length = x.size(1)

        # encoder
        encoder_hidden = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        encoder_outputs, encoder_hidden = self.encoder(x, encoder_hidden)

        # decoder
        decoder_outputs = torch.zeros(target_length, batch_size, self.output_size, device=self.device)
        attentions = torch.zeros(target_length, batch_size, input_length)

        decoder_input = torch.tensor(self.pad_token, dtype=torch.long, device=self.device).expand(batch_size, 1)
        decoder_hidden = encoder_hidden

        for i in range(target_length):
            decoder_output, decoder_hidden, attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

            topv, topi = decoder_output.topk(1, dim=1)
            decoder_input = topi.detach()
            
            decoder_outputs[i] = decoder_output
            attentions[i] = attention.squeeze(2)

        return decoder_outputs, attentions

    def save(self, path, info=None):
        
        state = {
            'state_dict': self.state_dict(),
            'input_size': self.input_size,
            'output_size': self.output_size,
            'target_length': self.target_length,
            'embedding_size': self.embedding_size,
            'hidden_size': self.hidden_size,
            'attention_size': self.attention_size,
            'attention_name': self.attention_name
        }
        if info != None:
            state = {**info, **state}
        torch.save(state, path)