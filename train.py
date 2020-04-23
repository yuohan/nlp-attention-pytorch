import re
import io
import math
import unicodedata
from itertools import chain

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Tokenizer:

    def __init__(self):

        self.word_index = {'<PAD>': 0}
        self.index_to_word = {0: '<PAD>'}
        self.num_words = 2

    def tokenize(self, x):
        """ Tokenize x

        Parameters
        ----------
        x: List of string
        Input text list

        Return
        ----------
        token_x: List of sequence(List)
        Tokenized input x
        """
        token_x = []
        for text in x:
            token_text = []
            for word in text.split(' '):
                if word not in self.word_index:
                    self.word_index[word] = self.num_words
                    self.num_words += 1
                token = self.word_index[word]
                token_text.append(token)
            token_x.append(token_text)
        return token_x

    def pad(self, x, length=None):
        """ Pad x

        Parameters
        ----------
        x: List of sequence(List)
        Input sequence list
        length: uint
        Length to pad the sequence to. If None, use length of longest sequence in x.

        Return
        ------
        pad_x: List of sequence(List)
        Padded input x.
        """
        if length is None:
            length = max([len(seq) for seq in x])
        pad_token = self.word_index['<PAD>']
        pad_x = []
        for seq in x:
            pad_seq = seq
            if len(seq) < length:
                pad_length = length - len(seq)
                pad_seq = pad_seq + ([pad_token]*pad_length)
            pad_x.append(pad_seq)
        return pad_x

class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input, hidden):
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
        # embedded (batch_size, input_length, hidden_size)
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)

        return output, hidden

class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.w = nn.Linear(hidden_size*2, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        Parameters
        ----------
        decoder_hidden: tensor (num_layers*num_directions, batch_size, hidden_size)
        encoder_outputs: tensor (batch_size, input_length, hidden_size)

        Return
        ------
        conext: tensor (batch_size, hidden_size)
        score aligned: tensor (batch_size, input_length)
        """
        # calculate concat score
        # output of w: (batch_size, input_length, hidden_size*2)
        # output of v: (batch_size, input_length, 1)
        hidden = decoder_hidden.unsqueeze(1).expand(-1, encoder_outputs.size(1), -1)
        score = self.v(torch.tanh(self.w(torch.cat((hidden, encoder_outputs), dim=2))))
        # align (batch_size, input_length, 1)
        align = F.softmax(score, dim=1)
        # context vector (batch_size, hidden_size)
        context = torch.mul(align, encoder_outputs)
        context = torch.sum(context, dim=1)
        return context, align

class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size, attention):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.attention = attention
        self.out = nn.Linear(hidden_size*2, output_size)

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
        # embedded (batch_size, 1, hidden_size)
        embedded = self.embedding(input)
        # output (batch_size, 1, hidden_size)
        # hidden (num_layers*num_directions, batch_size, hidden_size)
        output, hidden = self.gru(embedded, hidden)

        # get context vector
        context, score = self.attention(hidden[-1], encoder_outputs)

        #attentional hidden state
        attn_hidden = torch.tanh(self.out(torch.cat((context, hidden[-1]), dim=1)))
        output = F.log_softmax(attn_hidden, dim=1)

        return output, hidden, score

def load_data(input, target):

    # load data
    with open(f'data/{input}', 'rt') as f:
        source = f.read().split('\n')

    with open(f'data/{target}', 'rt') as f:
        target = f.read().split('\n')

    return source, target

def preprocess(x, y):

    tokenizer_x = Tokenizer()
    tokenizer_y = Tokenizer()
    # tokenize
    x = tokenizer_x.tokenize(x)
    y = tokenizer_y.tokenize(y)
    # pad
    x = tokenizer_x.pad(x)
    y = tokenizer_y.pad(y)

    return x, y, tokenizer_x, tokenizer_y

def train(x, y, x_tk, y_tk, encoder, decoder, optimizer, criterion, device):

    # x: tensor(batch_size, input_length)

    optimizer.zero_grad()
    loss = 0

    batch_size = y.size(0)
    target_length = y.size(1)

    # train encoder
    encoder_hidden = torch.zeros(1, batch_size, encoder.hidden_size, device=device)
    encoder_outputs, encoder_hidden = encoder(x, encoder_hidden)

    # train decoder
    pad_token = y_tk.word_index['<PAD>']
    decoder_input = torch.tensor(pad_token, dtype=torch.long, device=device).expand(batch_size, 1)
    decoder_hidden = encoder_hidden

    for i in range(target_length):
        decoder_output, decoder_hidden, attention = decoder(decoder_input, decoder_hidden, encoder_outputs)

        topv, topi = decoder_output.topk(1, dim=1)
        decoder_input = topi.detach()

        loss += criterion(decoder_output, y[:,i])

    loss.backward()
    optimizer.step()

    return loss.item() / target_length

def train_loop(x, y, x_tk, y_tk, encoder, decoder, device, epochs, batch_size, learning_rate):

    parameters = chain(encoder.parameters(), decoder.parameters())
    optimizer = optim.SGD(parameters, lr=learning_rate)
    criterion = nn.NLLLoss()

    train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.1, random_state=42)

    for epoch in range(epochs):

        for batch in range(math.ceil(len(train_x) / batch_size)):
            batch_x = torch.tensor(train_x[batch*batch_size: (batch+1)*batch_size], dtype=torch.long, device=device)
            batch_y = torch.tensor(train_y[batch*batch_size: (batch+1)*batch_size], dtype=torch.long, device=device)

            loss = train(batch_x, batch_y, x_tk, y_tk, encoder, decoder, optimizer, criterion, device)

            print (f'\repoch {epoch:3} batch {batch:5} training loss {loss:.4f}', end='')
        print ('\n')

def main():

    en, fr = load_data('en', 'fr')

    x, y, x_tk, y_tk = preprocess(fr, en)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 2
    batch_size = 256
    hidden_size = 512
    learning_rate = 0.01
    
    encoder = Encoder(x_tk.num_words, hidden_size).to(device)
    attention = Attention(hidden_size)
    decoder = Decoder(hidden_size, y_tk.num_words, attention).to(device)

    train_loop(x, y, x_tk, y_tk, encoder, decoder, device, epochs, batch_size, learning_rate)
    
if __name__ == '__main__':
    main()