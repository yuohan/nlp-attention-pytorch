import re
import io
import math
import unicodedata

from sklearn.model_selection import train_test_split

import torch
import torch.optim as optim

from seq_to_seq import Seq2seq

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

def train(x, y, model, optimizer, criterion, target_length):

    # x: tensor(batch_size, input_length)

    optimizer.zero_grad()
    outputs, _ = model(x, target_length)
    
    loss = 0
    for i in range(target_length):
        loss += criterion(outputs[i], y[:,i])
    loss.backward()
    optimizer.step()

    return loss.item() / target_length

def validate(x, y, model, criterion, target_length):

    with torch.no_grad():
        outputs, _ = model(x, target_length)

        loss = 0
        for i in range(target_length):
            loss += criterion(outputs[i], y[:,i])
        
    return loss.item() / target_length

def train_loop(x, y, model, epochs, batch_size, learning_rate, device):

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.1, random_state=42)
    val_x = torch.tensor(val_x, dtype=torch.long, device=device)
    val_y = torch.tensor(val_y, dtype=torch.long, device=device)

    target_length = len(train_y[0])

    for epoch in range(epochs):

        for batch in range(math.ceil(len(train_x) / batch_size)):
            batch_x = torch.tensor(train_x[batch*batch_size: (batch+1)*batch_size], dtype=torch.long, device=device)
            batch_y = torch.tensor(train_y[batch*batch_size: (batch+1)*batch_size], dtype=torch.long, device=device)

            train_loss = train(batch_x, batch_y, model, optimizer, criterion, target_length)

            print (f'\repoch {epoch+1:3} training loss {train_loss:.4f}', end='')

        valid_loss = validate(val_x, val_y, model, criterion, target_length)
        print (f'\repoch {epoch+1:3} training loss {train_loss:.4f} validation loss {valid_loss:.4f}')

def main():

    en, fr = load_data('en', 'fr')

    x, y, x_tk, y_tk = preprocess(fr, en)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 10
    batch_size = 64
    embedding_size = 256
    hidden_size = 1024
    attention_size = 10
    learning_rate = 0.001

    attention = 'additive'
    # [additive, dot, multiplicative, concat]
    pad_token = x_tk.word_index['<PAD>']

    model = Seq2seq(x_tk.num_words, y_tk.num_words, embedding_size, hidden_size, attention_size, attention, pad_token, device).to(device)

    train_loop(x, y, model, epochs, batch_size, learning_rate, device)
    
if __name__ == '__main__':
    main()