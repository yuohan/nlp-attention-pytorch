import re
import io
import math
import argparse
import unicodedata

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

from data import Corpus
from seq_to_seq import Seq2seq

def load_data(filepath):

    # load data
    with open(filepath, 'rt') as f:
        data = f.read().split('\n')

    return preprocess(data)

def preprocess(x):

    tokenizer_x = Tokenizer()
    # tokenize
    x = tokenizer_x.tokenize(x)
    # pad
    x = tokenizer_x.pad(x)

    return x, tokenizer_x

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

def train_loop(x, y, model, epochs, batch_size, learning_rate, device, path):

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(ignore_index=0)

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

        model.save(path)

def main(source, target, save_path, attention_name, epochs, 
        batch_size, embedding_size, hidden_size, attention_size, learning_rate):

    x, x_tk = load_data(source)
    y, y_tk = load_data(target)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pad_token = x_tk.word_index['<PAD>']

    model = Seq2seq(x_tk.num_words, y_tk.num_words, embedding_size, hidden_size, attention_size, attention_name, pad_token, device).to(device)

    train_loop(x, y, model, epochs, batch_size, learning_rate, device, save_path)
    x_tk.save('source_corpus.pkl')
    y_tk.save('target_corpus.pkl')
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a Translator')

    parser.add_argument('source', type=str,
                    help='File path of source dataset')
    parser.add_argument('target', type=str,
                    help='File path of target dataset')
    parser.add_argument('--save', type=str, default='model.pkl',
                    help='Model save path')
    parser.add_argument('--type', type=str, default='concat', ## [additive, dot, multiplicative, concat]
                    help='Attention score function')
    parser.add_argument('--epoch', type=int, default=20,
                    help='Number of epochs')
    parser.add_argument('--batch', type=int, default=64,
                    help='Number of batch sizes')
    parser.add_argument('--embed', type=int, default=256,
                    help='Number of embedding dim both encoder and decoder')
    parser.add_argument('--hidden', type=int, default=512,
                    help='Number of features of hidden layer')
    parser.add_argument('--attn', type=int, default=10,
                    help='Number of features of attention')
    parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate')

    args = parser.parse_args()

    main(args.source, args.target, args.save, args.type, args.epoch, args.batch, args.embed, args.hidden, args.attn, args.lr)