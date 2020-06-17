import time
import math
import yaml
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from torchtext.data.utils import get_tokenizer

from seq2seq import Seq2seq

def make_datasets(batch_size, device):

    input_lang = Field(tokenize=get_tokenizer('spacy', language='de'),
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

    target_lang = Field(tokenize=get_tokenizer('spacy', language='en'),
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

    train_data, valid_data, _ = Multi30k.splits(exts = ('.de', '.en'), fields = (src, trg))
    input_lang.build_vocab(train_data)
    target_lang.build_vocab(train_data)

    train_iterator, valid_iterator = BucketIterator.splits((train_data, valid_data), batch_size=batch_size, device=device)
    return train_iterator, input_lang, target_lang

def train(input_tensor, target_tensor, model, optimizer, criterion):

    target_length = target_tensor.size(0)

    optimizer.zero_grad()
    outputs, _ = model(input_tensor, target_length)

    loss = 0
    for i in range(target_length):
        loss += criterion(outputs[i], target_tensor[i])
    loss.backward()
    optimizer.step()

    return loss.item() / target_length

def as_minute(s):
    m = math.floor(s / 60)
    s -= m * 60
    return f'{m:2}m {int(s):2}s'

def print_log(start, now, loss, cur_step, total_step):

    s = now - start
    es = s / (cur_step/total_step)
    rs = es - s

    print(f'\r{as_minute(s)} (-{as_minute(rs)}({cur_step}/{total_step}) {loss:.4f}', end='')

def train_loop(pairs, model, epochs, batch_size, learning_rate, pad_token, device, print_every=5.0):

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.NLLLoss(ignore_index=pad_token)

    for epoch in range(1, epochs+1):

        print (f'epoch: {epoch}/{epochs}')

        batch_loss_total = 0
        start = time.time()
        last = start
        for input_tensor, target_tensor, idx, steps_per_epoch in generate_batch(pairs, batch_size, device):
            loss = train(input_tensor, target_tensor, model, optimizer, criterion)
            batch_loss_total += loss

            now = time.time()
            if (now - last) > print_every or idx == steps_per_epoch:
                last = now
                batch_loss_avg = batch_loss_total / idx
                print_log(start, now, batch_loss_avg, idx, steps_per_epoch)
        print()

def main(data_path, input_lang_name, target_lang_name, save_path,
        epochs, batch_size, learning_rate, dec_name, attn_name,
        attn_size, hidden_size, embed_size, num_layers):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_iterator, input_lang, target_lang = make_datasets(batch_size, device)
    PAD_TOKEN = target_lang.vocab.stoi['<pad>']
    SOS_TOKEN = target_lang.vocab.stoi['<sos>']

    model = Seq2seq(dec_name, attn_name, input_lang.num_words, target_lang.num_words, embed_size, hidden_size, attn_size, num_layers, SOS_TOKEN, device).to(device)
    train_loop(pairs, model, epochs, batch_size, learning_rate, PAD_TOKEN, device)

    model.save(f'{save_path}/model.pth', input_lang, target_lang)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a Translator')

    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                    help='Configuration file path')

    args = parser.parse_args()
    kwargs = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    main(**kwargs)