import yaml
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from torchtext.data.utils import get_tokenizer

from seq2seq import Seq2seq
from transformer import Transformer

def make_datasets(input_lang_name, target_lang_name, batch_size, device):

    input_lang = Field(tokenize=get_tokenizer('spacy', language=input_lang_name),
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

    target_lang = Field(tokenize=get_tokenizer('spacy', language=target_lang_name),
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

    exts = ('.'+input_lang_name, '.'+target_lang_name)
    train_data, valid_data, _ = Multi30k.splits(exts=exts, fields=(input_lang, target_lang))
    input_lang.build_vocab(train_data)
    target_lang.build_vocab(train_data)

    train_iterator, valid_iterator = BucketIterator.splits((train_data, valid_data), batch_size=batch_size, device=device)
    return train_iterator, valid_iterator, input_lang, target_lang

class Trainer:

    def __init__(self, model, train_loader, val_loader, ignore_index):

        self.model = model
        self.data_loaders = {'train':train_loader, 'val':val_loader}
        self.ignore_index = ignore_index

    def train(self, epochs, learning_rate):

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.NLLLoss(ignore_index=self.ignore_index)
        epoch_loss = {'train':0, 'val':0}

        for epoch in range(epochs):
            print (f'Epoch {epoch+1}/{epochs}')

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                epoch_loss[phase] = 0
                for batch in tqdm(self.data_loaders[phase]):
                    src = batch.src
                    trg = batch.trg
                    trg_len = trg.size(0)

                    optimizer.zero_grad()

                    loss = 0
                    with torch.set_grad_enabled(phase == 'train'):
                        output, _ = self.model(src, trg)

                        d_out = output.shape[-1]
                        output = output.contiguous().view(-1, d_out)
                        trg = trg[1:].contiguous().view(-1)

                        loss += criterion(output, trg)
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        epoch_loss[phase] += loss.item()/ trg_len

                epoch_loss[phase] /= len(self.data_loaders[phase])

            for phase in ['train', 'val']:
                print(f'{phase} Loss: {epoch_loss[phase]:.4f}')
            print('-'*10)

def main(data_path, save_path, src_lang, tgt_lang,
        epochs, batch_size, learning_rate, model_params):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_iterator, val_iterator, src_field, tgt_field = make_datasets(src_lang, tgt_lang, batch_size, device)
    
    tgt_sos = tgt_field.vocab.stoi[tgt_field.init_token]
    src_pad = src_field.vocab.stoi[src_field.pad_token]
    tgt_pad = tgt_field.vocab.stoi[tgt_field.pad_token]

    input_dim = len(src_field.vocab)
    output_dim = len(tgt_field.vocab)

    if model_name == 'Transformer':

        max_len = model_params['max_len']
        model_dim = model_params['model_dim']
        ff_dim = model_params['ff_dim']
        num_layers = model_params['num_layers']
        num_heads = model_params['num_heads']
        drop_prob = model_params['drop_prob']
        model = Transformer(src_pad, tgt_pad, max_len, input_dim, output_dim, model_dim, ff_dim, num_layers, num_heads, drop_prob)

    else:
        attn_name = model_params['attn_name']
        embed_dim = model_params['embed_dim']
        hidden_dim = model_params['hidden_dim']
        attn_dim = model_params['attn_dim']
        num_layers = model_params['num_layers']
        model = Seq2seq(model_name, attn_name, tgt_sos, input_dim, output_dim, embed_dim, hidden_dim, attn_dim, num_layers)
    
    model.to(device)
    trainer = Trainer(model, train_iterator, val_iterator, tgt_pad)
    trainer.train(epochs, learning_rate)

    model.save(f'{save_path}/model.pth', input_lang, target_lang)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a Translator')

    parser.add_argument('src', type=str, help='Source language')
    parser.add_argument('tgt', type=str, help='Target language')
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--save-dir', type=str)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                    help='Configuration file path')

    args = parser.parse_args()
    model_params = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    main(args.data_dir, args.save_dir, args.src, args.tgt, args.epochs, args.batch, args.lr, model_params)