import yaml
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from torchtext.data.utils import get_tokenizer

from seq2seq import Seq2seq

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

    def __init__(self, model, train_loader, val_loader, input_lang, target_lang):

        self.model = model
        self.data_loaders = {'train':train_loader, 'val':val_loader}
        self.pad_token = target_lang.vocab.stoi[target_lang.pad_token]

    def train(self, epochs, learning_rate):

        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.NLLLoss(ignore_index=self.pad_token)
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

def main(data_path, input_lang_name, target_lang_name, save_path,
        epochs, batch_size, learning_rate, dec_name, attn_name,
        attn_size, hidden_size, embed_size, num_layers):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_iterator, val_iterator, input_lang, target_lang = make_datasets(input_lang_name, target_lang_name, batch_size, device)
    SOS_TOKEN = target_lang.vocab.stoi[target_lang.init_token]

    model = Seq2seq(dec_name, attn_name, len(input_lang.vocab), len(target_lang.vocab), embed_size, hidden_size, attn_size, num_layers, SOS_TOKEN, device).to(device)
    trainer = Trainer(model, train_iterator, val_iterator, input_lang, target_lang)
    trainer.train(epochs, learning_rate)

    model.save(f'{save_path}/model.pth', input_lang, target_lang)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a Translator')

    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                    help='Configuration file path')

    args = parser.parse_args()
    kwargs = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    main(**kwargs)