import spacy
import torch
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

from seq2seq import Seq2seq
from transformer import Transformer

def load_model(model_path, device):

    state = torch.load(model_path, map_location=device)

    params = state['parameter']
    if params['name'] == 'Transformer':
        params.pop('name')
        model = Transformer(**params)
    else:
        model = Seq2seq(**params)
    model.to(device)
    model.load_state_dict(state['state_dict'])

    return model, state['src_lang'], state['tgt_lang'], state['src_vocab'], state['tgt_vocab']

def translate(model, src, tgt_vocab, max_len=50):

    model.eval()

    dec_outputs = [tgt_vocab.stoi['<sos>']]

    if type(model).__name__ == 'Transformer':
        # encoder
        src_mask = model.get_src_mask(src)
        with torch.no_grad():
            enc_outputs = model.encoder(src, src_mask)
        # decoder
        
        for i in range(max_len):
            dec_input = torch.tensor(dec_outputs, dtype=torch.long, device=src.device).unsqueeze(1)
            tgt_mask = model.get_tgt_mask(dec_input)

            with torch.no_grad():
                output, _ = model.decoder(dec_input, enc_outputs, src_mask, tgt_mask)
            
            pred = output.argmax(2)[-1,:].item()
            
            dec_outputs.append(pred)
            if pred == tgt_vocab.stoi['<eos>']:
                break
    else: #seq2seq
        # encoder
        with torch.no_grad():
            enc_outputs, enc_hidden = model.encoder(src)
        # decoder
        dec_hidden = enc_hidden
        for i in range(max_len):
            dec_input = torch.tensor(dec_outputs[-1], dtype=torch.long, device=src.device).view(1,1)

            with torch.no_grad():
                dec_output, dec_hidden, _ = model.decoder(dec_input, dec_hidden, enc_outputs)

            topv, topi = dec_output.topk(1, dim=1)
            pred = topi.item()

            dec_outputs.append(pred)
            if pred == tgt_vocab.stoi['<eos>']:
                break

    translated = [tgt_vocab.itos[i] for i in dec_outputs]
    return translated

def main(text, model_path, show_attention=False):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # load model
    model, src_lang, tgt_lang, src_vocab, tgt_vocab = load_model(model_path, device)

    # text to tensor
    tokens = ['<sos>'] + [token.text.lower() for token in spacy.load(src_lang)(text)] + ['<eos>']
    indices = [src_vocab.stoi[token] for token in tokens]
    src = torch.tensor(indices, dtype=torch.long, device=device).unsqueeze(1)

    translated = translate(model, src, tgt_vocab)

    print ('Input text:')
    print (text)
    print ('Translated text:')
    print (translated)

    if show_attention:
        xticklabels = text.split(' ') + ['<EOS>']
        yticklabels = result[0].split(' ') + ['<EOS>']
        attention = attention[:len(yticklabels)][:len(xticklabels)]
        ax = sns.heatmap(attention, cmap=sns.light_palette("orange", as_cmap=True))
        ax.set_xticklabels(xticklabels, rotation='horizontal')
        ax.set_yticklabels(yticklabels, rotation='horizontal')
        ax.xaxis.set_ticks_position('top')
        plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Translate with pretraied model')

    parser.add_argument('text', type=str,
                    help='Text to be translated')
    parser.add_argument('--model', type=str)
    parser.add_argument('--show-attention', action='store_true',
                    help='Plot attention')

    args = parser.parse_args()
    main(args.text, args.model, args.show_attention)