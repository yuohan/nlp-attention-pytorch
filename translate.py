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

def translate(model, input_data, target_lang, max_length, device):

    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.long, device=device).view(-1, 1)
        input_length = input_tensor.size(0)

        output, attention = model(input_tensor, max_length)
        topv, topi = output.topk(1, dim=2)
        predicted_words = target_lang.to_text([topi.squeeze().tolist()])

    return predicted_words, attention.cpu().numpy()

def main(text, model_path, show_attention=False):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # load model
    model, src_lang, tgt_lang, src_vocab, tgt_vocab = load_model(model_path, device)

    # load tokenizer
    input_lang = Tokenizer(None)
    input_lang.load(input_lang_path)

    target_lang = Tokenizer(None)
    target_lang.load(target_lang_path)

    input_data = input_lang.to_token([preprocess(text)])

    result, attention = translate(model, input_data, target_lang, target_lang.max_len, device)

    print ('Input text:')
    print (text)
    print ('Translated text:')
    print (result[0])

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