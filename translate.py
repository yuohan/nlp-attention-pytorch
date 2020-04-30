import re
import yaml
import argparse
import unicodedata
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from tokenizer import Tokenizer
from seq2seq import Seq2seq, load_model

def preprocess(s):
    # Turn a Unicode string to plain ASCII, thanks to
    # http://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )
    # Lowercase, trim, and remove non-letter characters 
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r'([.!?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    return s

def translate(model, input_data, target_lang, max_length, device):

    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.long, device=device).view(1,-1)
        input_length = input_tensor.size(1)

        output, attention = model(input_tensor, max_length)
        topv, topi = output.topk(1, dim=2)
        predicted_words = target_lang.to_text([topi.squeeze().tolist()])

    return predicted_words, attention.cpu().numpy()

def main(text, model_path, input_lang_path, target_lang_path, show_attention=False):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # load model
    model = load_model(model_path, device)

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
    parser.add_argument('--config', type=str, default='configs/translate_config.yaml',
                    help='Configuration file path')
    parser.add_argument('--show-attention', action='store_true',
                    help='Plot attention')

    args = parser.parse_args()
    kwargs = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    main(args.text, **kwargs, show_attention=args.show_attention)