import argparse
import torch

from data import Corpus
from seq_to_seq import Seq2seq

def translate(x):
    
    x = torch.tensor(x, dtype=torch.long).view(1,-1)
    with torch.no_grad():
        outputs, attentions = model(x)

    topv, topi = outputs.topk(1, dim=2)
    return topi.squeeze().tolist()

def main(text, model_path, source_path, target_path):

    # load model
    state = torch.load(model_path, map_location=torch.device('cpu'))
    attention_name = state['attention_name']
    attention_size = state['attention_size']
    embedding_size = state['embedding_size']
    hidden_size = state['hidden_size']
    input_size = state['input_size']
    output_size = state['output_size']
    target_length = state['target_length']

    model = Seq2seq(input_size, output_size, target_length, embedding_size, hidden_size, attention_size, attention_name, 0, 'cpu')
    model.load_state_dict(state['state_dict'])

    # load corpus
    source_corpus = Corpus()
    source_corpus.load(source_path)

    target_corpus = Corpus()
    target_corpus.load(target_path)

    x = source_corpus.tokenize([text])

    result = translate(x)
    pred_text = ' '.join(en_tk.index_to_word[tk] for tk in tesult)

    print ('Input text:')
    print (text)
    print ('Translated text:')
    print (pred_text)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Translate with pretraied model')

    parser.add_argument('text', type=str,
                    help='Text to be translated')
    parser.add_argument('--model', type=str, default='model.pth',
                    help='Pretrained model path')
    parser.add_argument('--source', type=str, default='source.pkl',
                    help='Source corpus path')
    parser.add_argument('--target', type=str, default='target.pkl',
                    help='Targer corpus path')
    args = parser.parse_args()
    main(args.text, args.model, args.source, args.target)