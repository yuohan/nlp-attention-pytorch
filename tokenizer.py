import pickle

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

class Tokenizer:

    def __init__(self, name):

        self.name = name
        self.word2index = {'<PAD>': PAD_TOKEN, '<SOS>': SOS_TOKEN, '<EOS>': EOS_TOKEN}
        self.index2word = {PAD_TOKEN: '<PAD>', SOS_TOKEN: '<SOS>', EOS_TOKEN: '<EOS>'}
        self.num_words = 3
        self.max_len = 0

    def to_token(self, x):
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
                if word not in self.word2index:
                    self.word2index[word] = self.num_words        
                    self.index2word[self.num_words] = word
                    self.num_words += 1
                token = self.word2index[word]
                token_text.append(token)
            #append EOS token
            token_text.append(EOS_TOKEN)
            if len(token_text) > self.max_len:
                self.max_len = len(token_text)
            token_x.append(token_text)
        return token_x

    def to_text(self, x):
        """ Reconstruct token to text

        Parameters
        ----------
        x: List of list
        Input sequence list

        Return
        ----------
        text: List of string
        """
        text_list = []
        for seq in x:
            end_pos = seq.index(EOS_TOKEN) if EOS_TOKEN in seq else len(seq)
            text = ' '.join([self.index2word[t] for t in seq[:end_pos]])
            text_list.append(text)
        return text_list

    def save(self, path):
        with open(path, 'wb') as f:
            obs = {'name': self.name, 'corpus':self.word2index, 'max_len':self.max_len}
            pickle.dump(obs, f)

    def load(self, path):
        with open(path, 'rb') as f:
            obs = pickle.load(f)
            self.name = obs['name']
            self.word2index = obs['corpus']
            self.index2word = {idx: word for word, idx in self.word2index.items()}
            self.num_words = len(self.word2index)
            self.max_len = obs['max_len']