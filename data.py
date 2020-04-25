import pickle

class Corpus:

    def __init__(self):

        self.word_index = {'<PAD>': 0}
        self.index_to_word = {0: '<PAD>'}
        self.num_words = 1

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
                    self.index_to_word[self.num_words] = word
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

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.word_index, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.word_index = pickle.load(f)
            self.index_to_word = {idx: word for word, idx in self.word_index.items()}