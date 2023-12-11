import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, random_split

import string as st

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

import torchtext
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
from torchtext import data
from torchtext import vocab

def data_preprocess(path):
    df = pd.read_csv(path)
    df['text'] = df['text'].astype(str).apply(str.lower)
    df['stars'] = df['stars'].apply(lambda x: x - 1)
    texts = df.text
    labels = df.stars

    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    punctuations = st.punctuation

    X = []
    for text in list(texts):
        temp_list = []
        tokens = word_tokenize(str(text))
        for token in tokens:
            if token not in punctuations:
                if token == 'not':
                    temp_list.append(token)
                elif token not in stop_words and '...' not in token:
                    stem = ps.stem(token)
                    temp_list.append(stem)
        X.append(' '.join(temp_list))

    return X, labels


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

        unique_words = set()
        for string in X:
            words = string.split()
            unique_words.update(words)

        unique_words = list(unique_words)
        embedding_dim = 100
        global_vectors = GloVe(name='6B', dim=embedding_dim) # 42B, 840B

        corpus_size = len(unique_words)
        weights_matrix = np.zeros((corpus_size, embedding_dim))

        for i, word in enumerate(unique_words):
            word_vector = global_vectors.get_vecs_by_tokens(word)
            if word_vector.sum().item() == '0':
                weights_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim, ))
            else:
                weights_matrix[i] = word_vector

        self.unique_words = unique_words
        self.weights_matrix = weights_matrix

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sentence = self.X[idx]
        label = self.y[idx]

        indices = [self.unique_words.index(word) for word in sentence.split()]

        return {
            'input': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

def get_dataset(path):
    X, labels = data_preprocess(path)
    dataset = CustomDataset(X=X, y=labels)
    valid_size = 1000
    test_size = 1000
    train_size = len(dataset) - valid_size - test_size
    train_dataset, valid_dataset, test_dataset = random_split(dataset, 
                                                            [train_size, valid_size, test_size],
                                                            generator=torch.Generator().manual_seed(0))
    return train_dataset, valid_dataset, test_dataset