import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torchtext.vocab import GloVe

class Data(Dataset):
    
    def __init__(self, path):
        df = pd.read_csv(path)
        X = df.X
        self.y = df.labels

        unique_words = set()
        for string in X:
            words = str(string).split()
            unique_words.update(words)
        
        word_to_id = {}
        unique_words = list(unique_words)
        embedding_dim = 100
        corpus_size = len(unique_words)
        global_vectors = GloVe(name='6B', dim=embedding_dim) # 42B, 840B
        weights_matrix = np.zeros((corpus_size, embedding_dim))
        for i, word in enumerate(unique_words):
            word_to_id[word] = i
            word_vector = global_vectors.get_vecs_by_tokens(word)
            if word_vector.sum().item() == '0':
                weights_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim, ))
            else:
                weights_matrix[i] = word_vector

        self.unique_words = unique_words
        self.weights_matrix = weights_matrix
        
        self.X = []
        
        for sentence in X:
            indices = [word_to_id[word] for word in str(sentence).split()]
            self.X.append(indices)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        indices = self.X[idx]
        label = self.y[idx]

        return {
            'input': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }