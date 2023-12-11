import torch
import torch.nn as nn

class Classifier(nn.Module):
  def __init__(self, corpus_size, embedding_dim, hidden_size, 
               load_embed=False, weights_matrix=None, 
               trainable_embedding=False, dropout=0.1, num_layers=2):
      super().__init__()
      self.num_layers = num_layers
      self.hidden_size = hidden_size
      self.dropout = dropout
      self.class_size = 5

      self.embedding = nn.Embedding(corpus_size, embedding_dim)

      if load_embed and weights_matrix is not None:
        self.embedding.load_state_dict({'weight': torch.tensor(weights_matrix)})

      self.embedding.weight.requires_grad = trainable_embedding

      self.gru = nn.GRU(embedding_dim, hidden_size, self.num_layers, 
                      dropout=self.dropout, bidirectional=True, batch_first=True)

      self.output = nn.Linear(hidden_size*2, self.class_size)

  def forward(self, x):
      embedding = self.embedding(x)
      out, _ = self.gru(embedding)
      out = out[:, -1, :]
      out = self.output(out)
      return out
  
class GRUNet(nn.Module):
  def __init__(self, corpus_size, embedding_dim, hidden_size,
               trainable_embedding=False, load_embed=False, 
               weights_matrix=None, num_layers=2, p=0.1,
               bidirectional=True):
      super(GRUNet, self).__init__()
      self.hidden_size = hidden_size
      self.num_layers = num_layers
      self.p = p
      self.output_dim = 5

      self.embedding = nn.Embedding(corpus_size, embedding_dim)
      if load_embed and weights_matrix is not None:
        self.embedding.load_state_dict({'weight': torch.tensor(weights_matrix)})
      self.embedding.weight.requires_grad = trainable_embedding

      self.gru = nn.GRU(embedding_dim, hidden_size, self.num_layers, 
                        bidirectional=bidirectional, batch_first=True)
      
      if not bidirectional:
          self.num_directions = 1
          self.linear = nn.Linear(self.hidden_size, self.hidden_size)
      else:
          self.num_directions = 2
          self.linear = nn.Linear(self.hidden_size * 2, self.hidden_size)

      self.fc = nn.Sequential(
         self.linear,
         nn.Dropout(p=self.p),
         nn.ReLU(),
         nn.Linear(self.hidden_size, self.output_dim)
      )

  def forward(self, x):
      embedding = self.embedding(x)
      y, _ = self.gru(embedding, None)
      y = self.fc(y[:, -1, :])
      return y