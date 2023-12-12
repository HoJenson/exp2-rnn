import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from model import Classifier, GRUNet
from torch.utils.data import DataLoader, random_split
from data import data_preprocess, CustomDataset


class Lab3Model(object):
    def __init__(self, batch_size=64, num_workers=10, path='data/yelp.csv'):
        
        X, labels = data_preprocess(path)
        self.dataset = CustomDataset(X=X, y=labels)
        valid_size = 1000
        test_size = 1000
        train_size = len(self.dataset) - valid_size - test_size
        self.train_dataset, self.valid_dataset, self.test_dataset = random_split(self.dataset, 
                                                                [train_size, valid_size, test_size],
                                                                generator=torch.Generator().manual_seed(0))
        self.train_dataloader = DataLoader(self.train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True,
                                           num_workers=num_workers,
                                           drop_last=True,
                                           collate_fn=collate_fn)
        self.valid_dataloader = DataLoader(self.valid_dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=num_workers,
                                           drop_last=True,
                                           collate_fn=collate_fn)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.net = None
        self.lr = None
        self.optimizer = None
        self.device = None
        self.schedule = None
        self.fig_name = None
        self.loss_list = {"train": [], "val": []}
        self.acc_list = {"train": [], "val": []}

    def train(self, lr=0.01, epochs=10, device='cuda', wait=8, lrd=False, hidden_size=128, 
              num_layers=2, p=0.1, bidirectional=True):
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        print(self.device)
        self.lr = lr
        corpus_size, embedding_dim = self.dataset.weights_matrix.shape
        self.net = GRUNet(corpus_size=corpus_size, 
                          embedding_dim=embedding_dim, 
                          hidden_size=hidden_size,
                          load_embed=True,
                          weights_matrix=self.dataset.weights_matrix, 
                          num_layers=num_layers, 
                          p=p,
                          bidirectional=bidirectional).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)        
        
        if lrd:
            self.schedule = ReduceLROnPlateau(self.optimizer, 'min', patience=1, verbose=True)
        
        total_params = sum([param.nelement() for param in self.net.parameters() if param.requires_grad])
        print(">>> Total params: {}".format(total_params))
        
        print(">>> Start training")
        min_val_loss = np.inf
        delay = 0
        for epoch in range(epochs):
            self.net.train()
            train_loss = 0.0
            train_acc = 0.0
            for data in tqdm(self.train_dataloader):
                self.optimizer.zero_grad()
                inputs = data['input']
                labels = data["label"]
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.net(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                train_acc += self.acc(labels=labels.cpu().numpy(), outputs=outputs.detach().cpu().numpy())
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            
            train_loss = train_loss / len(self.train_dataloader)
            train_acc = train_acc / len(self.train_dataloader)
            self.loss_list['train'].append(train_loss)
            self.acc_list['train'].append(train_acc)

            self.net.eval()
            val_loss = 0.0
            val_acc = 0.0
            for data in self.valid_dataloader:
                inputs = data["input"]
                labels = data["label"]
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.net(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                val_loss += loss.item()
                val_acc += self.acc(labels=labels.cpu().numpy(), outputs=outputs.detach().cpu().numpy())
            
            val_loss = val_loss / len(self.valid_dataloader)
            val_acc = val_acc / len(self.valid_dataloader)
            self.loss_list['val'].append(val_loss)
            self.acc_list['val'].append(val_acc)
            
            print(f"Epoch {epoch}: train loss {train_loss:10.6f}, acc {train_acc:7.4f}, "
                      f"val loss {val_loss:10.6f}, acc {val_acc:7.4f}, ")
            

            if lrd:
                self.schedule.step(val_loss)

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_loss_acc = val_acc
                print(f"Update min_val_loss to {min_val_loss:10.6f}")
                delay = 0
            else:
                delay = delay + 1

            if delay > wait:
                break
        
        print(">>> Finished training")
        self.plot_loss()
        self.plot_acc()
        print(">>> Finished plot loss")
        return min_val_loss_acc
    
    def acc(self, labels, outputs):
        pre_labels = np.argmax(outputs, axis=1)
        labels = labels.reshape(len(labels))
        acc = np.sum(pre_labels == labels) / len(pre_labels)
        return acc
    
    def plot_loss(self):
        plt.figure()
        train_loss = self.loss_list['train']
        val_loss = self.loss_list['val']
        plt.plot(train_loss, c="red", label="train_loss")
        plt.plot(val_loss, c="blue", label="val_loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("CrossEntropyLoss")
        plt.title("CrossEntropyLoss of Train and Validation in each Epoch")
        plt.savefig(f"fig/{self.fig_name}_loss.png")

    def plot_acc(self):
        plt.figure()
        train_acc = self.acc_list['train']
        val_acc = self.acc_list['val']
        plt.plot(train_acc, c="red", label="train_acc")
        plt.plot(val_acc, c="blue", label="val_acc")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy of Train and Validation in each Epoch")
        plt.savefig(f"fig/{self.fig_name}_acc.png")

    def test(self):
        test_data_loader = DataLoader(dataset=self.test_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=self.num_workers,
                                      drop_last=True,
                                      collate_fn=collate_fn)
        test_acc = 0.0
        self.net.eval()
        for data in test_data_loader:
            inputs = data["input"]
            labels = data["label"]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.net(inputs)
            test_acc += self.acc(labels=labels.cpu().numpy(), outputs=outputs.detach().cpu().numpy())
        test_acc = test_acc / len(test_data_loader)
        return test_acc

def collate_fn(batch):
    inputs = [item['input'] for item in batch]
    labels = [item['label'] for item in batch]

    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)

    return {
        'input': inputs_padded,
        'label': torch.stack(labels)
    }