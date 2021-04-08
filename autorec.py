import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch.optim as optim
import pandas as pd


class AutoRec(nn.Module):
    
    def __init__(self, num_users, num_hidden, dropout=0.05):
        super(AutoRec, self).__init__()
        self.encoder = nn.Linear(num_users, num_hidden)
        self.decoder = nn.Linear(num_hidden, num_users)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input):
        hidden = torch.sigmoid(self.dropout(self.encoder(input)))
        pred = self.decoder(hidden)
        if self.training:
            return pred * np.sign(input)
        else:
            return pred

def train():
    ## CONFIG
    num_epochs = 100
    
    train_data = pd.read_csv(
        "ml-1m.train.rating", 
        sep='\t', header=None, names=['user', 'item', 'rating'], 
        usecols=[0, 1,2], dtype={0: np.int32, 1: np.int32, 2:np.int32})

    train_data_matrix = train_data.pivot(index='user', columns='item', values='rating')

    test_data = pd.read_csv(
        "ml-1m.test.rating", 
        sep='\t', header=None, names=['user', 'item', 'rating'], 
        usecols=[0, 1,2], dtype={0: np.int32, 1: np.int32, 2:np.int32})

    test_data_matrix = test_data.pivot(index='user', columns='item',values='rating')

    model = AutoRec(num_users=500, num_hidden=20)

    train_dataloader = torch.utils.data.DataLoader(train_data_matrix, batch_size=32)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    running_loss = 0.0

    for epoch in range(num_epochs):
        for i, data in enumerate(train_dataloader):
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = F.mse_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if epoch % 10 == 0:    # print every 10 epochs
            print('[%d] loss: %.3f' %
                (epoch, running_loss / 10))
            running_loss = 0.0
