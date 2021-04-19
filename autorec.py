import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch.optim as optim
import pandas as pd
import random

class AutoRec(nn.Module):
    
    def __init__(self, num_users, num_hidden, dropout=0.20):
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

    def loss(self, predictions, input, optimizer, mask_input, lambda_value):
        cost = 0
        temp2 = 0
        print(predictions)
        print((predictions - input) * mask_input)
        cost += ((predictions - input) * mask_input).pow(2).sum()
        print(cost)
        rmse = cost

        for i in optimizer.param_groups:
            for j in i['params']:
                # print(type(j.data), j.shape,j.data.dim())
                if j.data.dim() == 2:
                    temp2 += torch.nansum(torch.t(j.data).pow(2))

        cost += temp2 * lambda_value * 0.5
        return cost, rmse

def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index+2))
    return 0


def metrics(predictions, test_data, train_data_matrix, top_k):
    HR, NDCG = [], []

    i = 0
    for index, row in test_data.iterrows():
        user, item, label = row
        i = i + 1
        user_predictions = predictions[user, :]
        user_trained = train_data_matrix[user, :]
        #user_trained = np.nan_to_num(user_trained, nan=0)
        user_unrated = np.where(user_trained == 0)[0]
        random100 = np.random.choice(user_unrated, 100)

        user_ns_predictions = user_predictions[random100].detach().numpy()
        user_item_prediction = user_predictions[item].detach().numpy()
        listed_preds = user_ns_predictions.tolist()
        listed_preds.append(user_item_prediction)
        unranked_preds = np.array(listed_preds)
        sorted_inds = np.argsort(unranked_preds)[:top_k]
        if 100 in sorted_inds:
            index = np.where(sorted_inds == 100)[0]
            NDCG.append(np.reciprocal(np.log2(index + 2)))
            HR.append(1)
        else:
            NDCG.append(0)
            HR.append(0)
    return np.mean(HR), np.mean(NDCG)

def train():
    ## CONFIG
    num_epochs = 100
    
    train_data = pd.read_csv(
        "NCF/datasets/ml-1m.train.rating", 
        sep='\t', header=None, names=['user', 'item', 'rating'], 
        usecols=[0, 1,2], dtype={0: np.int32, 1: np.int32, 2:np.int32})

    train_data_matrix = train_data.pivot(index='user', columns='item', values='rating').to_numpy()

    num_users = train_data_matrix.shape[0]
    num_items = train_data_matrix.shape[1]

    #print(train_data_matrix.to_numpy())

    #train_data_matrix = np.nan_to_num(train_data_matrix, nan=0)
    #train_data_matrix = (train_data_matrix>0).astype(float)

    test_data = pd.read_csv(
        "NCF/datasets/ml-1m.test.rating", 
        sep='\t', header=None, names=['user', 'item', 'rating'], 
        usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2:np.int32})

    test_data_matrix = test_data.pivot(index='user', columns='item',values='rating').to_numpy()
    #test_data_matrix = np.nan_to_num(test_data_matrix, nan=0)
    #test_data_matrix = (test_data_matrix>0).astype(float)

    model = AutoRec(num_users=num_users, num_hidden=20)

    #train_dataloader = torch.utils.data.DataLoader(train_data_matrix, batch_size=32)


    torch_data_mat = torch.from_numpy(train_data_matrix).float().T

    train_mask = (torch_data_mat>0).float()
    torch_test_data_mat = torch.from_numpy(test_data_matrix).float().T
    test_mask = (torch_test_data_mat>0).float()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    #print(enumerate(train_dataloader))
    loss_function = nn.MSELoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(torch_data_mat)
        cost, rmse = model.loss(outputs, torch_data_mat, optimizer, train_mask, 1)
        cost.backward()
        optimizer.step()
        if epoch % 10 == 0:    # print every 10 epochs
            #test_preds = model(torch_test_data_mat)
            #test_cost, test_rmse = model.loss(test_preds, torch_test_data_mat, optimizer, test_mask, 1)
            print('[%d] train loss: %.3f' %(epoch, rmse/np.nansum(torch_data_mat)))
            #print('[%d] test loss: %.3f' %(epoch, test_rmse/np.nansum(torch_test_data_mat)))
            #hitrate, ndcg = metrics(outputs.T, test_data, train_data_matrix, 10)
            #print('[%d] hitrate: %.3f ncdf: %.3f' %(epoch, hitrate, ndcg))

if __name__ == "__main__":
    train()
