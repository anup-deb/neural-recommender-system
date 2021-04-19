import torch
import torch.nn as nn
import numpy as np
from torch.autograd import variable
from data import get_data, pinterest
import math
import time
import argparse
import torch.utils.data as Data
import torch.optim as optim
import json


class Autorec(nn.Module):
    def __init__(self,args, num_users,num_items):
        super(Autorec, self).__init__()

        self.args = args
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_units = args.hidden_units
        self.lambda_value = args.lambda_value

        self.encoder = nn.Sequential(
            nn.Linear(self.num_items, self.hidden_units),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_units, self.num_items),
        )


    def forward(self,torch_input):

        encoder = self.encoder(torch_input)
        decoder = self.decoder(encoder)

        return decoder

    def loss(self,decoder,input,optimizer,mask_input):
        cost = 0
        temp2 = 0

        cost += (( decoder - input) * mask_input).pow(2).sum()
        rmse = cost

        for i in optimizer.param_groups:
            for j in i['params']:
                if j.data.dim() == 2:
                    temp2 += torch.t(j.data).pow(2).sum()

        cost += temp2 * self.lambda_value * 0.5
        return cost,rmse

def train(epoch, dataset):

    RMSE = 0
    cost_all = 0
    for step, (batch_x, batch_mask_x, batch_y) in enumerate(loader):

        batch_x = batch_x.type(torch.FloatTensor).cuda()
        batch_mask_x = batch_mask_x.type(torch.FloatTensor).cuda()

        decoder = rec(batch_x)
        loss, rmse = rec.loss(decoder=decoder, input=batch_x, optimizer=optimer, mask_input=batch_mask_x)
        optimer.zero_grad()
        loss.backward()
        optimer.step()
        cost_all += loss
        RMSE += rmse

    RMSE = np.sqrt(RMSE.detach().cpu().numpy() / (train_mask_r == 1).sum())
    print('epoch ', epoch,  ' train RMSE : ', RMSE)
    return RMSE
    #numbers['train RMSE'].append(RMSE)
    #numbers['test loss'].append(loss)
    #numbers['train acc'].append(train_top1)
    #numbers['test acc'].append(top1)
    #numbers['time'].append(time)
    #json.dump(numbers, open(os.path.join(args.model_path, 'values_run-{}.json'.format(args.run)), 'w'))

def test(epoch, dataset):

    test_r_tensor = torch.from_numpy(test_r).type(torch.FloatTensor).cuda()
    test_mask_r_tensor = torch.from_numpy(test_mask_r).type(torch.FloatTensor).cuda()

    #print(f"Test R tensor {test_r_tensor}, {test_r_tensor.shape}")
    decoder = rec(test_r_tensor)
    #print(f"Test Decoder prev {decoder}, {decoder.shape}")
    decoder = torch.from_numpy(np.clip(decoder.detach().cpu().numpy(),a_min=1,a_max=5)).cuda()

    unseen_user_test_list = list(user_test_set - user_train_set)
    unseen_item_test_list = list(item_test_set - item_train_set)

    for user in unseen_user_test_list:
        for item in unseen_item_test_list:
            if test_mask_r[user,item] == 1:
                decoder[user,item] = 3
    
    #print(f"Test Decoder after {decoder}")

    mse = ((decoder - test_r_tensor) * test_mask_r_tensor).pow(2).sum()
    RMSE = mse.detach().cpu().numpy() / (test_mask_r == 1).sum()
    RMSE = np.sqrt(RMSE)

    print('epoch ', epoch, ' test RMSE : ', RMSE)
    return RMSE

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='I-AutoRec ')
    parser.add_argument('--hidden_units', type=int, default=500)
    parser.add_argument('--lambda_value', type=float, default=1)

    parser.add_argument('--train_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)

    parser.add_argument('--optimizer_method', choices=['Adam', 'RMSProp'], default='Adam')
    parser.add_argument('--grad_clip', type=bool, default=False)
    parser.add_argument('--base_lr', type=float, default=1e-3)
    parser.add_argument('--decay_epoch_step', type=int, default=50, help="decay the learning rate for each n epochs")

    parser.add_argument('--random_seed', type=int, default=1000)
    parser.add_argument('--display_step', type=int, default=1)
    parser.add_argument('--dataset', choices=['movielens', 'pinterest'], required=True)
    parser.add_argument('--outputname', type=str, required=True)

    args = parser.parse_args()

    np.random.seed(args.random_seed)

    if args.dataset == 'movielens':
        num_users = 6040
        num_items = 3952
        num_total_ratings = 1000209
        data_name = 'original'
        path = "./%s" % data_name + "/"
        train_ratio = 0.9
        train_r,train_mask_r,test_r,test_mask_r,user_train_set,item_train_set,user_test_set,item_test_set = get_data(path, num_users, num_items, num_total_ratings, train_ratio)
    elif args.dataset == 'pinterest':
        num_users = 55187
        num_items = 9916
        train_r,train_mask_r,test_r,test_mask_r,user_train_set,item_train_set,user_test_set, item_test_set = pinterest()

    print(train_r)
    print(train_r.shape)
    print(test_r)

    args.cuda = torch.cuda.is_available()

    rec = Autorec(args,num_users,num_items)
    if args.cuda:
        rec.cuda()

    optimer = optim.Adam(rec.parameters(), lr = args.base_lr, weight_decay=1e-4)

    num_batch = int(math.ceil(num_users / args.batch_size))

    torch_dataset = Data.TensorDataset(torch.from_numpy(train_r),torch.from_numpy(train_mask_r),torch.from_numpy(train_r))
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    numbers = {'train_rmse': [], 'test_rmse': []}
    for epoch in range(args.train_epoch):
        numbers['train_rmse'].append(train(epoch=epoch, dataset=args.dataset))
        numbers['test_rmse'].append(test(epoch=epoch, dataset=args.dataset))
        json.dump(numbers, open('runs/values_run-{}.json'.format(args.outputname), 'w'))

    torch.save(rec, f"runs/autorec_{args.output_name}.pth")
