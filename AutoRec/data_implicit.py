import numpy as np
import argparse
import math
import pandas as pd
import scipy.sparse as sp


def get_data(path, num_users, num_items, num_total_ratings, train_ratio):

    fp = open(path + "ratings.dat")

    user_train_set = set()
    user_test_set = set()
    item_train_set = set()
    item_test_set = set()

    train_r = np.zeros((num_users, num_items))
    test_r = np.zeros((num_users, num_items))

    train_mask_r = np.zeros((num_users, num_items))
    test_mask_r = np.zeros((num_users, num_items))

    random_perm_idx = np.random.permutation(num_total_ratings)
    train_idx = random_perm_idx[0:int(num_total_ratings * train_ratio)]
    test_idx = random_perm_idx[int(num_total_ratings * train_ratio):]

    lines = fp.readlines()

    ''' Train '''
    for itr in train_idx:
        line = lines[itr]
        user, item, rating, _ = line.split("::")
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        train_r[user_idx, item_idx] = 1
        train_mask_r[user_idx, item_idx] = 1

        user_train_set.add(user_idx)
        item_train_set.add(item_idx)

    ''' Test '''
    for itr in test_idx:
        line = lines[itr]
        user, item, rating, _ = line.split("::")
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        test_r[user_idx, item_idx] = 1
        test_mask_r[user_idx, item_idx] = 1

        user_test_set.add(user_idx)
        item_test_set.add(item_idx)

    return train_r,train_mask_r,test_r,test_mask_r,user_train_set,item_train_set,user_test_set,item_test_set


def pinterest():
    """ We load all the three file here to save time in each epoch. """
    train_data = pd.read_csv("pinterest-20.train.rating", sep='\t', header=None, names=['user', 'item'], usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

    test_data = pd.read_csv("pinterest-20.test.rating", sep='\t', header=None, names=['user', 'item'], usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

    user_num = max(train_data['user'].max(), test_data['user'].max()) + 1
    item_num = max(train_data['item'].max(), test_data['item'].max()) + 1

    user_train_set = set()
    user_test_set = set()
    item_train_set = set()
    item_test_set = set()

    train_mask_r = np.zeros((user_num, item_num))
    test_mask_r = np.zeros((user_num, item_num))

    train_data_val = train_data.values.tolist()
    test_data_val = test_data.values.tolist()

    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data_val:
        train_mat[x[0], x[1]] = 1.0
        train_mask_r[x[0], x[1]] = 1.0

        user_train_set.add(x[0])
        item_train_set.add(x[1])

    test_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in test_data_val:
        test_mat[x[0], x[1]] = 1.0
        test_mask_r[x[0], x[1]] = 1.0

        user_test_set.add(x[0])
        item_test_set.add(x[1])

    #test_data = []
    #with open(config.test_negative, 'r') as fd:
    #		line = fd.readline()
    #	while line != None and line != '':
    #		arr = line.split('\t')
    #		u = eval(arr[0])[0]
    #		test_data.append([u, eval(arr[0])[1]])
    #		for i in arr[1:]:
    #			test_data.append([u, int(i)])
    #		line = fd.readline()

    return train_mat.toarray(),train_mask_r,test_mat.toarray(),test_mask_r,user_train_set,item_train_set,user_test_set,item_test_set