import torch
import torch.utils.data as data
import model
import config
import evaluate
import data_utils

if __name__ == '__main__':
    myModel = torch.load("../ncf_models/ml-1m-NeuMF-end.pth")

    train_data, test_data, user_num, item_num, train_mat = data_utils.load_all()

    # construct the train and test datasets
    train_dataset = data_utils.NCFData(
        train_data, item_num, train_mat, 0, False)
    test_dataset = data_utils.NCFData(
        test_data, item_num, train_mat, 0, False)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=1, shuffle=False, num_workers=4)
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=1, shuffle=False, num_workers=0)

    # train_rmse = evaluate.rmse(myModel, train_loader)
    # print("Train rmse is {}", train_rmse)

    test_rmse = evaluate.rmse(myModel, test_loader)
    print("Test rmse is ", test_rmse)


