import numpy as np
import torch


def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def metrics(model, test_loader, top_k):
	HR, NDCG = [], []

	for user, item, label in test_loader:
		user = user.cuda()
		item = item.cuda()

		predictions = model(user, item)
		_, indices = torch.topk(predictions, top_k)
		recommends = torch.take(
				item, indices).cpu().numpy().tolist()

		gt_item = item[0].item()
		HR.append(hit(gt_item, recommends))
		NDCG.append(ndcg(gt_item, recommends))

	return np.mean(HR), np.mean(NDCG)

def rmse(model, loader):
	sumsquares = 0
	i = 1
	total = len(loader)
	print("Beginning evaluation")
	for user, item, label in loader:

		i=i+1
		user = user.cuda()
		item = item.cuda()
		label = label.cuda()
		predictions = model(user, item)
		if (i%10000==0):
			print(i+1, "out of", total, "data points evaluated")
			print("predictions", predictions)
			print("predictions", label)
		msediff = ((predictions - label)**2).sum()
		sumsquares+=msediff

	return torch.sqrt(sumsquares/total)