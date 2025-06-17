# import sys
# sys.path.append('/scratch/rt62/xc5888/UCS-MIA/')
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import argparse
import torch


def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="index for GPU, -1 for CPU")
    args = parser.parse_args()

    return args



class UserItemRatingDataset(Dataset):
    """
    Wrapper, convert <userID, itemID, rating> Tensor into PyTorch Dataset
    """
    def __init__(self, user_tensor, item_tensor, target_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, item):
        return self.user_tensor[item], self.item_tensor[item], self.target_tensor[item]

    def __len__(self):
        return len(self.user_tensor)


class MatrixFactorization(torch.nn.Module):
    def __init__(self, max_index_user, max_index_item, num_hidden_features=200):
        super(MatrixFactorization, self).__init__()
        self.user_emb = torch.nn.Embedding(num_embeddings=max_index_user+1, embedding_dim=num_hidden_features)
        self.item_emb = torch.nn.Embedding(num_embeddings=max_index_item+1, embedding_dim=num_hidden_features)

    def forward(self, user, item):
        u = self.user_emb(user)
        v = self.item_emb(item)
        return (u * v).sum(1)


if __name__ == "__main__":

    # 数据检查
    file_path = "processedData/ml1m_data.csv"
    data = pd.read_csv(file_path, sep=',', engine='python')
    max_index_user = data['uid'].max()

    # for dataset ml-1m
    max_index_item = data['iid'].max()

    # for dataset ml-100k
    # max_index_item = 1682

    # for dataset RecSys Challenge
    # max_index_item = 1306050

    # matrix factorization
    args = get_parameter()
    training_data = data

    training_data_array = training_data.to_numpy()
    users, items, ratings = training_data_array[:, 0].tolist(), training_data_array[:, 1].tolist(), training_data_array[:, 2].tolist()
    users_tensors, items_tensors, ratings_tensors = torch.LongTensor(users), torch.LongTensor(items), torch.FloatTensor(ratings)

    dataset = UserItemRatingDataset(user_tensor=users_tensors,
                                    item_tensor=items_tensors,
                                    target_tensor=ratings_tensors)

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    mf_model = MatrixFactorization(max_index_user=max_index_user, max_index_item=max_index_item).to(args.device)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(mf_model.parameters(), lr=0.01, momentum=0.9)

    epoch = 50
    for t in range(epoch):
        loss_epoch = []
        mf_model.train()
        for _, (user, item, rating) in enumerate(dataloader):
            user, item, rating = user.to(args.device), item.to(args.device), rating.to(args.device)
            pre_rating = mf_model(user, item)
            # print(pre_rating)
            # print(pre_rating.view(-1))
            loss = loss_func(pre_rating, rating)
            # print(loss.item())
            loss_epoch.append(loss.item())
            # backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("for epoch {}, the loss is {}".format(t, sum(loss_epoch) / len(loss_epoch)))

    item_e = mf_model.item_emb.weight.data.cpu()
    item_embedding = pd.DataFrame(item_e.numpy())
    item_embedding.to_csv('contentData/MovieLens_itemsEmbedding_200.csv', index=False, header=None)

    user_e = mf_model.user_emb.weight.data.cpu()
    user_embedding = pd.DataFrame(user_e.numpy())
    user_embedding.to_csv('contentData/MovieLens_usersEmbedding_200.csv', index=False, header=None)




