import numpy as np
import pandas as pd
from datetime import datetime


def process_ml1m():
    rating_file_path = "ml-1m/ratings.dat"
    data = pd.read_csv(rating_file_path, sep="::", header=None, engine='python')
    data.columns = ['uid', 'iid', 'rating', 'time']
    data['time'] = data['time'].astype(int)
    data = data.sort_values(by=['uid', 'time'])
    data.to_csv("ml1m_data.csv", index=False)
    # data['time'] = data['time'].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
    last_10 = data.groupby('uid').tail(10).reset_index(drop=True) # 作为 test_data

    last_10.to_csv("last_10_interactions_per_user.csv", index=False)

    last_10['is_test'] = True
    data_with_flag = data.merge(last_10[['uid', 'iid', 'rating', 'time', 'is_test']],
                                on=['uid', 'iid', 'rating', 'time'],
                                how='left')
    train = data_with_flag[data_with_flag['is_test'].isna()].drop(columns='is_test').reset_index(drop=True)
    train.to_csv("train_data.csv", index=False)

    print("the number of unique users ", data['uid'].nunique())

    # print("the process finished")


def split_users():
    # 把所有的 user ids 划分成两部分， 即 member ids 和  non-member ids
    train_data = pd.read_csv("processedData/train_data.csv")
    unique_user_ids = train_data['uid'].unique()
    # 随机挑选 6040/2 用户 作为 member ids ， 剩下的作为 non-member ids
    np.random.seed(20)
    half_users = np.random.choice(unique_user_ids, size=len(unique_user_ids)//2, replace=False)
    members_ids = half_users
    non_members_ids = list(set(unique_user_ids) - set(members_ids))

    return members_ids, non_members_ids

    # print(members_ids)
    # print(len(members_ids))
    # print(non_members_ids)
    # print(len(non_members_ids))


def split_data():
    mem_ids, nonmem_ids = split_users()
    train_data = pd.read_csv("processedData/train_data.csv")
    test_data = pd.read_csv("processedData/last_10_interactions_per_user.csv")

    members_train_data = train_data[train_data['uid'].isin(mem_ids)].reset_index(drop=True)
    nonmems_train_data = train_data[train_data['uid'].isin(nonmem_ids)].reset_index(drop=True)

    members_train_data.to_csv("training_data_members.csv", index=False)
    nonmems_train_data.to_csv("training_data_nonmems.csv", index=False)

    members_test_data = test_data[test_data['uid'].isin(mem_ids)].reset_index(drop=True)
    nonmems_test_data = test_data[test_data['uid'].isin(nonmem_ids)].reset_index(drop=True)

    members_test_data.to_csv("testing_data_members.csv", index=False)
    nonmems_test_data.to_csv("testing_data_nonmems.csv", index=False)








if __name__ == "__main__":
    # process_ml1m()
    # split_users()
    split_data()

    # root_path = "D:/CodeTraining/DropoutNet-Data/recsys2017.pub/recsys2017.pub/eval/warm/"
    # train_file = root_path + "train.csv"
    # train_file_data = pd.read_csv(train_file, delimiter=",", header=None, dtype=np.int32)
    # train_file_data.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    # print(train_file_data.head())
    # print(train_file_data.shape[0])
    # unique_user_ids = train_file_data['user_id'].nunique()
    # print("the number of unique users is ", unique_user_ids)
    #
    # rating_unique_values = train_file_data['rating'].unique()
    # print(rating_unique_values)  #输出值是 [0 1 2 3 5]

