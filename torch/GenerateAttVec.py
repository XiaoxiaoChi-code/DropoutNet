import os
import torch
import pandas as pd
import torch.nn.functional as F
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.datasets import dump_svmlight_file


def item_converting(row, title_list, genre_list):
    title_idx = torch.tensor([[title_list.index(str(row['title']))]]).long()
    title_idx_flatten = title_idx.view(-1)
    title_idx_oneHot = F.one_hot(title_idx_flatten, num_classes=len(title_list))

    num_genre = len(genre_list)
    genre_idx = torch.zeros(1, num_genre).long()
    for genre in str(row['genre']).split("|"):
        idx = genre_list.index(genre)
        genre_idx[0, idx] = 1

    return torch.cat((title_idx_oneHot, genre_idx), 1)


def user_convert_to_oneHotVector(row, gender_list, age_list, occupation_list, zipcode_list,
                                 num_gender, num_age, num_occupation, num_zipcode):
    gender_idx = torch.tensor([[gender_list.index(str(row['gender']))]]).long()
    gender_idx_flatten = gender_idx.view(-1)
    gender_idx_oneHot = F.one_hot(gender_idx_flatten, num_classes=num_gender)

    # encode for age
    age_idx = torch.tensor([[age_list.index(str(row['age']))]]).long()
    age_idx_flatten = age_idx.view(-1)
    age_idx_oneHot = F.one_hot(age_idx_flatten, num_classes=num_age)

    # encode for occupation
    # print(str(row['occupation']))
    occupation_idx = torch.tensor([[occupation_list.index(str(row['occupation']))]]).long()
    occupation_idx_flatten = occupation_idx.view(-1)
    occupation_idx_oneHot = F.one_hot(occupation_idx_flatten, num_classes=num_occupation)

    # encode for zip-code
    zipcode_idx = torch.tensor([[zipcode_list.index(str(row['zipcode'][:5]))]]).long()
    zipcode_idx_flatten = zipcode_idx.view(-1)
    zipcode_idx_oneHot = F.one_hot(zipcode_idx_flatten, num_classes=num_zipcode)

    # 记录每个属性对应的 0-1 向量的长度
    assert torch.concat((gender_idx_oneHot, age_idx_oneHot, occupation_idx_oneHot,
                             zipcode_idx_oneHot), 1).shape[1] == gender_idx_oneHot.shape[1] + age_idx_oneHot.shape[1] + occupation_idx_oneHot.shape[1] + zipcode_idx_oneHot.shape[1]

    # 打印每个属性 对应的0-1向量的维度
    print('the dimension of gender is', gender_idx_oneHot.shape[1])
    print('the dimension of age is ', age_idx_oneHot.shape[1])
    print('the dimension of occupation is ', occupation_idx_oneHot.shape[1])
    print('the dimension of zipcode is ', zipcode_idx_oneHot.shape[1])

    return torch.concat((gender_idx_oneHot, age_idx_oneHot, occupation_idx_oneHot, zipcode_idx_oneHot), 1)


def load_list(fname):
    list = []
    with open(fname, encoding="utf-8") as f:
        for line in f.readlines():
            list.append(line.strip())
    return list


def generate(master_path="contentData"):
    dataset_path = "ml-1m/features"

    # user content
    gender_list = load_list("{}/m_gender.txt".format(dataset_path))
    age_list = load_list("{}/m_age.txt".format(dataset_path))
    occupation_list = load_list("{}/m_occupation.txt".format(dataset_path))
    print(occupation_list)
    zipcode_list = load_list("{}/m_zipcode.txt".format(dataset_path))

    # movie content
    movie_content = pd.read_csv("ml-1m/movies.dat", sep="::", engine='python', header=None, encoding='latin-1')
    movie_content.columns = ['movieID', 'title', 'genre']
    movie_title_list = movie_content['title'].unique().tolist()
    # 提取 unique genres
    movie_genres = movie_content['genre'].tolist()
    movie_genre_list = []
    for i in movie_genres:
        single_genre_list = str(i).split("|")
        for genre in single_genre_list:
            if genre not in movie_genre_list:
                movie_genre_list.append(genre)

    title_list = movie_title_list
    genre_list = movie_genre_list

    user_data = pd.read_csv("ml-1m/users.dat", sep="::", header=None, engine='python')
    user_data.columns = ["userID", "gender", "age", "occupation", "zipcode"]

    rating_data = pd.read_csv("ml-1m/ratings.dat", sep="::", header=None, engine='python')
    rating_data.columns = ['uid', 'movie_id', 'rating', 'date']

    item_data = pd.read_csv("ml-1m/movies.dat", sep="::", header=None, engine='python', encoding='latin-1')
    item_data.columns = ['movie_id', 'title', 'genre']

    num_gender = user_data['gender'].nunique()
    num_age = user_data['age'].nunique()
    num_occupation = user_data['occupation'].nunique()
    num_zipcode = user_data['zipcode'].nunique()

    # hashmap for user profile
    if not os.path.exists("{}/m_user_dict.pkl".format(master_path)):
        user_dict = {}
        for idx, row in user_data.iterrows():
            # u_info = user_converting(row, gender_list, age_list, occupation_list, zipcode_list)
            u_info = user_convert_to_oneHotVector(row, gender_list, age_list, occupation_list, zipcode_list,
                                                  num_gender, num_age, num_occupation, num_zipcode)
            # 将得到的 user one-hot feature vector 存储到这个变量中保存
            print("this is u_info ", u_info)
            user_dict[row['userID']] = u_info
        pickle.dump(user_dict, open("{}/m_user_dict.pkl".format(master_path), "wb"))

    else:
        user_dict = pickle.load(open("{}/m_user_dict.pkl".format(master_path), "rb"))

    if not os.path.exists("{}/m_movie_dict.pkl".format(master_path)):
        movie_dict = {}
        for idx, row in item_data.iterrows():
            m_info = item_converting(row, title_list, genre_list)
            # m_info = item_converting(row, rate_list, genre_list, director_list, actor_list)

            # 将得到的 item one-hot feature vector 放进这个变量中存储
            print("this is m_info ", m_info)
            movie_dict[row['movie_id']] = m_info
        pickle.dump(movie_dict, open("{}/m_movie_dict.pkl".format(master_path), "wb"))
    else:
        movie_dict = pickle.load(open("{}/m_movie_dict.pkl".format(master_path), "rb"))

    return user_dict, movie_dict


if __name__ == "__main__":

    user_dict, movie_dict = generate()

