import numpy as np
import pandas as pd
import pickle


def load_user_content():
    user_content = pickle.load(open("contentData/m_user_dict.pkl", 'rb'))
    user_ids = sorted(user_content.keys())
    # print(user_ids)
    num_users = max(user_ids) + 1
    user_feature_dim = len(user_content[max(user_ids)][0])
    # print(user_feature_dim)
    user_array = np.zeros((num_users, user_feature_dim))
    # print(user_array.shape)
    for uid, vec in user_content.items():
        # print(vec.numpy())
        user_array[uid] = vec.numpy()
    return user_array


if __name__ == "__main__":
    user_content = load_user_content()
    print(user_content.shape)
    lower_endpoint = 9
    upper_endpoint = 29
    indices = np.arange(lower_endpoint, upper_endpoint+1)
    perturb_user_content = []
    for user in user_content[1:]:
        # print(user)
        non_zero_indices = np.nonzero(np.array(user))[0]
        # print(non_zero_indices)
        ori_index = non_zero_indices[2] # 这个值肯定会落在 [9,29] 这个区间内
        left_indices = indices[indices != ori_index]
        np.random.seed(2022)
        random_index = np.random.choice(left_indices)
        user[random_index] = 1
        user[ori_index] = 0
        perturb_user_content.append(user)

    perturb_user_content = np.array(perturb_user_content)

    zero_row = np.zeros((1, perturb_user_content.shape[1]))
    perturb_user_content = np.vstack((zero_row, perturb_user_content))

    np.savetxt("contentData/ml-1m_user_content-perturb.txt", perturb_user_content)

    print("perturbing user content information is completed. ")

    perturbed_content = np.loadtxt("contentData/ml-1m_user_content-perturb.txt")
    print(perturbed_content.shape)

