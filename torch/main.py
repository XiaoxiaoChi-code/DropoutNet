import numpy as np
import pandas as pd
import scipy
import torch
import datetime
from sklearn import datasets
from tqdm import tqdm
import argparse
import os
import pickle

import utils
import data
import model

# n_users = 1497020 + 1
# n_items = 1306054 + 1

def main():
    # data_path           = args.data_dir
    checkpoint_path     = args.checkpoint_path
    tb_log_path         = args.tb_log_path
    model_select        = args.model_select

    rank_out            = args.rank
    user_batch_size     = 1000
    n_scores_user       = 2500
    data_batch_size     = 100
    dropout             = args.dropout
    recall_at           = range(50, 550, 50)
    eval_batch_size     = 1000
    max_data_per_step   = 2500000
    eval_every          = args.eval_every
    num_epoch           = 10

    _lr = args.lr
    _decay_lr_every = 50
    _lr_decay = 0.1

    experiment = '%s_%s' % (
        datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'),
        '-'.join(str(x / 100) for x in model_select) if model_select else 'simple'
    )
    print('running: ' + experiment)

    # dat = load_data(data_path)
    dat = load_data()
    u_pref_scaled = dat['u_pref_scaled']
    v_pref_scaled = dat['v_pref_scaled']
    eval_warm = dat['eval_warm']
    eval_cold_user = dat['eval_cold_user']
    # eval_cold_item = dat['eval_cold_item']
    user_content = dat['user_content']
    item_content = dat['item_content']
    u_pref = dat['u_pref']
    v_pref = dat['v_pref']
    user_indices = dat['user_indices']

    timer = utils.timer(name='main').tic()

    # append pref factors for faster dropout
    v_pref_expanded = np.vstack([v_pref_scaled, np.zeros_like(v_pref_scaled[0, :])])
    v_pref_last = v_pref_scaled.shape[0]
    u_pref_expanded = np.vstack([u_pref_scaled, np.zeros_like(u_pref_scaled[0, :])])
    u_pref_last = u_pref_scaled.shape[0]
    timer.toc('initialized numpy data')

    # prep eval
    eval_batch_size = eval_batch_size
    timer.tic()
    eval_warm.init_tf(u_pref_scaled, v_pref_scaled, user_content, item_content, eval_batch_size)
    timer.toc('initialized eval_warm').tic()
    eval_cold_user.init_tf(u_pref_scaled, v_pref_scaled, user_content, item_content, eval_batch_size)
    timer.toc('initialized eval_cold_user').tic()
    # eval_cold_item.init_tf(u_pref_scaled, v_pref_scaled, user_content, item_content, eval_batch_size)
    # timer.toc('initialized eval_cold_item').tic()

    dropout_net = model.get_model(latent_rank_in=u_pref.shape[1],
                               user_content_rank=user_content.shape[1],
                               item_content_rank=item_content.shape[1],
                               model_select=model_select,
                               rank_out=rank_out)

    row_index = np.copy(user_indices)
    n_step = 0
    best_cold_user = 0
    best_cold_item = 0
    best_warm = 0
    n_batch_trained = 0
    best_step = 0
    optimizer = torch.optim.SGD(dropout_net.parameters(), args.lr, momentum=0.9)
    crit = torch.nn.MSELoss()
    d_train = torch.device(args.model_device)
    d_eval = torch.device(args.inf_device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=_decay_lr_every, gamma=_lr_decay)
    dropout_net.to(d_train)
    dropout_net.train()

    for epoch in range(num_epoch):
        np.random.shuffle(row_index)
        for b in utils.batch(row_index, user_batch_size):
            n_step += 1
            # prep targets
            target_users = np.repeat(b, n_scores_user)
            target_users_rand = np.repeat(np.arange(len(b)), n_scores_user)
            target_items_rand = [np.random.choice(v_pref.shape[0], n_scores_user) for _ in b]
            target_items_rand = np.array(target_items_rand).flatten()
            target_ui_rand = np.transpose(np.vstack([target_users_rand, target_items_rand]))
            
            preds_pref = np.matmul(u_pref[b, :], v_pref.T)
            preds_pref = torch.tensor(preds_pref)
            target_scores, target_items = torch.topk(preds_pref, k=n_scores_user, sorted=True)
            random_scores = preds_pref.detach().cpu().numpy()[target_ui_rand[:,0],target_ui_rand[:,1]]


            # merge topN and randomN items per user
            target_scores = np.append(target_scores, random_scores)
            target_items = np.append(target_items, target_items_rand)
            target_users = np.append(target_users, target_users)

            n_targets = len(target_scores)
            perm = np.random.permutation(n_targets)
            n_targets = min(n_targets, max_data_per_step)
            data_batch = [(n, min(n + data_batch_size, n_targets)) for n in range(0, n_targets, data_batch_size)]
            f_batch = 0
            pbar = tqdm(data_batch, desc='ubatch')
            
            for (start, stop) in pbar:
                batch_perm = perm[start:stop]
                batch_users = target_users[batch_perm]
                batch_items = target_items[batch_perm]
                if dropout != 0:
                    n_to_drop = int(np.floor(dropout * len(batch_perm)))
                    perm_user = np.random.permutation(len(batch_perm))[:n_to_drop]
                    perm_item = np.random.permutation(len(batch_perm))[:n_to_drop]
                    batch_v_pref = np.copy(batch_items)
                    batch_u_pref = np.copy(batch_users)
                    batch_v_pref[perm_user] = v_pref_last
                    batch_u_pref[perm_item] = u_pref_last
                else:
                    batch_v_pref = batch_items
                    batch_u_pref = batch_users

                Uin = u_pref_expanded[batch_u_pref, :]
                Vin = v_pref_expanded[batch_v_pref, :]
                # Ucontent = user_content[batch_users, :].todense()
                Ucontent = user_content[batch_users, :]
                # Vcontent = item_content[batch_items, :].todense()
                Vcontent = item_content[batch_items, :]
                targets = target_scores[batch_perm]
                
                # Uin = torch.tensor(Uin).to(d_train)
                # Vin = torch.tensor(Vin).to(d_train)
                # Ucontent = torch.tensor(Ucontent).to(d_train)
                # Vcontent = torch.tensor(Vcontent).to(d_train)
                # targets = torch.tensor(targets).to(d_train)

                Uin = torch.tensor(Uin).to(torch.float32).to(d_train)
                Vin = torch.tensor(Vin).to(torch.float32).to(d_train)
                Ucontent = torch.tensor(Ucontent).to(torch.float32).to(d_train)
                Vcontent = torch.tensor(Vcontent).to(torch.float32).to(d_train)
                targets = torch.tensor(targets).to(torch.float32).to(d_train)
                
                preds, U_embedding, V_embedding = dropout_net.forward(Uin, Vin, Ucontent, Vcontent)
                loss = crit(preds, targets)
                loss_out = loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                f_batch += loss_out
                if np.isnan(f_batch):
                    raise Exception('f is nan')
                n_batch_trained += 1
                pbar.set_description(f'updates={n_batch_trained/1000:.0f}k f={loss_out:.4f} f_tot={f_batch:.2f}')
            # step after every ubatch, decay is based on # of ubatch
            scheduler.step()

            if n_step % eval_every == 0:
                dropout_net.to(d_eval)
                dropout_net.eval()

                recall_warm      = dropout_net.evaluate(recall_k=recall_at, eval_data=eval_warm,      name="warm",      device=d_eval)
                recall_cold_user = dropout_net.evaluate(recall_k=recall_at, eval_data=eval_cold_user, name="cold_user", device=d_eval)

                # recall_cold_item = dropout_net.evaluate(recall_k=recall_at, eval_data=eval_cold_item, device=d_eval)

                dropout_net.to(d_train)
                dropout_net.train()

                # checkpoint
                # agg_cur = np.sum(recall_warm + recall_cold_user + recall_cold_item)
                agg_cur = np.sum(recall_warm + recall_cold_user)
                # agg_best = np.sum(best_warm + best_cold_user + best_cold_item)
                agg_best = np.sum(best_warm + best_cold_user)
                if agg_cur > agg_best:
                    best_cold_user = recall_cold_user
                    # best_cold_item = recall_cold_item
                    best_warm      = recall_warm
                    best_step      = n_step

                timer.toc('%d [%d]b [%d]tot f=%.2f best[%d]' % (
                    n_step, len(data_batch), n_batch_trained, f_batch, best_step
                )).tic()
                print ('\t\t'+' '.join([('@'+str(i)).ljust(6) for i in recall_at]))
                # print('warm start\t%s\ncold user\t%s\ncold item\t%s' % (
                #     ' '.join(['%.4f' % i for i in recall_warm]),
                #     ' '.join(['%.4f' % i for i in recall_cold_user]),
                #     ' '.join(['%.4f' % i for i in recall_cold_item])
                # ))

                print('warm start\t%s\ncold user\t%s' % (
                    ' '.join(['%.4f' % i for i in recall_warm]),
                    ' '.join(['%.4f' % i for i in recall_cold_user]),

                ))


# def load_data(data_path):
#     timer = utils.timer(name='main').tic()
#     split_folder = os.path.join(data_path, 'warm')
#
#     u_file                  = os.path.join(data_path, 'trained/warm/U.csv.bin')
#     v_file                  = os.path.join(data_path, 'trained/warm/V.csv.bin')
#     user_content_file       = os.path.join(data_path, 'user_features_0based.txt')
#     item_content_file       = os.path.join(data_path, 'item_features_0based.txt')
#     train_file              = os.path.join(split_folder, 'train.csv')
#     test_warm_file          = os.path.join(split_folder, 'test_warm.csv')
#     test_warm_iid_file      = os.path.join(split_folder, 'test_warm_item_ids.csv')
#     test_cold_user_file     = os.path.join(split_folder, 'test_cold_user.csv')
#     test_cold_user_iid_file = os.path.join(split_folder, 'test_cold_user_item_ids.csv')
#     test_cold_item_file     = os.path.join(split_folder, 'test_cold_item.csv')
#     test_cold_item_iid_file = os.path.join(split_folder, 'test_cold_item_item_ids.csv')
#
#     dat = {}
#     # load preference data
#     timer.tic()
#     u_pref = np.fromfile(u_file, dtype=np.float32).reshape(n_users, 200)
#     v_pref = np.fromfile(v_file, dtype=np.float32).reshape(n_items, 200)
#     dat['u_pref'] = u_pref
#     dat['v_pref'] = v_pref
#
#     timer.toc('loaded U:%s,V:%s' % (str(u_pref.shape), str(v_pref.shape))).tic()
#
#     # pre-process
#     _, dat['u_pref_scaled'] = utils.prep_standardize(u_pref)
#     _, dat['v_pref_scaled'] = utils.prep_standardize(v_pref)
#     timer.toc('standardized U,V').tic()
#
#     # load content data
#     timer.tic()
#     user_content, _ = datasets.load_svmlight_file(user_content_file, zero_based=True, dtype=np.float32)
#     dat['user_content'] = user_content.tolil(copy=False)
#     timer.toc('loaded user feature sparse matrix: %s' % (str(user_content.shape))).tic()
#     item_content, _ = datasets.load_svmlight_file(item_content_file, zero_based=True, dtype=np.float32)
#     dat['item_content'] = item_content.tolil(copy=False)
#     timer.toc('loaded item feature sparse matrix: %s' % (str(item_content.shape))).tic()
#
#     # load split
#     timer.tic()
#     train = pd.read_csv(train_file, delimiter=",", header=None, dtype=np.int32).values.ravel().view(
#         dtype=[('uid', np.int32), ('iid', np.int32), ('inter', np.int32), ('date', np.int32)])
#     dat['user_indices'] = np.unique(train['uid'])
#     timer.toc('read train triplets %s' % train.shape).tic()
#
#     dat['eval_warm'] = data.load_eval_data(test_warm_file, test_warm_iid_file, name='eval_warm', cold=False,
#                                            train_data=train)
#     dat['eval_cold_user'] = data.load_eval_data(test_cold_user_file, test_cold_user_iid_file, name='eval_cold_user',
#                                                 cold=True,
#                                                 train_data=train)
#     dat['eval_cold_item'] = data.load_eval_data(test_cold_item_file, test_cold_item_iid_file, name='eval_cold_item',
#                                                 cold=True,
#                                                 train_data=train)
#     return dat


def load_content_data():
    # load content data
    # users' content
    user_content = pickle.load(open("contentData/m_user_dict.pkl", 'rb'))
    user_ids = sorted(user_content.keys())
    print(user_ids)
    num_users = max(user_ids) + 1
    user_feature_dim = len(user_content[max(user_ids)][0])
    print(user_feature_dim)
    user_array = np.zeros((num_users, user_feature_dim))
    print(user_array.shape)
    for uid, vec in user_content.items():
        # print(vec.numpy())
        user_array[uid] = vec.numpy()

    # items' content
    movie_content = pickle.load(open("contentData/m_movie_dict.pkl", 'rb'))
    movie_ids = sorted(movie_content.keys())
    num_movies = max(movie_ids) + 1
    item_feature_dim = len(movie_content[max(movie_ids)][0])

    movie_array = np.zeros((num_movies, item_feature_dim))
    for iid, vec in movie_content.items():
        movie_array[iid] = vec.numpy()

    return user_array, movie_array


def load_data():
    # preference data file
    user_preference_file = "contentData/MovieLens_masked_usersEmbedding_200.csv"
    item_preference_file = "contentData/MovieLens_itemsEmbedding_200.csv"

    # content data file
    users_content_vectors, items_content_vectors = load_content_data()

    train_file = "processedData/training_data_members.csv"
    test_warm_file = "processedData/testing_data_members.csv"
    test_warm_iid_file = "contentData/item_ids.txt"

    test_cold_user_file = "processedData/testing_data_nonmems.csv"
    test_cold_user_iid_file = "contentData/item_ids.txt"

    dat = {}
    u_pref = pd.read_csv(user_preference_file, header=None).to_numpy()
    v_pref = pd.read_csv(item_preference_file, header=None).to_numpy()

    dat['u_pref'] = u_pref
    dat['v_pref'] = v_pref

    # pre-process
    _, dat['u_pref_scaled'] = utils.prep_standardize(u_pref)
    _, dat['v_pref_scaled'] = utils.prep_standardize(v_pref)

    dat['item_content'] = items_content_vectors
    dat['user_content'] = users_content_vectors

    train = pd.read_csv(train_file, delimiter=',', dtype=np.int32).values.ravel().view(
        dtype=[('uid', np.int32), ('iid', np.int32), ('rating', np.int32), ('time', np.int32)]
    )

    dat['user_indices'] = np.unique(train['uid'])
    dat['eval_warm'] = data.load_eval_data(test_warm_file, test_warm_iid_file, name='eval_warm', cold=False,
                                           train_data=train)
    dat['eval_cold_user'] = data.load_eval_data(test_cold_user_file, test_cold_user_iid_file, name='eval_cold_user',
                                                cold=True, train_data=train)

    return dat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo script to run DropoutNet on RecSys data",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--data-dir', type=str, required=True, help='path to eval in the downloaded folder')

    parser.add_argument('--model_device', type=str, default='cuda:0', help='device to use for training')
    parser.add_argument('--inf_device', type=str, default='cpu', help='device to use for inference')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='path to dump checkpoint data from TensorFlow')
    parser.add_argument('--tb_log_path', type=str, default=None,
                        help='path to dump TensorBoard logs')
    parser.add_argument('--model_select', nargs='+', type=int,
                        default=[800, 400],
                        help='specify the fully-connected architecture, starting from input,'
                             ' numbers indicate numbers of hidden units',
                        )
    parser.add_argument('--rank', type=int, default=200, help='output rank of latent model')
    parser.add_argument('--dropout', type=float, default=0.5, help='DropoutNet dropout')
    parser.add_argument('--eval_every', type=int, default=2, help='evaluate every X user-batch')

    # 我把 learning rate 调小了十倍，原作者设置的值是 0.005
    parser.add_argument('--lr', type=float, default=0.0005, help='starting learning rate')


    args = parser.parse_args()
    main()
