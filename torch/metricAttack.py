import torch
import pandas as pd
import time
import numpy as np


num_latent = 200

# 读取item 的 embedding vectors
items_embeddings = pd.read_csv("contentData/MovieLens_itemsEmbedding.csv", sep=',', header=None).to_numpy()

# member_rec = open("recommendationLists/ml-1m_member_recommendations.txt", 'r')
member_rec = open("ml-1m_member_recommendations.txt", 'r')
member_recommendations_lists = []
for line in member_rec.readlines():
    line = [int(float(x)) for x in line.split(' ')]
    member_recommendations_lists.append(line)
member_recommendations_lists = np.array(member_recommendations_lists)
print("the shape of recommendations for members, ", member_recommendations_lists.shape)


nonmem_rec = open("recommendationLists/ml-1m_cold_user_recommendations.txt", 'r')
nonmem_recommendations_list = []
for line in nonmem_rec.readlines():
    line = [int(float(x)) for x in line.split(' ')]
    nonmem_recommendations_list.append(line)
nonmem_recommendations_array = np.array(nonmem_recommendations_list)
print("the shape of recommendations for non-members ", nonmem_recommendations_array.shape)


interaction_member = {}
with open("processedData/training_data_members.csv", 'r') as member_interactions:
    next(member_interactions)
    for line in member_interactions.readlines():
        line = [int(x) for x in line.strip().split(',')]
        # print(line)
        sessionID = line[0]
        itemID = line[1]
        interaction_member.setdefault(sessionID, []).append(itemID)
members_ids = list(interaction_member.keys()) # 插入顺序
print("the keys of member interactions ", interaction_member.keys())


# read interactions for nonmembers
interaction_nonmem = {}
with open("processedData/training_data_nonmems.csv", 'r') as nonmem_interactions:
    next(nonmem_interactions)
    for line in nonmem_interactions.readlines():
        line = [int(x) for x in line.strip().split(',')]
        sessionID = line[0]
        itemID = line[1]
        interaction_nonmem.setdefault(sessionID, []).append(itemID)
nonmembers_ids = list(interaction_nonmem.keys())
print("the keys of nonmem interactions ", interaction_nonmem.keys())


# read recommendations for reference users
reference_rec = open("recommendationLists/ml-1m_reference_recommendations.txt", 'r')
recommend_reference = []
for line in reference_rec.readlines():
    line = [int(float(x)) for x in line.split(' ')]
    recommend_reference.append(line)
recommend_reference = np.array(recommend_reference)
print("the shape of recommendations for members, ", recommend_reference.shape)




memSimilarity1 = {}
memSimilarity2 = {}

member_S1 = []
member_S2 = []
memberS1minusS2 = []

member_count = 0
num_of_member = 0



vector_member1 = {} # vectors for member interactions
for key, value in interaction_member.items():
    temp_vector = torch.zeros(num_latent)
    length = len(value)
    for i in range(len(value)):
        # print(items_embeddings.shape)
        # print(value[i])
        # print(items_embeddings[168])
        temp_vector = temp_vector + items_embeddings[value[i]]
    temp_vector = temp_vector/ length
    vector_member1[key] = temp_vector

print("the number of keys in vector_member1 is ", len(vector_member1.keys()))
print("all keys of vector_member1 is ", vector_member1.keys())




vector_member2 = {}
for i in range(member_recommendations_lists.shape[0]):
    temp_vector = torch.zeros(num_latent)
    value = member_recommendations_lists[i]
    length = len(value)
    for j in range(len(value)):
        temp_vector = temp_vector + items_embeddings[value[j]]
    if length != 0:
        temp_vector = temp_vector / length
    vector_member2[members_ids[i]] = temp_vector
print("keys in vector_shadow_member2 are ", vector_member2.keys())



vector_member3 = {}
for i in range(member_recommendations_lists.shape[0]):
    temp_vector = torch.zeros(num_latent)
    value = recommend_reference[i]
    length = len(value)
    for j in range(len(value)):
        temp_vector = temp_vector + items_embeddings[value[j]]
    if length != 0:
        temp_vector = temp_vector / length
    vector_member3[members_ids[i]] = temp_vector
print("keys in vector reference recommendations are ", vector_member3.keys())




predicted_value = []
groundTruth_label = []

for key, _ in vector_member1.items():
    memSimilarity1[key] = 1.0 / (torch.sqrt(torch.sum(torch.pow(torch.subtract(vector_member1[key], vector_member2[key]), 2), dim=0)).tolist())
    member_S1.append(memSimilarity1[key])
    memSimilarity2[key] = 1.0 / (torch.sqrt(torch.sum(torch.pow(torch.subtract(vector_member2[key], vector_member3[key]), 2),dim=0)).tolist())
    member_S2.append(memSimilarity2[key])
    memberS1minusS2.append(memSimilarity1[key] - memSimilarity2[key])

    num_of_member = num_of_member + 1
    predicted_value.append(memSimilarity1[key] - memSimilarity2[key])
    groundTruth_label.append(1)

    if memSimilarity1[key] > memSimilarity2[key]:
        member_count = member_count + 1

# np.savetxt("../MetricAttackResults/BERT_100K_memberS1minusS2.csv", memberS1minusS2, delimiter=',')
print("total number of shadow members is {} ".format(num_of_member))
print("the number of members when S1 > S2 is {} ".format(member_count))

print("--------------------------------------------------------------------------------------------------")


nonmemSimilarity1 = {}
nonmemSimilarity2 = {}

nonmember_S1 = []
nonmember_S2 = []

nonmemS1minusS2 = []

nonmem_count = 0
num_of_nonmem = 0

vector_nonmem1 = {}
for key, value in interaction_nonmem.items():
    temp_vector = torch.zeros(num_latent)
    length = len(value)
    for i in range(len(value)):
        temp_vector = temp_vector + items_embeddings[value[i]]
    temp_vector = temp_vector / length
    vector_nonmem1[key] = temp_vector
print("keys in vector_nonmem1 are ", vector_nonmem1.keys())

vector_nonmem2 = {} # vectors for nonmembers recommendations
for i in range(nonmem_recommendations_array.shape[0]):
    temp_vector = torch.zeros(num_latent)
    value = nonmem_recommendations_array[i]
    length = len(value)
    for j in range(len(value)):
        temp_vector = temp_vector + items_embeddings[value[j]]
    temp_vector = temp_vector / length
    vector_nonmem2[nonmembers_ids[i]] = temp_vector
print("keys in vector_nonmem2 are ", vector_nonmem2.keys())




vector_nonmem3 = {} # vectors for nonmembers recommendations
for i in range(nonmem_recommendations_array.shape[0]):
    temp_vector = torch.zeros(num_latent)
    value = recommend_reference[3020+i]
    length = len(value)
    for j in range(len(value)):
        temp_vector = temp_vector + items_embeddings[value[j]]
    temp_vector = temp_vector / length
    vector_nonmem3[nonmembers_ids[i]] = temp_vector

print("keys in vector_nonmem3 are ", vector_nonmem3.keys())


for key, _ in vector_nonmem1.items():
    nonmemSimilarity1[key] = 1.0 / (torch.sqrt(
        torch.sum(torch.pow(torch.subtract(vector_nonmem1[key], vector_nonmem2[key]), 2),
                  dim=0)).tolist())
    nonmember_S1.append(nonmemSimilarity1[key])
    nonmemSimilarity2[key] = 1.0 / (torch.sqrt(
        torch.sum(torch.pow(torch.subtract(vector_nonmem2[key], vector_nonmem3[key]), 2),
                  dim=0)).tolist())
    nonmember_S2.append(nonmemSimilarity2[key])
    nonmemS1minusS2.append(nonmemSimilarity1[key] - nonmemSimilarity2[key])

    num_of_nonmem = num_of_nonmem + 1

    predicted_value.append(nonmemSimilarity1[key] - nonmemSimilarity2[key])
    groundTruth_label.append(0)

    if nonmemSimilarity1[key] < nonmemSimilarity2[key]:
        nonmem_count = nonmem_count + 1

# np.savetxt("../MetricAttackResults/BERT_100K_nonmemS1minusS2.csv", nonmemS1minusS2, delimiter=',')
print("total number of shadow non-members is {} ".format(num_of_nonmem))
print("the number of non-members when S1 < S2 is {} ".format(nonmem_count))


ASR = (member_count + nonmem_count) / (num_of_member + num_of_nonmem)

falsePositive = num_of_nonmem - nonmem_count
falsePositiveRate = falsePositive / num_of_nonmem
truePositiveRate = member_count / num_of_member

print("attack success rate is {}.".format(ASR))
print("false positive rate is {}".format(falsePositiveRate))
print("true positive rate is {}".format(truePositiveRate))



print("**************************************************************")
print("calculating FPRs, TPRs, and all thresholds:  ")


# 获取所有唯一的预测值并排序（作为阈值）
thresholds = np.sort(np.unique(predicted_value))[::-1]

# 计算每个阈值下的 TPR 和 FPR
results = []
for thresh in thresholds:
    predictions = (np.array(predicted_value) >= thresh).astype(int)
    TP = np.sum((predictions == 1) & (np.array(groundTruth_label) == 1))
    FP = np.sum((predictions == 1) & (np.array(groundTruth_label) == 0))
    FN = np.sum((predictions == 0) & (np.array(groundTruth_label) == 1))
    TN = np.sum((predictions == 0) & (np.array(groundTruth_label) == 0))

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    results.append((thresh, FPR, TPR))

# 转为 DataFrame 并打印
roc_data = pd.DataFrame(results, columns=["Threshold", "FPR", "TPR"])
print(roc_data)
roc_data.to_csv('roc_data_ml-100k-bert.csv', index=False)
