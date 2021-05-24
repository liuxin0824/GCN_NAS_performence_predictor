import json
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import math

from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.utils.data as Data

from models import GCN
from models import GCN2





def normalization(data):
  _range = np.max(data) - np.min(data)
  return (data - np.min(data)) / _range


def id_to_list(id_list):
  arch_list = []
  for id in id_list:
    arch_name = "arch" + str(id)
    arch_list.append(arch_name)
  return arch_list


def list_to_data(arch_list, all_data):
  input_data = []
  output_data = []
  for arch in arch_list:
    input_data.append(all_data[arch]["arch"])
    output_data.append(all_data[arch]["acc"])
  return input_data, np.array(output_data)












model = torch.load("./mymodel.pth")



with open("./Track2_stage2_test.json", 'r') as load_f:
  load_dict = json.load(load_f)



test_arch_id  = range(1,100002)    #//为整数除法



 #固定数据集测试集选项



#index前面加上arch用于查找
test_arch_list = id_to_list(test_arch_id)


test_x, test_y = list_to_data(test_arch_list, load_dict)







def normalize(mx):
  """Row-normalize sparse matrix"""
  rowsum = np.array(mx.sum(1))
  r_inv = np.power(rowsum, -1).flatten()
  r_inv[np.isinf(r_inv)] = 0.
  r_mat_inv = np.diag(r_inv)
  mx = r_mat_inv.dot(mx)
  return mx



train_eye1 = np.eye(16*13)
#train_eye1 = np.eye(16*16)
train_eye1 = torch.FloatTensor(train_eye1) #使用单位阵作为数据（放弃feature仅使用拓扑图
train_eye = np.zeros([13*16,2*16+6+7])#####前32维为16乘2 16block每个block分为两部分，标志同一个opt，后13对应13种超参选择

for i in range(16):
    for j in range(6):
        train_eye[i*13+j][2*i] = 1
    for j in range(7):
        train_eye[i * 13 + j +6][2*i+1] = 1

for index in range(16):
    for i,j in enumerate(range(32,45)):
        train_eye[index*13+i][j] = 1

train_eye = torch.FloatTensor(train_eye) #使用增强版的feature


def traindata_to_adj(train_x, predict_y):
    for j, single_x in enumerate(train_x):

        adj = np.zeros((16 * 13, 16 * 13))
        #adj = np.zeros((16 * 16, 16 * 16))
        if j % 1000 == 0:
            print(j)
        for i, four_meta in enumerate(single_x):
            if i < 15:
                adj_a_1 = i * 13 + four_meta[2] - 1
                adj_a_2 = (four_meta[0] - 1) * 13 + single_x[four_meta[0] - 1][2] - 1

                adj[adj_a_1][adj_a_2] = 1


                #adj[adj_a_2][adj_a_1] = 1

        adj = normalize(adj + np.eye(adj.shape[0]))
        adj = torch.FloatTensor(adj)
        output = model(train_eye1, adj)
        predict_y[j] = output.data


    return predict_y

predict_y = traindata_to_adj(test_x,test_y)


def data_alter(arch_list, all_data ,predict):

  for i,arch in enumerate(arch_list):
    all_data[arch]["acc"] = predict[i]
  return all_data


all_data = data_alter(test_arch_list,load_dict,predict_y)


'''
test_loss = test_loss/len(test_y)

print('TEST MSE LOSS:{:.8f}'.format(test_loss))

test_loss = math.sqrt(test_loss)

print('TEST RMSE LOSS:{:.8f}'.format(test_loss))

x = range(len(test_y))

plt.plot(x, test_y, color="r", linestyle="--", marker=".", label="True")
plt.plot(x, predict_y, color="b", linestyle="--", marker=".", label="Predicted")

plt.xticks([i for i in range(len(test_y))])


plt.legend()

plt.show()

'''



with open("./outcome.json", 'w') as f:
  json.dump(all_data, f)

print(all_data)
