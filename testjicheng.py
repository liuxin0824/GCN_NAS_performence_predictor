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












model1 = torch.load("./mymodel1.pth")
model2 = torch.load("./mymodel2.pth")
model3 = torch.load("./mymodel3.pth")
model4 = torch.load("./mymodel4.pth")
model5 = torch.load("./mymodel5.pth")
model6 = torch.load("./mymodel6.pth")
model7 = torch.load("./mymodel7.pth")
model8 = torch.load("./mymodel8.pth")
model9 = torch.load("./mymodel9.pth")

model10 = torch.load("./mymodel10.pth")
'''
model11 = torch.load("./mymodel11.pth")
model12 = torch.load("./mymodel12.pth")
model13 = torch.load("./mymodel13.pth")
model14 = torch.load("./mymodel14.pth")
model15 = torch.load("./mymodel15.pth")'''
'''
model16 = torch.load("./mymodel16.pth")
model17 = torch.load("./mymodel17.pth")
model18 = torch.load("./mymodel18.pth")
model19 = torch.load("./mymodel19.pth")
model20 = torch.load("./mymodel20.pth")
'''



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

        output_list = []
        output = model1(train_eye1, adj).data
        output_list.append((output))
        output = model2(train_eye1, adj).data
        output_list.append((output))
        output = model3(train_eye1, adj).data
        output_list.append((output))
        output = model4(train_eye1, adj).data
        output_list.append((output))
        output = model5(train_eye1, adj).data
        output_list.append((output))
        output = model6(train_eye1, adj).data
        output_list.append((output))
        output = model7(train_eye1, adj).data
        output_list.append((output))
        output = model8(train_eye1, adj).data
        output_list.append((output))
        output = model9(train_eye1, adj).data
        output_list.append((output))
        output = model10(train_eye1, adj).data
        output_list.append((output))
        '''
        output = model11(train_eye1, adj).data
        output_list.append((output))
        output = model12(train_eye1, adj).data
        output_list.append((output))
        output = model13(train_eye1, adj).data
        output_list.append((output))
        output = model14(train_eye1, adj).data
        output_list.append((output))
        output = model15(train_eye1, adj).data
        output_list.append((output))'''
        '''
        output = model16(train_eye1, adj).data
        output_list.append((output))
        output = model17(train_eye1, adj).data
        output_list.append((output))
        output = model18(train_eye1, adj).data
        output_list.append((output))
        output = model19(train_eye1, adj).data
        output_list.append((output))
        output = model20(train_eye1, adj).data
        output_list.append((output))
        '''
        #output_list.sort()

        outputdata = 0
        for i in range(0,10):
            outputdata+=output_list[i]


        predict_y[j] = outputdata/10




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



with open("./outcomejicheng.json", 'w') as f:
  json.dump(all_data, f)

print(all_data)
