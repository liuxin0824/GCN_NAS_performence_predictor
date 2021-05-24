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

def input_transfer_array(input_data):

  new_input_data = []
  for input in input_data:
    new_input = []
    for cell in input:
      new_input+=cell
    new_input_data.append(new_input)
  return np.array(new_input_data)

with open("./Track2_final_archs.json", 'r') as load_f:
  load_dict = json.load(load_f)





all_arch_id = range(1,len(load_dict)+1) #1 —— 200

num_test_arch = len(all_arch_id) // 4    #//为整数除法

num_train_arch = len(all_arch_id) - num_test_arch

train_arch_id = all_arch_id[:num_train_arch] #固定数据集测试集选项
test_arch_id = all_arch_id[-num_test_arch:]
#test_arch_id = np.random.choice(a=all_arch_id, size=num_test_arch, replace=False)
#train_arch_id = [id for id in all_arch_id if id not in test_arch_id]

train_arch_list = id_to_list(train_arch_id)#index前面加上arch用于查找
test_arch_list = id_to_list(test_arch_id)

train_x, train_y = list_to_data(train_arch_list, load_dict)
test_x, test_y = list_to_data(test_arch_list, load_dict)


train_x_g,test_x_g = np.array(train_x),np.array(test_x)
train_x, test_x = input_transfer_array(train_x), input_transfer_array(test_x)


train_x, test_x = normalization(train_x), normalization(test_x)



adj = np.zeros((16,16))
adj[0][1] = 1
adj[1][2] = 1
adj[2][3] = 1
adj[3][4] = 1
adj[4][5] = 1
adj[5][6] = 1
adj[6][7] = 1
adj[7][8] = 1
adj[8][9] = 1
adj[9][10] = 1
adj[10][11] = 1
adj[11][12] = 1
adj[12][13] = 1
adj[13][14] = 1
adj[14][15] = 1
adj = adj.T+adj
e = np.eye(15)




def normalize(mx):
  """Row-normalize sparse matrix"""
  rowsum = np.array(mx.sum(1))
  r_inv = np.power(rowsum, -1).flatten()
  r_inv[np.isinf(r_inv)] = 0.
  r_mat_inv = np.diag(r_inv)
  mx = r_mat_inv.dot(mx)
  return mx


adj = normalize(adj + np.eye(adj.shape[0]))

adj = torch.FloatTensor(adj)      #adj 已经变成tensor形式
print(adj)


train_y = torch.FloatTensor(train_y) #label已变成tensor
print(train_y)

train_x_g = torch.FloatTensor(train_x_g) #train数据转换成tensor
print(train_x_g)

#进行批处理
batch_size = 5
torch_dataset = Data.TensorDataset(train_x_g, train_y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0        # set multi-work num read data
)



random_seed = 100
torch.manual_seed(random_seed)

model = GCN(nfeat=train_x_g.shape[2],
            nhid=3,
            nclass=2,
            dropout=0.5)
print(model)

optimizer = optim.Adam(model.parameters(),
                       lr=0.001, weight_decay=5e-3)

criterion = nn.MSELoss()

number_epoch = 300



for epoch in range(number_epoch):
  # Convert numpy arrays to torch tensors

  epoch_loss = 0
  for i,(batch_data_label) in enumerate(loader):

    batch_data = batch_data_label[0]
    batch_label = batch_data_label[1]

    batch_loss = torch.zeros(1)

    for j,solid_data in enumerate(batch_data):
      output = model(solid_data,adj)
      loss = criterion(output, batch_label[j])
      batch_loss += loss


    optimizer.zero_grad()
    batch_loss.backward()
    epoch_loss = epoch_loss+batch_loss.item()

    optimizer.step()
    batch_loss_mse = batch_loss.item()/batch_size
    batch_loss_rmse = math.sqrt(batch_loss_mse)
    print('Epoch [{}/{}], Batch [{}/{}], BatchMSELoss: {:.8f},batchRMSELoss: {:.8f}'.format(epoch + 1,
                                                                                  number_epoch,
                                                                                  i+1,
                                                                                  len(loader),
                                                                                  batch_loss_mse,
                                                                                  batch_loss_rmse))

  epoch_loss = epoch_loss/len(train_x_g)  #mse loss
  epoch_loss_r = math.sqrt(epoch_loss)  # rmse loss
  print('Epoch [{}/{}], MSELoss: {:.8f},RMSELoss: {:.8f}'.format(epoch + 1, number_epoch, epoch_loss,epoch_loss_r))







'''
##################
clf = MLPRegressor(hidden_layer_sizes=(100,), activation="relu", solver='adam',
                   alpha=0.0001, batch_size=256, learning_rate="adaptive",
                   learning_rate_init=0.1, max_iter=2000)
print
clf.fit(train_x, train_y)

print(clf.score(train_x, train_y), clf.score(test_x, test_y))
'''
#######
test_x_g = torch.FloatTensor(test_x_g)
predict_y = []
test_y_tensor = torch.FloatTensor(test_y) #label已变成tensor
test_loss = 0
for i, trainexample in enumerate(test_x_g):
  output = model(trainexample, adj)
  loss = criterion(output, test_y_tensor[i])
  test_loss = test_loss + loss.item() #RMSE LOSS
  predict_y.append(output.data)

test_loss = test_loss/len(test_y)

print('TEST MSE LOSS:{:.8f}'.format(test_loss))

test_loss = math.sqrt(test_loss)

print('TEST RMSE LOSS:{:.8f}'.format(test_loss))

print('random_seed: {} ,num_epoch: {} ,batch_size: {}'.format(random_seed,number_epoch,batch_size))#打印随机seed，记录模型的初值选择


x = range(len(test_y))

plt.plot(x, test_y, color="r", linestyle="--", marker=".", label="True")
plt.plot(x, predict_y, color="b", linestyle="--", marker=".", label="Predicted")

plt.xticks([i for i in range(len(test_y))])


plt.legend()

plt.show()



