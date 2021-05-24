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



''' 
def traindata_to_adj_ceshiban(train_x):  #消除没用的2 4 维尝试 13->6(7)效果一般
    adj_list = []
    for j,single_x in enumerate(train_x):

        adj = np.zeros((16*7, 16*7))

        for i,four_meta in enumerate(single_x):
            if i<15:
                adj_a_1 = i*7 + four_meta[2]-1
                adj_a_2 = (four_meta[0]-1)*7+ single_x[four_meta[0]-1][2]-1
                adj[adj_a_1][adj_a_2] = 1
                #adj[adj_a_2][adj_a_1] = -1



        adj_list.append(adj)


    return adj_list

'''
def normalization(data):
  _range = np.max(data) - np.min(data)
  return (data - np.min(data)) / _range

def id_to_list(id_list):
  arch_list = []
  for id in id_list:
    arch_name = "arch_few_shot_" + str(id)
    arch_list.append(arch_name)
  return arch_list

def list_to_data(arch_list, all_data):
  input_data = []
  output_data = []
  for arch in arch_list:
    input_data.append(all_data[arch]["arch"])
    output_data.append(all_data[arch]["acc"])
  return input_data, np.array(output_data)
'''
def input_transfer_array(input_data):

  new_input_data = []
  for input in input_data:
    new_input = []
    for cell in input:
      new_input+=cell
    new_input_data.append(new_input)
  return np.array(new_input_data)
'''

def traindata_to_adj(train_x):
    adj_list = []
    for j,single_x in enumerate(train_x):

        adj = np.zeros((16*13, 16*13))
       # adj = np.zeros((16 * 16, 16 * 16))

        for i,four_meta in enumerate(single_x):
            if i<15:
                adj_a_1 = i*13 + four_meta[2]-1
                adj_a_2 = (four_meta[0]-1)*13+ single_x[four_meta[0]-1][2]-1
                adj[adj_a_1][adj_a_2] = 1
                #adj[adj_a_2][adj_a_1] = 1



        adj_list.append(adj)


    return adj_list


with open("./Track2_stage2_few_show_trainning.json", 'r') as load_f:
  load_dict = json.load(load_f)

#############################################################生成最终数据区

'''
with open("./Track2_stage1_test_10w_dict_sort_fix.json", 'r') as load_f_final:
    load_dict_final = json.load(load_f_final)

id_list_final = range(201, 100201)

arch_list_final = id_to_list(id_list_final)

final_test_x, final_test_y = list_to_data(arch_list_final, load_dict_final)

'''
##########################################################################





all_arch_id = range(1,len(load_dict)+1) #1-30

num_test_arch = 31    #//为整数除法

num_train_arch = len(all_arch_id)

train_arch_id = all_arch_id[:num_train_arch] #固定数据集测试集选项
test_arch_id = all_arch_id[-num_test_arch:]

#train_arch_id = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,2,10,18,26]
#test_arch_id = [4,6,8,12,14,16,20,22,24,28,30]


#train_arch_id = all_arch_id[50:200] #固定数据集测试集选项
#test_arch_id = all_arch_id[:50]

#test_arch_id = np.random.choice(a=all_arch_id, size=num_test_arch, replace=False)
#train_arch_id = [id for id in all_arch_id if id not in test_arch_id]

train_arch_list = id_to_list(train_arch_id)#index前面加上arch用于查找
test_arch_list = id_to_list(test_arch_id)


train_x, train_y = list_to_data(train_arch_list, load_dict)
test_x, test_y = list_to_data(test_arch_list, load_dict)
print(len(train_x))



######更换拓扑图的邻接矩阵



adj_list = traindata_to_adj(train_x)
np.set_printoptions(threshold=np.inf)#打印长np不省略

adj_train = np.array(adj_list)




adj_list_test = traindata_to_adj(test_x)
adj_test = np.array(adj_list_test)


'''
adj_list_test_final = traindata_to_adj(final_test_x)
adj_test_final = np.array(adj_list_test_final)
'''



######



#train_x_g,test_x_g = np.array(train_x),np.array(test_x)
#train_x, test_x = input_transfer_array(train_x), input_transfer_array(test_x)


#train_x, test_x = normalization(train_x), normalization(test_x)







def normalize(mx):
  """Row-normalize sparse matrix"""
  rowsum = np.array(mx.sum(1))
  r_inv = np.power(rowsum, -1).flatten()
  r_inv[np.isinf(r_inv)] = 0.
  r_mat_inv = np.diag(r_inv)
  mx = r_mat_inv.dot(mx)
  return mx


############################################################################################处理新版adj

adj_train_normal = []
for adj1 in adj_train:
    adj1 = normalize(adj1 + np.eye(adj1.shape[0]))

    adj_train_normal.append(adj1)

adj_test_normal = []
for adj1 in adj_test:
    adj1 = normalize(adj1 + np.eye(adj1.shape[0]))

    adj_test_normal.append(adj1)
'''
adj_test_normal_final = []
for adj1 in adj_test_final:
    adj1 = normalize(adj1 + np.eye(adj1.shape[0]))

    adj_test_normal_final.append(adj1)
'''

torch.set_printoptions(threshold=np.inf)#打印长tensor不省略

adj_train_normal = torch.FloatTensor(adj_train_normal)  #更改的拓扑邻接矩阵转化成了tensor
adj_test_normal = torch.FloatTensor(adj_test_normal)
'''
adj_test_normal_final = torch.FloatTensor(adj_test_normal_final)
'''
np.set_printoptions(threshold=np.inf)#打印长np不省略


train_eye1 = np.eye(16*13)
#train_eye1 = np.eye(16*16)
train_eye = np.zeros([13*16,2*16+6+7])#####前32维为16乘2 16block每个block分为两部分，标志同一个opt，后13对应13种超参选择

for i in range(16):
    for j in range(6):
        train_eye[i*13+j][2*i] = 1
    for j in range(7):
        train_eye[i * 13 + j +6][2*i+1] = 1

for index in range(16):
    for i,j in enumerate(range(32,45)):
        train_eye[index*13+i][j] = 1

print(train_eye)
train_eye = torch.FloatTensor(train_eye) #使用增强版的feature


train_eye1 = torch.FloatTensor(train_eye1) #使用单位阵作为数据（放弃feature仅使用拓扑图


#########################################################################################################

#adj = normalize(adj + np.eye(adj.shape[0]))

#adj = torch.FloatTensor(adj)      #adj 已经变成tensor形式
#print(adj)


train_y = torch.FloatTensor(train_y) #label已变成tensor
print(train_y)

#train_x_g = torch.FloatTensor(train_x_g) #train数据转换成tensor
#print(train_x_g)

#进行批处理
#batch_size = len(train_x)
batch_size = 5
torch_dataset = Data.TensorDataset(adj_train_normal, train_y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0        # set multi-work num read data
)



random_seed = 190
torch.manual_seed(random_seed)

model = GCN(nfeat=16*13,
            nhid=64,
            nclass=24,
            dropout=0.1) #208 64 16
'''
model = GCN(nfeat=16*16,
            nhid=32,
            nclass=4,
            dropout=0.5)
            '''
print(model)

weight_decay = 0
lr = 0.001
optimizer = optim.Adam(model.parameters(),
                       lr=lr, weight_decay=weight_decay)

criterion = nn.MSELoss()

number_epoch = 800


for epoch in range(number_epoch):
  # Convert numpy arrays to torch tensors

  epoch_loss = 0
  for i,(batch_data_label) in enumerate(loader):

    batch_data = batch_data_label[0]
    batch_label = batch_data_label[1]

    batch_loss = torch.zeros(1)

    for j,solid_data in enumerate(batch_data):
      output = model(train_eye1,solid_data)
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

  epoch_loss = epoch_loss/len(train_x) #mse loss
  epoch_loss_r = math.sqrt(epoch_loss)  # rmse loss
  print('Epoch [{}/{}], MSELoss: {:.8f},RMSELoss: {:.8f}'.format(epoch + 1, number_epoch, epoch_loss,epoch_loss_r))
  if(epoch_loss_r<0.0001):
      break







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
##test_x_g = torch.FloatTensor(test_x_g)
predict_y = []
test_y_tensor = torch.FloatTensor(test_y) #label已变成tensor
test_loss = 0
for i, trainexample in enumerate(adj_test_normal):
  output = model(train_eye1, trainexample)
  loss = criterion(output, test_y_tensor[i])
  test_loss = test_loss + loss.item() #RMSE LOSS
  predict_y.append(output.data)

test_loss = test_loss/len(test_y)

print('TEST MSE LOSS:{:.8f}'.format(test_loss))

test_loss = math.sqrt(test_loss)

print('TEST RMSE LOSS:{:.8f}'.format(test_loss))

print('random_seed: {} ,num_epoch: {} ,batch_size: {} \n weight_decay: {},  lr: {}'.format(random_seed,number_epoch,batch_size,weight_decay,lr))#打印随机seed，记录模型的初值选择
print('if dropout: f' )
print(model)
'''
for i, trainexample in enumerate(adj_test_normal_final):
  output = model(train_eye1, trainexample)
  final_test_y[i] = output.data

def data_alter(arch_list, all_data, final_test_y):

  for i,arch in enumerate(arch_list):
    all_data[arch]["acc"] = final_test_y[i]
  return all_data

all_data_new = data_alter(arch_list_final,load_dict_final,final_test_y)

with open("./outcome.json", 'w') as f:
  json.dump(all_data_new, f)

'''
########################################下面为画出test
torch.save(model,"./mymodel20.pth")
x = range(len(test_y))

plt.plot(x, test_y, color="r", linestyle="--", marker=".", label="True")
plt.plot(x, predict_y, color="b", linestyle="--", marker=".", label="Predicted")

plt.xticks([i for i in range(len(test_y))])


plt.legend()

plt.show()
