import json
import numpy as np

import torch






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















with open("./outcome_big3_1624_21633.json", 'r') as load_f:
  load_dict = json.load(load_f)

with open("./outcome_wuchajixiao_seed20_21754.json", 'r') as load_f:
  load_dict1 = json.load(load_f)

with open("./outcome_seed15comb_21757.json", 'r') as load_f:
  load_dict2 = json.load(load_f)







test_arch_id  = range(1,100002)    #//为整数除法



 #固定数据集测试集选项



#index前面加上arch用于查找
test_arch_list = id_to_list(test_arch_id)


test_x, test_y = list_to_data(test_arch_list, load_dict)
test_x1, test_y1 = list_to_data(test_arch_list, load_dict1)
test_x2, test_y2 = list_to_data(test_arch_list, load_dict2)


outcome = (3*test_y+test_y1+test_y2)/5



def data_alter(arch_list, all_data ,predict):

  for i,arch in enumerate(arch_list):
    all_data[arch]["acc"] = predict[i]
  return all_data


all_data = data_alter(test_arch_list,load_dict,outcome)
#all_data = data_alter(test_arch_list,load_dict,outcome_test)


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



with open("./outcome_combine.json", 'w') as f:
  json.dump(all_data, f)

print(all_data)