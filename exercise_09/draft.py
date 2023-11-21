from ctypes import sizeof
import numpy as np
import pandas as pd
import os
import torch

# df = pd.DataFrame([[2, 3], [5, 6],[4,7],[5,8], [8, '1 4 5 7']],
#      index=['cobra', 'viper', 'sidewinder','asdd','dddd'],
#      columns=['max_speed', 'shield'])

# print(df.shape)

# a = df.loc['sidewinder']['shield']
# print(type(a))
# img = np.array([
#             int(item) for item in a.split()
#         ]).reshape((2, 2))
# print(img)

# img_new = np.expand_dims(img, axis=2).astype(np.uint8)
# print(img_new)

# print(list(df.columns))
    

# print(list(df.columns))
# print(type(df.iloc[0][['max_speed', 'shield']].values.reshape(1,2)))
# file_name = "training.csv"
# i2dl_exercises_path = os.path.dirname(os.path.abspath(os.getcwd()))
# data_root = os.path.join(i2dl_exercises_path, "datasets", "facial_keypoints")
# csv_file = os.path.join(data_root, file_name)
# test_df = pd.read_csv(csv_file)
# print(type(test_df.loc[1]['Image']))
# print(test_df.loc[1]['Image'])


# x = torch.tensor([[[1,2,3],[3,4,5]]])
# print(x.shape)
# x = torch.unsqueeze(x, 0)
# print(x.shape)

# x = np.random.randn(10,3,6,6)
# x_mean = np.mean(x, axis=(0,2,3),keepdims=True)
# print(x_mean.shape)
# print((x-x_mean).shape)

a = np.random.rand(4,4)

com = np.array([a,a,a])#变为三通道 
print(a.shape)
print(a)
print(com.shape)
