import os
from PIL import Image
import numpy as np
from torchvision import transforms
import torch

# class _List(object): 
#     def __init__(self, _list):
#         self._list = _list
 
#     def __getitem__(self, key):
#         if isinstance(key, int):
#             return self._list[key]
#         elif isinstance(key, slice):
#             return [self[ii] for ii in range(*key.indices(len(self)))]
#     def __len__(self):
#         return len(self._list) 

 

# c = _List(range(10))
# key=slice(1, 5, None)
# print(key.indices(len(c)))

# i2dl_exercises_path = os.path.dirname(os.path.abspath(os.getcwd()))
# data_root = os.path.join(i2dl_exercises_path, 'datasets','segmentation')


# image_paths_file=f'{data_root}/segmentation_data/train.txt'
# root_dir_name = os.path.dirname(image_paths_file)
# # with open(image_paths_file) as f:
# #     image_names = f.read().splitlines()
# # print(image_names[0:2])

# image_names = ['11_16_s.bmp']
# img_id = image_names[0].replace('.bmp', '')  #image_names[0] :  ['11_16_s.bmp']
# print(img_id)


# img = Image.open(os.path.join(root_dir_name,
#                                       'images',
#                                       img_id + '.bmp')).convert('RGB')



# to_tensor = transforms.ToTensor()
# img = to_tensor(img)

# print(img.size())


# target = Image.open(os.path.join(root_dir_name,
#                                          'targets',
#                                          img_id + '_GT.bmp'))

# target = np.array(target, dtype=np.int64)

# target_labels = target[..., 0]  
# print(target.shape)

# print((target == [0,0,0]).shape)

# x = np.all(target == [0,0,0],axis=2)
# print(x.shape)
# print(x[0][0])

# a = np.ones((5,5))
# a[0][0:5] = 0
# print(a)
# print(type(np.unique(a)))

# a = torch.randn(3,5)
# # a = a.view(*a.size(),-1)
# # print(a.shape)
# # scatter_dim = len(a.size())
# # a_tensor = a.view(*a.size(),-1)
# # zeros = torch.zeros(*a.size(), 10, dtype=a.dtype3
# # print(zeros.shape)

# # x = zeros.scatter(scatter_dim, a_tensor, 1)
# # print(x)
# print(a)

# x,y = torch.max(a, 1)
# print(y)


a = torch.randn([1, 3, 4, 4])

b = torch.tensor([[2, 0, 2, 0],
        [2, 2, 2, 2],
        [2, 0, 0, 1],
        [2, 1, 2, 1]])

_, c  = torch.max(a, axis=1)
print(c)
b = b.unsqueeze(0)
print(b.shape)
loss_func = torch.nn.CrossEntropyLoss()
loss = loss_func(a,c)
print(loss)
