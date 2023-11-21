import numpy as np

dataset = [{'data': 2},{'data:',4},{'data:',6},{'data:',8}]

batch_dict = {}

batch_dict = {'data':np.array(2)}

batch_dict['data'] = np.hstack((batch_dict['data'],4))
batch_dict['data'] = np.hstack((batch_dict['data'],6))

print(batch_dict)
print(batch_dict['data'].shape)

arr = np.array([[2],[4],[6]])

print(arr)
print(np.array([2,4,6]).shape)
batch_dict = {}
batch_dict['data'] = arr.flatten()
print(batch_dict)