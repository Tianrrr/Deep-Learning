import numpy as np

# def rel_error(x, y):
#     """ returns relative error """
#     return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)
# value = 1 / (1 + np.exp(-x))
# truth = np.array([[0.37754067, 0.39913012, 0.42111892, 0.44342513],
#                                [0.46596182, 0.48863832, 0.51136168, 0.53403818],
#                                [0.55657487, 0.57888108, 0.60086988, 0.62245933]])


# error = rel_error(truth, value)
# print(error < 1e-8)

x = np.random.randn(5, 1)
# print(x)

# x[x < 0] = 0
# print(x)
# cache = x

# cache[cache > 0] = 1
# dx = cache
# print(x)

# n = 2
# x = np.random.randn(n, 2, 3)


# print(x)
# x = x.reshape(n,-1)
# print(x)
N, M = 3,5
x = np.arange(24).reshape(4,6)
# print(x)
# xmean = np.mean(x, axis=0, keepdims=True)
# print(xmean.shape)
# xx = xmean.reshape(1,-1).T
# print(xx.shape)
# dw = np.zeros((6,2))
# for i in range(dw.shape[1]):
#     dw[:,i] = xx

# print(dw)
# print(x)
# x_mean = np.mean(x,axis=0)
# print(x_mean)

# one = np.ones([10,6])
# print(one)
# xx = one * x_mean
# print(xx.T)



# a = np.ones([1,6])
# print(a)
# b = np.arange(24).reshape(6,4)
# print(b)
# print(a.dot(b).flatten())


# W = np.ones([3,5])
# print(W)
# w = np.sum(W)
# print(w)

# split={'train': 0.6, 'val': 0.2, 'test': 0.2}
# split_values = [v for k,v in split.items()]

# print(split_values)

# a = np.array([2,3,4,5,6,7])
# print(len(a))

# y_out = np.random.randn(8,10)
# N = y_out.shape
# print(N)
# y_truth_one_hot = np.zeros_like(y_out)
# print(y_truth_one_hot)
# label = [1,2,3,4]
# rand_perm = np.random.permutation(4)
# print(rand_perm)
# a = list(np.array(label)[rand_perm])
# print(a)
# print(a[0].shape)


# images = ['W1','b1','W2','b2']
# labels = [1,4,6,4]

# dict = {}
# for k,v in zip(images,labels):
#     dict[k] = v
# print(dict)

    

# w = np.random.randn(3,4)
# config = {}
# config.setdefault('momentum', 0.9)
# v = config.get('velocity', np.zeros_like(w))
# print(config)

# optim_configs = {}
# for p in dict:   # 遍历的是params的key （W1,b1,W2,b2...)
#     d = {k: v for k, v in optim_configs.items()}
#     print(d)
#     optim_configs[p] = d
#     print(optim_configs)



# a = np.random.randn(3,5) * 10
# index = np.argmax(a, axis=1)
# print(a)
# print(index.shape)


# a = np.array([3,4,6,2,3,5])
# print(a.shape == True)
t = 1
validate = t == 0
print(not validate)