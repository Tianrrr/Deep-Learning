from math import prod
import numpy as np
from itertools import product

# labels = [0,3,4,5,6,7,8,9,1,2]
# print(type(labels[0]))
# images = ['a','b','c','d','e','f','g','h','i','k']
# np.random.seed(0)
# rand_perm = np.random.permutation(len(images))
# idx = rand_perm[:6]
# print(type(list(np.array(labels)[idx])[0]))


# a = np.arange(24).reshape(4,6)
# a = np.flip(a,0)
# print(a)

# x = np.random.randn(3,5) * 10
# print(x)


# x[x < 0] = 0.001
# x[x > 0] = 1
# print(x)

dict_0 = {
  "Alphabet": ["a","b","c","d","e"],
  "Number": [1,2,3,4,5]}



# money = car.pop("dollars",10000)

# print(car)
# print(money)


# optim_configs = {}
# for p in car:
#     d = {k: v for k, v in optim_configs.items()}
#     optim_configs[p] = d

# print(optim_configs)

for i in product(*dict_0.values()):
    print(i)