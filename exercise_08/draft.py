from re import A
import numpy as np
a = np.random.randn(3,5)
print(a)

a = np.random.rand(*a.shape) < 0.5
print(a)
