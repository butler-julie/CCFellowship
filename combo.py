import numpy as np
from itertools import product

list1 = np.arange(10, 540, 75)
list2 = [None, 'tanh', 'relu', 'sigmoid']
list3 = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
list4 =  [0.0, 0.05, 0.1, 0.2, 0.25, 0.5]

output = list(product(list1, list2, list3, list4))

print(output)
