cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def overlap(x1, w1, x2, w2):
    l1 = x1 - w1 / 2
    l2 = x2 - w2 / 2
    left = l1 > l2 ? l1 : l2
    r1 = x1 + w1 / 2
    r2 = x2 + w2 / 2
    right = r1 < r2 ? r1 : r2
    return right - left
