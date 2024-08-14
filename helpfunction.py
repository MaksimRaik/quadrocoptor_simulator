import numpy as np

def add1(M):
    M=np.asarray(M)
    return np.vstack((M,np.ones(M.shape[1])))