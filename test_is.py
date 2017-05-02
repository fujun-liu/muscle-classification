'''
test_is.py
'''
import numpy as np

def WeightedReservoir(weight, k):
    N = weight.size
    if N <= k:
        return range(N)
    weight = weight/np.sum(weight)
    inidce = range(k)
    w_sum = .0
    for i in range(k):
        w_sum += weight[i]/k
    for i in range(k, N):
        w_sum += weight[i]/k
        p = weight[i]/w_sum
        if np.random.rand() < p:
            inidce[np.random.randint(0,k)] = i
    return inidce

if __name__ == "__main__":
    a = range(10)
    print a
    print WeightedReservoir(np.array(a, dtype=float), 3)
            
    

