import numpy as np
from numpy import linalg as LA

"""Vectors: n x m"""
a = np.array([1,1,1,1,1,1])         # 1 x 6
b = np.array([2,2,2,2,2,2])         # 1 x 6
X = np.array([[3,3,3,3,3,3],
              [3,3,3,3,3,3],
              [3,3,3,3,3,3],
              [3,3,3,3,3,3]])       # 4 x 6
Z = np.array([[4,4,4,4,4,4],
             [4,4,4,4,4,4],
             [4,4,4,4,4,4]])        # 3 x 6
# print("Z\n",Z.shape)
# print("X\n",X.shape)
# print("a\n",a.shape)
# print("b\n",b.shape)


"""1b: vectorizing the sum y=sum(a_i&b_i)"""
y1 = a@b
print(f"exercise 1b: the sum is {y1}\n")

"""1c: for-loop as vectorized code"""
y2 = a@b
print(f"exercise 1c: for-loop as mathmatical vector operation and vectorized code {y2}\n")

"""1d: nested for-loop as vectorized code"""
y3 = a@X.T
# print(f"Transposen til X: {X.T}")
print(f"exercise 1d: nested for-loops: {y3}\n")

"""1e: vectorize the algorithm"""
a_tmp = np.tile(np.array([a]), (4,1))
print("a_tmp: \n", a_tmp)
y4 = np.sum(a_tmp*X, axis=1)
print(f"exercise 1e: vectorize the algotithm, elementwise multiplication: {y4}\n")

"""1f: matrix multiplication / dot product"""
y6 = Z@X.T                      # fasiten sa dette
# print(y6)
y5 = np.dot(Z, X.T)             # men dette fungerer også
print(f"exercise 1f: matrix multiplication / dot protduct:\n {y5}\n")

"""1g: """
