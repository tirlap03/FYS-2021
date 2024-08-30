import numpy as np

"""First numpy exercise"""
# print(np.__version__)

# print(np.show_config())

"""2"""
# print(np.info(np.add))


"""3: none of the elements are non-zero"""
# x = np.array([1,2,3,4])
# print(f"original array: {x}")

# print("Array contains 0?")
# print(np.all(x))

# x = np.array([0,1,2,3])
# print("Array does not contain 0?")
# print(np.all(x))


"""4: any elements are non-zero"""
# x = np.array([0,0,0,1])
# print("Array contains 0?")
# print(np.any(x))

# x = np.array([0,0,0])
# print(f"arrat contains 0? {np.any(x)}")

"""5: elements are finite (not infinity/not a number)"""
# Creating an array containing elements 1, 0, NaN (not a number), and infinity
# x = np.array([1,0,np.nan, np.inf])
# print(x)

# print(f"is array finite? {np.isfinite(x)}")

"""6: positiv or negative infinity"""
# x = np.array([1,-2,np.inf,-np.inf,0,3.5])
# res_pos_inf = np.isposinf(x)
# res_neg_inf = np.isneginf(x)

# print(f"elementwise positive infinity check: {res_pos_inf}")
# print(f"elementwise negative infinity check: {res_neg_inf}")


"""7: elementwise test for NaN"""
# x = np.array([1,np.nan, 5])
# y = np.array([1,2,3])

# res_NaN_x = np.isnan(x)
# res_NaN_y = np.isnan(y)

# print(f"elementwise NaN check for array x: {res_NaN_x}")
# print(f"elementwise NaN check for y: {res_NaN_y}")

"""8: elementwise complex numbers, real numbers, scalar type"""
# x = np.array([1,np.pi,-1,(1j)])

# res_comp = np.iscomplex(x)
# res_real = np.isreal(x)
# res_scal_1 = np.isscalar(3.1)
# res_scal_2 = np.isscalar([3.1])

# print(f"elementwise check: complex: {res_comp}, real: {res_real}, scalar; {res_scal_1}, scalar {res_scal_2}")

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""2:convert list of values into 1D array"""
# x = [12.23, 13.32, 100, 36.32]
# print(f"OG list: {x}")

# a = np.array(x)
# print(f"1D array: {a}")


"""3: 3x3matrix from 2-10"""
# x = np.arange(2,11).reshape(3,3)

# print(x)

"""4:null vector with size 10, updating 6th number to 11"""
# # Creating an array filled with zeroes of length 10
# x = np.zeros(10)
# print(x)

# # Updating 6th number
# x[6] = 11
# print(x)

"""5:array with values ranging from 12 to 38"""
# x = np.arange(12,38)
# print(x)

"""6:reversed array (first element becomes last)"""
# y = x[::-1]
# print(y)

"""7:convert array to floating type"""
# x = np.array([1,2,3,4])
# print(x)

# b = np.asarray(x)
# print(b)
# c = np.float64(x)
# print(c)

"""8:2D array with 1 on border and 0 insode"""
# x = np.ones((5,5))
# print(x)

# # setting the border elements to 1 and inner elements to 0
# x[1:-1, 1:-1] = 0
# print(x)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Define two 2x2 matrices (for 1 and 2)
# p = [[1,0], [0,1]]
# q = [[1,2], [3,4]]
# print("p:",p)
# print("q:",q)
# print()

"""1:multiplication on two matrices"""

# result1 = np.matmul(p, q)
# print(result1)
# result2 = np.dot(p,q)
# print(result2)

"""2:outer product of two given vectors"""
# result3 = np.outer(p,q)
# print(result3)

"""3:cross product of two given vectors"""
# result4 = np.cross(p,q)
# print(result4)
# result5 = np.cross(q,p)
# print(result5)

"""4:determinant of a given square array"""
from numpy import linalg as LA
# x = np.array([[1,0],[1,2]])
# print(x)
# print(np.linalg.det(x))

"""5:evaluation of einstein's summation convention 
of two given multidimensional arrays
"""
a = np.array([1,2,3])
b = np.array([0,1,0])
print(f"a:{a},b: {b}")

# einsteins summation convention of 1D arrays
res_1D = np.einsum("n,n", a, b)
print("1D array: \n", res_1D)

# 3x3 arrays
x = np.arange(9).reshape(3,3)
y = np.arange(3,12).reshape(3,3)
print(x)
print(y)

res_2D = np.einsum("mk,kn",x,y)
print("2D array: \n", res_2D)
