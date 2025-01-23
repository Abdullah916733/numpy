import numpy as np
import matplotlib.pyplot as plt

arr = np.array([1, 2, 3, 4])

# print(arr.ndim)
# print(arr.shape)
# print(arr.size)
# print(type(arr))

max_1d = np.ones((3, 4), dtype=int)
# print(max_1d)

max_2d = np.zeros((3, 4), dtype=int)
# print(max_2d)

max_bool = np.zeros((3, 4), dtype=bool)
# print(max_bool)

em_max = np.empty((3, 3))
# print(em_max)

arr_num = np.arange(1, 13)
# print(arr_num)

resh_num = arr_num.reshape(2, 6)
# print(resh_num)

# print(resh_num.ravel())

# print(resh_num.transpose())


resh_num1 = np.arange(1, 13).reshape(3,4)
resh_num2 = np.arange(1, 13).reshape(3,4)

# print(resh_num1)

# print(np.add(resh_num1, resh_num2))
# print(np.divide(resh_num1, resh_num2))
# print(np.subtract(resh_num1, resh_num2))
# print(np.multiply(resh_num1, resh_num2))


# print(resh_num1.max())
# print(resh_num1.argmax())
# print(resh_num1.min())
# print(resh_num1.argmin())


# print(resh_num1)
# print(resh_num1.max(axis=1))
# print(resh_num1.max(axis=0))


# print(resh_num1)
# print(np.sum(resh_num1, axis=1))
# print(np.sum(resh_num1, axis=0))


mx = np.arange(1,101).reshape(10,10)
# print(mx)

# print(mx[:, 1:])
# print(mx[:, 1:3])
# print(mx[:, 1:3].ndim)
# print(mx[:, 1:3].dtype)
# print(mx[:, 1:3].itemsize)

arr1 = np.array([1,2,3,4,5])
arr2 = np.array([6,7,8,9,10])

combine = np.concatenate((arr1, arr2))

# print(combine)

vcomb = np.vstack((arr1,arr2))
# print(vcomb)

hcomb = np.hstack((arr1, arr2))
# print(hcomb)

sparr = np.split(combine, 5)
# print(sparr)

x_sin = np.arange(0, 3*np.pi, 0.1)
# print(x_sin)

y_sin = np.sin(x_sin)
# print(y_sin)

# plt.plot(x_sin, y_sin)
# plt.show()


y_cos = np.cos(x_sin)
# plt.plot(x_sin, y_cos)
# plt.show()

y_tan = np.tan(x_sin)
plt.plot(x_sin, y_tan)
plt.show()




