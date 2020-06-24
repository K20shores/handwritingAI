import numpy as np

x = np.array([[1,1], [1,1]])
w = np.array([[.1,.1], [.1,.1]])
b = np.array([[1], [1]])
print(x)
print(w)
print(b)

z = np.dot(w,x) + b
print(z)
print()

xs = np.hstack((x, x))
print(xs)
z = np.dot(w,xs) + b
print(z)
print(np.sum(z, 0))
print(np.sum(z, 1))

y = np.array([[1,1], [0,0]])
print(x - y)

"""
before
delta_nabla_b (3,10,1)
delta_nabla_w[0] (10,784)
delta_nabla_w[1] (10, 10)
delta_nabla_w[2] (10, 10)

expected after
delta_nabla_b (3,10,10)
delta_nabla_w[0] (10, 10, 784)
delta_nabla_w[1] (10, 10, 10)
delta_nabla_w[2] (10, 10, 10)
"""
