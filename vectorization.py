import numpy as np
from time import time

n = 10000000
x = np.random.rand(n)
y = np.random.rand(n)

# adding two lists elementwise

# non vectorized version

start_time = time()
z1 = []
for k in range(n):
    z1.append(x[k] + y[k])
end_time = time()
t1 = end_time - start_time
print(f"The code took {t1} seconds")

# vectorized version
start_time = time()
z2 = x + y 
end_time = time()
t2 = end_time - start_time
print(f"The code took {t2} seconds")

print((z1 == z2).all())

print(t1/t2)

# matrix multiplication
n = 200
A = np.random.rand(n,n)
B = np.random.rand(n,n)

C1 = np.zeros((n,n))

start_time = time()
for i in range(n):
    # get i'th row of A
    row_a = A[i]
    for j in range(n):
        # get j'th column of B
        col_b = B[:,j]

        # compute the i, j entry of C1 as the dot product of i'th row of A with j'th column of B
        entry = 0
        for k in range(n):
            entry += row_a[k] * col_b[k]
        C1[i,j] = entry 
end_time = time()
t3 = end_time - start_time
print(f"The code took {t3} seconds")

# vectorized
start_time = time()
# numpy matrix multiplication
C2 = np.dot(A,B)
end_time = time()
t4 = end_time - start_time        
print(f"The code took {t4} seconds")

print(np.mean(abs(C1-C2)))

print(t3/t4)


