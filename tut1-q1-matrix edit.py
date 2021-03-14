#!/usr/bin/env python
# coding: utf-8

# # Complexity of Matrix Operations
# 
# We will implement different matrix operations by hand, test the implementation and compare the speed with the optimized numpy library. We will also study the computational complexity of these matrix operations, and estimate the performance of our computer.

# ## Implementation of Matrix Operations (6 points)
# 
# Implement the matrix operations $\mathbf{{x}}^T\mathbf{{x}}$, $\mathbf{{A}}\mathbf{{x}}$, $\mathbf{{A}}\mathbf{{A}}$ and $\mathbf{{x}}^T\mathbf{{A}}\mathbf{{x}}$ in Python without using the numpy package. Verify your implementation. 

# In[233]:



def x_dot_x(x):
    # YOUR CODE HERE
    ans = 0
    for i in range(len(x)):
        ans += x[i]*x[i]
    return ans


# In[234]:


assert( x_dot_x( [1,2] ) == 5)


# In[235]:


def A_dot_x (A, x):
    # YOUR CODE HERE
    
    # create vector b with zeros
    b = []
    for p in range(len(x)):
        b.append(0)
    
    # multiply each element with corresponding xi and add to b
    for i in range(len(x)):
        coeff = x[i]
        for j in range (len(A[i])):
            anj = A[i][j]
            b[j] += coeff*anj
    return b


# In[236]:


assert( A_dot_x( [ [1,2], [2,1] ], [0,1] ) == [2,1] )


# In[237]:


def A_dot_A (A):
    # YOUR CODE HERE
    
    # create array A2 with zeros
    A2 = []
    for p in range(len(A)):
        A2.append([])
        for q in range(len(A[p])):
            A2[p].append(0)
    
    # multiply each element with corresponding xi and add to b
    for i in range(len(A)):
        for j in range (len(A[i])):
            for k in range(len(A)):
                A2[i][j] += A[i][k] * A[k][j]
    return A2


# In[238]:


assert( A_dot_A( [ [1,1], [2,0] ] ) == [ [3,1], [2,2] ])


# In[239]:


def x_dot_A_dot_x (A,x):
    # A dot x
    Ax = A_dot_x(A, x)
    
    # x dot Ax
    ans = 0
    for i in range(len(x)):
        ans += x[i]*Ax[i]
    return ans


# In[240]:


assert( x_dot_A_dot_x( [ [1,0], [0,1] ], [2,1] ) == 5)


# ## Measurement and Visualization of Run-Time (7 points)
# 
# For each of the implemented matrix operations, measure the execution
# time as a function of $n$, up to execution times on the order of one 
# second. Compare the execution times of your implementation with a
# dedicated matrix library, e.g. the numpy package within python. Plot
# the execution times for all matrix operations and both
# implementations.

# In[241]:


# ploting comparison dot product of x_dot_x
import matplotlib.pyplot as plt

import numpy as np
import time


# YOUR CODE HERE

## x dot x comparison
# calc runtimes for x of length 1 to 100 using own method
x_lengths = np.arange(1, 100) # 50 000 000
xx_periods_ours = []
xx_periods_numpy = []
for x_len in x_lengths:
    
    # create vector x
    vect = np.arange(1,x_len+1)
    
    # time the execution of x*x
    init = time.time()
    b = x_dot_x(vect)
    duration = time.time() - init
    
    xx_periods_ours.append(duration)
    
    # numpy's runtime
    init = time.time()
    b = vect.dot(vect)
    duration = time.time() - init
    
    xx_periods_numpy.append(duration)
# print(xx_periods_ours)
# print(xx_periods_numpy)

# plot results

# fig, axs = plt.subplots(2)
# fig.suptitle('x dot x execution time')
# axs[0].plot(x_lengths, xx_periods_ours)
# axs[1].plot(x_lengths, xx_periods_numpy)

plt.plot(x_lengths, xx_periods_ours, ".", xx_periods_numpy, ".")
plt.xlabel('Vector length 1xn')
plt.ylabel('X dot X execution time (s)')
plt.show()


# In[242]:


# ploting comparison dot product of A_dot_x

# already created x_lengths
Ax_periods_ours = []
Ax_periods_numpy = []
for x_len in x_lengths:
    # create vector
    vect = np.arange(1,x_len+1)
    
    # create matrix of same length as vector ( x_len by x_len ) matrix
    #A = np.zeros(shape=(x_len, x_len))
    ##print(A)
    #for i in range(x_len):
    #    for j in range(x_len):
    #        A[i][j] = i+j
    
    shape = x_len,x_len # shape of matrix, x_len by x_len matrix
    row_len, col_len = np.indices(shape) # rows and column indices
    A = ((37.1*(row_len+1) + 91.7*(col_len+1)**2)%20.0)-10.0 # how every element will be calculated using the indices
    
    
    # our custom Ax duration
    init = time.time()
    b = A_dot_x (A, vect)
    duration = time.time() - init
    
    Ax_periods_ours.append(duration)
    
    # numpy's Ax duration
    init = time.time()
    b = A.dot(vect)
    duration = time.time() - init
    
    Ax_periods_numpy.append(duration)

plt.plot(x_lengths, Ax_periods_ours, ".", Ax_periods_numpy, ".")
plt.xlabel('Vector length 1xn')
plt.ylabel('A dot x execution time (s)')
plt.show()


# In[243]:


# ploting comparison dot product of A_dot_A (A)

# already created x_lengths
AA_periods_ours = []
AA_periods_numpy = []
for x_len in x_lengths:
    
    # create matrix of same length as vector ( x_len by x_len ) matrix
    
    shape = x_len,x_len # shape of matrix, x_len by x_len matrix
    row_len, col_len = np.indices(shape) # rows and column indices
    A = ((37.1*(row_len+1) + 91.7*(col_len+1)**2)%20.0)-10.0 # how every element will be calculated using the indices
    
    
    # our custom AA duration
    init = time.time()
    b = A_dot_A (A)
    duration = time.time() - init
    
    AA_periods_ours.append(duration)
    
    # numpy's AA duration
    init = time.time()
    b = A.dot(A)
    duration = time.time() - init
    
    AA_periods_numpy.append(duration)

plt.plot(x_lengths, AA_periods_ours, ".", AA_periods_numpy, ".")
plt.xlabel('Matrix size nxn')
plt.ylabel('A dot A execution time (s)')
plt.show()


# In[244]:


# ploting comparison dot product of x_dot_A_dot_x (A,x)

# already created x_lengths
xAx_periods_ours = []
xAx_periods_numpy = []
for x_len in x_lengths:
    # create vector
    vect = np.arange(1,x_len+1)
    
    # create matrix of same length as vector ( x_len by x_len ) matrix
    shape = x_len,x_len # shape of matrix, x_len by x_len matrix
    row_len, col_len = np.indices(shape) # rows and column indices
    A = ((37.1*(row_len+1) + 91.7*(col_len+1)**2)%20.0)-10.0 # how every element will be calculated using the indices
    
    
    # our custom xAx duration
    init = time.time()
    b = x_dot_A_dot_x(A,vect)
    duration = time.time() - init
    
    xAx_periods_ours.append(duration)
    
    # numpy's xAx duration
    init = time.time()
    b = vect.dot(A.dot(vect))
    duration = time.time() - init
    
    xAx_periods_numpy.append(duration)

plt.plot(x_lengths, xAx_periods_ours, ".", xAx_periods_numpy, ".")
plt.xlabel('Vector length 1xn')
plt.ylabel('x dot A dot x execution time (s)')
plt.show()


# Present your results in a clear and understandable form. Make sure all features you refer to in the discussion below can easily be identified.

# ## Interpretation (7 points)
# 
# Base your answers to the following questions on your implementation and measurements above. Explain your reasoning. Refer to the plot(s) and other results where appropriate.

# How do the runtimes of the implementation in pure Python and numpy compare? Can you explain the differences?

# YOUR ANSWER HERE

# Based on the plot(s) from the previous part, compare the computational complexity of the different matrix operations. Do the results agree with your expectations?

# YOUR ANSWER HERE

# How many floating point operations per second do the algorithms achieve? It is sufficient to quote a few examples.
# 
# On which hardware did you execute the tests? Are your results in line with the FLOPS of your computer?

# YOUR ANSWER HERE
