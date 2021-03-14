#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 18:55:04 2020

@author: rayhaan
"""

import time
import numpy  as np
from functools import reduce
import matplotlib.pyplot as plt

#MATRIX MULTIPLICATION FUNCTION
def matmult(A, B):
    C = [[0 for row in range(len(A))] for col in range(len(B[0]))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k]*B[k][j]
    return C

#DOT PRODUCT FUNCTION
def dot(a,b):
    dotp = 0
    for e,f in zip(a,b):
        dotp += e*f
    return dotp

#MATRIX TIMES VECTOR FUNCTION
def mtvec(m, v):
    nrows = len(m)
    w = [None] * nrows
    for row in range(nrows):
        w[row] = reduce(lambda x,y: x+y, map(lambda x,y: x*y, m[row], v))
    return w

InputList1 = np.arange(0,1001,100)
nlist =[]
clist = []


start = time.time()
for a in InputList1:
    
    #CREATING VECTORS
    xr = np.arange(1,a+1)
    xc = xr.reshape(a,1)
    
    #CREATING MATRIX
    shape = a,a # Defines the shape of the Matrix, in this case a,a implies a matrxi with a rows and a columns
    xi, yi = np.indices(shape) # xi is the rows indices and yi is the column indices
    M = ((37.1*(xi+1) + 91.7*(yi+1)**2)%20.0)-10.0 # Indicates how every element will be calculated using the indices
    
    ##START TIME
    start = time.time()
    
    #DOT PRODUCT
    #xtx = int(dot(xr,xr)) #Tranpose multiplied by vector
    
    #MATRIX VECTOR PRODUCT
    Mx = mtvec(M,xc)
    
    #VECTOR MATRIX VECTOR PRODUCT
    xtMx = dot(xr,Mx)
    
    #MATRIX MATRIX MULTIPLICATION
    #MM = matmult(M,M)
    
    t = time.time() - start
    clist.append(t)
    
    
    
#    
t = 0
start = 0
InputList2 = np.arange(0,7001,700)
#

for a in InputList2:
    
    #CREATING VECTORS
    xr = np.arange(1,a+1)
    xc = xr.reshape(a,1)
    
    #CREATING MATRIX
    shape = a,a # Defines the shape of the Matrix, in this case a,a implies a matrxi with a rows and a columns
    xi, yi = np.indices(shape) # xi is the rows indices and yi is the column indices
    M = ((37.1*(xi+1) + 91.7*(yi+1)**2)%20.0)-10.0 # Indicates how every element will be calculated using the indices
    
    #START TIME
    start = time.time()
    
    #NUMPY DOT PRODUCT
    #xtxn = xr.dot(xr)
    
    
    #MATRIX VECTOR PRODUCT NUMPY
    #Mxn = M.dot(xc)
    
    #VECTOR MATRIX VECTOR PRODUCT NUMPY
    a = xr.dot(M).dot(xc)[0]
    t = time.time() - start
    nlist.append(t)


#PLOTTING RESULTS
fig = plt.figure()

ax = fig.add_subplot(111)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')

ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

ax1.scatter(InputList1, clist, color = 'b',label = 'Custom algorithm')
ax2.scatter(InputList2, nlist, color = 'r', label = 'Numpy package')

ax.set_xlabel('Array size (1 x n)(n x n)(n x 1)')
ax.set_ylabel('Execution time (s)')

ax1.legend()
ax2.legend()

plt.plot()

plt.savefig('xtMx.png')



