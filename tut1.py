
2.#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 08:02:10 2020

@author: rayhaan
"""

"CP TUT 1 "  "PRNMOG001"
import numpy  as np
from functools import reduce
import json
from collections import OrderedDict as odic


vecprod= odic()
mprod = odic()

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

#CREATING AND WRITING TO JSON FILE DEFINED
def WJSON(path, fileName, data):
    filePathNameWExt = './' + path + '/' + fileName + '.json'
    with open(filePathNameWExt, 'w') as fp:
        json.dump(data, fp)


InputList = [5,10,20,100,1000]

   
for a in InputList:
    
    #CREATING VECTORS
    xr = np.arange(1,a+1)
    xc = xr.reshape(a,1)
    
    #CREATING MATRIX
    shape = a,a # Defines the shape of the Matrix, in this case a,a implies a matrxi with a rows and a columns
    xi, yi = np.indices(shape) # xi is the rows indices and yi is the column indices
    M = ((37.1*(xi+1) + 91.7*(yi+1)**2)%20.0)-10.0 # Indicates how every element will be calculated using the indices
    
    #DOT PRODUCT
    xtx = int(dot(xr,xr)) #Tranpose multiplied by vector
    
    #MATRIX VECTOR PRODUCT
    Mx = mtvec(M,xc)
    
    #VECTOR MATRIX VECTOR PRODUCT
    xtMx = int(dot(xr,Mx)[0])
    
    #MATRIX MATRIX MULTIPLICATION
    MM = matmult(M,M)
    
    ##ADDING DATA TO DICTIONARY
    vecprod[str(a)] = xtx  
    mprod[str(a)] = xtMx
    
xtxp = {'xtx': vecprod}
mprodp = {'xtMx':mprod}
dic1 = odic()
dic1 = xtxp
dic1.update(mprodp)
WJSON('./','Results',dic1)
        
    
                
