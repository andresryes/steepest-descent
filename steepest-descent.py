import numpy as np
from numpy import linalg
import scipy as sp
from scipy import optimize
import math
from sympy import symbols, diff

#we receive a 2x2 matrix
A = np.matrix([[2, 1],[1,2]])
B = np.matrix([[1, 0],[0,2]])
C = np.matrix([[5, 1],[1,4]])

A = B
#vector b 2x1
b = np.matrix([[3], [4]])
u,v = symbols('u v', real=True)
print("A:")
print(A)
print("b:")
print(b)

#function
#f = A.A1[3]*v
#print(diff(f, v))

#print(diff(f1, v))
#print(gradientef([1,1]))
#using the linalg library
print("exact solution using numpy:")
print(linalg.solve(A,b))

#Steepest Descent 
def f(x):
    u = x[0]
    v = x[1]
    return 0.5*(u*(u*A.A1[0] + v*A.A1[2]) + v*(u*A.A1[1] + v*A.A1[3])) - b.A1[0]*u - b.A1[1]*v

#the gradient of the previous function, it indicates the direction
def gradientef(x):
    u,v = symbols('u v', real=True)
    f1 = 0.5*(u*(u*A.A1[0] + v*A.A1[2]) + v*(u*A.A1[1] + v*A.A1[3])) - b.A1[0]*u - b.A1[1]*v
    du = str(diff(f1, u))
    dv = str(diff(f1, v))
    u = x[0]
    v = x[1]
    y=[eval(du),eval(dv)]
    return y  

#alpha function of one variable, indicates the step in each iteration/jump
def funcion(alpha,x,s):
    u = x[0]+alpha*s[0]
    v = x[1]+alpha*s[1]
    y1 = 0.5*(u*(u*A.A1[0] + v*A.A1[2]) + v*(u*A.A1[1] + v*A.A1[3])) - b.A1[0]*u - b.A1[1]*v
    return y1

#this is the method itself, receives xk first guess, the function, tolerance
def desc(xk,f,tol):
    #initial error
    error=1
    #this is the array with all the dots to graphic 
    X=[]
    #array with all vectors 
    S=[]
    #append to the dots array the initial guess
    X.append(xk)
    #make this algorithm while the error is still bigger than the tolerance 
    #level
    while(error>tol):
        #calculates the first direction of the steepest descent using the 
        #negative of the gradient
        sk=np.dot(-1.0,gradientef(xk))  
        #optimizes alpha funciton using Golden Ration optimization of Python
        ak1=sp.optimize.golden(funcion,args=(xk,sk))
        #getting the direction using the step size
        sk_n=np.dot(ak1,sk)
        #append the direction
        S.append(sk_n)
        #calculating our new xk+1
        xk1=xk+np.dot(ak1,sk)
        #append the new dot
        X.append(xk1)
        #calculating the error
        error=np.linalg.norm(xk1-xk)
        #changing xk to the new value
        xk=xk1
    return xk,X,S

y=desc([10, 15],f,0.0000001)
print("the approx solution is: "+str(y[0]))
Xk=y[1] 
Sk=y[2] 
print("iterations:")
print(len(Sk))

X1=[]
Y1=[]
Z1=[]