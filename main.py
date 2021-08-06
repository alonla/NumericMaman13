import numpy as np
import matplotlib.pyplot as plt
#Get two vectors X = (x_0,x_1,...,x_n), Y = (y_0,y_1,...,y_n) and return the interpolating polynomial using the difference method
def Interpolation(X, F): #calculate devided difference table F
	for i in range(1, len(X)):
		for j in range(1, i + 1):
			F[i,j] = (F[i,j-1] - F[i-1,j-1])/(X[i]-X[i-j])
	print(F)
	return F; 

#formula for the intepolation polynom 
# Pn(x) = F0,0 + sum(i=0->n)Fi,i mul(j=0->i-1)(x-xj)

def XVAL(i,n):
	return (-5 + 10 * (i-1)/(n-1));

def YVAL(xi): #calculate y values according to x values. 
	return 1/(1+np.power(xi,2));

def f(t):#the function to interpolate
    return (1/(1+np.power(t,2)))
def CalcMult(t,i,X): 
    #calculate the multiplications in the interpolation formula for mult(x-xj)
    sum = 1; 
    for j in range(0,i-1):
        sum = sum * (t-X[j]);
    return sum; 
def IntPol(t,A,X): #calculate the interpolation polynom
    sum = A[0]; 
    for i in range(1,len(Y)):
        sum = sum + A[i]*CalcMult(t,i,X)
    return sum;

#MAIN
#F - contains the table of devided differences. 
# X - vector of x values
# Y - y values of the x's

X = [];
Y = [];
n = 5;

F = np.zeros((n,n))
#Get x vals 
for i in range(1, n+1):
	X.append(XVAL(i,n));
	Y.append(YVAL(X[i-1]))
#move on F and insert values to first column
for i in range(0,n):
	F[i,0] = Y[i];
Interpolation(X,F)

# the enteries of the interpolation polynom
for i in range(0,n):
	print(F[i,i])

    
t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

# the coefficients of the interpolation polynom
A = [];
for i in range(0,n):
    A.append(F[i,i]);

#plot 
plt.figure()
plt.subplot(211)
plt.plot(t2, f(t2), 'k', t2, IntPol(t2,A,X),'b')

    
#running that show that calculating the divided difference works.
'''X = [];
X.append(1);X.append(1.3);X.append(1.6);X.append(1.9);X.append(2.2)

F = np.array([[0.7651977,0,0,0,0], [0.6200860,0,0,0,0],[0.4554022,0,0,0,0],[0.2818186,0,0,0,0],[0.1103623,0,0,0,0]]);
print(F) #table of divided difference 
Interpolation(X,F);'''

for i in (0,5):
	print("hello");
	print("end");