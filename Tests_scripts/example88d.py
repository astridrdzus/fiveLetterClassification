#.-*-coding: utf-8-*-
#--------------------------------------------------#
#           Astrid Giselle Rodriguez Us
#--------------------------------------------------#
#Programa: Ejemplo 8.8 Finite Difference Method, Numerical Analysis, T. Sauer
#Descripción
#Fecha:

from math import*
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as la
from matplotlib import  cm
from matplotlib.animation import FuncAnimation
from matplotlib import  cm


plt.ion () #activa modo interactivo

#valores del problem
M = 100                                 #numero de puntos discretos
N = 60
xo = 0.0                                #region de solucion
xf = 1.0
yo = 1.0
yf = 2.0

dx = (xf-xo)/M                          #discretización espacial
dy = (yf-yo)/N
x = np.arange(xo,xf+dx,dx)              #posiciones espaciales discretas
y = np.arange(yo,yf+dy,dy)              #en la region de solucion

#condiciones de frontera espacial y funcion del problema
def g1(x): return  np.log(x**2+1.0)
def g2(x): return np.log(x**2+4.0)
def g3(y): return 2.0*np.log(y)
def g4(y): return np.log(y**2+1.0)
def f(x,y): return 0.0

#genera matrices de trabajo
X= np.zeros((M+1,N+1))
Y= np.zeros((M+1,N+1))
W = np.zeros((M+1, N+1))
for i in range(M+1):
    for j in range(N+1):
        X[i,j] = x[i]
        Y[i,j] = y[j]

#Solución directa del problema, declara variables
MN = (M+1)*(N+1)
A = np.zeros((MN,MN))
b = np.zeros((MN,1))

#condiciones de frontera
for j in range(1,N):
    i = 0
    k = i*(N+1) +j
    A[k,k] = 1.0
    b[k] = g3(y[j])
    i = M
    k = i*(N+1) + j
    A[k,k] = 1.0
    b[k] = g4(y[j])
for i in range (1,M):
    j = 0
    k = i*(N+1) +j
    A[k,k] = 1.0
    b[k] = g1(x[i])
    j = N
    k = i*(N+1)+j
    A[k,k] = 1.0
    b[k] = g2(x[i])

A[0,0] = 1.0
b[0] = 0.5*(g1(x[0])+g3(y[0]))
A[N,N] = 1.0
b[N] = 0.5*(g2(x[0])+ g3(y[N]))
A[M*(N+1), M*(N+1)] = 1.0
b[M*(N+1)] = 0.5*(g1(x[M])+g4(y[0]))
A[M*(N+1)+N,M*(N+1)+N] = 1.0
b[M*(N+1)+N] = 0.5*(g2(x[M])+g4(y[N]))

#llena matriz para puntos incógnitas

dx2 = 1.0/dx**2
dy2 = 1.0/dy**2
dxdy = -2.0*(dx2+dy2)
for i in range(1,M):
    for j in range(1,N):
        k = i*(N+1) + j
        b[k] = f(x[i],y[j])
        A[k,(i-1)*(N+1)+j] = dx2
        A[k,(i+1)*(N+1)+j] = dx2
        A[k,k] = dxdy
        A[k,i*(N+1)+ j-1] = dy2
        A[k,i*(N+1)+j+1] = dy2

#soluciona, actualiza superficie solucion
out = np.linalg.solve(A,b)
for i in range(M+1):
    for j in range(N+1):
        k = i*(N+1) + j
        W[i,j] = out[k]

#grafica superficie solucion
fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.set_xlim(xo,xf), ax.set_ylim(yo,yf), ax.set_zlim(0.0,2.0)
ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('w(x,y'), ax.set_title('Solucion')
#ax.plot_surface(X,Y,W, rstride=1, cstride=1, cmap = cm.coolwarm, linewidth=0, antialiased=False)
#ax.plot_wireframe(X,Y,W, rstride=1, cstride=1, cmap = cm.coolwarm, antialiased=False)
ax.plot_wireframe(X,Y,W)
#cierra todas las ventanas
input('presiona cualquier tecla para continuar')
plt.close('all')