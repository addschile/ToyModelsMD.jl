import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from numba import jit

def potential(x,y):
    A=np.array([-200,-100,-170,15])
    a=np.array([-1,-1,-6.5,0.7])
    b=np.array([0,0,11,0.6])
    c=np.array([-10,-10,-6.5,0.7])
    x0=np.array([1,0,-0.5,-1])
    y0=np.array([0,0.5,1.5,1])
    potterms=A*np.exp(a*(x-x0)**2+b*(x-x0)*(y-y0)+c*(y-y0)**2)
    return np.sum(potterms)

for k in range(1,21):
  Mx=101
  My=101
  x=np.linspace(-1.5,1.5,Mx)
  y=np.linspace(-0.5,2.0,My)
  xx,yy=np.meshgrid(x,y)
  V=np.zeros((My,Mx))
  for i in range(Mx):
      for j in range(My):
          V[j,i]=potential(x[i],y[j])
          if (V[j,i]>400):
              V[j,i]=400
  plt.contourf(x,y,V,levels=25,cmap='rainbow')
  for j in range(1,11):
    traj = np.loadtxt('init_traj_%d%d.txt'%(k,j))
    plt.plot(traj[:,0],traj[:,1])
  plt.colorbar()
  plt.xlabel(r'$x$')
  plt.ylabel(r'$y$')
  plt.title(r'Epoch: %d'%(k))
  plt.savefig('init_trajs_%d.png'%(k))
  plt.show()
