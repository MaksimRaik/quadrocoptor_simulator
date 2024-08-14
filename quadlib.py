import numpy as np
import matplotlib.pyplot as plt
import helpfunction as hp
import scipy as sp

## draw function

def figure3D():

    return plt.subplots(subplot_kw=dict(projection='3d'))

def clean_draw3D( ax, xlim1 = -10, xlim2 = 10, ylim1 = -10, ylim2 = 10, zlim1 = -10, zlim2 = 10):

    ax.clear()
    ax.set_xlim(xlim1, xlim2)
    ax.set_ylim(ylim1, ylim2)
    ax.set_zlim(zlim1, zlim2)

def draw3H(ax, M, col, shadow=False, mirror=1):  # mirror=-1 in case z in directed downward
    #print(M[0])
    ax.plot(mirror * M[0], M[1], mirror * M[2], color=col)
    if shadow:
        ax.plot(mirror * M[0], M[1], 0 * M[2], color='gray')

def circle3H(r):
    theta = np.arange(0, 2 * np.pi, 0.01)
    x = r * np.cos(theta) + np.zeros(theta.size)
    y = r * np.sin(theta) + np.zeros(theta.size)
    z = np.zeros(theta.size)
    return hp.add1(np.asarray([x, y, z]))

def tran3H(x, y, z):

    Matrix = np.eye(4)

    Matrix[-1,:] = np.asarray([x, y, z, 0.0])

    return Matrix

def ToH(R,v = np.zeros((3,1))):  # transformation matrix to homogenous
    H = np.hstack((R,v))
    V = np.vstack((H, np.asarray([0,0,0,1])))
    return V

def adjoint(w):
    if isinstance(w, (float, int)): return np.asarray([[0,-w] , [w,0]])

    return np.asarray([[0,-w[2],w[1]] , [w[2],0,-w[0]] , [-w[1],w[0],0]])
def adjoint_inv(A):
    if A.size==4:  return A[1,0]  # A is 2x2
    return np.asarray([[A[2,1]],[A[0,2]],[A[1,0]]]) # A is 3x3
def expw(w):

    return sp.linalg.expm(adjoint(w))

def euler_matrix(phi, theta, psi):

    return expw([0,0,psi]) @ expw([0,theta,0]) @ expw([phi,0,0])

def eulerderivative(phi,theta,psi):

    return np.asarray([[1,np.sin(phi)*np.tan(theta),np.cos(phi)*np.tan(theta)],
                       [0, np.cos(phi),-np.sin(phi)],
                       [0,np.sin(phi) / np.cos(theta),np.cos(phi)/np.cos(theta)]])

def eulerH(phi,theta,psi):

    return ToH(expw([0,0,psi]) @ expw([0,theta,0]) @ expw([phi,0,0]))

def draw_quadrotor3D(ax, p, R, alpha, l, mirror=-1):

    Ca=np.hstack((circle3H(0.3*l),[[0.3*l,-0.3*l],[0,0],[0,0],[1,1]]))
    T = np.dot(tran3H(p[0], p[1], p[2]), ToH(R))  # I replaced tran3H(*) to avoid warning
    C0 = np.dot(np.dot(np.dot(T, tran3H(0, l, 0)), eulerH(0, 0, alpha[0])), Ca)  # we rotate the blades
    C1 = T @ tran3H(-l, 0, 0) @ eulerH(0, 0, -alpha[1]) @ Ca
    C2 = T @ tran3H(0, -l, 0) @ eulerH(0, 0, alpha[2]) @ Ca
    C3 = T @ tran3H(l, 0, 0) @ eulerH(0, 0, -alpha[3]) @ Ca
    M = T @ hp.add1(np.asarray([[l, -l, 0, 0, 0], [0, 0, 0, l, -l], [0, 0, 0, 0, 0]]))
    draw3H(ax, M, 'grey', True, mirror)  # body
    draw3H(ax, C0, 'green', True, mirror)
    draw3H(ax, C1, 'black', True, mirror)
    draw3H(ax, C2, 'red', True, mirror)
    draw3H(ax, C3, 'blue', True, mirror)
