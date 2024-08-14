import quadlib as qdlb
import numpy as np
import matplotlib.pyplot as plt
import time

fig, ax = qdlb.figure3D()

ech = 30

def draw_quad(X, alpha):
    qdlb.clean_draw3D(ax,-ech,ech,-ech,ech,-ech,ech)
    R = qdlb.euler_matrix(X[3], X[4], X[5])
    qdlb.draw_quadrotor3D(ax, X, R, alpha, 5 * l)

def f(X, W):

    B = np.asarray( [ [beta, beta, beta, beta],[-beta * l, 0., beta * l, 0.],[0., -beta * l, 0., beta * l], [-delta, delta, -delta, delta] ] )

    X = X.flatten()

    x, y, z, phi, theta, psi = list(X[0:6])

    vr = X[6:9]

    wr = X[9:12]

    W2 = W * np.fabs(W)

    tau = B @ W2.flatten()

    R = qdlb.euler_matrix(phi, theta, psi)

    delta_vr = -np.dot(qdlb.adjoint(wr), vr) + np.dot(np.linalg.inv(R), np.asarray([0., 0., 9.81])) + np.asarray([0.,0.,-tau[0] / m])

    delta_p = np.dot( R, vr)

    dangles = np.dot( qdlb.eulerderivative(phi, theta, psi), wr )

    delta_wr = np.dot( np.dot( np.dot(np.linalg.inv(I), (-qdlb.adjoint(wr))), I), wr ) + tau[1:4]

    delta_X = np.vstack((delta_p, dangles, delta_vr, delta_wr)).flatten()

    return delta_X

def control(X):

    return np.ones(4) * 5.

x = np.asarray([0,0,-5, 0., 0, 0, 10, 10, 0, 0, 0, 0])

alpha = np.asarray([ 0.,0.,0.,0. ]).T

delta_t = 0.1

I = np.eye(3) * 10.

I[-1][-1] = 20.

beta = 2.

delta = 1.

m = 10.

l = 1.

for t in np.arange(0, 5, delta_t):

    omega = control(x)

    x = x + delta_t * f(x, omega)
    alpha = alpha + delta_t * omega
    draw_quad(x, alpha)
    time.sleep(0.001)
    plt.show()

time.sleep(1)
