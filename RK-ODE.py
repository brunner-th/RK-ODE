# -*- coding: utf-8 -*-

# -----------------------------------------------------------------
#
# name: thomas brunner
# email: brunner.th@hotmail.com
# nr: 12018550
#
# -----------------------------------------------------------------

from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.figsize"] = (10, 7)


def DESolverND(
    f,
    y_0,
    delta_t,
    t_max,
    A=[[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 1 / 2, 0, 0], [0, 0, 1, 0]],
    b=[1 / 6, 1 / 3, 1 / 3, 1 / 6],
    c=[0, 1 / 2, 1 / 2, 1],
    plot=True,
):

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    # f is a DE in the form ydash = func(y,t)
    
    t_vec = np.linspace(0, t_max, int(t_max / delta_t), dtype=float)

    # check butcher tableau:

    IsExplicit(A)

    def findK(tn, yn):

        k_vector = np.zeros((len(b), len(y_0)), dtype=float)
        # print(k_vector)
        for i in range(len(b)):
            summe = np.zeros_like(y_0)

            for j in range(len(b)):

                summe = np.add(summe, A[i, j] * k_vector[j, :])
                # print(summe)
            if np.shape(yn)[0] == 2:
                k_vector[i, :] = f(
                    tn + delta_t * c[i],
                    np.add(yn[:], delta_t * summe[:])[0],
                    np.add(yn[:], delta_t * summe[:])[1],
                )
            elif np.shape(yn)[0] == 6:
                k_vector[i, :] = f(
                    tn + delta_t * c[i],
                    np.add(yn[:], delta_t * summe[:])[0],
                    np.add(yn[:], delta_t * summe[:])[1],
                    np.add(yn[:], delta_t * summe[:])[2],
                    np.add(yn[:], delta_t * summe[:])[3],
                    np.add(yn[:], delta_t * summe[:])[4],
                    np.add(yn[:], delta_t * summe[:])[5],
                )
            
            elif np.shape(yn)[0] == 1:
                k_vector[i, :] = f(
                    tn + delta_t * c[i],
                    np.add(yn[:], delta_t * summe[:])[0],
                )
            
            elif np.shape(yn)[0] == 3:
                k_vector[i, :] = f(
                    tn + delta_t * c[i],
                    np.add(yn[:], delta_t * summe[:])[0],
                    np.add(yn[:], delta_t * summe[:])[1],
                    np.add(yn[:], delta_t * summe[:])[2],
                )
                
            
        return k_vector

    yn = np.zeros((len(t_vec), len(y_0)), dtype=float)

    yn[0, :] = y_0

    for i in range(len(t_vec) - 1):
        summe2 = np.zeros_like(y_0, dtype=float)

        k = findK(t_vec[i], yn[i])

        for j in range(len(b)):
            summe2 += b[j] * k[j, :]
        yn[i + 1, :] = yn[i, :] + delta_t * summe2
    if np.array_equal(A, [[0]]):
        Method = "Explicit Euler"
    elif np.array_equal(A, [[0, 0], [1 / 2, 0]]):
        Method = "Explicit Midpoint"
    elif np.array_equal(
        A, [[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 1 / 2, 0, 0], [0, 0, 1, 0]]
    ):
        Method = "Runge-Kutta-4"
    else:
        Method = "Unknown Method"
    

    if plot is True:

        plt.plot(
            t_vec, yn[:, 0], color="firebrick"
        )  # markers left out for visibility , marker = "+"
        #plt.plot(t_vec, yn[:, 1], color="steelblue", label="Omega")
        
        String = "Method: " + Method + ", Delta_t = " + str(delta_t)
        plt.title(String)
        
        plt.grid()
        plt.show()
    return yn


def IsExplicit(Butcher_Tableau):

    if np.allclose(Butcher_Tableau, np.tril(Butcher_Tableau)) and np.allclose(
        np.diag(Butcher_Tableau), np.zeros(len(Butcher_Tableau))
    ):
        print("Butcher Tableau suggests that the Method is explicit")
    else:
        print("Butcher Tableau suggests that the Method is implicit")





function_pend = lambda t, phi, w: [w, -9.81 * sin(phi)]

function_ex = lambda t, x: [3*x]
 
DESolverND(function_ex, [3], 0.01, 10)  # hier geb ich phi,omega ein, RK4

DESolverND(function_pend, [9.9*pi / 10, 0], 0.1, 10)  # hier geb ich phi,omega ein, RK4







