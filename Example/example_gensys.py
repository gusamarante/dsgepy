"""
This script implents
"""

from pydsge import gensys
import numpy as np

g0 = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [-1.1, 0, 1, 1],
               [0, 1, 0, 0]])

g1 = np.array([[0,   0, 1, 0],
               [0,   0, 0, 1],
               [0,   0, 0, 0],
               [0, 0.7, 0, 0]])

c = np.zeros((4, 1))

psi = np.array([[0,  0],
                [0,  0],
                [4,  1],
                [3, -2]])

pi = np.array([[1, 0],
               [0, 1],
               [0, 0],
               [0, 0]])

G1, C, impact, fmat, fwt, ywt, gev, eu, loose = gensys(g0, g1, c, psi, pi)
