#Importacion de librerias
import numpy as np
from sympy import *
#Matrices de traslación homogéneas

def translation_x(x):
    return np.array([[1, 0, 0, x],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

def translation_y(y):
    return np.array([[1, 0, 0, 0],
                      [0, 1, 0, y],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

def translation_z(z):
    return np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                        [0, 0, 1, z],
                        [0, 0, 0, 1]])

#Matrices de rotación homogéneas

def rot_x(qx):
    return np.array([[1, 0, 0, 0],
                      [0, np.cos(qx), -np.sin(qx), 0],
                        [0, np.sin(qx), np.cos(qx), 0],
                        [0, 0, 0, 1]])

def rot_y(qy):
    return np.array([[np.cos(qy), 0, np.sin(qy), 0],
                      [0, 1, 0, 0],
                        [-np.sin(qy), 0, np.cos(qy), 0],
                        [0, 0, 0, 1]])

def rot_z(qz):
    return np.array([[np.cos(qz), -np.sin(qz), 0, 0],
                      [np.sin(qz), np.cos(qz), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

#Matrices de rotación simbólicas

def rot_xs(qx):
    return np.array([[1, 0, 0, 0],
                      [0, cos(qx), -sin(qx), 0],
                        [0, sin(qx), cos(qx), 0],
                        [0, 0, 0, 1]])

def rot_ys(qy):
    return np.array([[cos(qy), 0, sin(qy), 0],
                      [0, 1, 0, 0],
                        [-sin(qy), 0, cos(qy), 0],
                        [0, 0, 0, 1]])

def rot_zs(qz):
    return np.array([[cos(qz), -sin(qz), 0, 0],
                      [sin(qz), cos(qz), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])



#Matriz del algoritmo de Denavit-Hartenberg
def Transformation_DH(theta, d, a, alpha):
    return np.linalg.multi_dot([rot_z(theta), translation_z(d), translation_x(a), rot_x(alpha)])

#Matriz del algoritmo de Denavit-Hartenberg simbólica
def Transformation_DHS(theta, d, a, alpha):
    return np.linalg.multi_dot([rot_zs(theta), translation_z(d), translation_x(a), rot_xs(alpha)])

#Invertir matrices simbólicamente
def Inversa_S(M):
    return Matrix(M).inv()

#Función simbólica de la matriz noap
n_x = Symbol("n_x")
n_y = Symbol("n_y")
n_z = Symbol("n_z")
o_x = Symbol("o_x")
o_y = Symbol("o_y")
o_z = Symbol("o_z")
a_x = Symbol("a_x")
a_y = Symbol("a_y")
a_z = Symbol("a_z")
p_x = Symbol("p_x")
p_y = Symbol("p_y")
p_z = Symbol("p_z")

def Tnoap():
    T= Matrix([[n_x, o_x, a_x, p_x],
                  [n_y, o_y, a_y, p_y],
                  [n_z, o_z, a_z, p_z],
                  [0,0,0,1]])
    return T


"Cálculo de la cinemática directa e inversa de forma analítica mediante Algoritmo de Denavit-Hartenberg y sympy"
q_1s=Symbol("q_1")
q_2s=Symbol("q_2")
alpha_s = Symbol("alpha") #alpha = -45.4
l_1s=Symbol("l_1")
l_2s=Symbol("l_2")
l_3s=Symbol("l_3")

#Matrices de transformación en cada articulación


A_1s = Transformation_DHS(q_1s-pi/2, 0 ,l_1s, 0)
A_2s = Transformation_DHS(q_2s, 0 ,l_2s, 0)
A_3s = Transformation_DHS(alpha_s, 0, l_3s, 0)

#Matriz resultante transformacion
T_03s= simplify(np.linalg.multi_dot([A_1s, A_2s, A_3s]))
P_xs = T_03s[0][3]
P_ys = T_03s[1][3]
