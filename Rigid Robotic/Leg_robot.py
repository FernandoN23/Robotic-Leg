"""
Forward and Inverse Kinematics Leg Robot
Trabajo Dirigido Avanzado
"""


import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
import Matrices_Homogeneas_y_DH as MH

"Función auxiliar de jacobiano para el cálculo de la cinemática inversa"
def jacobian(f, x, epsilon=1e-1): 
        n = len(x)
        m = len(f(x))
        jacobian_matrix = np.zeros((m, n))
        f_x = f(x)
        
        for i in range(n):
            perturbation = np.zeros(n)
            perturbation[i] = epsilon
            f_x_plus = f(x + perturbation)
            jacobian_matrix[:, i] = (f_x_plus - f_x) / epsilon
            
        return jacobian_matrix

"Definición de la clase del robot"
class Leg_robot():
    def __init__(self, val_list=[0,0]):
        self.val_list = val_list
        self.poses = np.array([])
        self.ax3 = None #subplot 3D
        self.ax = None #subplot 2D
        self.memoriaX=[]
        self.memoriaY=[]
        self.link_x = [0]
        self.link_y = [0]
    "Obtener las poses del robot"
    def get_pose_to_origin_leg_robot(self):

        origen = np.identity(4) 

        #Transformación Primer Link(pata superior)
        Z1 = np.linalg.multi_dot([MH.rot_z(np.deg2rad(self.val_list[0])), MH.translation_z(0)])
        X1 = np.linalg.multi_dot([MH.translation_x(160), MH.rot_x(0)])

        #Transformación Segundo link (pata inferior)
        Z2 = np.linalg.multi_dot([MH.rot_z(np.deg2rad(self.val_list[1])), MH.translation_z(0)])
        X2 = np.linalg.multi_dot([MH.translation_x(150), MH.rot_x(0)]) 

        #Transformación Tercer link (tobillo-efector)
        Z3 = np.linalg.multi_dot([MH.rot_z(np.deg2rad(-45.4)), MH.translation_z(0)])
        X3 = np.linalg.multi_dot([MH.translation_x(20), MH.rot_x(0)]) 

        #Sistema de referencia efector:
        Z4 = np.linalg.multi_dot([MH.rot_z(np.deg2rad(0)), MH.translation_z(0)])
        X4 = np.linalg.multi_dot([MH.translation_x(0), MH.rot_x(0)]) 
        #Union de links
        A1 = np.linalg.multi_dot([origen,Z1])
        A2 = np.linalg.multi_dot([A1,X1,Z2])
        A3 = np.linalg.multi_dot([A2,X2,Z3])
        A4 = np.linalg.multi_dot([A3,X3,Z4,X4])
        #Poses de todos los ejes
        poses = np.array([origen,A1,A2,A3,A4])
        self.poses = poses
        return self.poses
    

    "Dibujar los ejes del Algoritmo de Denavit-Hartenberg y la curva del efector"
    def draw_axes_and_effector(self, poses, name="", color="k"):
        #Rescatamos las poses del efector para dibujar en el plano
        X_pos = poses[4][0][3]
        Y_pos = poses[4][1][3]
        #Se guardan los valores anteriores en variables de memoria para conservar los puntos dibujados
        self.memoriaX.append(X_pos)
        self.memoriaY.append(Y_pos)

        #Iteramos para cada cambio en las poses
        for pose in poses:
            
            "Gráfico de los ejes en las variables articuladas"
            #Configuración de labels de cada eje
            self.ax3.set_xlabel('X [mm]')
            self.ax3.set_ylabel('Y [mm]')
            self.ax3.set_zlabel('Z [mm]')
            #Configuración de los límites de cada eje
            self.ax3.set_xlim(-600, 600)
            self.ax3.set_ylim(-600, 600)
            self.ax3.set_zlim(0, 300)

            #Graficar los ejes
            self.ax3.scatter(xs=[0], ys=[0], zs=[0], marker='o', color=color)
            origin_pose = np.transpose(pose)[3, 0:3]            
            x_rot = np.linalg.multi_dot([pose, [1, 0, 0, 0]])
            y_rot = np.linalg.multi_dot([pose, [0, 1, 0, 0]])
            z_rot = np.linalg.multi_dot([pose, [0, 0, 1, 0]])
            self.ax3.quiver(origin_pose[0], origin_pose[1], origin_pose[2], x_rot[0],
                    x_rot[1], x_rot[2], length=100, normalize=True, color='r')
            self.ax3.quiver(origin_pose[0], origin_pose[1], origin_pose[2], y_rot[0],
                    y_rot[1], y_rot[2], length=20, normalize=True, color='g')
            self.ax3.quiver(origin_pose[0], origin_pose[1], origin_pose[2], z_rot[0],
                    z_rot[1], z_rot[2], length=50, normalize=True, color='b')
            self.ax3.scatter(xs=[origin_pose[0]], ys=[origin_pose[1]],
                    zs=[origin_pose[2]], marker='o')
            "Dibujo de la curva descrita por el efector"

            #Configuración de labels de cada eje
            self.ax.set_xlabel('X [mm]')
            self.ax.set_ylabel('Y [mm]')

            #Configuración de los límites de cada eje
            self.ax.set_xlim(-600, 600)
            self.ax.set_ylim(-600, 600)
            #Graficar lo que dibuja el efector 
            self.ax3.plot(self.memoriaX, self.memoriaY, "k") #Dibujo en el plot 3D
            self.ax.plot(self.memoriaX, self.memoriaY, "k") #Dibujo en el plot 2D
            #Titulo del gráfico:
            self.ax.set_title("Curva cinemática de la pata robótica")

    "Función para graficar"
    def plotter(self):
        #Definición de gráficos
        fig = plt.figure()
        fig.set_size_inches(12,6)
        self.ax3 = fig.add_subplot(121, projection='3d', position=[-0.15, 0.2, 0.8, 0.8])
        self.ax = fig.add_subplot(122, position=[0.65, 0.4, 0.3, 0.5])
        self.draw_axes_and_effector(self.poses) #Dibujar los ejes y ejes de las poses obtenidas

        "Sliders y botones interactivos"
        # Creacion de sliders
        self.slider1_ax = plt.axes([0.2, 0.1, 0.65, 0.03])
        self.slider2_ax = plt.axes([0.2, 0.05, 0.65, 0.03])
        resetax = plt.axes([0.8, 0.015, 0.1, 0.04])

        self.slider1 = Slider(self.slider1_ax, 'Hombro', -90, 90, valinit=self.val_list[0])
        self.slider2 = Slider(self.slider2_ax, 'Codo', -90, 90, valinit=self.val_list[1])
        
        #Agregar boton de reseteo
        button = plt.Button(resetax, 'Reset', color='white', hovercolor='0.975')
        button.on_clicked(self.reset)
        self.slider1.on_changed(self.update)
        self.slider2.on_changed(self.update)

        #plt.ion() #solo activar para el caso iterativo de varios ángulos
        plt.show()
        plt.pause(0.05)
    "Actualización del gráfico"
    def update(self, val):
        #Limpiar el plot
        self.ax3.clear()
        self.ax.clear()
        slider_values = np.array([self.slider1.val, self.slider2.val])
        self.val_list = slider_values
        self.get_pose_to_origin_leg_robot()
        self.draw_axes_and_effector(self.poses)
        pass

    "Reinicio de los sliders"
    def reset(self, event):
        self.slider1.reset()
        self.slider2.reset()
        self.memoriaX = []
        self.memoriaY = []
        self.ax.clear()
        pass
    
    "Cálculo de cinemática directa"
    def forward_kinematics(self, parametros):
        
        self.val_list = parametros
        self.get_pose_to_origin_leg_robot()
        
        #Matriz de transformación homogénea
        XYZ = np.array([self.poses[4][0][3], self.poses[4][1][3]], dtype=float) # Reemplazar por columna correspondiente de la matriz de transformacion homogenea (self.poses)
        # Pose final del end effector
        pose = np.array([XYZ[0], XYZ[1]])
        return pose

    "Cálculo de cinemática inversa"
    def inverse_kinematics(self, target_pos,theta0=[0,0]): 
        "Obtener las poses del efector"
        def F(theta):
            self.val_list = theta
            self.get_pose_to_origin_leg_robot()
            XYZ = np.array([self.poses[4][0][3], self.poses[4][1][3]], dtype=float)
            f1 = XYZ[0]
            f2 = XYZ[1]
            return np.array([f1, f2])
        # TODO: Implementar la cinematica inversa del SCARA
        def NewtonRaphson(f,theta0, target_pos, tol=1e-3, max_iter=100):
            theta = theta0
            i = 0
            for _ in range(max_iter):
                try:
                    jacobiano_val=np.linalg.inv(jacobian(f, theta))
                    
                except:
            #usar la pseudo inversa si es singular
                    jacobiano_val=np.linalg.pinv(jacobian(f, theta))
                MSQ = np.linalg.multi_dot([jacobiano_val, jacobiano_val.transpose()])      
                
                if np.abs(np.linalg.det(MSQ))>0:
                    theta_new = theta - np.linalg.multi_dot([jacobiano_val,(f(theta) - target_pos)])
                else:
                    theta_new = theta + np.random.rand(len(theta))
                print(theta_new)
                #print((theta_new))
                if abs(f(theta_new)[0] - target_pos[0]) < tol and abs(f(theta_new)[1] - target_pos[1])  < tol:
                    print('Error por componente: ',[abs(f(theta_new)[0] - target_pos[0]),abs(f(theta_new)[1] - target_pos[1])])
                    print('Iteraciones: ' ,i)
                    return theta_new 
                theta = theta_new
                i += 1
                
                print(theta)
            raise ValueError("No se pudo encontrar la solución después de %d iteraciones" % max_iter)
        pos = NewtonRaphson(F,theta0,target_pos)
        
        return pos


if __name__ == "__main__":
    Leg = Leg_robot()
    "Invocar cinemática directa"
    
    param_joints = [0,90]# Ejemplo de parametros de los 4 joints, modificar por cualquier otro valor que sea factible
    dk = Leg.forward_kinematics(param_joints)
    print('Cinematica directa:', dk)

    "Invocar cinemática inversa a partir del resultado de cinemática directa"
    inicio = time.time()
    ik = Leg.inverse_kinematics(dk) #modificar las variables de esta función dependiendo de cómo funciona su fn de c. inversa
    print('Cinematica inversa:', ik)
    fin = time.time()
    print('Tiempo de ejecución: ', fin-inicio, '[s]' )
    #param_joints debe ser igual o equivalente a ik
    
    " Ejecutar el plotter"
    Leg.plotter()