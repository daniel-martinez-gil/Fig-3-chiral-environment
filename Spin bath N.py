import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint



#epsilon and delta of the system

epsilon = 0
delta = 1
#epsilon and delta of the environment
Ei = 50
Di = 1


#número de moléculas
N = 12
#acomplamiento \Lambda
c_momento_momento = 0
#número de realizaciones para hacer la media n = 1000 or 2000
numero = 2000


z0 = 0.9999
phi0 = 0


# Definimos las funciones
def f1(t, z, phi):
    return -2 * delta * np.sqrt(1 - z**2) * np.sin(phi)

def f2(t, z, phi, zs, c_momento_momento):
    return (2 * epsilon + 2 * delta * (z / np.sqrt(1 - z**2)) * np.cos(phi) 
            +   c_momento_momento * np.sum(zs))

def fz(t, phi, z):
    return -2 * Di * np.sqrt(1 - z**2) * np.sin(phi)

def fphi(t, z, phi, z_s, c_momento_momento, Ei):
    return 2 * Ei + 2 * Di * (z / np.sqrt(1 - z**2)) * np.cos(phi) + N* c_momento_momento * z_s

# Función que agrupa todo el sistema de ecuaciones
def fun1(A, t, c_momento_momento, Ei):
    z, phi, zs, phis = A[0], A[1], A[2:N+2], A[N+2:]
    dz = f1(t, z, phi)
    dphi = f2(t, z, phi, zs, c_momento_momento)
    dzs = np.array([fz(t, phis[i], zs[i]) for i in range(N)])
    dphis = np.array([fphi(t, zs[i], phis[i], z, c_momento_momento,Ei) for i in range(N)])
    return np.concatenate(([dz, dphi], dzs, dphis))

# Condiciones iniciales
zs0 = np.random.random(N) * 1.999999 - 1
phis0 = np.random.random(N) * 2 * np.pi

A0 = np.concatenate(([z0, phi0], zs0, phis0))


"""
\epsilon, \delta and \Lambda are comparative measures. I considered 
Ei = 5e-11, Di = 1e-12, c_momento_momento (Lambda) = 1e-12. The results are the
same as considering Ei = 50, Di = 1, c_momento_momento = 1 becasuse the ratio
between them is the same (Ei/Di = 50), and the same with Lambda.
Only the time of the simulation changes, but the result is the same.
In the simulation I consider the first case, considering t = 100 to ensure 
convergence of the results. Otherwisethe time has to be considered with a correction
factor proportional to the order of magnitude of 
Ei, Di, Lambda, which is more computationally demandant.
"""

# Tiempo de integración
#pasos de tiempo 
pt = 2000
t = np.linspace(0, 100, pt)

# Resolución del sistema
sol = odeint(fun1, A0, t, args = (c_momento_momento, Ei))

# Graficación
z = sol[:, 0]
phi = sol[:, 1]
zi = sol[:, 2:N+2]
phii = sol[:, N+2:]



#Para hacer la media

# array de soluciones

def media(c_momento_momento, Ei):
    soluciones = [None] * numero
    for i in range(0,numero,1):
        zs0 = np.random.random(N) * 1.99 - 1
        phis0 = np.random.random(N) * 2 * np.pi
        #zs0 = np.zeros(N)
        #phis0 = np.ones(N)*np.pi
        
        A0 = np.concatenate(([z0, phi0], zs0, phis0))
    
        # Resolución del sistema
        sol = odeint(fun1, A0, t, args = (c_momento_momento, Ei))
        soluciones[i] = sol[:,0]
        
    #calcular la media para todas las simulaciones
    media = []
    for i in range(0,pt,1):
        medias = 0
        for j in range(0,numero,1):
            medias += soluciones[j][i]
        media.append(medias/numero)
        
    return media
        




plt.ylabel('$<Z(t)>_n$')
plt.xlabel('t')
plt.plot(t,media(c_momento_momento, Ei), 'black', label = '$\Lambda = $' +str(c_momento_momento))
plt.ylim(-1.1,1.1)
plt.legend(framealpha = 1, loc = 'lower right')
#plt.savefig('N12.png', bbox_inches = 'tight')
plt.show()






"""
# Abrir el archivo en modo escritura
with open("N12L1Ei0.txt", "w") as archivo:
    # Escribir cada elemento de la lista en una línea separada
    for elemento in media(c_momento_momento, Ei):
        archivo.write(str(elemento) + "\n")
"""
"""
#dispersión
sigma = []
for i in range(0,pt,1):
    suma = 0
    for j in range (0,numero,1):
        suma += (media[i]-soluciones[j][i])**2
    sigma.append(np.sqrt(suma/numero))


plt.plot(t,sigma)
plt.title('Dispersión')
plt.ylabel('$\sigma(t)$')
plt.xlabel('t')
plt.show()

dispersion_media = np.sum(sigma[500:])/(500)
print(dispersion_media)
"""









