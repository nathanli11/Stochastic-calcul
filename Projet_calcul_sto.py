import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

#Parametres
T = 3  # Temps final
n = 1000  # Nombre de pas de temps
N = 10  # Nombre de trajectoires
pas = T / n

a = 0.1
r0 = 0.015
b = 1.5 * r0
gamma = 0.015
alpha = 0.1


#Fonctions f et g
def f(x):
    return (x/2*T)

def g(x):
    return 1/(1+x)

#Fonction sigma
def sigma(t, x):
    return alpha * (1 + f(t) + g(x))

#Initialisation des matrices pour r, S0 et S_R
r = r0 * np.ones((n+1, N))
S0 = np.ones((n+1, N))
S_R = np.ones((n+1, N))
S = np.ones((n+1, N))

#Simulation
dates = np.linspace(0, T, n+1)
for j in range(N):
    for i in range(1, n+1):
        #Calcul de r, S0 et S_R
        delta_W = np.sqrt(pas) * npr.randn()
        r[i, j] = r[i-1, j] + a * (b - r[i-1, j]) * pas + gamma * delta_W
        S0[i, j] = S0[i-1, j] + r[i-1, j] * S0[i-1, j] * pas
        delta_B = np.sqrt(pas) * npr.randn()
        S_R[i, j] = S_R[i-1, j] + sigma(dates[i-1], S_R[i-1, j]) * S_R[i-1, j] * delta_B
        S[i,j] = S_R[i, j] * S0[i, j]


#1 Graphe des résultats pour r
plt.plot(dates, r)  
plt.title('Simulation de r avec Schéma d\'Euler')
plt.xlabel('Temps')
plt.ylabel('r')
plt.show()

#2 Graphe des résultats pour S0
plt.plot(dates, S0)  
plt.title('Simulation de S0 avec Schéma d\'Euler')
plt.xlabel('Temps')
plt.ylabel('S0')
plt.show()

#3 Graphe des résultats pour S_R
plt.plot(dates, S_R)  
plt.title(r"Simulation de $\tilde{S}$ avec Schéma d'Euler")
plt.xlabel('Temps')
plt.ylabel(r'$\tilde{S}$')
plt.show()

#4 Graphe des résultats pour S
plt.plot(dates, S) 
plt.title('Simulation de S avec Schéma d\'Euler')
plt.xlabel('Temps')
plt.ylabel('S')
plt.show()

#Question 6
#Fonction payoff pour option put
def payoff(K, S):
    return max(K - S, 0)

K1 = S[0,0]
gT1 = np.array([payoff(K1, S[-1, i]) / np.exp(r0*pas*n) for i in range(N)])
print("Payoff de la fonction gT1:", gT1)

K2 = S[0,0]*2
#Calcul de la moyenne intégrale de S sur T
mean_S = np.mean(S, axis=0) #np.mean(S, axis=0) * T * (1/T)
gT2 = np.array([payoff(K2, mean_S[i]) / np.exp(r0*pas*n) for i in range(N)])
print("Payoff de la fonction gT2 :", gT2)

