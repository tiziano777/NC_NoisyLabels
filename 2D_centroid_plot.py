import numpy as np
import matplotlib.pyplot as plt

# Parametri
n = 6  # Distanza dei centroidi da zero lungo l'asse x
varianza = 7
num_punti = 1500

# Posizionamento dei due centroidi c1 e c2 in 2D
c1 = np.array([-n, 0])  # Spostato a -n sull'asse x
c2 = np.array([n, 0])   # Spostato a +n sull'asse x

# Generazione dei punti attorno ai centroidi
punti_c1 = np.random.multivariate_normal(c1, np.eye(2) * varianza, num_punti)
punti_c2 = np.random.multivariate_normal(c2, np.eye(2) * varianza, num_punti)

# Retta di separazione fissa su x = 0
x_sep = np.full(100, 0)  # Linea verticale a x = 0
y_sep = np.linspace(-8, 8, 100)  # Intervallo di y per tracciare la retta

# Visualizzazione
plt.figure(figsize=(8, 8))

# Plottiamo i punti per c1 (rossi) e c2 (verdi)
plt.scatter(punti_c1[:, 0], punti_c1[:, 1], color='red', alpha=0.6, label='sample mapping of c1')
plt.scatter(punti_c2[:, 0], punti_c2[:, 1], color='green', alpha=0.6, label='sample mapping of c2')

# Plottiamo i centroidi (nero)
plt.scatter(c1[0], c1[1], color='black', s=200, label='c1')
plt.scatter(c2[0], c2[1], color='black', s=200, label='c2')

# Plottiamo la retta di separazione in blu
plt.plot(x_sep, y_sep, color='blue', linestyle='-', linewidth=2, label='label collapse boundary')

# Impostiamo i limiti degli assi
plt.xlim(-n-5, n+5)  # Spazio orizzontale che copre i centroidi
plt.ylim(-8, 8)      # Spazio verticale fisso

# Impostazioni aggiuntive del grafico
plt.axhline(0, color='grey', linewidth=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Point distribution around label centroid')
plt.legend()

# Mostriamo il grafico
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')

plt.savefig('centroid.png')

plt.show()
exit(0)
