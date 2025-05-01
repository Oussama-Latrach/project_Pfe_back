import numpy as np

# Charger le fichier .npy
data = np.load('data.npy')

# Modifier les valeurs :
# Si la 6e colonne vaut 4 ou 5 → on met 3
data[(data[:, 5] == 4) | (data[:, 5] == 5), 5] = 3

# Si la 6e colonne vaut 6 → on met 4
data[data[:, 5] == 6, 5] = 4

# Sauvegarder le fichier modifié
np.save('data.npy', data)
