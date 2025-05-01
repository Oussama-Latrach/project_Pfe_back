import numpy as np

# Charger le fichier .npy
data = np.load('inference_zone.npy')  # Remplace par le chemin de ton fichier

# Vérification du nombre de colonnes
if data.shape[1] < 6:
    raise ValueError("Le fichier ne contient pas au moins 6 colonnes.")

# Garder uniquement les 5 premières colonnes
data_reduit = data[:, :5]

# Sauvegarder dans un nouveau fichier
np.save('inference_zone.npy', data_reduit)

print("Fichier sauvegardé avec les 5 premières colonnes.")
