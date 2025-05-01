import numpy as np
from collections import Counter

# Chargement du fichier .npy
data = np.load("inference_zone.npy")  # Remplace par le chemin vers ton fichier .npy


# 1. Nombre total de points
total_points = data.shape[0]
print(f"Nombre total de points : {total_points}")

# 2. Nombre de points par classe
# La dernière colonne contient les classes
classes = data[:, -1]
compte_classes = Counter(classes)
print("\nNombre de points par classe :")
for k, v in sorted(compte_classes.items()):
    print(f"  - Classe {k}: {v} points")

# 3. Vérification des attributs (formes des données)
print("\nVérification des attributs dans les données :")
# Afficher les 5 premières lignes pour vérifier les attributs
print(data[:5])

# 4. Affichage de la forme (shape) du fichier
shape = data.shape
print(f"\nShape du fichier .npy : {shape}")

# 5. Valeurs uniques des classes
unique_classes = np.unique(classes)
print("\nValeurs uniques des classes dans le fichier :")
print(unique_classes)

# 6. Vérification de l'ordre des colonnes
# L'ordre des colonnes devrait être : x, y, z, ReturnNumber, NumberOfReturns, classification
print("\nOrdre des attributs dans les colonnes :")
print("1. x - Coordonnée X")
print("2. y - Coordonnée Y")
print("3. z - Coordonnée Z")
print("4. ReturnNumber - Numéro de retour")
print("5. NumberOfReturns - Nombre de retours")
print("6. classification - Classe de chaque point")
