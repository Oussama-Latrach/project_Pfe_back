"""
Data preprocessing pipeline for point cloud data1.
Converts LAZ files to normalized numpy arrays with class remapping.
"""

import laspy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

def remap_classification(value, classes_cibles):
    """Remap non-target classes to class 1 (Unclassified)."""
    return value if value in classes_cibles else 1

# Configuration
input_file = "inference_zone.laz"
output_file = "inference_zone.npy"
target_classes = [1, 2, 3, 4, 5, 6]  # Classes to keep

print(" Starting data1 preprocessing...")

# Load LAZ file
print(f" Loading LAZ file: {input_file}")
las = laspy.read(input_file)

# Extract attributes
print(" Extracting point cloud attributes...")
x = las.x
y = las.y
z = las.z
rn = las.return_number
nr = las.number_of_returns
cls = las.classification

# Remap classes
print(" Remapping classes...")
vectorized_remap = np.vectorize(lambda x: remap_classification(x, target_classes))
cls_remapped = vectorized_remap(cls)

# Normalize coordinates
print(" Normalizing coordinates...")
scaler = MinMaxScaler()
xyz = np.column_stack((x, y, z))
xyz_normalized = scaler.fit_transform(xyz)

# Combine features
print(" Combining features...")
data = np.column_stack((xyz_normalized, rn, nr, cls_remapped))

# Save processed data1
print(f" Saving processed data1 to {output_file}")
np.save(output_file, data)

# Display sample data1
print("\n Sample of processed data1 (first 5 points):")
print(data[:5])

# Display class distribution
class_counts = Counter(cls_remapped)
print("\n Class distribution after remapping:")
for cls, count in sorted(class_counts.items()):
    print(f"  - Class {cls}: {count:,} points")

print("\n Preprocessing completed successfully!")