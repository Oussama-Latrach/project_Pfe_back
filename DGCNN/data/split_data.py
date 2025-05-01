"""
Data splitting utility.
Divides dataset into train/val/test sets with 70%/15%/15% split.
"""

import numpy as np

# Configuration
input_file = "data.npy"
output_files = {
    'train': "train.npy",
    'val': "val.npy",
    'test': "test.npy"
}

print(" Starting data1 splitting...")

# Load data1
print(f" Loading data1 from {input_file}")
data = np.load(input_file)
print(f" Total points: {data.shape[0]:,}")

# Shuffle data1
print(" Shuffling data1...")
np.random.seed(42)  # For reproducibility
np.random.shuffle(data)

# Calculate split sizes
n_total = data.shape[0]
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val

print("\n Split sizes:")
print(f"  - Train: {n_train:,} points ({n_train/n_total:.0%})")
print(f"  - Val:   {n_val:,} points ({n_val/n_total:.0%})")
print(f"  - Test:  {n_test:,} points ({n_test/n_total:.0%})")

# Split data1
print("\n Splitting data1...")
train_data = data[:n_train]
val_data = data[n_train:n_train + n_val]
test_data = data[n_train + n_val:]

# Save splits
for name, path in output_files.items():
    print(f" Saving {name} set to {path}")
    np.save(path, locals()[f"{name}_data"])

print("\n Data splitting completed successfully!")