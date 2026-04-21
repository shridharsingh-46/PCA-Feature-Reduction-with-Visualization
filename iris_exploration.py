import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

data = load_iris()

X = data.data
y = data.target
feature_names = data.feature_names

df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print(df.head().to_markdown())

import seaborn as sns
import matplotlib.pyplot as plt

# Generate pairplot
sns.pairplot(df, hue='target', palette='viridis')
plt.title('Iris Dataset Pairplot')
print("Pairplot saved to iris_pairplot.png")

from sklearn.preprocessing import StandardScaler

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n### Feature Scaling Verification ###")
print(f"Mean of scaled features: {X_scaled.mean(axis=0)}")
print(f"Std of scaled features: {X_scaled.std(axis=0)}")

# Store scaled data for PCA next
np.save('X_scaled.npy', X_scaled)

from sklearn.decomposition import PCA

# Applying PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\n### PCA Results ###")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.2%}")

# Store PCA data for visualization next
np.save('X_pca.npy', X_pca)

import matplotlib.pyplot as plt

# 2D PCA Visualization
plt.figure(figsize=(10, 6))

for i, target_name in enumerate(data.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=target_name, s=50, alpha=0.7)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Iris Dataset 2D Visualization")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('pca_visualization.png')
print("PCA visualization saved to pca_visualization.png")

from mpl_toolkits.mplot3d import Axes3D

# Applying 3D PCA
pca3 = PCA(n_components=3)
X_pca3 = pca3.fit_transform(X_scaled)

print(f"\n### 3D PCA Results ###")
print(f"Explained variance ratio (3 components): {pca3.explained_variance_ratio_}")
print(f"Total variance retained (3 components): {sum(pca3.explained_variance_ratio_):.2%}")

# 3D PCA Visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for i, target_name in enumerate(data.target_names):
    ax.scatter(X_pca3[y == i, 0], X_pca3[y == i, 1], X_pca3[y == i, 2], label=target_name, s=50, alpha=0.7)

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
ax.set_title("3D PCA - Iris Dataset Visualization")
ax.legend()
plt.savefig('pca_3d_visualization.png')
print("3D PCA visualization saved to pca_3d_visualization.png")
