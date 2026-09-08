import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load the dataset
df = pd.read_csv("lncRNA_5_Cancers.csv")

print("Dataset shape:", df.shape)
print(df.head())

print("\nCancer classes:")
print(df["Class"].value_counts())

# Separate the data
# First column = sample ID
sample_ids = df["Ensembl_ID"]

# Middle 12,309 columns = lncRNA expression features
X = df.iloc[:, 1:-1]

# Last column = cancer class labels
y = df["Class"]

print("\nFeature Matrix shape:", X.shape)
print("Label shape:", y.shape)

# Task 1: PCA

#Standardize the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# print("Standardized data shape:", X_scaled.shape)

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print("PCA shape:", X_pca.shape)

# Create a DataFrame for PCA results
pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Class": y.values
})

# Plot five cancer types
plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Class",
    s=50,
    alpha=0.7
)

plt.title("PCA of lncRNA Expression Data")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")

# Set range for x and y axes
plt.xlim(-30, 30)
plt.ylim(-30, 30)

plt.legend(title="Cancer Type")

plt.tight_layout()
plt.show()

# Task 2: Violin Plots for PCA Components
# PC1: 
plt.figure(figsize=(10, 6))

sns.violinplot(
    data=pca_df,
    x="Class",
    y="PC1",
    inner="box"
)

plt.title("Distribution of PC1 by Cancer Type")
plt.xlabel("Cancer Type")
plt.ylabel("PC1 Value")

plt.tight_layout()
plt.show()

# PC2:
plt.figure(figsize=(10, 6))
sns.violinplot(
    data=pca_df,
    x="Class",
    y="PC2",
    inner="box"
)
plt.title("Distribution of PC2 by Cancer Type")
plt.xlabel("Cancer Type")
plt.ylabel("PC2 Value")
plt.tight_layout()
plt.show()

# Task 3: t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    init="pca",
    learning_rate="auto"
)

X_tsne = tsne.fit_transform(X)
print("t-SNE shape:", X_tsne.shape)

tsne_df = pd.DataFrame({
    "t-SNE 1": X_tsne[:, 0],
    "t-SNE 2": X_tsne[:, 1],
    "Class": y.values
})

print(tsne_df.head())

# Plot two t-SNE components
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=tsne_df,
    x="t-SNE 1",
    y="t-SNE 2",
    hue="Class",
    s=50,
    alpha=0.7
)

plt.title("t-SNE of lncRNA Expression Data")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")

plt.legend(title="Cancer Type")

plt.tight_layout()
plt.show()

# Task 4: Violin Plots for t-SNE Components
# t-SNE 1:
plt.figure(figsize=(10, 6))
sns.violinplot(
    data=tsne_df,
    x="Class",
    y="t-SNE 1",
    inner="box"
)

plt.title("Distribution of t-SNE 1 by Cancer Type")
plt.xlabel("Cancer Type")
plt.ylabel("t-SNE 1")

plt.tight_layout()
plt.show()

# t-SNE 2:
plt.figure(figsize=(10, 6))

sns.violinplot(
    data=tsne_df,
    x="Class",
    y="t-SNE 2",
    inner="box"
)

plt.title("Distribution of t-SNE 2 by Cancer Type")
plt.xlabel("Cancer Type")
plt.ylabel("t-SNE 2")

plt.tight_layout()
plt.show()