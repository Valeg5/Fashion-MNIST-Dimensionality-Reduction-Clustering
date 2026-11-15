# Dimensionality Reduction & Unsupervised Clustering on Fashion-MNIST

## Authors
* **Valentinos Goudas**
* * **Olga Saridou**
* **Eleni Georgoudi**
* **Thrasyvoulos Vasdavanos**

Department of Applied Informatics, University of Macedonia (Academic Year 2025–2026)

---

An empirical machine learning benchmark investigating the interplay between **Dimensionality Reduction techniques** and **Unsupervised Clustering algorithms** on high-dimensional image data. Developed as part of the *Machine Learning* curriculum at the **University of Macedonia**.

The project evaluates how linear, deep neural, and non-linear manifold projection spaces impact cluster formation, separation, and ground-truth label alignment across the 70,000-image Fashion-MNIST benchmark.

---

## Dataset & Preprocessing

* **Dataset:** Fashion-MNIST (70,000 grayscale images across 10 fashion categories).
* **Dimensionality:** $28 \times 28$ pixels ($784$ features per instance).
* **Data Splits:** 48,000 training, 12,000 validation, and 10,000 test images.
* **Normalization:** Min-Max pixel scaling $[0, 1]$ ($\text{pixel} / 255$) to ensure uniform convergence and prevent numerical instability.

---

## Dimensionality Reduction Architectures

Three contrasting projection paradigms were implemented and benchmarked:

1. **Principal Component Analysis (PCA):** Linear orthogonal transformation projecting $784\text{D} \rightarrow 2\text{D}$, maximizing explained variance.
2. **Deep Stacked Autoencoders (SAE):** Symmetric neural network reducing dimensionality through hierarchical non-linear compression ($784 \rightarrow 128 \rightarrow 64 \rightarrow 32$ latent bottleneck $\rightarrow 64 \rightarrow 128 \rightarrow 784$), evaluated on visual reconstruction fidelity against original inputs.
3. **t-Distributed Stochastic Neighbor Embedding (t-SNE):** Probabilistic non-linear manifold projection into 2D space, preserving local neighborhood structures and forming distinct visual cluster islands.

---

## Evaluated Clustering Algorithms

The compressed feature spaces were piped into three clustering paradigms:

* **MiniBatch K-Means:** Scalable centroid-based partitioning optimized for large-scale image datasets.
* **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** Non-parametric density clustering capable of discovering arbitrary cluster geometries and filtering noise.
* **Agglomerative (Hierarchical) Clustering:** Bottom-up linkage strategy revealing multi-level taxonomic relationships.

---

## Evaluation Metrics

Cluster quality and alignment were rigorously validated using internal and external indices:
* **Silhouette Score:** Cohesion vs. separation balance $[-1, 1]$.
* **Calinski-Harabasz Index:** Variance ratio criterion (higher indicates dense, well-separated clusters).
* **Davies-Bouldin Index:** Similarity between each cluster and its closest counterpart (lower is better).
* **Adjusted Rand Index (ARI):** Concordance with true ground-truth class labels, adjusted for chance.

---

## Benchmark Results & Key Findings

### Comparative Performance Across Reduction Spaces

| Reduction Method | Clustering Algorithm | Dimensionality | Silhouette Score | Davies-Bouldin | Adjusted Rand Index (ARI) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Raw Data** | MiniBatch K-Means | 784D | 0.144 | 1.986 | 0.317 |
| **PCA** | MiniBatch K-Means | 2D | 0.151 | 2.002 | 0.351 |
| **PCA** | DBSCAN | 2D | 0.370 | 0.926 | 0.403 |
| **PCA** | Agglomerative | 2D | 0.146 | 1.854 | 0.364 |
| **t-SNE** | MiniBatch K-Means | 2D | **0.393** | **0.830** | **0.418** |
| **t-SNE** | Agglomerative | 2D | 0.386 | 0.813 | 0.411 |
| **Deep SAE** | MiniBatch K-Means | 32D | 0.144 | 1.950 | 0.300 |
| **Deep SAE** | Agglomerative | 32D | 0.154 | 1.780 | 0.326 |

### Core Insights
* **Manifold Learning Superiority (t-SNE):** Projecting high-dimensional pixel data via t-SNE drastically boosted clustering performance across all algorithms, yielding the highest agreement with true labels (**$\text{ARI} = 0.418$** with MiniBatch K-Means).
* **Linear vs. Non-Linear Trade-offs:** PCA provided near-instantaneous computation ($~2.6\text{s}$) with acceptable baseline separation, whereas t-SNE and Deep SAE required substantial training latency ($~300\text{s}$ to $550\text{s}$) in exchange for capturing non-linear geometric manifolds.
* **Autoencoder Latent Space:** The Stacked Autoencoder successfully compressed features down to a 32-dimensional bottleneck while retaining high image reconstruction fidelity, creating balanced feature representations well-suited for hierarchical clustering.

---

## Tech Stack

* **Language:** Python 3.x
* **Deep Learning Framework:** TensorFlow / Keras (Autoencoders)
* **Machine Learning & Analytics:** `scikit-learn`, `numpy`, `pandas`
* **Visualization:** `matplotlib`, `seaborn`
