from mnist_loader import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS,Isomap,SpectralEmbedding
#load MNISt data set
X,labels = mnist.load_mnist(test_set=True,selection=(2,4,6))

#turn the dimensionality to 2D using t-SNE
tsne = TSNE(n_components=2, random_state=50)
X_reduced = tsne.fit_transform(X)

plt.figure(figsize=(15,12))
plt.scatter(X_reduced[:, 0],X_reduced[:, 1], c=labels, cmap='jet' )
plt.axis('off')
plt.title("original data")
plt.colorbar()
plt.show()

#apply PCA
pca = PCA()
X_pca = pca.fit_transform(X)
tsne = TSNE(n_components=2, random_state=50)
X_reduced = tsne.fit_transform(X_pca)
plt.figure(figsize=(15,12))
plt.scatter(X_reduced[:, 0],X_reduced[:, 1], c=labels, cmap='jet' )
plt.axis('off')
plt.title("PCA applied data")
plt.colorbar()
plt.show()

#apply MDS
mds = MDS()
X_mds = mds.fit_transform(X)
tsne = TSNE(n_components=2, random_state=50)
X_reduced = tsne.fit_transform(X_mds)
plt.figure(figsize=(15,12))
plt.scatter(X_reduced[:, 0],X_reduced[:, 1], c=labels, cmap='jet' )
plt.axis('off')
plt.title("MDS applied data")
plt.colorbar()
plt.show()

#apply IsoMAP
iso = Isomap()
X_iso = iso.fit_transform(X)
tsne = TSNE(n_components=2, random_state=50)
X_reduced = tsne.fit_transform(X_iso)
plt.figure(figsize=(15,12))
plt.scatter(X_reduced[:, 0],X_reduced[:, 1], c=labels, cmap='jet' )
plt.axis('off')
plt.title("ISOMAP applied data")
plt.colorbar()
plt.show()

#apply LE
le = SpectralEmbedding()
X_le = le.fit_transform(X)
tsne = TSNE(n_components=2, random_state=50)
X_reduced = tsne.fit_transform(X_le)
plt.figure(figsize=(15,12))
plt.scatter(X_reduced[:, 0],X_reduced[:, 1], c=labels, cmap='jet' )
plt.axis('off')
plt.title("LE applied data")
plt.colorbar()
plt.show()
