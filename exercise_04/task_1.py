import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import make_circles,make_swiss_roll
from sklearn.manifold import MDS,Isomap,SpectralEmbedding

#generate circles
X, y = make_circles(n_samples=500, factor=.3, noise=.1)
rgb=np.array(['r','g','b'])
plt.figure()
#draw the original circle
plt.scatter(X[:,0],X[:,1],color=rgb[y])
plt.title("original circle")
plt.show()
#apply pca
pca = PCA()
X_pca = pca.fit_transform(X)
plt.figure()

plt.scatter(X_pca[:,0],X_pca[:,1],color=rgb[y])
plt.title("circle after pca")
plt.show()

#apply MDS
mani = MDS()
X_mds = mani.fit_transform(X)
plt.figure()

plt.scatter(X_mds[:,0],X_mds[:,1],color=rgb[y])
plt.title("circle after MDS")
plt.show()

#apply IsoMap
iso = Isomap()
X_iso = iso.fit_transform(X)
plt.figure()

plt.scatter(X_iso[:,0],X_iso[:,1],color=rgb[y])
plt.title("circle after IsoMap")
plt.show()
#apply Laplacian Eigenmaps
le = SpectralEmbedding()
X_le = le.fit_transform(X)
plt.figure()

plt.scatter(X_le[:,0],X_le[:,1],color=rgb[y])
plt.title("circle after Laplacian Eigenmaps")
plt.show()

#generate swiss roll and draw the original roll
X, y = make_swiss_roll(n_samples=1000, noise=0.1)

ax = plt.figure().add_subplot( projection='3d')
ax.scatter(X[:,0],X[:,1],X[:,2],c=y,cmap=plt.cm.Spectral)
plt.axis('tight')
plt.xticks([]), plt.yticks([])
plt.title("original roll")
plt.show()
#apply pca
pca = PCA()
X_pca = pca.fit_transform(X)

ax = plt.figure().add_subplot( projection='3d')
ax.scatter(X[:,0],X[:,1],X[:,2],c=y,cmap=plt.cm.Spectral)
plt.axis('tight')
plt.xticks([]), plt.yticks([])
plt.title("roll after PCA")
plt.show()

#apply MDS
mani = MDS()
X_mds = mani.fit_transform(X)

ax = plt.figure().add_subplot( projection='3d')
ax.scatter(X[:,0],X[:,1],X[:,2],c=y,cmap=plt.cm.Spectral)
plt.axis('tight')
plt.xticks([]), plt.yticks([])
plt.title("roll after MDS")
plt.show()

#apply IsoMap
iso = Isomap()
X_iso = iso.fit_transform(X)

ax = plt.figure().add_subplot( projection='3d')
ax.scatter(X[:,0],X[:,1],X[:,2],c=y,cmap=plt.cm.Spectral)
plt.axis('tight')
plt.xticks([]), plt.yticks([])
plt.title("roll after IsoMap")
plt.show()

#apply Laplacian Eigenmaps
le = SpectralEmbedding()
X_le = le.fit_transform(X)

ax = plt.figure().add_subplot( projection='3d')
ax.scatter(X[:,0],X[:,1],X[:,2],c=y,cmap=plt.cm.Spectral)
plt.axis('tight')
plt.xticks([]), plt.yticks([])
plt.title("roll after Laplacian Eigenmaps")
plt.show()
