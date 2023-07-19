import numpy as np
from sklearn.decomposition import PCA

def fit_pca(trials):
    Z = np.vstack([trial.Z for trial in trials])
    pca = PCA(n_components=Z.shape[1])
    pca.fit(Z)
    return pca

def apply_pca(trials, pca):
    for trial in trials:
        trial.Z_pc = pca.transform(trial.Z)
    return trials
