import numpy as np
import time
import matplotlib.pyplot as plt

def distances(candidats, votants):

    c = np.array(candidats)
    v = np.array(votants)

    diff = v[:, None, :] - c[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))

print(distances([[0], [1]], [[0.5], [0.2],[0.8], [0.1], [0.9]]))

def tableau_des_preferences(candidats, votants):

    dist  = distances(candidats, votants)     # (nb_v, nb_c)
    ordre = np.argsort(dist, axis=1)

    nb_votants, nb_candidats = dist.shape
    tab  = np.empty((nb_votants, nb_candidats), dtype=np.int32)
    rows = np.arange(nb_votants)[:, None]
    tab[rows, ordre] = np.arange(1, nb_candidats + 1, dtype=np.int32)
    return tab

print(tableau_des_preferences([[0], [1]], [[0.5], [0.2],[0.8], [0.1], [0.9]]))