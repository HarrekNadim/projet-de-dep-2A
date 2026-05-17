from functools import partial
from Program_to_simulate_votes_V6 import *

# a = distribution_gaussienne(3, 300000)
# print(a[1])
# b = []
# for i in range(125):
#     b.append([1,2,3])
# for i in range(125):
#     b.append([2,3,1])
# for i in range(50):
#     b.append([3,1,2])

# print("Sondage : ", b)
# print(Borda_classique(b))
# plot_résultats(Borda_classique(b), title="Résultats méthode de Borda")
# plot_résultats(SM1T(b), title="Résultats méthode du scrutin majoritaire à un tour")
# plot_résultats(Copeland(b), title="Résultats méthode de Copeland")

generateur_custom = partial(distribution_gaussienne,centre=0.1,variance=0.5)

score = test_de_manipulabilite(SM2T, 3, 300,
                               generateur_votants=generateur_custom, candidats_fixes=[0.2, 0.21, 0.8],
                               nb_simulations=10000)

# candiats = distribution_uniforme(3)
# votants = distribution_uniforme(30)

# print(Borda_classique(tableau_des_preferences(candiats, votants)))