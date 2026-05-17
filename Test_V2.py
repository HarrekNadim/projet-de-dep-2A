from Program_to_simulate_votes_V7 import *

# # candidats = distribution_uniforme(3)
# # votants = distribution_uniforme(30)# # dist = distances(candidats, votants)
# # tdp = tableau_des_preferences(candidats, votants)

# # print(SM1T(tdp))
# # print(score_de_satisfaction(gagnant(SM1T(tdp)), dist))
# # print(SM2T(tdp))
# # print(score_de_satisfaction(gagnant(SM2T(tdp)), dist))
# # candidat_optimal(dist)
# # simulation_manipulabilite(SM2T, 3, 300,
# #                           strategie_manipulation=vote_utile_SM2T,
# #                          nb_simulations=100)
# # simulation_manipulabilite(Borda_classique, 3, 300,
# #                          generateur_votants=distribution_uniforme, genenerateur_candidats=distribution_uniforme,
# #                          nb_simulations=100)

# # résultats = comparer_manipulabilite(
# #     scrutins     = [SM1T, SM2T, Borda_classique, Copeland],
# #     strategie_manipulation=optimisation_vote_groupe,
# #     nb_candidats = 3,
# #     nb_votants   = 300,
# #     nb_simulations = 100
# # )

# # résultats = comparer_manipulabilite(
# #     scrutins     = [SM1T, SM2T, Borda_classique, Copeland],
# #     strategie_manipulation=vote_utile,
# #     nb_candidats = 3,
# #     nb_votants   = 300,
# #     nb_simulations = 10000
# # )



# # Comparer deux scrutins côte à côte
# # Z1 = satisfaction_nb_candidats_votants(SM2T, 6, 300)
# # Z2 = satisfaction_nb_candidats_votants(Copeland, 6, 300)
# # Z3 = satisfaction_nb_candidats_votants(Borda_classique, 6, 300)

# # optimiser_borda(
# #     nb_candidats=3, nb_votants=300,
# #     metriques=[satisfaction_relative, satisfaction_réel],
# #     strategies_manipulation=[vote_utile, manipulation_naïve],
# #     nb_simulations=300,        # paysages pour l'optimisation
# #     nb_simulations_manip=200,  # simulations pour la manipulabilité
# # )

résultats = consistance_satisfaction(
    scrutins                = [SM1T, SM2T, Borda_classique, Copeland],
    nb_candidats            = 10,
    nb_votants              = 300,
    generateur_candidats    = lambda n: distribution_trimodale(n, dim=1),
    generateur_votants      = lambda n: distribution_uniforme(n, dim=1),
    metriques               = [satisfaction_relative],
    strategies_manipulation = [manipulation_naïve, vote_utile],
    nb_simulations          = 100
)

# test_dimension(
#     scrutins                = [SM1T, SM2T, Borda_classique, Copeland],
#     nb_candidats            = 3,
#     nb_votants              = 300,
#     generateur_candidats    = lambda n: distribution_trimodale(n, dim=5),
#     generateur_votants      = lambda n: distribution_uniforme(n, dim=5),
#     metriques               = [satisfaction_relative],
#     nb_simulations=1000)


# print(distribution_gaussienne(5, centre=0.5, variance=0.1, dim=2))