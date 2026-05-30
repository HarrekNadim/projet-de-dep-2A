import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import pandas as pd

from Main import nb_simul, N, M
scenarios_politiques = {
    # --- Scénarios Classiques ---

    # Loi Uniforme
    "hasard": (1, 1),

    # Courbe en cloche
    "centre": (5, 5),

    # Beaucoup de "0" et de "1", personne au milieu
    "polarisation_extreme": (0.5, 0.5),

    # --- Scénarios Victoire/Défaite ---

    # Majorité écrasante
    "consensus": (9, 1),
}

# --- Modélisation ---

def gen_system(nbVotants, nbCandidats, repartitionVotants, repartitionCandidats, invPref):
    if invPref:
        Va = scenarios_politiques[repartitionVotants][1]
        Vb = scenarios_politiques[repartitionVotants][0]
    else:
        Va = scenarios_politiques[repartitionVotants][0]
        Vb = scenarios_politiques[repartitionVotants][1]
    Ca = scenarios_politiques[repartitionCandidats][0]
    Cb = scenarios_politiques[repartitionCandidats][1]
    v = generer_votes_loi_beta(nbVotants, Va, Vb)
    c = generer_votes_loi_beta(nbCandidats, Ca, Cb)
    return v, c

def generer_votes_loi_beta(n, a, b):
    votes = beta.rvs(a, b, size=n)
    return votes

def tableau_de_pref(system):
    tabV = system[0]
    tabC = system[1]
    n_votants = tabV.shape[0]
    n_candidats = tabC.shape[0]
    tabPref = np.empty((n_votants + n_candidats, n_candidats), dtype=float)
    tabPref[:n_votants, :] = np.abs(tabV[:, None] - tabC[None, :])
    tabPref[n_votants:, :] = np.abs(tabC[:, None] - tabC[None, :])
    return tabPref

def tab_de_pref_comp(tabPref):
    sorter = np.argsort(tabPref, axis=1)
    ranks = np.empty_like(sorter, dtype=int)
    ranks[np.arange(tabPref.shape[0])[:, None], sorter] = np.arange(1, tabPref.shape[1] + 1)
    return ranks


# --- Les Systèmes de Vote ---

def SMT1(tabPref):
    choix = np.argmin(tabPref, axis=1)
    nbVotes = np.bincount(choix, minlength=tabPref.shape[1])
    meilleurs = np.flatnonzero(nbVotes == nbVotes.max()).tolist()
    if len(meilleurs) > 1:
        print("Il y a égalité pour SMT1")
    return meilleurs

def SMT2(tabPref):
    C = SMT1(tabPref)
    if len(C) > 1:
        C1, C2 = C[0], C[1]
    else:
        C1 = C[0]
        tabPref2 = np.delete(tabPref, C1, axis=1)
        C2 = SMT1(tabPref2)[0]
        if C2 >= C1:
            C2 += 1
    diff = tabPref[:, C1] - tabPref[:, C2]
    S1 = np.sum(diff < 0) + 0.5 * np.sum(diff == 0)
    S2 = np.sum(diff > 0) + 0.5 * np.sum(diff == 0)
    if S1 > S2:
        return C1
    elif S2 > S1:
        return C2
    print("égalité pour SMT2")
    return C1, C2

def Borda(tabPref):
    ranks = tab_de_pref_comp(tabPref)
    scores = np.sum(tabPref.shape[1] - ranks, axis=0)
    meilleurs = np.flatnonzero(scores == scores.max()).tolist()
    return meilleurs

def Copeland(tabPref):
    m = tabPref.shape[1]
    R = np.zeros(m)
    for j in range(m - 1):
        for k in range(j + 1, m):
            col_j = tabPref[:, j]
            col_k = tabPref[:, k]
            Sj = np.sum(col_j < col_k)
            Sk = np.sum(col_j > col_k)
            ties = np.sum(col_j == col_k)
            Sj += 0.5 * ties
            Sk += 0.5 * ties
            if Sj < Sk:
                R[k] += 1
            elif Sj > Sk:
                R[j] += 1
            else:
                R[j] += 0.5
                R[k] += 0.5
    return np.flatnonzero(R == R.max()).tolist()


# --- Un unique vainqueur ---
def extraire_vainqueur(resultat):
    if isinstance(resultat, (list, tuple, np.ndarray)):
        return resultat[0]
    return resultat


# --- Calcul de la satisfaction relative ---
def satisfaction_relative(tabPref, vainqueur):
    """
    Calcule le taux de satisfaction relative pour un vainqueur donné.

    Pour chaque agent i :
      - si son candidat favori est le vainqueur  : contribution = 1
      - sinon : contribution = dist(i, favori) / dist(i, vainqueur)

    S = (1/n) * sum_i contrib_i

    tabPref[i, j] = |position_i - position_candidat_j|
    """
    n = tabPref.shape[0]
    dist_favori   = tabPref[np.arange(n), np.argmin(tabPref, axis=1)]  # min sur candidats
    dist_vainqueur = tabPref[:, vainqueur]

    # Votants dont le favori EST le vainqueur : contribution = 1
    # Votants dont le favori N'EST PAS le vainqueur : contribution = dist_favori / dist_vainqueur
    egal = (dist_vainqueur == 0)   # correspond à dist_favori == 0 aussi (vainqueur = favori)
    contrib = np.where(egal, 1.0, dist_favori / dist_vainqueur)
    return contrib.mean()


# --- Génère les stats ---
def simuler_elections(nb_simulations, N, M, repV, repC, invPref=False):
    accords = {"SMT1_vs_Copeland": 0, "SMT2_vs_Copeland": 0, "Borda_vs_Copeland": 0}

    satisfaction  = {"Sat_SMT1": 0,  "Sat_SMT2": 0,  "Sat_Borda": 0,  "Sat_Copeland": 0}
    satisfaction_rel = {"SatR_SMT1": 0, "SatR_SMT2": 0, "SatR_Borda": 0, "SatR_Copeland": 0}

    accords_counts = np.zeros(3, dtype=int)

    # Lignes 0-3 : insatisfaction absolue ; lignes 4-7 : satisfaction relative
    StockSatSimu  = np.empty((4, nb_simulations), dtype=float)  # insatisfaction
    StockSatRSimu = np.empty((4, nb_simulations), dtype=float)  # satisfaction relative

    D = np.empty(nb_simulations, dtype=float)

    for k in range(nb_simulations):
        system   = gen_system(N, M, repV, repC, invPref)
        tabPref  = tableau_de_pref(system)

        v_smt1     = extraire_vainqueur(SMT1(tabPref))
        v_smt2     = extraire_vainqueur(SMT2(tabPref))
        v_borda    = extraire_vainqueur(Borda(tabPref))
        v_copeland = extraire_vainqueur(Copeland(tabPref))

        accords_counts[0] += int(v_smt1    == v_copeland)
        accords_counts[1] += int(v_smt2    == v_copeland)
        accords_counts[2] += int(v_borda   == v_copeland)

        # --- Insatisfaction absolue ---
        n_agents  = tabPref.shape[0]
        best_pref = np.argmin(tabPref, axis=1)
        cPref_v   = tabPref[np.arange(n_agents), best_pref]

        I_smt1     = np.sum(np.abs(cPref_v - tabPref[:, v_smt1]))     / n_agents
        I_smt2     = np.sum(np.abs(cPref_v - tabPref[:, v_smt2]))     / n_agents
        I_borda    = np.sum(np.abs(cPref_v - tabPref[:, v_borda]))    / n_agents
        I_copeland = np.sum(np.abs(cPref_v - tabPref[:, v_copeland])) / n_agents

        StockSatSimu[0, k] = I_smt1     * n_agents
        StockSatSimu[1, k] = I_smt2     * n_agents
        StockSatSimu[2, k] = I_borda    * n_agents
        StockSatSimu[3, k] = I_copeland * n_agents

        D[k] = I_smt1 - I_borda

        # --- Satisfaction relative ---
        StockSatRSimu[0, k] = satisfaction_relative(tabPref, v_smt1)
        StockSatRSimu[1, k] = satisfaction_relative(tabPref, v_smt2)
        StockSatRSimu[2, k] = satisfaction_relative(tabPref, v_borda)
        StockSatRSimu[3, k] = satisfaction_relative(tabPref, v_copeland)

    accords["SMT1_vs_Copeland"]  = accords_counts[0] / nb_simulations * 100
    accords["SMT2_vs_Copeland"]  = accords_counts[1] / nb_simulations * 100
    accords["Borda_vs_Copeland"] = accords_counts[2] / nb_simulations * 100

    sat_values = np.sum(StockSatSimu, axis=1) / tabPref.shape[0] / nb_simulations * 100
    satisfaction["Sat_SMT1"]     = sat_values[0]
    satisfaction["Sat_SMT2"]     = sat_values[1]
    satisfaction["Sat_Borda"]    = sat_values[2]
    satisfaction["Sat_Copeland"] = sat_values[3]

    satr_values = StockSatRSimu.mean(axis=1)
    satisfaction_rel["SatR_SMT1"]     = satr_values[0]
    satisfaction_rel["SatR_SMT2"]     = satr_values[1]
    satisfaction_rel["SatR_Borda"]    = satr_values[2]
    satisfaction_rel["SatR_Copeland"] = satr_values[3]

    StockSatSimuNorm  = StockSatSimu  / tabPref.shape[0]
    # StockSatRSimu est déjà normalisé dans [0, 1]

    return accords, satisfaction, satisfaction_rel, StockSatSimuNorm, StockSatRSimu, D


# --- Lancement des simulations ---
print(f"Lancement de {nb_simul} simulations par scénario avec {M} candidats...\n")

resultats_Condorcet   = {}
resultats_Satisfaction = {}
resultats_SatisfactionRel = {}
resultats_StockSat    = {}
resultats_StockSatR   = {}
resultats_D           = {}

for scenario in scenarios_politiques.keys():
    res = simuler_elections(nb_simul, N, M, repV=scenario, repC=scenario)
    resultats_Condorcet[scenario]      = res[0]
    resultats_Satisfaction[scenario]   = res[1]
    resultats_SatisfactionRel[scenario] = res[2]
    resultats_StockSat[scenario]       = res[3]
    resultats_StockSatR[scenario]      = res[4]
    resultats_D[scenario]              = res[5]


# --- Test statistique ---
def test_borda_vs_smt1(D, alpha=0.05):
    from scipy.stats import norm
    n          = len(D)
    m_hat      = np.mean(D)
    sigma_hat  = np.std(D, ddof=1)
    T_n        = np.sqrt(n) * m_hat / sigma_hat
    p_valeur   = 1 - norm.cdf(T_n)
    z_critique = norm.ppf(1 - alpha)
    return {
        "n": n, "m_hat": m_hat, "sigma_hat": sigma_hat,
        "T_n": T_n, "p_valeur": p_valeur,
        "z_critique": z_critique, "alpha": alpha,
        "rejet_H0": T_n > z_critique,
    }

scenario_test = "polarisation_extreme"
res_test = test_borda_vs_smt1(resultats_D[scenario_test], alpha=0.05)
print(f"\n{'='*60}")
print(f"TEST STATISTIQUE — scénario : {scenario_test}")
print(f"{'='*60}")
print(f"  T_n = {res_test['T_n']:.4f}   z_critique = {res_test['z_critique']:.4f}   p = {res_test['p_valeur']:.6f}")
if res_test["rejet_H0"]:
    print("  => On REJETTE H0 : Borda est significativement meilleur que SMT1.")
else:
    print("  => On NE REJETTE PAS H0.")
print(f"{'='*60}\n")


# --- Fonctions de tracé ---

def _base_plot(resultats_stock, nb_simul, ylabel, title, filename):
    """Tracé générique (courbes + bandeau ±1σ) pour insatisfaction ou satisfaction."""
    labels = ["SMT1", "SMT2", "Borda", "Copeland"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    scenarios    = list(resultats_stock.keys())
    n_scenarios  = len(scenarios)
    cols = 2
    rows = (n_scenarios + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axes = axes.flatten()
    x = np.arange(1, nb_simul + 1)

    for idx, scenario in enumerate(scenarios):
        ax    = axes[idx]
        stock = resultats_stock[scenario]
        for m_idx in range(4):
            y  = stock[m_idx]
            mu = np.mean(y)
            sigma = np.std(y)
            ax.plot(x, y, label=labels[m_idx], color=colors[m_idx], linewidth=1.2, alpha=0.85)
            ax.axhline(mu, color=colors[m_idx], linestyle="--", linewidth=0.9, alpha=0.6)
            ax.fill_between(x, mu - sigma, mu + sigma, color=colors[m_idx], alpha=0.08)
        ax.set_title(f"Scénario : {scenario}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Simulation")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    for idx in range(n_scenarios, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=150)
    plt.show()


def _base_boxplot(resultats_stock, ylabel, title, filename):
    """Boxplot générique pour insatisfaction ou satisfaction."""
    labels = ["SMT1", "SMT2", "Borda", "Copeland"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    scenarios   = list(resultats_stock.keys())
    n_scenarios = len(scenarios)
    cols = 2
    rows = (n_scenarios + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axes = axes.flatten()

    for idx, scenario in enumerate(scenarios):
        ax    = axes[idx]
        stock = resultats_stock[scenario]
        bp = ax.boxplot(
            [stock[m] for m in range(4)],
            labels=labels,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2)
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(f"Scénario : {scenario}", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    for idx in range(n_scenarios, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=150)
    plt.show()


# Insatisfaction absolue
_base_plot(
    resultats_StockSat, nb_simul,
    ylabel="Insatisfaction normalisée",
    title="Insatisfaction par simulation — comparaison des modes de scrutin",
    filename="satisfaction_par_scenario.png"
)
_base_boxplot(
    resultats_StockSat,
    ylabel="Insatisfaction normalisée",
    title="Distribution de l'insatisfaction — boîtes à moustaches",
    filename="boxplot_par_scenario.png"
)

# Satisfaction relative
_base_plot(
    resultats_StockSatR, nb_simul,
    ylabel="Satisfaction relative (∈ [0, 1])",
    title="Satisfaction relative par simulation — comparaison des modes de scrutin",
    filename="satisfaction_relative_par_scenario.png"
)
_base_boxplot(
    resultats_StockSatR,
    ylabel="Satisfaction relative (∈ [0, 1])",
    title="Distribution de la satisfaction relative — boîtes à moustaches",
    filename="boxplot_satisfaction_relative_par_scenario.png"
)