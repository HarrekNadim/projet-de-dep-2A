import numpy as np
import time
import matplotlib.pyplot as plt
import itertools
from functools import partial
import mplcursors

try:
    import cupy as cp
    cp.array([1])
    GPU_AVAILABLE = True
    print("✅ GPU détecté — CuPy activé")
except Exception as e:
    print(f"❌ Erreur CuPy : {type(e).__name__}: {e}")
    import numpy as cp
    GPU_AVAILABLE = False
    print("⚠️  CUDA incomplet — fallback sur NumPy CPU")


# ─────────────────────────────────────────────
# UTILITAIRES GPU / CPU
# ─────────────────────────────────────────────

def to_numpy(x):
    if GPU_AVAILABLE:
        return cp.asnumpy(x)
    return np.asarray(x)

def to_cp(x):
    if isinstance(x, cp.ndarray):
        return x
    return cp.array(x)

def _nom_scrutin(scrutin):
    """Retourne le nom d'un scrutin, même si c'est un functools.partial."""
    if isinstance(scrutin, partial):
        return scrutin.func.__name__
    return scrutin.__name__


# ─────────────────────────────────────────────
# GÉNÉRATION DES DONNÉES
#
# Convention : tous les tableaux d'agents ont la forme (nb, dim)
#   nb  = nombre d'agents (votants ou candidats)  — lignes
#   dim = nombre de dimensions de l'espace        — colonnes
#
# Exemples :
#   distribution_uniforme(100, dim=1)  → shape (100, 1)
#   distribution_uniforme(50,  dim=3)  → shape (50,  3)
#
# Pour fixer dim dans les fonctions de simulation, utiliser partial :
#   gen2D = partial(distribution_uniforme, dim=2)
#   simulation_manipulabilite(SM1T, 4, 100, generateur_votants=gen2D, ...)
# ─────────────────────────────────────────────

def distribution_uniforme(nb, dim=1):
    """
    Génère nb agents uniformément dans [0, 1]^dim.

    Retourne un tableau CuPy de forme (nb, dim).
    """
    return cp.array(np.random.random((nb, dim)))


def distribution_gaussienne(nb, centre=0, variance=1, dim=1):
    """
    Génère nb agents selon une loi normale N(centre, variance²) dans R^dim.
    Les dim coordonnées sont indépendantes et identiquement distribuées.

    Retourne un tableau CuPy de forme (nb, dim).
    """
    return cp.array(np.random.normal(centre, variance, (nb, dim)))


def distribution_3_pics(nb, pics=(0.2, 0.5, 0.8),
                        variances=(0.05, 0.05, 0.05),
                        poids=(1, 1, 1), dim=1):
    """
    Génère nb agents selon un mélange de 3 gaussiennes.
    En dim > 1, chaque dimension est tirée indépendamment avec les mêmes paramètres.

    Paramètres :
        nb        : nombre d'agents
        pics      : centres des 3 gaussiennes
        variances : variances de chaque gaussienne
        poids     : poids relatifs (n'ont pas besoin d'être normalisés)
        dim       : dimension de l'espace

    Retourne un tableau CuPy de forme (nb, dim).
    """
    poids_np    = np.array(poids, dtype=float) / np.sum(poids)
    tailles     = np.round(poids_np * nb).astype(int)
    tailles[-1] = nb - tailles[:-1].sum()

    colonnes = []
    for _ in range(dim):
        échantillons = np.concatenate([
            np.random.normal(loc=pics[k], scale=np.sqrt(variances[k]), size=tailles[k])
            for k in range(3)
        ])
        np.random.shuffle(échantillons)
        colonnes.append(échantillons)

    return cp.array(np.column_stack(colonnes))   # (nb, dim)


def distribution_trimodale(nb, centres=(0.2, 0.5, 0.8),
                           variances=(0.05, 0.05, 0.05),
                           amplitudes=(1, 1, 1), dim=1):
    """
    Alias de distribution_3_pics avec des paramètres nommés différemment.

    Retourne un tableau CuPy de forme (nb, dim).
    """
    return distribution_3_pics(nb, pics=centres, variances=variances,
                               poids=amplitudes, dim=dim)


# ─────────────────────────────────────────────
# VISUALISATION DES DISTRIBUTIONS
# ─────────────────────────────────────────────

def plot_distribution(*groupes,
                      etiquettes=None,
                      noms_dims=None,
                      titre="Distribution des agents",
                      bins=30,
                      alpha=0.6,
                      afficher_kde=True,
                      afficher_candidats=None):

    # ── Préparation ───────────────────────────────────────────────────────────
    if len(groupes) == 0:
        raise ValueError("Fournir au moins un groupe d'agents.")

    groupes_np = [to_numpy(to_cp(g)) for g in groupes]
    groupes_np = [g.reshape(-1, 1) if g.ndim == 1 else g for g in groupes_np]

    dim = groupes_np[0].shape[1]
    if any(g.shape[1] != dim for g in groupes_np):
        raise ValueError("Tous les groupes doivent avoir le même nombre de dimensions.")

    if etiquettes is None:
        etiquettes = [f"Groupe {i+1}" for i in range(len(groupes_np))]
    etiquettes = list(etiquettes) + [
        f"Groupe {i+1}" for i in range(len(etiquettes), len(groupes_np))
    ]

    if noms_dims is None:
        noms_dims = [f"Dim {d+1}" for d in range(dim)]

    couleurs = plt.cm.get_cmap("tab10")(np.linspace(0, 0.9, max(len(groupes_np), 3)))

    cands_np = None
    if afficher_candidats is not None:
        cands_np = to_numpy(to_cp(afficher_candidats))
        if cands_np.ndim == 1:
            cands_np = cands_np.reshape(-1, 1)

    # ── dim = 1 : histogramme + KDE ──────────────────────────────────────────
    if dim == 1:
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.suptitle(titre, fontsize=14, fontweight='bold')

        all_vals = np.concatenate([g[:, 0] for g in groupes_np])
        x_range  = np.linspace(all_vals.min() - 0.05, all_vals.max() + 0.05, 300)

        for i, (g, label) in enumerate(zip(groupes_np, etiquettes)):
            vals = g[:, 0]
            ax.hist(vals, bins=bins, density=True, alpha=alpha,
                    color=couleurs[i], label=label, edgecolor='white', linewidth=0.5)
            if afficher_kde:
                # KDE gaussien — sans scipy
                bw = max(1.06 * np.std(vals) * len(vals) ** (-0.2), 1e-9)
                kde = np.array([
                    np.mean(np.exp(-0.5 * ((x - vals) / bw) ** 2)
                            / (bw * np.sqrt(2 * np.pi)))
                    for x in x_range
                ])
                ax.plot(x_range, kde, color=couleurs[i], linewidth=2.5)

        if cands_np is not None:
            for j, cx in enumerate(cands_np[:, 0]):
                ax.axvline(cx, color='red', linewidth=1.8, linestyle='--',
                           label='Candidat' if j == 0 else None)

        ax.set_xlabel(noms_dims[0], fontsize=12)
        ax.set_ylabel("Densité", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()

    # ── dim = 2 : scatter 2D ──────────────────────────────────────────────────
    elif dim == 2:
        fig, ax = plt.subplots(figsize=(8, 7))
        fig.suptitle(titre, fontsize=14, fontweight='bold')

        for i, (g, label) in enumerate(zip(groupes_np, etiquettes)):
            ax.scatter(g[:, 0], g[:, 1], alpha=alpha, s=15,
                       color=couleurs[i], label=label, edgecolors='none')

        if cands_np is not None:
            ax.scatter(cands_np[:, 0], cands_np[:, 1],
                       marker='*', s=250, color='red', zorder=5,
                       edgecolors='darkred', linewidths=0.8, label='Candidats')

        ax.set_xlabel(noms_dims[0], fontsize=12)
        ax.set_ylabel(noms_dims[1], fontsize=12)
        ax.legend(fontsize=10, markerscale=1.5)
        ax.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()

    # ── dim ≥ 3 : matrice pairwise ────────────────────────────────────────────
    else:
        fig, axes = plt.subplots(dim, dim, figsize=(3 * dim, 3 * dim))
        fig.suptitle(titre, fontsize=14, fontweight='bold')

        for row in range(dim):
            for col in range(dim):
                ax = axes[row][col]

                if row == col:
                    # Diagonale : histogramme de la dimension row
                    for i, (g, label) in enumerate(zip(groupes_np, etiquettes)):
                        ax.hist(g[:, row], bins=bins, density=True, alpha=alpha,
                                color=couleurs[i], edgecolor='white', linewidth=0.4)
                    ax.set_xlabel(noms_dims[row], fontsize=9)
                    ax.tick_params(labelsize=7)
                    ax.grid(alpha=0.25, linestyle='--')

                else:
                    # Hors diagonale : scatter dim col (x) vs dim row (y)
                    for i, g in enumerate(groupes_np):
                        ax.scatter(g[:, col], g[:, row], alpha=alpha * 0.8,
                                   s=8, color=couleurs[i], edgecolors='none')
                    if cands_np is not None:
                        ax.scatter(cands_np[:, col], cands_np[:, row],
                                   marker='*', s=120, color='red', zorder=5,
                                   edgecolors='darkred', linewidths=0.5)
                    ax.set_xlabel(noms_dims[col], fontsize=9)
                    ax.set_ylabel(noms_dims[row], fontsize=9)
                    ax.tick_params(labelsize=7)
                    ax.grid(alpha=0.25, linestyle='--')

        # Légende commune
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=couleurs[i], markersize=9,
                               label=etiquettes[i])
                   for i in range(len(groupes_np))]
        if cands_np is not None:
            handles.append(plt.Line2D([0], [0], marker='*', color='w',
                                       markerfacecolor='red', markersize=12,
                                       markeredgecolor='darkred', label='Candidats'))
        fig.legend(handles=handles, loc='upper right', fontsize=10,
                   bbox_to_anchor=(0.98, 0.98), framealpha=0.9)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


# ─────────────────────────────────────────────
# DISTANCES ET TABLEAU DES PRÉFÉRENCES
# ─────────────────────────────────────────────

def distances(candidats, votants):

    c = to_cp(candidats)   # (nb_c, dim)
    v = to_cp(votants)     # (nb_v, dim)

    # diff[i, j, d] = v[i, d] - c[j, d]
    diff = v[:, None, :] - c[None, :, :]          # (nb_v, nb_c, dim)
    return cp.sqrt(cp.sum(diff ** 2, axis=2))      # (nb_v, nb_c)


def tableau_des_preferences(candidats, votants):

    dist  = distances(candidats, votants)     # (nb_v, nb_c)
    ordre = cp.argsort(dist, axis=1)

    nb_votants, nb_candidats = dist.shape
    tab  = cp.empty((nb_votants, nb_candidats), dtype=cp.int32)
    rows = cp.arange(nb_votants)[:, None]
    tab[rows, ordre] = cp.arange(1, nb_candidats + 1, dtype=cp.int32)
    return tab


# ─────────────────────────────────────────────
# SCRUTINS
# ─────────────────────────────────────────────

def SM1T(sondage):

    sondage = to_cp(sondage)
    votes = cp.argmin(sondage, axis=1)
    nb_candidats = sondage.shape[1]
    scores = cp.bincount(votes, minlength=nb_candidats)
    scores_np = to_numpy(scores)
    résultat = sorted(enumerate(scores_np), key=lambda x: x[1], reverse=True)
    return [[int(score), idx] for idx, score in résultat]


def SM2T(sondage):

    sondage = to_cp(sondage)
    votes = cp.argmin(sondage, axis=1)
    nb_candidats = sondage.shape[1]
    scores = cp.bincount(votes, minlength=nb_candidats)
    scores_np = to_numpy(scores)
    T1 = sorted([[int(s), i] for i, s in enumerate(scores_np)], reverse=True)
    c0, c1 = T1[0][1], T1[1][1]
    pref_c0 = sondage[:, c0] < sondage[:, c1]
    nb_c0 = int(cp.sum(pref_c0))
    nb_c1 = sondage.shape[0] - nb_c0
    T2 = sorted([[nb_c0, c0], [nb_c1, c1]], reverse=True)
    return T2


def Borda_classique(sondage):

    sondage = to_cp(sondage)
    nb_candidats = sondage.shape[1]
    scores = cp.sum(nb_candidats - sondage -1, axis=0)
    scores_np = to_numpy(scores)
    return sorted([[float(s), i] for i, s in enumerate(scores_np)], reverse=True)


def Borda_pondéré(sondage, poids):

    sondage = to_cp(sondage)
    nb_candidats = sondage.shape[1]
    if len(poids) != nb_candidats:
        raise ValueError("Le nombre de poids doit être égal au nombre de candidats.")
    poids_gpu = cp.array(poids, dtype=cp.float64)
    rangs_0indexed = sondage - 1
    scores_mat = poids_gpu[rangs_0indexed.astype(cp.int32)]
    scores = cp.sum(scores_mat, axis=0)
    scores_np = to_numpy(scores)
    return sorted([[float(s), i] for i, s in enumerate(scores_np)], reverse=True)


def Copeland(sondage):

    sondage = to_cp(sondage)
    rangs = sondage.astype(cp.float32)
    wins   = cp.sum(rangs[:, :, None] < rangs[:, None, :], axis=0)
    losses = cp.sum(rangs[:, :, None] > rangs[:, None, :], axis=0)
    victoires = (wins > losses).astype(cp.float32)
    egalites  = (wins == losses).astype(cp.float32) * 0.5
    nb_candidats = sondage.shape[1]
    idx = cp.arange(nb_candidats)
    victoires[idx, idx] = 0
    egalites[idx, idx]  = 0
    scores = cp.sum(victoires + egalites, axis=1)
    scores_np = to_numpy(scores)
    return sorted([[float(s), i] for i, s in enumerate(scores_np)], reverse=True)


# ─────────────────────────────────────────────
# TEST SM1T vs SM2T
# ─────────────────────────────────────────────

def test_SM1T_SM2T(nb_candidats, nb_votants, nb_simulations=1000, dim=1):
    """
    Compare SM1T et SM2T sur une grille de paramètres.
    Affiche une heatmap du nombre de divergences (gagnant T1 ≠ gagnant T2).
    """
    plt.close('all')
    cand_range = list(range(2, nb_candidats + 1))
    vot_range  = list(range(3, nb_votants + 1))
    gen = partial(distribution_uniforme, dim=dim)
    lst = []
    t0 = time.time()
    for k in cand_range:
        for j in vot_range:
            a = 0
            for _ in range(nb_simulations):
                candidats = gen(k)
                votants   = gen(j)
                tdp = tableau_des_preferences(candidats, votants)
                if SM1T(tdp)[0][1] != SM2T(tdp)[0][1]:
                    a += 1
            lst.append(a)
    print(f"⏱️  Durée : {time.time() - t0:.2f}s")
    Z = np.array(lst).reshape(len(cand_range), len(vot_range))
    extent = [vot_range[0], vot_range[-1], cand_range[0], cand_range[-1]]
    plt.imshow(Z, extent=extent, origin='lower', cmap='hot',
               interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.xlabel("Nombre de votants")
    plt.ylabel("Nombre de candidats")
    plt.title(f"Divergence SM1T / SM2T  (dim={dim})")
    plt.show()


# ─────────────────────────────────────────────
# SCORE DE SATISFACTION
# ─────────────────────────────────────────────

def satisfaction_relative(gagnant_idx, dist, votants=None):

    dist = to_cp(dist)
    dist_favori  = cp.min(dist, axis=1)
    dist_gagnant = dist[:, gagnant_idx]
    scores = cp.where(
        dist_gagnant > 0,
        dist_favori / dist_gagnant,
        cp.ones_like(dist_gagnant)
    )
    return float(cp.mean(scores))

def satisfaction_a_priori(gagnant_idx, tdp, votants=None):
    tdp = to_cp(tdp)
    rangs_gagnant = tdp[:, gagnant_idx]
    scores = 1 - (rangs_gagnant - 1) / (tdp.shape[1] - 1)
    return float(cp.mean(scores))

def satisfaction_maximale(dist):
    """Satisfaction maximale atteignable parmi tous les candidats."""
    best = 0.0
    for i in range(to_cp(dist).shape[1]):
        sat_i = satisfaction_relative(i, dist)
        if sat_i > best:
            best = sat_i
    return best


# ─────────────────────────────────────────────
# MANIPULABILITÉ SUR UN VOTE FIXÉ
# ─────────────────────────────────────────────

def _construire_bulletin_strategique(bulletins_coalition_np, favori, cible, nb_candidats):
    """Construit un bulletin stratégique : favori→1, cible→dernier, autres conservés."""
    milieu      = [c for c in range(nb_candidats) if c != favori and c != cible]
    rang_moyen  = {c: float(np.mean(bulletins_coalition_np[:, c])) for c in milieu}
    milieu_trié = sorted(milieu, key=lambda c: rang_moyen[c])
    bulletin    = [0] * nb_candidats
    bulletin[favori] = 1
    bulletin[cible]  = nb_candidats
    for rang, cand in enumerate(milieu_trié, start=2):
        bulletin[cand] = rang
    return bulletin


def manipulation_naïve(tdp_honnete, dist, scrutin):
    """
    Teste si un vote est manipulable par une stratégie naïve.
    Chaque coalition perdante applique un bulletin stratégique unique
    (favori→1, leader→dernier, autres en ordre moyen).

    Signature : (tdp, dist, scrutin) → bool
    """
    tdp_honnete = to_cp(tdp_honnete)
    nb_votants, nb_candidats = tdp_honnete.shape
    tdp_np = to_numpy(tdp_honnete)

    résultats_honnêtes = scrutin(tdp_honnete)
    gagnant_honnete    = résultats_honnêtes[0][1]

    if _nom_scrutin(scrutin) == "SM2T":
        votes_T1  = cp.argmin(tdp_honnete, axis=1)
        scores_T1 = to_numpy(cp.bincount(votes_T1, minlength=nb_candidats))
        cible     = int(np.argmax(scores_T1))
    else:
        cible = gagnant_honnete

    favoris_tous       = cp.argmin(tdp_honnete, axis=1)
    perdants_mask      = favoris_tous != gagnant_honnete
    indices_perdants   = to_numpy(cp.where(perdants_mask)[0])
    candidats_perdants = sorted(set(int(favoris_tous[i]) for i in indices_perdants))

    for favori in candidats_perdants:
        if favori == cible:
            continue
        indices_coalition = to_numpy(cp.where(favoris_tous == favori)[0])
        bulletin_strat = _construire_bulletin_strategique(
            tdp_np[indices_coalition], favori, cible, nb_candidats
        )
        tdp_test = tdp_honnete.copy()
        for idx in indices_coalition:
            tdp_test[idx] = cp.array(bulletin_strat, dtype=cp.int32)
        if scrutin(tdp_test)[0][1] == favori:
            return True
    return False


def vote_utile(tdp_honnete, dist, scrutin):
    """
    Stratégie de vote utile : les groupes dont le favori est éliminé
    reportent leurs voix vers le candidat viable qu'ils préfèrent.

    Signature : (tdp, dist, scrutin) → bool
    """
    tdp_honnete = to_cp(tdp_honnete)
    nb_votants, nb_candidats = tdp_honnete.shape
    tdp_np = to_numpy(tdp_honnete)

    if _nom_scrutin(scrutin) == "SM2T":
        votes_T1        = cp.argmin(tdp_honnete, axis=1)
        scores_T1       = to_numpy(cp.bincount(votes_T1, minlength=nb_candidats))
        T1_trié         = sorted(range(nb_candidats), key=lambda c: scores_T1[c], reverse=True)
        seuil_top2      = scores_T1[T1_trié[1]]
        gagnant_honnete = scrutin(tdp_honnete)[0][1]
        favoris_tous    = cp.argmin(tdp_honnete, axis=1)
        groupes = {c: to_numpy(cp.where(favoris_tous == c)[0])
                   for c in range(nb_candidats)
                   if int(cp.sum(favoris_tous == c)) > 0}
        eliminés = {c for c, idx in groupes.items()
                    if scores_T1[c] + len(idx) <= seuil_top2 and c != gagnant_honnete}
        viables = set()
        for c in range(nb_candidats):
            if c == gagnant_honnete:
                continue
            for e, idx_e in groupes.items():
                if e in eliminés and scores_T1[c] + len(idx_e) > seuil_top2:
                    viables.add(c)
                    break
        if not eliminés or not viables:
            return False
        tdp_test_np = tdp_np.copy()
        for éliminé in eliminés:
            indices_groupe = groupes[éliminé]
            rangs_moyens   = {v: float(np.mean(tdp_np[indices_groupe, v])) for v in viables}
            cible          = min(rangs_moyens, key=rangs_moyens.get)
            for i in indices_groupe:
                rang_cible       = tdp_np[i, cible]
                nouveau_bulletin = tdp_np[i].copy()
                for c in range(nb_candidats):
                    r = tdp_np[i, c]
                    if c == cible:
                        nouveau_bulletin[c] = 1
                    elif r < rang_cible:
                        nouveau_bulletin[c] = r + 1
                tdp_test_np[i] = nouveau_bulletin
        return scrutin(to_cp(tdp_test_np))[0][1] != gagnant_honnete

    else:
        résultats_honnêtes = scrutin(tdp_honnete)
        gagnant_honnete    = résultats_honnêtes[0][1]
        score_leader       = résultats_honnêtes[0][0]
        score_par_candidat = np.zeros(nb_candidats)
        for score, idx in résultats_honnêtes:
            score_par_candidat[idx] = score
        favoris_tous = cp.argmin(tdp_honnete, axis=1)
        groupes = {c: to_numpy(cp.where(favoris_tous == c)[0])
                   for c in range(nb_candidats)
                   if int(cp.sum(favoris_tous == c)) > 0}
        viables  = {c for c, idx in groupes.items()
                    if score_par_candidat[c] + len(idx) > score_leader}
        eliminés = {c for c in groupes if c not in viables and c != gagnant_honnete}
        if not eliminés or not viables:
            return False
        tdp_test_np = tdp_np.copy()
        for éliminé in eliminés:
            indices_groupe = groupes[éliminé]
            rangs_moyens   = {v: float(np.mean(tdp_np[indices_groupe, v])) for v in viables}
            cible          = min(rangs_moyens, key=rangs_moyens.get)
            for i in indices_groupe:
                rang_cible       = tdp_np[i, cible]
                nouveau_bulletin = tdp_np[i].copy()
                for c in range(nb_candidats):
                    r = tdp_np[i, c]
                    if c == cible:
                        nouveau_bulletin[c] = 1
                    elif r < rang_cible:
                        nouveau_bulletin[c] = r + 1
                tdp_test_np[i] = nouveau_bulletin
        return scrutin(to_cp(tdp_test_np))[0][1] != gagnant_honnete


def optimisation_vote_groupe(tdp_honnete, dist, scrutin):
    """
    Manipulation exhaustive : explore toutes les permutations de bulletin
    (favori fixé en rang 1) pour chaque coalition perdante.

    Signature : (tdp, dist, scrutin) → bool
    """
    tdp_honnete = to_cp(tdp_honnete)
    nb_votants, nb_candidats = tdp_honnete.shape
    résultats_honnêtes = scrutin(tdp_honnete)
    gagnant_honnete    = résultats_honnêtes[0][1]
    favoris_tous       = cp.argmin(tdp_honnete, axis=1)
    indices_perdants   = to_numpy(cp.where(favoris_tous != gagnant_honnete)[0])
    candidats_perdants = sorted(set(int(favoris_tous[i]) for i in indices_perdants))

    for favori in candidats_perdants:
        indices_coalition   = to_numpy(cp.where(favoris_tous == favori)[0])
        bulletins_originaux = tdp_honnete[indices_coalition].copy()
        autres = [c for c in range(nb_candidats) if c != favori]
        for perm in itertools.permutations(autres):
            bulletin = [0] * nb_candidats
            bulletin[favori] = 1
            for rang, cand in enumerate(perm, start=2):
                bulletin[cand] = rang
            tdp_test = tdp_honnete.copy()
            for idx in indices_coalition:
                tdp_test[idx] = cp.array(bulletin, dtype=cp.int32)
            if scrutin(tdp_test)[0][1] == favori:
                return True
        tdp_honnete[indices_coalition] = bulletins_originaux
    return False


# ─────────────────────────────────────────────
# SIMULATION DE MANIPULABILITÉ
# ─────────────────────────────────────────────

def simulation_manipulabilite(scrutin, nb_candidats, nb_votants,
                               strategie_manipulation=manipulation_naïve,
                               generateur_votants=distribution_uniforme,
                               genenerateur_candidats=distribution_uniforme,
                               nb_simulations=1000,
                               candidats_fixes=None):
    """
    Estime le taux de manipulabilité d'un scrutin.

    Pour utiliser un espace n-dimensionnel, passez les générateurs via partial :
        gen2D = partial(distribution_uniforme, dim=2)
        simulation_manipulabilite(SM1T, 4, 100,
                                  generateur_votants=gen2D,
                                  genenerateur_candidats=gen2D)
    """
    start            = time.perf_counter()
    nb_manipulations = 0
    scores_honnetes  = []

    for _ in range(nb_simulations):
        candidats = to_cp(candidats_fixes) if candidats_fixes is not None \
                    else genenerateur_candidats(nb_candidats)
        votants = generateur_votants(nb_votants)
        dist    = distances(candidats, votants)
        tdp     = tableau_des_preferences(candidats, votants)

        résultats_honnêtes = scrutin(tdp)
        gagnant_honnete    = résultats_honnêtes[0][1]
        scores_honnetes.append(satisfaction_relative(gagnant_honnete, dist))

        if strategie_manipulation(tdp, dist, scrutin):
            nb_manipulations += 1

    end       = time.perf_counter()
    taux      = nb_manipulations / nb_simulations
    s_hon_moy = float(np.mean(scores_honnetes)) if scores_honnetes else 0.0

    print(f"\n{'─'*55}")
    print(f"  Scrutin               : {_nom_scrutin(scrutin)}")
    print(f"  Stratégie             : {strategie_manipulation.__name__}")
    print(f"  Candidats / Votants   : {nb_candidats} / {nb_votants}")
    print(f"  Simulations           : {nb_simulations}")
    print(f"{'─'*55}")
    print(f"  Manipulations         : {nb_manipulations}/{nb_simulations}")
    print(f"  Taux de manipulation  : {taux:.4f}")
    print(f"  Satisfaction honnête  : {s_hon_moy:.4f}")
    print(f"  Durée                 : {end - start:.2f}s")
    print(f"{'─'*55}\n")

    return {
        "taux_manipulation"  : taux,
        "score_honnete_moyen": s_hon_moy,
        "scores_honnetes"    : scores_honnetes,
    }


def comparer_manipulabilite(scrutins, nb_candidats, nb_votants,
                             strategie_manipulation=manipulation_naïve,
                             generateur_votants=distribution_uniforme,
                             generateur_candidats=distribution_uniforme,
                             nb_simulations=1000,
                             candidats_fixes=None):
    """Compare le taux de manipulabilité de plusieurs scrutins sur les mêmes paysages."""
    start    = time.perf_counter()
    paysages = []
    for _ in range(nb_simulations):
        candidats = to_cp(candidats_fixes) if candidats_fixes is not None \
                    else generateur_candidats(nb_candidats)
        votants = generateur_votants(nb_votants)
        paysages.append((tableau_des_preferences(candidats, votants),
                         distances(candidats, votants)))

    résultats = {}
    for scrutin in scrutins:
        nb_manips, sats = 0, []
        for tdp, dist in paysages:
            g = scrutin(tdp)[0][1]
            sats.append(satisfaction_relative(g, dist))
            if strategie_manipulation(tdp, dist, scrutin):
                nb_manips += 1
        résultats[_nom_scrutin(scrutin)] = {
            "taux_manipulation"  : nb_manips / nb_simulations,
            "satisfaction_moyenne": float(np.mean(sats)),
        }

    end = time.perf_counter()
    print(f"\n{'═'*60}")
    print(f"  Stratégie : {strategie_manipulation.__name__}  |  "
          f"Candidats/Votants : {nb_candidats}/{nb_votants}  |  "
          f"Simulations : {nb_simulations}")
    print(f"{'═'*60}")
    print(f"  {'Scrutin':<22} {'Manip':>8}  {'Satisfaction':>13}")
    for nom, vals in sorted(résultats.items(), key=lambda x: x[1]["taux_manipulation"]):
        print(f"  {nom:<22} {vals['taux_manipulation']:>8.4f}  "
              f"{vals['satisfaction_moyenne']:>13.4f}")
    print(f"  Durée : {end - start:.2f}s\n")
    _plot_comparaison_manipulabilite(résultats, strategie_manipulation.__name__)
    return résultats


def _plot_comparaison_manipulabilite(résultats, nom_strategie=""):
    noms       = sorted(résultats, key=lambda n: résultats[n]["taux_manipulation"])
    taux_manip = [résultats[n]["taux_manipulation"]    for n in noms]
    taux_sat   = [résultats[n]["satisfaction_moyenne"] for n in noms]
    x = np.arange(len(noms)); width = 0.35
    fig, ax1 = plt.subplots(figsize=(max(8, len(noms) * 2), max(5, len(noms) * 0.9)))
    fig.suptitle(f"Manipulabilité & Satisfaction — {nom_strategie}", fontsize=13, fontweight='bold')
    bars1 = ax1.bar(x - width/2, taux_manip, width, color='tomato', alpha=0.85,
                    edgecolor='white', label='Taux de manipulation')
    ax1.set_ylabel("Taux de manipulation", fontsize=11, color='tomato')
    ax1.tick_params(axis='y', labelcolor='tomato')
    ax1.set_ylim([0, min(1.0, max(taux_manip)*1.35) if taux_manip else 1.0])
    for bar, v in zip(bars1, taux_manip):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f"{v:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold', color='tomato')
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, taux_sat, width, color='steelblue', alpha=0.85,
                    edgecolor='white', label='Satisfaction moyenne')
    ax2.set_ylabel("Satisfaction moyenne", fontsize=11, color='steelblue')
    ax2.tick_params(axis='y', labelcolor='steelblue')
    ax2.set_ylim([max(0, min(taux_sat)*0.92) if taux_sat else 0,
                  min(1.0, max(taux_sat)*1.08) if taux_sat else 1.0])
    for bar, v in zip(bars2, taux_sat):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                 f"{v:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold', color='steelblue')
    ax1.set_xticks(x); ax1.set_xticklabels(noms, rotation=15, ha='right', fontsize=10)
    ax1.set_xlabel("Méthode de vote", fontsize=11); ax1.grid(axis='y', alpha=0.25, linestyle='--')
    l1, lb1 = ax1.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, lb1+lb2, loc='upper left', fontsize=9)
    plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────
# CONSISTANCE DES SATISFACTIONS
# ─────────────────────────────────────────────

def consistance_satisfaction(scrutins, nb_candidats, nb_votants,
                              metriques=None, strategies_manipulation=None,
                              generateur_votants=distribution_uniforme,
                              generateur_candidats=distribution_uniforme,
                              nb_simulations=200):
    if metriques is None:
        metriques = [satisfaction_relative]
    if strategies_manipulation is None:
        strategies_manipulation = []

    noms_scrutins   = [_nom_scrutin(s) for s in scrutins]
    noms_strategies = [st.__name__ for st in strategies_manipulation]
    res_sat   = {m.__name__: {n: np.zeros(nb_simulations) for n in noms_scrutins}
                 for m in metriques}
    res_manip = {st.__name__: {n: np.zeros(nb_simulations, dtype=bool) for n in noms_scrutins}
                 for st in strategies_manipulation}

    start = time.perf_counter()
    for k in range(nb_simulations):
        candidats = generateur_candidats(nb_candidats)
        votants   = generateur_votants(nb_votants)
        dist      = distances(candidats, votants)
        tdp       = tableau_des_preferences(candidats, votants)
        for scrutin in scrutins:
            nom = _nom_scrutin(scrutin)
            res = scrutin(tdp); g = res[0][1]
            for m in metriques:
                res_sat[m.__name__][nom][k] = m(g, dist, votants)
            for st in strategies_manipulation:
                res_manip[st.__name__][nom][k] = st(tdp, dist, scrutin)
        print(f"\r  Progression : {k+1}/{nb_simulations}", end="", flush=True)
    print(f"\n  Durée : {time.perf_counter()-start:.1f}s")

    for m in metriques:
        nm = m.__name__
        print(f"\n{'═'*65}\n  Satisfaction — {nm}")
        print(f"  {'Scrutin':<22}  {'Moy':>7}  {'Méd':>7}  {'Std':>7}  {'Min':>7}  {'Max':>7}")
        for n in noms_scrutins:
            v = res_sat[nm][n]
            print(f"  {n:<22}  {np.mean(v):>7.4f}  {np.median(v):>7.4f}  "
                  f"{np.std(v):>7.4f}  {np.min(v):>7.4f}  {np.max(v):>7.4f}")
        print(f"{'═'*65}")

    if strategies_manipulation:
        for st in strategies_manipulation:
            ns = st.__name__
            print(f"\n{'═'*65}\n  Manipulation — {ns}")
            for n in noms_scrutins:
                v = res_manip[ns][n]
                print(f"  {n:<22}  {np.mean(v):>8.4f}  {int(np.sum(v)):>10}/{nb_simulations}")
            print(f"{'═'*65}")

    ylims_sat = {}
    for m in metriques:
        all_v = np.concatenate([res_sat[m.__name__][n] for n in noms_scrutins])
        mg = max(0.02, (all_v.max()-all_v.min())*0.08)
        ylims_sat[m.__name__] = (max(0.0, all_v.min()-mg), min(1.0, all_v.max()+mg))

    ylim_manip = None
    if strategies_manipulation:
        all_t = [float(np.mean(res_manip[ns][n])) for ns in noms_strategies for n in noms_scrutins]
        tmax, tmin = max(all_t) if all_t else 0.1, min(all_t) if all_t else 0.0
        mg = max(0.02, (tmax-tmin)*0.15)
        ylim_manip = (max(0.0, tmin-mg), min(1.0, tmax+mg*3))

    nb_lignes = len(metriques) + (1 if strategies_manipulation else 0)
    fig, axes = plt.subplots(nb_lignes, 2,
                             figsize=(20, min(10.0, 9.5/nb_lignes)*nb_lignes),
                             squeeze=False, dpi=96)
    fig.suptitle(f"Consistance — {nb_candidats} candidats · {nb_votants} votants · "
                 f"{nb_simulations} essais", fontsize=14, fontweight='bold')
    for row, m in enumerate(metriques):
        _plot_consistance_row(axes[row], res_sat[m.__name__], noms_scrutins,
                              m.__name__, ylim=ylims_sat[m.__name__])
    if strategies_manipulation:
        _plot_manipulation_row(axes[len(metriques)], res_manip,
                               noms_scrutins, noms_strategies, ylim=ylim_manip)
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.5, w_pad=2.0)
    plt.show()
    return {"satisfaction": res_sat, "manipulation": res_manip}


def _plot_consistance_row(axes_row, résultats_metrique, noms, nom_metrique, ylim=None):
    nb_scrutins = len(noms)
    couleurs    = plt.cm.get_cmap("tab10")(np.linspace(0, 0.9, nb_scrutins))
    nb_sims     = len(next(iter(résultats_metrique.values())))

    if ylim is None:
        all_v = np.concatenate([résultats_metrique[n] for n in noms])
        mg = max(0.02, (all_v.max()-all_v.min())*0.08)
        ylim = (max(0.0, all_v.min()-mg), min(1.0, all_v.max()+mg))
    ymin, ymax = ylim

    # --- PRÉPARATION DES DONNÉES POUR LES CHEVAUCHEMENTS ---
    # Dictionnaire pour grouper les noms par coordonnées exactes : {(x, y_arrondi): [nom1, nom2...]}
    points_info = {}
    for n in noms:
        for x_idx, y_val in enumerate(résultats_metrique[n], start=1):
            # On arrondit à 5 décimales pour éviter les faux négatifs liés à la précision des flottants
            coord = (x_idx, round(y_val, 5))
            if coord not in points_info:
                points_info[coord] = []
            points_info[coord].append(n)

    # --- GRAPHIQUE 1 : SCATTER PLOT ---
    ax = axes_row[0]
    xs = np.arange(1, nb_sims + 1)
    
    scatters = [] # On stocke les objets dessinés pour mplcursors
    for i, n in enumerate(noms):
        v = résultats_metrique[n]
        # Alignement parfait sur les entiers, pas de décalage
        sc = ax.scatter(xs, v, color=couleurs[i], linewidth=1.5, alpha=0.75, zorder=2,
                        label=f"{n}  (moy={np.mean(v):.3f}, std={np.std(v):.3f})")
        scatters.append(sc)

    # --- GESTION DU SURVOL (HOVER) ---
    cursor = mplcursors.cursor(scatters, hover=True)
    
    @cursor.connect("add")
    def on_add(sel):
        # Récupération des coordonnées pointées par la souris
        x_val = int(round(sel.target[0]))
        y_val = sel.target[1]
        coord = (x_val, round(y_val, 5))
        
        # Récupération de tous les noms partageant ce point
        noms_presents = points_info.get(coord, ["Inconnu"])
        
        # Formatage du texte de l'infobulle
        if len(noms_presents) > 1:
            texte = f"⚠️ Chevauchement ({len(noms_presents)}):\n" + "\n".join(noms_presents) + f"\n\nValeur: {y_val:.4f}"
        else:
            texte = f"{noms_presents[0]}\nValeur: {y_val:.4f}"
            
        sel.annotation.set_text(texte)
        # Style de l'infobulle
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9, ec="gray")

    ax.set_xticks(xs)
    ax.set_xlabel("Numéro de l'essai", fontsize=11)
    ax.set_ylabel(nom_metrique, fontsize=11)
    ax.set_title(f"{nom_metrique} — valeurs brutes", fontsize=11)
    ax.legend(fontsize=8, loc='best', framealpha=0.85)
    ax.set_ylim([ymin, ymax])
    ax.grid(alpha=0.3, linestyle='--')

    # --- GRAPHIQUE 2 : DISTRIBUTIONS (VIOLIN/BOXPLOT) ---
    ax = axes_row[1]
    esp = max(1.8, 0.9 + nb_scrutins * 0.3)
    pos = np.arange(nb_scrutins) * esp
    data = [résultats_metrique[n] for n in noms]
    
    parts = ax.violinplot(data, positions=pos, widths=esp*0.5, showmeans=False, showextrema=False)
    for j, pc in enumerate(parts['bodies']):
        pc.set_facecolor(couleurs[j]); pc.set_alpha(0.35)
        
    bp = ax.boxplot(data, positions=pos, widths=esp*0.15, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2.5),
                    whiskerprops=dict(linewidth=1.5, linestyle='--'),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markersize=3, alpha=0.35, markeredgewidth=0))
                    
    for j, patch in enumerate(bp['boxes']):
        patch.set_facecolor(couleurs[j]); patch.set_alpha(0.80)
        
    for j, n in enumerate(noms):
        med = np.median(résultats_metrique[n]); moy = np.mean(résultats_metrique[n])
        ax.text(pos[j]+esp*0.3, med, f"méd={med:.3f}\nmoy={moy:.3f}",
                fontsize=8, va='center', ha='left', color='#333333',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, ec='none'))
                
    ax.set_xticks(pos)
    ax.set_xticklabels(noms, fontsize=10, rotation=20, ha='right')
    ax.set_ylabel(nom_metrique, fontsize=11)
    ax.set_title(f"{nom_metrique} — distributions", fontsize=11)
    ax.set_xlim([-esp*0.6, pos[-1]+esp*1.2])
    ax.set_ylim([ymin, ymax])
    ax.grid(axis='y', alpha=0.3, linestyle='--')


def _plot_manipulation_row(axes_row, res_manip, noms_scrutins, noms_strategies, ylim=None):
    couleurs = plt.cm.get_cmap("tab10")(np.linspace(0, 0.9, len(noms_scrutins)))
    nb_s, nb_st = len(noms_scrutins), len(noms_strategies)
    esp = max(1.0, 0.6+nb_st*0.4); larg = esp*0.7/nb_st
    x = np.arange(nb_s)*(esp*nb_st+0.5)
    if ylim is None:
        all_t = [float(np.mean(res_manip[ns][n])) for ns in noms_strategies for n in noms_scrutins]
        tmax, tmin = max(all_t) if all_t else 0.1, min(all_t) if all_t else 0.0
        mg = max(0.02, (tmax-tmin)*0.15)
        ylim = (max(0.0, tmin-mg), min(1.0, tmax+mg*3))
    ymin, ymax = ylim
    ax = axes_row[0]
    for j, ns in enumerate(noms_strategies):
        taux = [float(np.mean(res_manip[ns][n])) for n in noms_scrutins]
        off  = (j-(nb_st-1)/2)*larg
        bars = ax.bar(x+off, taux, larg*0.85, color=[couleurs[k] for k in range(nb_s)],
                      alpha=0.80, edgecolor='white', linewidth=1.2, label=ns)
        for bar, t in zip(bars, taux):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+(ymax-ymin)*0.02,
                    f"{t:.3f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(noms_scrutins, fontsize=10, rotation=20, ha='right')
    ax.set_ylabel("Taux de manipulation", fontsize=11)
    ax.set_title("Taux de manipulation par scrutin et stratégie", fontsize=11)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.85)
    ax.set_ylim([ymin, ymax]); ax.grid(axis='y', alpha=0.3, linestyle='--')
    axes_row[1].set_visible(False)


# ─────────────────────────────────────────────
# TEST SUR LA DIMENSION
# ─────────────────────────────────────────────

def test_dimension(scrutins, nb_candidats, nb_votants,
                   metriques=None,
                   generateur_votants=distribution_uniforme,
                   generateur_candidats=distribution_uniforme,
                   nb_simulations=200):

    if metriques is None:
        metriques = [satisfaction_relative]

    noms_scrutins = [_nom_scrutin(s) for s in scrutins]

    # --- Détermination de la dimension maximale ---
    candidats_test = generateur_candidats(nb_candidats)
    votants_test   = generateur_votants(nb_votants)

    dim_max = min(candidats_test.shape[1], votants_test.shape[1])

    # résultats[metrique][scrutin][dimension] = tableau des simulations
    resultats = {
        m.__name__: {
            nom_scrutin: {
                d: np.zeros(nb_simulations)
                for d in range(1, dim_max + 1)
            }
            for nom_scrutin in noms_scrutins
        }
        for m in metriques
    }

    # ============================================================
    # SIMULATIONS
    # ============================================================

    start = time.perf_counter()

    for k in range(nb_simulations):

        # paysage politique complet
        candidats_full = generateur_candidats(nb_candidats)
        votants_full   = generateur_votants(nb_votants)

        for d in range(1, dim_max + 1):

            # projection dans les d premières dimensions
            candidats = candidats_full[:, :d]
            votants   = votants_full[:, :d]

            dist = distances(candidats, votants)
            tdp  = tableau_des_preferences(candidats, votants)

            for scrutin in scrutins:

                nom = _nom_scrutin(scrutin)

                res = scrutin(tdp)
                gagnant = res[0][1]

                for m in metriques:
                    resultats[m.__name__][nom][d][k] = (
                        m(gagnant, dist, votants)
                    )

        print(f"\r  Progression : {k+1}/{nb_simulations}",
              end="", flush=True)

    duree = time.perf_counter() - start
    print(f"\n  Durée : {duree:.1f}s")

    # ============================================================
    # AFFICHAGE TEXTE
    # ============================================================

    for m in metriques:

        nm = m.__name__

        print(f"\n{'═'*75}")
        print(f"  Analyse dimensionnelle — {nm}")
        print(f"{'═'*75}")

        for nom in noms_scrutins:

            print(f"\n  {nom}")
            print(f"  {'Dim':<6} {'Moy':>8} {'Méd':>8} {'Std':>8} "
                  f"{'Min':>8} {'Max':>8}")

            for d in range(1, dim_max + 1):

                vals = resultats[nm][nom][d]

                print(
                    f"  {d:<6} "
                    f"{np.mean(vals):>8.4f} "
                    f"{np.median(vals):>8.4f} "
                    f"{np.std(vals):>8.4f} "
                    f"{np.min(vals):>8.4f} "
                    f"{np.max(vals):>8.4f}"
                )

    # ============================================================
    # PLOTS
    # ============================================================

    for m in metriques:

        nm = m.__name__

        fig, axes = plt.subplots(
            1, 2,
            figsize=(18, 7),
            dpi=100
        )

        fig.suptitle(
            f"Impact de la dimension — {nm}\n"
            f"{nb_candidats} candidats · "
            f"{nb_votants} votants · "
            f"{nb_simulations} simulations",
            fontsize=14,
            fontweight='bold'
        )

        couleurs = plt.cm.get_cmap("tab10")(
            np.linspace(0, 0.9, len(noms_scrutins))
        )

        dimensions = np.arange(1, dim_max + 1)

        # ========================================================
        # GRAPHE 1 : fluctuation des satisfactions
        # ========================================================

        ax = axes[0]

        for i, nom in enumerate(noms_scrutins):

            moyennes = []
            stds = []

            for d in dimensions:

                vals = resultats[nm][nom][d]

                moyennes.append(np.mean(vals))
                stds.append(np.std(vals))

            moyennes = np.array(moyennes)
            stds = np.array(stds)

            ax.plot(
                dimensions,
                moyennes,
                marker='o',
                linewidth=2,
                color=couleurs[i],
                label=nom
            )

            ax.fill_between(
                dimensions,
                moyennes - stds,
                moyennes + stds,
                alpha=0.18,
                color=couleurs[i]
            )

        ax.set_xlabel("Dimension politique")
        ax.set_ylabel(nm)
        ax.set_title("Évolution de la satisfaction")
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend()

        # ========================================================
        # GRAPHE 2 : boxplots par dimension
        # ========================================================

        ax = axes[1]

        largeur_bloc = len(noms_scrutins) + 1.5

        positions = []
        labels = []

        for i, nom in enumerate(noms_scrutins):

            data = []
            pos = []

            for d in dimensions:

                vals = resultats[nm][nom][d]

                data.append(vals)

                p = d * largeur_bloc + i
                pos.append(p)

                positions.append(p)
                labels.append(str(d))

            bp = ax.boxplot(
                data,
                positions=pos,
                widths=0.7,
                patch_artist=True,
                showmeans=True,
                meanline=False,
                medianprops=dict(color='black', linewidth=2),
                meanprops=dict(
                    marker='D',
                    markerfacecolor='white',
                    markeredgecolor='black',
                    markersize=5
                )
            )

            for patch in bp['boxes']:
                patch.set_facecolor(couleurs[i])
                patch.set_alpha(0.75)

            # annotations statistiques
            for p, vals in zip(pos, data):

                moy = np.mean(vals)
                med = np.median(vals)
                std = np.std(vals)

                ax.text(
                    p + 0.15,
                    med,
                    f"μ={moy:.3f}\nσ={std:.3f}",
                    fontsize=7,
                    alpha=0.85
                )

        centres = [
            d * largeur_bloc + (len(noms_scrutins)-1)/2
            for d in dimensions
        ]

        ax.set_xticks(centres)
        ax.set_xticklabels(dimensions)

        ax.set_xlabel("Dimension politique")
        ax.set_ylabel(nm)
        ax.set_title("Distributions des satisfactions")
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # légende custom
        handles = [
            plt.Line2D(
                [0], [0],
                color=couleurs[i],
                lw=8
            )
            for i in range(len(noms_scrutins))
        ]

        ax.legend(handles, noms_scrutins)

        plt.tight_layout(rect=[0, 0, 1, 0.94])

    plt.show()

    return resultats
        
        

# ─────────────────────────────────────────────
# UTILITAIRES D'AFFICHAGE
# ─────────────────────────────────────────────

def gagnant(résultats):
    return résultats[0][1]


def plot_résultats(résultats, labels=None, title="Résultats du scrutin"):
    résultats_triés = sorted(résultats, key=lambda x: x[1])
    scores  = [s for s, _ in résultats_triés]
    indices = [i for _, i in résultats_triés]
    if labels is None:
        labels = [f"{i}" for i in indices]
    total = sum(scores); pourcentages = [s/total*100 for s in scores]
    bars = plt.bar(labels, scores, edgecolor='white', linewidth=2)
    for bar, score, pct in zip(bars, scores, pourcentages):
        plt.text(bar.get_x()+bar.get_width()/2., bar.get_height(),
                 f'{score:.2f}\n({pct:.1f}%)', ha='center', va='top', fontweight='bold', fontsize=10)
    plt.xlabel("Candidats", fontsize=12, fontweight='bold')
    plt.ylabel("Scores",    fontsize=12, fontweight='bold')
    plt.title(title,        fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--'); plt.tight_layout(); plt.show()


def plot_résultats_triés(résultats, title="Résultats du scrutin triés"):
    résultats_triés = sorted(résultats, key=lambda x: x[0], reverse=True)
    scores  = [s for s, _ in résultats_triés]
    indices = [f"Candidat {i}" for _, i in résultats_triés]
    total = sum(scores); pourcentages = [s/total*100 for s in scores]
    bars = plt.bar(indices, scores, edgecolor='white', linewidth=2)
    for bar, score, pct in zip(bars, scores, pourcentages):
        plt.text(bar.get_x()+bar.get_width()/2., bar.get_height(),
                 f'{score:.2f}\n({pct:.1f}%)', ha='center', va='top', fontweight='bold', fontsize=11)
    plt.xlabel("Candidats (classés)", fontsize=12, fontweight='bold')
    plt.ylabel("Scores",              fontsize=12, fontweight='bold')
    plt.title(title,                  fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right'); plt.tight_layout(); plt.show()


def plot_suffrage(votants, candidats, titre="Distribution du suffrage", noms_dims=None):
    """
    Visualise les positions des votants et des candidats dans leur espace.
    
    Paramètres :
        votants    : tableau de forme (nb_votants, dim) — positions des votants
        candidats  : tableau de forme (nb_candidats, dim) — positions des candidats
        titre      : titre de la figure
        noms_dims  : noms des dimensions (ex: ['Économie', 'Social'])
                     Par défaut : ['Dim 1', 'Dim 2', ...]
    
    Comportement selon dim :
        dim = 1 : histogramme + scatter avec votants et candidats
        dim = 2 : scatter 2D — votants en bleu, candidats en rouge (étoiles)
        dim ≥ 3 : matrice pairwise (scatter hors diagonale, histogramme diagonale)
    """
    votants = to_numpy(votants)
    candidats = to_numpy(candidats)
    
    if votants.ndim == 1:
        votants = votants.reshape(-1, 1)
    if candidats.ndim == 1:
        candidats = candidats.reshape(-1, 1)
    
    dim = votants.shape[1]
    
    if noms_dims is None:
        noms_dims = [f"Dim {i+1}" for i in range(dim)]
    
    # Cas 1D : histogramme superposé
    if dim == 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(votants, bins=30, alpha=0.6, label='Votants', color='blue', edgecolor='black')
        ax.scatter(candidats, [0]*len(candidats), marker='*', s=500, color='red', 
                   label='Candidats', edgecolor='darkred', linewidth=2, zorder=5)
        ax.set_xlabel(noms_dims[0], fontsize=12, fontweight='bold')
        ax.set_ylabel("Fréquence", fontsize=12, fontweight='bold')
        ax.set_title(titre, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11); ax.grid(alpha=0.3, linestyle='--')
        plt.tight_layout(); plt.show()
    
    # Cas 2D : scatter 2D
    elif dim == 2:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(votants[:, 0], votants[:, 1], alpha=0.5, s=50, 
                  color='blue', label=f'Votants (n={len(votants)})', edgecolor='none')
        ax.scatter(candidats[:, 0], candidats[:, 1], marker='*', s=800, 
                  color='red', label=f'Candidats (n={len(candidats)})', 
                  edgecolor='darkred', linewidth=2)
        ax.set_xlabel(noms_dims[0], fontsize=12, fontweight='bold')
        ax.set_ylabel(noms_dims[1], fontsize=12, fontweight='bold')
        ax.set_title(titre, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best'); ax.grid(alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout(); plt.show()
    
    # Cas 3D+ : matrice pairwise
    else:
        n_plots = dim
        fig, axes = plt.subplots(n_plots, n_plots, figsize=(3*n_plots, 3*n_plots), squeeze=False)
        fig.suptitle(titre, fontsize=14, fontweight='bold')
        
        for i in range(n_plots):
            for j in range(n_plots):
                ax = axes[i, j]
                
                if i == j:
                    # Diagonale : histogrammes
                    ax.hist(votants[:, i], bins=20, alpha=0.6, color='blue', edgecolor='black')
                    ax2 = ax.twinx()
                    ax2.scatter(candidats[:, i], [0]*len(candidats), marker='*', s=200, 
                              color='red', edgecolor='darkred', linewidth=1.5, zorder=5)
                    ax2.set_ylim(-1, 5)
                    ax2.set_yticks([])
                    ax.set_ylabel(noms_dims[i], fontsize=10)
                else:
                    # Hors diagonale : scatter
                    ax.scatter(votants[:, j], votants[:, i], alpha=0.4, s=30, 
                              color='blue', edgecolor='none')
                    ax.scatter(candidats[:, j], candidats[:, i], marker='*', s=300, 
                              color='red', edgecolor='darkred', linewidth=1.5)
                    ax.grid(alpha=0.2, linestyle='--')
                
                if i == n_plots - 1:
                    ax.set_xlabel(noms_dims[j], fontsize=10)
                if j == 0 and i != 0:
                    ax.set_ylabel(noms_dims[i], fontsize=10)
        
        plt.tight_layout()
        plt.show()


# ═══════════════════════════════════════════════════════════════
#  TESTS
# ═══════════════════════════════════════════════════════════════

def _ok(condition, message):
    statut = "✅ OK" if condition else "❌ ÉCHEC"
    print(f"  {statut}  —  {message}")
    return condition


def run_tests():
    """
    Suite de tests couvrant toutes les dimensions et toutes les fonctions.
    Convention vérifiée : agents en lignes → shape (nb, dim).
    """
    np.random.seed(42)
    erreurs = 0

    # ─── SECTION 1 : Formes des générateurs ───────────────────────────────────
    print("\n" + "═"*60)
    print("  SECTION 1 — Formes des tableaux générés  (attendu : (nb, dim))")
    print("═"*60)
    for dim in [1, 2, 3, 5]:
        for gen, nom in [
            (partial(distribution_uniforme,   dim=dim), "uniforme"),
            (partial(distribution_gaussienne, dim=dim), "gaussienne"),
            (partial(distribution_3_pics,     dim=dim), "3_pics"),
            (partial(distribution_trimodale,  dim=dim), "trimodale"),
        ]:
            arr = to_numpy(gen(50))
            ok  = arr.shape == (50, dim)
            if not ok: erreurs += 1
            _ok(ok, f"{nom}(50, dim={dim}) → shape {arr.shape}  (attendu (50, {dim}))")

    # ─── SECTION 2 : Matrice des distances ────────────────────────────────────
    print("\n" + "═"*60)
    print("  SECTION 2 — Matrice des distances")
    print("═"*60)

    np.random.seed(0)
    c1 = distribution_uniforme(4, dim=1)   # (4, 1)
    v1 = distribution_uniforme(7, dim=1)   # (7, 1)
    D     = to_numpy(distances(c1, v1))
    D_ref = np.abs(to_numpy(v1)[:, 0:1] - to_numpy(c1)[:, 0])
    ok = np.allclose(D, D_ref, atol=1e-6)
    if not ok: erreurs += 1
    _ok(ok, "dim=1 : distances == |v - c|  (valeur absolue)")

    for dim in [1, 2, 4]:
        c = distribution_uniforme(5, dim=dim)
        v = distribution_uniforme(9, dim=dim)
        D = to_numpy(distances(c, v))
        ok = D.shape == (9, 5)
        if not ok: erreurs += 1
        _ok(ok, f"distances shape  dim={dim} : {D.shape}  (attendu (9, 5))")

    ok = np.all(to_numpy(distances(distribution_uniforme(6, dim=3),
                                   distribution_uniforme(6, dim=3))) >= 0)
    if not ok: erreurs += 1
    _ok(ok, "Toutes les distances sont ≥ 0  (dim=3)")

    pt = distribution_uniforme(1, dim=3)
    d_self = float(to_numpy(distances(pt, pt))[0, 0])
    ok = abs(d_self) < 1e-9
    if not ok: erreurs += 1
    _ok(ok, f"Distance d'un point à lui-même = {d_self:.2e}  (attendu 0)")

    np.random.seed(7)
    a, b, cp_ = [distribution_uniforme(1, dim=2) for _ in range(3)]
    dab = float(to_numpy(distances(a, b))[0, 0])
    dbc = float(to_numpy(distances(b, cp_))[0, 0])
    dac = float(to_numpy(distances(a, cp_))[0, 0])
    ok  = dac <= dab + dbc + 1e-9
    if not ok: erreurs += 1
    _ok(ok, f"Inégalité triangulaire (dim=2) : {dac:.4f} ≤ {dab:.4f} + {dbc:.4f}")

    # ─── SECTION 3 : Tableau des préférences ──────────────────────────────────
    print("\n" + "═"*60)
    print("  SECTION 3 — Tableau des préférences")
    print("═"*60)
    for dim in [1, 2, 3]:
        nb_c, nb_v = 5, 20
        c = distribution_uniforme(nb_c, dim=dim)
        v = distribution_uniforme(nb_v, dim=dim)
        tdp = to_numpy(tableau_des_preferences(c, v))

        ok = tdp.shape == (nb_v, nb_c)
        if not ok: erreurs += 1
        _ok(ok, f"dim={dim} : shape {tdp.shape}  (attendu ({nb_v}, {nb_c}))")

        ok = all(sorted(tdp[i]) == list(range(1, nb_c+1)) for i in range(nb_v))
        if not ok: erreurs += 1
        _ok(ok, f"dim={dim} : chaque ligne est une permutation de [1..{nb_c}]")

        dm = to_numpy(distances(c, v))
        ok = all(int(np.argmin(dm[i])) == int(np.argmin(tdp[i])) for i in range(nb_v))
        if not ok: erreurs += 1
        _ok(ok, f"dim={dim} : le rang-1 correspond au candidat le plus proche")

    # ─── SECTION 4 : Scrutins ─────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  SECTION 4 — Scrutins (dim=1, 2, 3)")
    print("═"*60)
    scrutins = [SM1T, SM2T, Borda_classique, Copeland,
                partial(Borda_pondéré, poids=[4, 3, 2, 1])]
    for dim in [1, 2, 3]:
        nb_c, nb_v = 4, 30
        tdp = tableau_des_preferences(distribution_uniforme(nb_c, dim=dim),
                                      distribution_uniforme(nb_v, dim=dim))
        for scrutin in scrutins:
            nom = _nom_scrutin(scrutin)
            try:
                res = scrutin(tdp)
                if nom == "SM2T":
                    ok = (isinstance(res, list) and len(res) == 2
                          and res[0][0] >= res[1][0]
                          and all(0 <= r[1] < nb_c for r in res))
                else:
                    ok = (isinstance(res, list) and len(res) == nb_c
                          and all(res[i][0] >= res[i+1][0] for i in range(len(res)-1))
                          and sorted(r[1] for r in res) == list(range(nb_c)))
            except Exception as e:
                ok = False; print(f"    Exception {nom} : {e}")
            if not ok: erreurs += 1
            _ok(ok, f"dim={dim} : {nom} — format et tri corrects")

    # ─── SECTION 5 : Cas manuel dim=1 ─────────────────────────────────────────
    print("\n" + "═"*60)
    print("  SECTION 5 — Cas manuel dim=1  (4 candidats, 5 votants)")
    print("═"*60)
    # Candidats (4, 1) : 0.0  0.3  0.7  1.0
    # Votants   (5, 1) : 0.1  0.1  0.4  0.6  0.9
    #   Favori de chaque votant : 0, 0, 1, 2, 3 → SM1T gagnant = c0
    c_man = cp.array([[0.0], [0.3], [0.7], [1.0]])
    v_man = cp.array([[0.1], [0.1], [0.4], [0.6], [0.9]])
    tdp_man = tableau_des_preferences(c_man, v_man)
    tdp_np  = to_numpy(tdp_man)
    print("  Tableau des préférences (ligne=votant, colonne=rang du candidat j) :")
    for i, row in enumerate(tdp_np):
        print(f"    votant {i} : {list(row)}")
    g = SM1T(tdp_man)[0][1]
    ok = g == 0
    if not ok: erreurs += 1
    _ok(ok, f"SM1T : gagnant = candidat {g}  (attendu 0)")
    print(f"  Borda classique : {Borda_classique(tdp_man)}")

    # ─── SECTION 6 : Satisfaction ─────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  SECTION 6 — Score de satisfaction")
    print("═"*60)
    c_sat = cp.array([[0.5]])          # (1, 1)
    v_sat = distribution_uniforme(20, dim=1)
    s = satisfaction_relative(0, distances(c_sat, v_sat))
    ok = abs(s - 1.0) < 1e-9
    if not ok: erreurs += 1
    _ok(ok, f"1 seul candidat → satisfaction = {s:.6f}  (attendu 1.0)")

    for dim in [1, 2, 3]:
        c = distribution_uniforme(5, dim=dim)
        v = distribution_uniforme(50, dim=dim)
        s = satisfaction_relative(SM1T(tableau_des_preferences(c, v))[0][1],
                                  distances(c, v))
        ok = 0.0 <= s <= 1.0 + 1e-9
        if not ok: erreurs += 1
        _ok(ok, f"dim={dim} : satisfaction ∈ [0,1]  (obtenu {s:.4f})")

    # ─── SECTION 7 : Simulation complète multi-dim ────────────────────────────
    print("\n" + "═"*60)
    print("  SECTION 7 — Simulation complète (manipulabilité multi-dim)")
    print("═"*60)
    for dim in [1, 2, 3]:
        gen = partial(distribution_uniforme, dim=dim)
        try:
            res = simulation_manipulabilite(
                SM1T, nb_candidats=3, nb_votants=30,
                strategie_manipulation=manipulation_naïve,
                generateur_votants=gen, genenerateur_candidats=gen,
                nb_simulations=50)
            ok = 0.0 <= res["taux_manipulation"] <= 1.0
        except Exception as e:
            ok = False; print(f"    Exception dim={dim} : {e}")
        if not ok: erreurs += 1
        _ok(ok, f"dim={dim} : simulation SM1T — taux ∈ [0,1]")

    # ─── SECTION 8 : Déterminisme ─────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  SECTION 8 — Déterminisme")
    print("═"*60)
    np.random.seed(123)
    tdp_d = tableau_des_preferences(distribution_uniforme(4, dim=1),
                                    distribution_uniforme(40, dim=1))
    g1, g2 = gagnant(SM1T(tdp_d)), gagnant(SM1T(tdp_d))
    ok = g1 == g2
    if not ok: erreurs += 1
    _ok(ok, f"Même tableau → même gagnant  ({g1} == {g2})")

    # ─── SECTION 9 : Générateurs multi-dim ───────────────────────────────────
    print("\n" + "═"*60)
    print("  SECTION 9 — Générateurs dim > 1  (shape attendu : (100, dim))")
    print("═"*60)
    for dim in [2, 3, 4]:
        for gen, nom in [
            (partial(distribution_uniforme,   dim=dim), "uniforme"),
            (partial(distribution_gaussienne, dim=dim), "gaussienne"),
            (partial(distribution_3_pics,     dim=dim), "3_pics"),
            (partial(distribution_trimodale,  dim=dim), "trimodale"),
        ]:
            arr = to_numpy(gen(100))
            ok  = arr.shape == (100, dim)
            if not ok: erreurs += 1
            _ok(ok, f"{nom} dim={dim} → shape {arr.shape}  (attendu (100, {dim}))")

    # ─── Bilan ────────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    if erreurs == 0:
        print("  🎉  Tous les tests sont passés avec succès !")
    else:
        print(f"  ⚠️   {erreurs} test(s) en échec — voir les détails ci-dessus.")
    print("═"*60 + "\n")
    return erreurs == 0


# ─────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_tests()

    print("\n" + "═"*60)
    print("  EXEMPLES D'UTILISATION")
    print("═"*60)

    print("\n▶  Exemple 1 — Scrutins en dim=1")
    c = distribution_uniforme(5, dim=1)
    v = distribution_uniforme(200, dim=1)
    tdp = tableau_des_preferences(c, v)
    print(f"  SM1T     : {SM1T(tdp)}")
    print(f"  Borda    : {Borda_classique(tdp)}")
    print(f"  Copeland : {Copeland(tdp)}")

    print("\n▶  Exemple 2 — Scrutins en dim=2")
    c2 = distribution_uniforme(5, dim=2)
    v2 = distribution_uniforme(200, dim=2)
    tdp2 = tableau_des_preferences(c2, v2)
    print(f"  SM1T     : {SM1T(tdp2)}")
    print(f"  Copeland : {Copeland(tdp2)}")

    print("\n▶  Exemple 3 — Simulation manipulabilité dim=3")
    gen3D = partial(distribution_uniforme, dim=3)
    simulation_manipulabilite(Borda_classique, nb_candidats=4, nb_votants=50,
                               generateur_votants=gen3D, genenerateur_candidats=gen3D,
                               nb_simulations=100)

    print("\n▶  Exemple 4 — plot_distribution dim=1 (uniforme vs gaussienne)")
    plot_distribution(
        distribution_uniforme(500, dim=1),
        distribution_gaussienne(500, centre=0.5, variance=0.15, dim=1),
        etiquettes=["Uniforme", "Gaussienne"],
        noms_dims=["Position idéologique"],
        titre="Comparaison dim=1 — uniforme vs gaussienne"
    )

    print("\n▶  Exemple 5 — plot_distribution dim=2 (votants + candidats)")
    v2_viz = distribution_gaussienne(400, centre=0.5, variance=0.2, dim=2)
    c2_viz = distribution_uniforme(5, dim=2)
    plot_distribution(
        v2_viz,
        etiquettes=["Votants"],
        noms_dims=["Économie", "Social"],
        titre="Votants gaussiens + candidats uniformes (dim=2)",
        afficher_candidats=c2_viz
    )

    print("\n▶  Exemple 6 — plot_distribution dim=3 (matrice pairwise)")
    plot_distribution(
        distribution_uniforme(300, dim=3),
        distribution_3_pics(300, dim=3),
        etiquettes=["Uniforme", "3 pics"],
        noms_dims=["Axe A", "Axe B", "Axe C"],
        titre="Matrice pairwise dim=3"
    )

