import random
import numpy as np
import time
import matplotlib.pyplot as plt
import itertools
from functools import partial

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
# UTILITAIRE : conversion GPU → CPU
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
# ─────────────────────────────────────────────

def distribution_uniforme(nb,dim=1):
    return cp.array([np.random.random(nb) for i in range(dim)])

def distribution_gaussienne(nb, centre=0, variance=1,dim=1):
    return cp.array([np.random.normal(centre, variance, nb) for i in range(dim)])



def distribution_3_pics(nb, pics=(0.2, 0.5, 0.8), variances=(0.05, 0.05, 0.05), poids=(1, 1, 1),dim=1):
    """
    Génère une distribution avec 3 pics gaussiens.

    Paramètres :
        nb        : nombre de points à générer
        pics      : tuple de 3 positions (centres des pics)
        variances : tuple de 3 variances (largeur de chaque pic)
        poids     : tuple de 3 poids (importance relative de chaque pic)
                    — n'ont pas besoin d'être normalisés

    Retourne :
        tableau CuPy 1-D de taille nb
    """
    poids      = np.array(poids, dtype=float)
    poids      = poids / poids.sum()                      # normalisation
    tailles    = np.round(poids * nb).astype(int)
    tailles[-1] = nb - tailles[:-1].sum()                 # ajuster pour avoir exactement nb points

    échantillons = np.concatenate([
        np.random.normal(loc=pics[k], scale=np.sqrt(variances[k]), size=tailles[k])
        for k in range(3)
    ])
    np.random.shuffle(échantillons)
    return cp.array(échantillons)




def distribution_trimodale(nb, centres=[0.2, 0.5, 0.8], variances=[0.05, 0.05, 0.05], amplitudes=[1, 1, 1]):
    """
    Génère une distribution avec 3 pics gaussiens.

    Paramètres :
        nb         : nombre de points à générer
        centres    : positions des 3 pics (dans [0, 1] par défaut)
        variances  : écart-type de chaque pic (un par pic)
        amplitudes : poids relatifs de chaque pic
                     ex: [2, 1, 1] → le 1er pic a 2x plus de points

    Retourne :
        tableau CuPy 1-D de taille nb
    """
    amplitudes = np.array(amplitudes, dtype=float)
    amplitudes = amplitudes / amplitudes.sum()
    tailles    = np.round(amplitudes * nb).astype(int)
    tailles[-1] = nb - tailles[:-1].sum()

    points = np.concatenate([
        np.random.normal(c, v, n)
        for c, v, n in zip(centres, variances, tailles)
    ])
    np.random.shuffle(points)
    return cp.array(points)


# ─────────────────────────────────────────────
# TABLEAU DES PRÉFÉRENCES
# ─────────────────────────────────────────────

def distances(candidats, votants):
    c = to_cp(candidats)
    v = to_cp(votants)
    return cp.abs(v[:, None] - c[None, :])

def tableau_des_preferences(candidats, votants):
    c = to_cp(candidats)
    v = to_cp(votants)
    dist  = cp.abs(v[:, None] - c[None, :])
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
    scores = cp.sum(nb_candidats - sondage, axis=0)
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

def test_SM1T_SM2T(nb_candidats, nb_votants, nb_simulations=1000):
    plt.close('all')
    cand_range = list(range(2, nb_candidats + 1))
    vot_range  = list(range(3, nb_votants + 1))
    lst = []
    t0 = time.time()
    for k in cand_range:
        for j in vot_range:
            a = 0
            for _ in range(nb_simulations):
                candidats = distribution_uniforme(k)
                votants   = distribution_uniforme(j)
                tdp = tableau_des_preferences(candidats, votants)
                T1, T2 = SM2T(tdp)
                if T1[0][1] != T2[0][1]:
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
    plt.title("Divergence SM1T / SM2T")
    plt.show()


# ─────────────────────────────────────────────
# SCORE DE SATISFACTION
# ─────────────────────────────────────────────

def satisfaction_relative(gagnant_idx, dist, votants=None):
    """
    Satisfaction relative : rapport entre la distance au favori absolu
    et la distance au gagnant réel.

    score_i = dist(votant_i, favori) / dist(votant_i, gagnant)

    Retourne un float dans [0, 1] — 1 = le gagnant est le favori de tous.
    """
    dist = to_cp(dist)
    dist_favori  = cp.min(dist, axis=1)
    dist_gagnant = dist[:, gagnant_idx]
    scores = cp.where(
        dist_gagnant > 0,
        dist_favori / dist_gagnant,
        cp.ones_like(dist_gagnant)
    )
    return float(cp.mean(scores))

def satisfaction_maximale(dist):
    max = 0
    for i in range(dist.shape[1]):
        sat_i = satisfaction_relative(i, dist)
        if sat_i > max:
            max = sat_i
    return max

# def satisfaction_réel(gagnant_idx, dist, votants=None):
#     """
#     Satisfaction réelle : position du gagnant sur l'axe
#     [candidat imaginaire optimal → candidat réel le plus éloigné].

#     Le candidat imaginaire optimal est placé à la moyenne des positions
#     des votants. Sa distance moyenne à l'électorat est la déviation
#     absolue moyenne : d_optimal = mean_i(|votant_i − mean(votants)|).

#     score = 1 − (d_gagnant − d_optimal) / (d_pire − d_optimal)

#     Retourne un float dans [0, 1] :
#         1 = le gagnant est au centre de gravité de l'électorat
#         0 = le gagnant est le candidat réel le plus éloigné de l'électorat

#     Paramètres :
#         gagnant_idx : indice 0-based du candidat gagnant
#         dist        : tableau 2-D (nb_votants × nb_candidats)
#         votants     : tableau 1-D des positions des votants
#                       si None, fallback sur le meilleur candidat réel
#     """
#     dist          = to_cp(dist)
#     dist_moyennes = to_numpy(cp.mean(dist, axis=0))   # (nb_candidats,)

#     if votants is not None:
#         v         = to_numpy(to_cp(votants))
#         mu        = float(np.mean(v))
#         d_optimal = float(np.mean(np.abs(v - mu)))
#     else:
#         d_optimal = float(np.min(dist_moyennes))     # fallback : meilleur candidat réel

#     d_pire    = float(np.max(dist_moyennes))
#     d_gagnant = float(dist_moyennes[gagnant_idx])

#     ecart_total = d_pire - d_optimal
#     if ecart_total == 0:
#         return 1.0

#     return float(max(0.0, 1.0 - (d_gagnant - d_optimal) / ecart_total))


# ─────────────────────────────────────────────
# MANIPULABILITÉ SUR UN VOTE FIXÉ
# ─────────────────────────────────────────────

def _construire_bulletin_strategique(bulletins_coalition_np, favori, gagnant, nb_candidats):
    """
    Construit le bulletin stratégique unique pour une coalition :
        - favori  → rang 1
        - gagnant → rang nb_candidats (dernier)
        - autres  → ordre honnête moyen de la coalition (rangs 2 … n-1)
    """
    milieu      = [c for c in range(nb_candidats) if c != favori and c != gagnant]
    rang_moyen  = {c: float(np.mean(bulletins_coalition_np[:, c])) for c in milieu}
    milieu_trié = sorted(milieu, key=lambda c: rang_moyen[c])
    bulletin = [0] * nb_candidats
    bulletin[favori]  = 1
    bulletin[gagnant] = nb_candidats
    for rang, cand in enumerate(milieu_trié, start=2):
        bulletin[cand] = rang
    return bulletin


def manipulation_naïve(tdp_honnete, dist, scrutin):
    """
    Teste si un vote est manipulable par une stratégie naïve.

    Chaque coalition perdante applique un bulletin stratégique unique :
      - favori  → rang 1
      - cible   → rang dernier
      - autres  → ordre honnête moyen conservé

    Pour SM2T, la cible est le gagnant du T1 (le candidat avec le plus de
    voix au premier tour) plutôt que le gagnant final T2 — car c'est au T1
    que la manipulation est possible en affaiblissant le leader.

    Retourne True si au moins une coalition réussit, False sinon.
    """
    tdp_honnete = to_cp(tdp_honnete)
    dist        = to_cp(dist)
    nb_votants, nb_candidats = tdp_honnete.shape
    tdp_np = to_numpy(tdp_honnete)

    résultats_honnêtes = scrutin(tdp_honnete)
    gagnant_honnete    = résultats_honnêtes[0][1]

    # Pour SM2T : cibler le gagnant T1 plutôt que le gagnant T2
    if _nom_scrutin(scrutin) == "SM2T":
        votes_T1   = cp.argmin(tdp_honnete, axis=1)
        scores_T1  = to_numpy(cp.bincount(votes_T1, minlength=nb_candidats))
        cible      = int(np.argmax(scores_T1))   # leader au T1
    else:
        cible = gagnant_honnete

    favoris_tous       = cp.argmin(tdp_honnete, axis=1)
    perdants_mask      = favoris_tous != gagnant_honnete
    indices_perdants   = to_numpy(cp.where(perdants_mask)[0])
    candidats_perdants = sorted(set(int(favoris_tous[i]) for i in indices_perdants))

    for favori in candidats_perdants:
        if favori == cible:
            continue   # la coalition qui soutient déjà le leader n'a pas d'intérêt à manipuler
        indices_coalition = to_numpy(cp.where(favoris_tous == favori)[0])
        bulletin_strat = _construire_bulletin_strategique(
            tdp_np[indices_coalition], favori, cible, nb_candidats
        )
        bulletin_gpu = cp.array(bulletin_strat, dtype=cp.int32)
        tdp_test = tdp_honnete.copy()
        for idx in indices_coalition:
            tdp_test[idx] = bulletin_gpu
        if scrutin(tdp_test)[0][1] == favori:
            return True

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
    Estime le taux de manipulabilité d'un scrutin sur nb_simulations paysages.

    À chaque simulation, un nouveau vote est généré et
    strategie_manipulation(tdp, dist, scrutin) est appelée.
    Chaque True renvoyé compte comme une manipulation réussie.

    Paramètres :
        scrutin                : fonction de scrutin à tester
        nb_candidats           : nombre de candidats
        nb_votants             : nombre de votants
        strategie_manipulation : fonction (tdp, dist, scrutin) → bool
                                 par défaut : manipulation_naïve
        generateur_votants     : générateur de positions des votants
        genenerateur_candidats : générateur de positions des candidats
        nb_simulations         : nombre de paysages électoraux testés
        candidats_fixes        : positions fixes des candidats (optionnel)

    Retourne un dict avec :
        taux_manipulation   : fraction de simulations manipulables
        score_honnete_moyen : satisfaction moyenne sous vote sincère
        scores_honnetes     : liste brute des scores de satisfaction honnêtes
    """
    start            = time.perf_counter()
    nb_manipulations = 0
    scores_honnetes  = []

    for _ in range(nb_simulations):
        # ── 1. Génération du paysage ──────────────────────────────────────────
        if candidats_fixes is not None:
            candidats    = to_cp(candidats_fixes)
            nb_candidats = len(candidats_fixes)
        else:
            candidats = genenerateur_candidats(nb_candidats)

        votants = generateur_votants(nb_votants)
        dist    = distances(candidats, votants)
        tdp     = tableau_des_preferences(candidats, votants)

        # ── 2. Score de satisfaction honnête ──────────────────────────────────
        résultats_honnêtes = scrutin(tdp)
        gagnant_honnete    = résultats_honnêtes[0][1]   # 0-based
        scores_honnetes.append(satisfaction_relative(gagnant_honnete, dist))

        # ── 3. Test de manipulabilité via la stratégie fournie ────────────────
        if strategie_manipulation(tdp, dist, scrutin):
            nb_manipulations += 1

    # ── 4. Synthèse ───────────────────────────────────────────────────────────
    end       = time.perf_counter()
    taux      = nb_manipulations / nb_simulations
    s_hon_moy = float(np.mean(scores_honnetes)) if scores_honnetes else 0.0

    print(f"\n{'─'*55}")
    print(f"  Scrutin               : {scrutin.__name__}")
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


def vote_utile(tdp_honnete, dist, scrutin):
    """
    Stratégie de vote utile — fonctionne pour tous les scrutins.

    Pour SM2T, la viabilité et le gagnant de référence sont calculés sur
    les scores T1 (et non T2), ce qui permet aux groupes éliminés du T1
    de rediriger leurs voix vers un candidat qui peut passer le premier tour.

    Algorithme :
      1. Calculer les scores par candidat (T1 pour SM2T, résultat direct sinon).
      2. Identifier les groupes "éliminés" : score[favori] + taille ≤ score[leader]
      3. Chaque groupe éliminé vote pour son candidat viable préféré.
      4. Vérifier si le nouveau scrutin change le gagnant final.

    Signature : (tdp, dist, scrutin) → bool
    """
    tdp_honnete = to_cp(tdp_honnete)
    dist        = to_cp(dist)
    nb_votants, nb_candidats = tdp_honnete.shape
    tdp_np = to_numpy(tdp_honnete)

    # ── Scores de référence ───────────────────────────────────────────────────
    if _nom_scrutin(scrutin) == "SM2T":
        votes_T1           = cp.argmin(tdp_honnete, axis=1)
        scores_T1          = to_numpy(cp.bincount(votes_T1, minlength=nb_candidats))
        T1_trié            = sorted(range(nb_candidats), key=lambda c: scores_T1[c], reverse=True)
        seuil_top2         = scores_T1[T1_trié[1]]   # score du 2ème au T1
        gagnant_honnete    = scrutin(tdp_honnete)[0][1]

        favoris_tous = cp.argmin(tdp_honnete, axis=1)
        groupes = {}
        for c in range(nb_candidats):
            indices = to_numpy(cp.where(favoris_tous == c)[0])
            if len(indices) > 0:
                groupes[c] = indices

        # Éliminé = ne peut pas atteindre le top 2 même avec toutes ses voix
        eliminés = {c for c, idx in groupes.items()
                    if scores_T1[c] + len(idx) <= seuil_top2
                    and c != gagnant_honnete}

        # Viable = non-éliminé, non-gagnant, ET peut entrer dans le top 2
        # si un groupe éliminé lui transfère ses voix
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

        tdp_test = to_cp(tdp_test_np)
        return scrutin(tdp_test)[0][1] != gagnant_honnete

    else:
        résultats_honnêtes = scrutin(tdp_honnete)
        gagnant_honnete    = résultats_honnêtes[0][1]
        score_leader       = résultats_honnêtes[0][0]
        score_par_candidat = np.zeros(nb_candidats)
        for score, idx in résultats_honnêtes:
            score_par_candidat[idx] = score

        favoris_tous = cp.argmin(tdp_honnete, axis=1)
        groupes = {}
        for c in range(nb_candidats):
            indices = to_numpy(cp.where(favoris_tous == c)[0])
            if len(indices) > 0:
                groupes[c] = indices

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

        tdp_test = to_cp(tdp_test_np)
        return scrutin(tdp_test)[0][1] != gagnant_honnete

def optimisation_vote_groupe(tdp_honnete, dist, scrutin):
    """
    Stratégie de manipulation par optimisation exhaustive.

    Pour chaque coalition perdante, explore toutes les permutations possibles
    du bulletin (en gardant uniquement la contrainte que le favori est en rang 1)
    et retourne True dès qu'une permutation fait gagner le favori.

    C'est la stratégie la plus puissante — elle trouve toutes les manipulations
    possibles, là où manipulation_naïve n'en teste qu'une seule par coalition.

    Signature identique à manipulation_naïve : (tdp, dist, scrutin) → bool
    """
    tdp_honnete = to_cp(tdp_honnete)
    dist        = to_cp(dist)
    nb_votants, nb_candidats = tdp_honnete.shape
    tdp_np = to_numpy(tdp_honnete)

    # ── Résultat honnête ──────────────────────────────────────────────────────
    résultats_honnêtes = scrutin(tdp_honnete)
    gagnant_honnete    = résultats_honnêtes[0][1]   # 0-based

    favoris_tous       = cp.argmin(tdp_honnete, axis=1)
    perdants_mask      = favoris_tous != gagnant_honnete
    indices_perdants   = to_numpy(cp.where(perdants_mask)[0])
    candidats_perdants = sorted(set(int(favoris_tous[i]) for i in indices_perdants))

    # ── Tester chaque coalition avec toutes les permutations ──────────────────
    for favori in candidats_perdants:
        indices_coalition   = to_numpy(cp.where(favoris_tous == favori)[0])
        bulletins_originaux = tdp_honnete[indices_coalition].copy()

        # Tous les candidats sauf le favori — on permute leur ordre
        autres = [c for c in range(nb_candidats) if c != favori]

        for perm in itertools.permutations(autres):
            # Construire le bulletin : favori=1, puis les autres dans l'ordre de la permutation
            bulletin = [0] * nb_candidats
            bulletin[favori] = 1
            for rang, cand in enumerate(perm, start=2):
                bulletin[cand] = rang

            bulletin_gpu = cp.array(bulletin, dtype=cp.int32)
            tdp_test = tdp_honnete.copy()
            for idx in indices_coalition:
                tdp_test[idx] = bulletin_gpu

            résultat_test = scrutin(tdp_test)
            if résultat_test[0][1] == favori:
                return True

        # Restaurer avant de passer à la coalition suivante
        tdp_honnete[indices_coalition] = bulletins_originaux

    return False




def comparer_manipulabilite(scrutins, nb_candidats, nb_votants,
                             strategie_manipulation=manipulation_naïve,
                             generateur_votants=distribution_uniforme,
                             generateur_candidats=distribution_uniforme,
                             nb_simulations=1000,
                             candidats_fixes=None):
    """
    Compare le taux de manipulabilité de plusieurs méthodes de vote
    sur les mêmes paysages électoraux.

    Paramètres :
        scrutins               : liste de fonctions de scrutin à comparer
        nb_candidats           : nombre de candidats
        nb_votants             : nombre de votants
        strategie_manipulation : fonction (tdp, dist, scrutin) → bool
        generateur_votants     : générateur de positions des votants
        generateur_candidats   : générateur de positions des candidats
        nb_simulations         : nombre de paysages électoraux testés
        candidats_fixes        : positions fixes des candidats (optionnel)

    Retourne un dict { nom_scrutin → taux_manipulation }
    """
    start = time.perf_counter()

    # Pré-générer tous les paysages une seule fois — même base pour tous les scrutins
    paysages = []
    for _ in range(nb_simulations):
        if candidats_fixes is not None:
            candidats = to_cp(candidats_fixes)
            n_cand    = len(candidats_fixes)
        else:
            candidats = generateur_candidats(nb_candidats)
            n_cand    = nb_candidats
        votants = generateur_votants(nb_votants)
        dist    = distances(candidats, votants)
        tdp     = tableau_des_preferences(candidats, votants)
        paysages.append((tdp, dist, n_cand))

    # Tester chaque scrutin sur les mêmes paysages
    résultats = {}
    for scrutin in scrutins:
        nb_manips    = 0
        sats_honnetes = []
        for tdp, dist, _ in paysages:
            # Satisfaction honnête
            res_honnete = scrutin(tdp)
            gagnant_h   = res_honnete[0][1]
            sats_honnetes.append(satisfaction_relative(gagnant_h, dist))
            # Manipulabilité
            if strategie_manipulation(tdp, dist, scrutin):
                nb_manips += 1
        résultats[scrutin.__name__] = {
            "taux_manipulation"  : nb_manips / nb_simulations,
            "satisfaction_moyenne": float(np.mean(sats_honnetes)),
        }

    # ── Affichage console ─────────────────────────────────────────────────────
    end = time.perf_counter()
    print(f"\n{'═'*60}")
    print(f"  Comparaison de manipulabilité & satisfaction")
    print(f"  Stratégie   : {strategie_manipulation.__name__}")
    print(f"  Candidats / Votants : {nb_candidats} / {nb_votants}")
    print(f"  Simulations : {nb_simulations}")
    print(f"{'═'*60}")
    print(f"  {'Scrutin':<22} {'Manip':>8}  {'Satisfaction':>13}")
    print(f"  {'─'*22} {'─'*8}  {'─'*13}")
    for nom, vals in sorted(résultats.items(), key=lambda x: x[1]["taux_manipulation"]):
        print(f"  {nom:<22} {vals['taux_manipulation']:>8.4f}  "
              f"{vals['satisfaction_moyenne']:>13.4f}")
    print(f"{'─'*60}")
    print(f"  Durée : {end - start:.2f}s")
    print(f"{'═'*60}\n")

    _plot_comparaison_manipulabilite(résultats, strategie_manipulation.__name__)

    return résultats


def _plot_comparaison_manipulabilite(résultats, nom_strategie=""):
    # Trier par taux de manipulation croissant
    noms       = sorted(résultats, key=lambda n: résultats[n]["taux_manipulation"])
    taux_manip = [résultats[n]["taux_manipulation"]   for n in noms]
    taux_sat   = [résultats[n]["satisfaction_moyenne"] for n in noms]

    x      = np.arange(len(noms))
    width  = 0.35
    height = max(5, len(noms) * 0.9)

    fig, ax1 = plt.subplots(figsize=(max(8, len(noms) * 2), height))
    fig.suptitle(f"Manipulabilité & Satisfaction — {nom_strategie}",
                 fontsize=13, fontweight='bold')

    # ── Barres manipulabilité (axe gauche, rouge) ─────────────────────────────
    bars1 = ax1.bar(x - width / 2, taux_manip, width,
                    color='tomato', alpha=0.85, edgecolor='white', label='Taux de manipulation')
    ax1.set_ylabel("Taux de manipulation", fontsize=11, color='tomato')
    ax1.tick_params(axis='y', labelcolor='tomato')
    ax1.set_ylim([0, min(1.0, max(taux_manip) * 1.35)])

    for bar, v in zip(bars1, taux_manip):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{v:.3f}", ha='center', va='bottom', fontsize=9,
                 fontweight='bold', color='tomato')

    # ── Barres satisfaction (axe droit, bleu) ─────────────────────────────────
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, taux_sat, width,
                    color='steelblue', alpha=0.85, edgecolor='white', label='Satisfaction moyenne')
    ax2.set_ylabel("Satisfaction moyenne", fontsize=11, color='steelblue')
    ax2.tick_params(axis='y', labelcolor='steelblue')
    sat_min = max(0, min(taux_sat) * 0.92)
    ax2.set_ylim([sat_min, min(1.0, max(taux_sat) * 1.08)])

    for bar, v in zip(bars2, taux_sat):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f"{v:.3f}", ha='center', va='bottom', fontsize=9,
                 fontweight='bold', color='steelblue')

    # ── Axes & légende ────────────────────────────────────────────────────────
    ax1.set_xticks(x)
    ax1.set_xticklabels(noms, rotation=15, ha='right', fontsize=10)
    ax1.set_xlabel("Méthode de vote", fontsize=11)
    ax1.grid(axis='y', alpha=0.25, linestyle='--')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# OPTIMISATION DES COEFFICIENTS DE BORDA
# ─────────────────────────────────────────────

def optimiser_borda(nb_candidats, nb_votants,
                    generateur_votants=distribution_uniforme,
                    generateur_candidats=distribution_uniforme,
                    metriques=None,
                    strategies_manipulation=None,
                    nb_simulations=300,
                    nb_simulations_manip=200,
                    méthode="differential_evolution",
                    verbose=True):
    """
    Optimise les coefficients de Borda pour maximiser la satisfaction moyenne,
    sur plusieurs métriques, et compare la manipulabilité avec Borda classique.

    Paramètres :
        nb_candidats              : nombre de candidats
        nb_votants                : nombre de votants
        generateur_votants        : générateur de positions des votants
        generateur_candidats      : générateur de positions des candidats
        metriques                 : liste de fonctions (gagnant_idx, dist, votants) → float
                                    par défaut : [satisfaction_relative]
        strategies_manipulation   : liste de fonctions (tdp, dist, scrutin) → bool
                                    si None, pas de graphe de manipulation
        nb_simulations            : paysages pré-générés pour l'optimisation
        nb_simulations_manip      : simulations pour le calcul de manipulabilité
        méthode                   : "differential_evolution" ou "nelder-mead"
        verbose                   : affiche la progression

    Retourne :
        dict {
            metrique.__name__ → {
                "poids_optimaux"              : liste de nb_candidats poids,
                "satisfaction"                : satisfaction optimale,
                "satisfaction_borda_classique": satisfaction baseline,
                "historique"                  : convergence de l'optimisation,
            }
        }
        + clé "manipulation" si strategies_manipulation est fourni
    """
    from scipy.optimize import differential_evolution, minimize

    if metriques is None:
        metriques = [satisfaction_relative]
    if strategies_manipulation is None:
        strategies_manipulation = []

    # ── Pré-générer les paysages ──────────────────────────────────────────────
    if verbose:
        print(f"  Pré-génération de {nb_simulations} paysages…")
    paysages = []
    for _ in range(nb_simulations):
        candidats = generateur_candidats(nb_candidats)
        votants   = generateur_votants(nb_votants)
        dist      = distances(candidats, votants)
        tdp       = tableau_des_preferences(candidats, votants)
        paysages.append((tdp, dist, votants))

    résultats = {}

    # ── Optimisation par métrique ─────────────────────────────────────────────
    for metrique in metriques:
        nm = metrique.__name__
        if verbose:
            print(f"\n{'─'*55}")
            print(f"  Métrique : {nm}")

        historique = []

        def objectif(params):
            milieu = np.clip(np.sort(params)[::-1], 0, 1)
            poids  = np.concatenate([[1.0], milieu, [0.0]])
            sats = []
            for tdp, dist, votants in paysages:
                res = Borda_pondéré(tdp, poids)
                sats.append(metrique(res[0][1], dist, votants))
            val = -float(np.mean(sats))
            historique.append(-val)
            return val

        poids_classique = np.array([nb_candidats - 1 - r for r in range(nb_candidats)],
                                   dtype=float)
        poids_classique = poids_classique / poids_classique[0]

        sats_classique = [metrique(Borda_classique(tdp)[0][1], dist, votants)
                          for tdp, dist, votants in paysages]
        sat_classique  = float(np.mean(sats_classique))

        if verbose:
            print(f"  Baseline Borda classique : {sat_classique:.4f}")
            print(f"  Optimisation en cours ({méthode})…")

        start = time.perf_counter()
        n_params = nb_candidats - 2

        if n_params == 0:
            poids_optimaux = [1.0, 0.0]
            sat_optimale   = sat_classique
        else:
            bornes = [(0.0, 1.0)] * n_params
            if méthode == "differential_evolution":
                res_opt = differential_evolution(
                    objectif, bornes,
                    maxiter=200, tol=1e-4, seed=42, workers=1,
                    callback=(lambda xk, convergence: print(
                        f"\r    sat={historique[-1]:.4f}  (best={max(historique):.4f})",
                        end="", flush=True)) if verbose else None
                )
            else:
                x0 = np.linspace(1.0, 0.0, n_params + 2)[1:-1]
                res_opt = minimize(objectif, x0, method="Nelder-Mead",
                                   options={"maxiter": 5000, "xatol": 1e-4, "fatol": 1e-4})
            if verbose:
                print()
            milieu         = np.clip(np.sort(res_opt.x)[::-1], 0, 1)
            poids_optimaux = list(np.concatenate([[1.0], milieu, [0.0]]))
            sat_optimale   = -res_opt.fun

        end = time.perf_counter()

        if verbose:
            print(f"\n{'═'*55}")
            print(f"  Borda classique : {sat_classique:.4f}")
            print(f"  Borda optimal   : {sat_optimale:.4f}  ({sat_optimale - sat_classique:+.4f})")
            print(f"  Poids optimaux  : " +
                  "  ".join(f"[r{r+1}] {p:.3f}" for r, p in enumerate(poids_optimaux)))
            print(f"  Durée           : {end - start:.1f}s")
            print(f"{'═'*55}")

        résultats[nm] = {
            "poids_optimaux"              : poids_optimaux,
            "satisfaction"                : sat_optimale,
            "satisfaction_borda_classique": sat_classique,
            "historique"                  : historique,
            "poids_classique"             : list(poids_classique),
        }

    # ── Calcul de manipulabilité ──────────────────────────────────────────────
    manip_résultats = {}   # {strategie_name → {scrutin_name → taux}}
    if strategies_manipulation:
        if verbose:
            print(f"\n  Calcul manipulabilité ({nb_simulations_manip} simulations)…")

        scrutins_a_tester = {"Borda_classique": Borda_classique}
        for nm, res in résultats.items():
            poids = res["poids_optimaux"]
            nom   = f"Borda_opt_{nm}"
            scrutins_a_tester[nom] = partial(Borda_pondéré, poids=poids)

        for strategie in strategies_manipulation:
            ns = strategie.__name__
            manip_résultats[ns] = {}
            for nom_scrutin, scrutin_fn in scrutins_a_tester.items():
                nb_manip = 0
                for _ in range(nb_simulations_manip):
                    candidats = generateur_candidats(nb_candidats)
                    votants   = generateur_votants(nb_votants)
                    dist_m    = distances(candidats, votants)
                    tdp       = tableau_des_preferences(candidats, votants)
                    if strategie(tdp, dist_m, scrutin_fn):
                        nb_manip += 1
                manip_résultats[ns][nom_scrutin] = nb_manip / nb_simulations_manip
                if verbose:
                    print(f"    {ns} / {nom_scrutin} : {nb_manip / nb_simulations_manip:.3f}")

        résultats["manipulation"] = manip_résultats

    # ── Plots ─────────────────────────────────────────────────────────────────
    nb_metriques  = len(metriques)
    has_manip     = bool(strategies_manipulation)
    nb_lignes     = nb_metriques + (1 if has_manip else 0)

    fig, axes = plt.subplots(
        nb_lignes, 1,
        figsize=(10, max(4, 9.5 / nb_lignes) * nb_lignes),
        squeeze=False, dpi=96
    )
    fig.suptitle(
        f"Optimisation Borda — {nb_candidats} candidats · {nb_votants} votants · {nb_simulations} simulations",
        fontsize=14, fontweight='bold'
    )

    couleurs = {"classique": "tomato", "optimal": "steelblue"}
    cmap     = plt.cm.get_cmap("tab10")

    for row, metrique in enumerate(metriques):
        nm  = metrique.__name__
        res = résultats[nm]
        poids_opt = res["poids_optimaux"]
        poids_cls = res["poids_classique"]
        sat_opt   = res["satisfaction"]
        sat_cls   = res["satisfaction_borda_classique"]
        rangs     = list(range(1, nb_candidats + 1))

        ax = axes[row][0]
        ax.plot(rangs, poids_opt, 'o-', color=couleurs["optimal"], linewidth=2,
                markersize=7, label=f'Borda optimal — {nm} ({sat_opt:.3f})')
        ax.plot(rangs, poids_cls, 's--', color=couleurs["classique"], linewidth=1.5,
                markersize=6, label=f'Borda classique ({sat_cls:.3f})')
        ax.set_xlabel("Rang du candidat", fontsize=10)
        ax.set_ylabel("Poids", fontsize=10)
        ax.set_title(f"Profil des poids — {nm}", fontsize=11)
        ax.set_xticks(rangs)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, linestyle='--')

    # Graphe manipulation
    if has_manip:
        ax_manip = axes[nb_metriques][0]
        noms_scrutins   = list(next(iter(manip_résultats.values())).keys())
        noms_strategies = list(manip_résultats.keys())
        nb_s  = len(noms_scrutins)
        nb_st = len(noms_strategies)
        largeur    = 0.7 / nb_st
        x          = np.arange(nb_s, dtype=float)
        couleurs_s = cmap(np.linspace(0, 0.9, nb_s))

        for j, ns in enumerate(noms_strategies):
            taux   = [manip_résultats[ns][n] for n in noms_scrutins]
            offset = (j - (nb_st - 1) / 2) * largeur
            bars   = ax_manip.bar(x + offset, taux, largeur * 0.85,
                                  color=[couleurs_s[k] for k in range(nb_s)],
                                  alpha=0.80, edgecolor='white', label=ns)
            for bar, t in zip(bars, taux):
                ax_manip.text(bar.get_x() + bar.get_width() / 2,
                              bar.get_height() + 0.01,
                              f"{t:.3f}", ha='center', va='bottom',
                              fontsize=8, fontweight='bold')

        ax_manip.set_xticks(x)
        ax_manip.set_xticklabels(noms_scrutins, fontsize=9, rotation=15, ha='right')
        ax_manip.set_ylabel("Taux de manipulation", fontsize=10)
        ax_manip.set_title("Manipulabilité : Borda classique vs optimal(s)", fontsize=11)
        ax_manip.legend(fontsize=9, loc='upper right')
        taux_max = max(t for ns in noms_strategies for t in manip_résultats[ns].values())
        ax_manip.set_ylim([0, min(1.0, taux_max * 1.4 + 0.05)])
        ax_manip.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.5, w_pad=2.0)
    plt.show()

    return résultats


# ─────────────────────────────────────────────
# CONSISTANCE DES SATISFACTIONS SUR PLUSIEURS ESSAIS
# ─────────────────────────────────────────────

def consistance_satisfaction(scrutins, nb_candidats, nb_votants,
                              metriques=None,
                              strategies_manipulation=None,
                              generateur_votants=distribution_uniforme,
                              generateur_candidats=distribution_uniforme,
                              nb_simulations=200):
    if metriques is None:
        metriques = [satisfaction_relative]
    if strategies_manipulation is None:
        strategies_manipulation = []

    noms_scrutins   = [s.__name__ for s in scrutins]
    noms_strategies = [st.__name__ for st in strategies_manipulation]

    res_sat = {
        m.__name__: {n: np.zeros(nb_simulations) for n in noms_scrutins}
        for m in metriques
    }
    res_manip = {
        st.__name__: {n: np.zeros(nb_simulations, dtype=bool) for n in noms_scrutins}
        for st in strategies_manipulation
    }

    start = time.perf_counter()

    for k in range(nb_simulations):
        candidats = generateur_candidats(nb_candidats)
        votants   = generateur_votants(nb_votants)
        dist      = distances(candidats, votants)
        tdp       = tableau_des_preferences(candidats, votants)

        for scrutin in scrutins:
            res       = scrutin(tdp)
            gagnant_h = res[0][1]
            for metrique in metriques:
                res_sat[metrique.__name__][scrutin.__name__][k] = metrique(gagnant_h, dist, votants)
            for strategie in strategies_manipulation:
                res_manip[strategie.__name__][scrutin.__name__][k] = strategie(tdp, dist, scrutin)

        print(f"\r  Progression : {k + 1}/{nb_simulations}", end="", flush=True)

    print(f"\n  Durée : {time.perf_counter() - start:.1f}s")

    # ── Synthèse console — satisfaction ──────────────────────────────────────
    for m in metriques:
        nm = m.__name__
        print(f"\n{'═'*65}")
        print(f"  Satisfaction — {nm}")
        print(f"  Candidats / Votants / Essais : {nb_candidats} / {nb_votants} / {nb_simulations}")
        print(f"{'─'*65}")
        print(f"  {'Scrutin':<22}  {'Moy':>7}  {'Méd':>7}  {'Std':>7}  {'Min':>7}  {'Max':>7}")
        print(f"  {'─'*22}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
        for n in noms_scrutins:
            v = res_sat[nm][n]
            print(f"  {n:<22}  {np.mean(v):>7.4f}  {np.median(v):>7.4f}  "
                  f"{np.std(v):>7.4f}  {np.min(v):>7.4f}  {np.max(v):>7.4f}")
        print(f"{'═'*65}")

    # ── Synthèse console — manipulation ──────────────────────────────────────
    if strategies_manipulation:
        for st in strategies_manipulation:
            ns = st.__name__
            print(f"\n{'═'*65}")
            print(f"  Manipulation — {ns}")
            print(f"{'─'*65}")
            print(f"  {'Scrutin':<22}  {'Taux':>8}  {'Nb manip':>10}")
            print(f"  {'─'*22}  {'─'*8}  {'─'*10}")
            for n in noms_scrutins:
                v = res_manip[ns][n]
                print(f"  {n:<22}  {np.mean(v):>8.4f}  {int(np.sum(v)):>10}/{nb_simulations}")
            print(f"{'═'*65}")

    # ── Calcul des bornes Y globales par métrique (zoom sur les données) ──────
    ylims_sat = {}
    for m in metriques:
        all_vals = np.concatenate([res_sat[m.__name__][n] for n in noms_scrutins])
        marge    = max(0.02, (all_vals.max() - all_vals.min()) * 0.08)
        ylims_sat[m.__name__] = (
            max(0.0,  all_vals.min() - marge),
            min(1.0,  all_vals.max() + marge),
        )

    # Bornes Y pour la manipulation
    ylim_manip = None
    if strategies_manipulation:
        all_taux = [
            float(np.mean(res_manip[ns][n]))
            for ns in noms_strategies
            for n  in noms_scrutins
        ]
        taux_max  = max(all_taux) if all_taux else 0.1
        taux_min  = min(all_taux) if all_taux else 0.0
        marge_m   = max(0.02, (taux_max - taux_min) * 0.15)
        ylim_manip = (
            max(0.0, taux_min - marge_m),
            min(1.0, taux_max + marge_m * 3),   # un peu plus de place pour les annotations
        )

    # ── Plots ─────────────────────────────────────────────────────────────────
    nb_metriques  = len(metriques)
    nb_strategies = len(strategies_manipulation)
    nb_lignes     = nb_metriques + (1 if nb_strategies > 0 else 0)

    hauteur_par_ligne = min(10.0, 9.5 / nb_lignes)
    fig, axes = plt.subplots(
        nb_lignes, 2,
        figsize=(20, hauteur_par_ligne * nb_lignes),
        squeeze=False,
        dpi=96
    )
    fig.suptitle(
        f"Consistance — {nb_candidats} candidats · {nb_votants} votants · {nb_simulations} essais",
        fontsize=14, fontweight='bold'
    )

    for row, m in enumerate(metriques):
        _plot_consistance_row(
            axes[row], res_sat[m.__name__], noms_scrutins, m.__name__,
            ylim=ylims_sat[m.__name__]
        )

    if nb_strategies > 0:
        _plot_manipulation_row(
            axes[nb_metriques], res_manip, noms_scrutins, noms_strategies,
            ylim=ylim_manip
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.5, w_pad=2.0)
    plt.show()

    return {"satisfaction": res_sat, "manipulation": res_manip}


def _plot_consistance_row(axes_row, résultats_metrique, noms, nom_metrique, ylim=None):
    nb_scrutins = len(noms)
    couleurs    = plt.cm.get_cmap("tab10")(np.linspace(0, 0.9, nb_scrutins))
    nb_sims     = len(next(iter(résultats_metrique.values())))
    fenetre     = max(10, nb_sims // 15)

    # Bornes Y : fournies ou calculées localement en dernier recours
    if ylim is None:
        all_vals = np.concatenate([résultats_metrique[n] for n in noms])
        marge    = max(0.02, (all_vals.max() - all_vals.min()) * 0.08)
        ylim     = (max(0.0, all_vals.min() - marge),
                    min(1.0, all_vals.max() + marge))
    ymin, ymax = ylim

    # ── Graphe 1 : moyennes glissantes ────────────────────────────────────────
    ax = axes_row[0]
    for i, n in enumerate(noms):
        v         = résultats_metrique[n]
        glissante = np.convolve(v, np.ones(fenetre) / fenetre, mode='valid')
        x_gl      = np.arange(fenetre, nb_sims + 1)
        ax.plot(x_gl, glissante, color=couleurs[i], linewidth=2,
                label=f"{n}  (moy={np.mean(v):.3f}, std={np.std(v):.3f})")
    ax.set_xlabel("Numéro de l'essai", fontsize=11)
    ax.set_ylabel(nom_metrique, fontsize=11)
    ax.set_title(f"{nom_metrique} — moyenne glissante (fenêtre={fenetre})", fontsize=11)
    ax.legend(fontsize=8, loc='best', framealpha=0.85)
    ax.set_ylim([ymin, ymax])
    ax.grid(alpha=0.3, linestyle='--')

    # ── Graphe 2 : violin + boxplot ───────────────────────────────────────────
    ax = axes_row[1]
    espacement = max(1.8, 0.9 + nb_scrutins * 0.3)
    positions  = np.arange(nb_scrutins) * espacement
    data       = [résultats_metrique[n] for n in noms]

    parts = ax.violinplot(data, positions=positions, widths=espacement * 0.5,
                          showmeans=False, showextrema=False)
    for j, pc in enumerate(parts['bodies']):
        pc.set_facecolor(couleurs[j])
        pc.set_alpha(0.35)

    bp = ax.boxplot(data, positions=positions, widths=espacement * 0.15,
                    patch_artist=True,
                    medianprops=dict(color='black', linewidth=2.5),
                    whiskerprops=dict(linewidth=1.5, linestyle='--'),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markersize=3, alpha=0.35, markeredgewidth=0))
    for j, patch in enumerate(bp['boxes']):
        patch.set_facecolor(couleurs[j])
        patch.set_alpha(0.80)

    marge_droite = espacement * 0.3
    for j, n in enumerate(noms):
        med = np.median(résultats_metrique[n])
        moy = np.mean(résultats_metrique[n])
        ax.text(positions[j] + marge_droite, med,
                f"méd={med:.3f}\nmoy={moy:.3f}",
                fontsize=8, va='center', ha='left', color='#333333',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, ec='none'))

    ax.set_xticks(positions)
    ax.set_xticklabels(noms, fontsize=10, rotation=20, ha='right')
    ax.set_ylabel(nom_metrique, fontsize=11)
    ax.set_title(f"{nom_metrique} — distributions", fontsize=11)
    # Marge supplémentaire à droite pour les annotations texte
    ax.set_xlim([-espacement * 0.6, positions[-1] + espacement * 1.2])
    ax.set_ylim([ymin, ymax])
    ax.grid(axis='y', alpha=0.3, linestyle='--')


def _plot_manipulation_row(axes_row, res_manip, noms_scrutins, noms_strategies, ylim=None):
    couleurs_scrutins = plt.cm.get_cmap("tab10")(np.linspace(0, 0.9, len(noms_scrutins)))
    nb_scrutins   = len(noms_scrutins)
    nb_strategies = len(noms_strategies)

    espacement = max(1.0, 0.6 + nb_strategies * 0.4)
    largeur    = espacement * 0.7 / nb_strategies
    x          = np.arange(nb_scrutins) * (espacement * nb_strategies + 0.5)

    # Bornes Y fournies ou calculées localement
    if ylim is None:
        all_taux = [float(np.mean(res_manip[ns][n]))
                    for ns in noms_strategies for n in noms_scrutins]
        taux_max  = max(all_taux) if all_taux else 0.1
        taux_min  = min(all_taux) if all_taux else 0.0
        marge_m   = max(0.02, (taux_max - taux_min) * 0.15)
        ylim = (max(0.0, taux_min - marge_m),
                min(1.0, taux_max + marge_m * 3))
    ymin, ymax = ylim

    ax = axes_row[0]
    for j, ns in enumerate(noms_strategies):
        taux   = [float(np.mean(res_manip[ns][n])) for n in noms_scrutins]
        offset = (j - (nb_strategies - 1) / 2) * largeur
        bars   = ax.bar(x + offset, taux, largeur * 0.85,
                        color=[couleurs_scrutins[k] for k in range(nb_scrutins)],
                        alpha=0.80, edgecolor='white', linewidth=1.2,
                        label=ns)
        for bar, t in zip(bars, taux):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (ymax - ymin) * 0.02,
                    f"{t:.3f}", ha='center', va='bottom',
                    fontsize=8, fontweight='bold', rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(noms_scrutins, fontsize=10, rotation=20, ha='right')
    ax.set_ylabel("Taux de manipulation", fontsize=11)
    ax.set_title("Taux de manipulation par scrutin et stratégie", fontsize=11)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.85)
    ax.set_ylim([ymin, ymax])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    axes_row[1].set_visible(False)


# ─────────────────────────────────────────────
# SATISFACTION EN FONCTION DES PARAMÈTRES
# ─────────────────────────────────────────────

def satisfaction_nb_candidats_votants(scrutin, nb_candidats_max, nb_votants_max,
                                      generateur_votants=distribution_uniforme,
                                      generateur_candidats=distribution_uniforme,
                                      nb_simulations=200,
                                      metrique=satisfaction_relative):
    """
    Calcule la satisfaction moyenne en fonction du nombre de candidats
    et du nombre de votants, puis affiche une heatmap des résultats.

    Paramètres :
        scrutin              : fonction de scrutin à analyser
        nb_candidats_max     : nombre de candidats maximum (à partir de 2)
        nb_votants_max       : nombre de votants maximum (à partir de 3)
        generateur_votants   : générateur de positions des votants
        generateur_candidats : générateur de positions des candidats
        nb_simulations       : nombre de simulations par cellule
        metrique             : fonction (gagnant_idx, dist) → float
                               par défaut : satisfaction_relative
                               alternative : satisfaction_réel

    Retourne :
        Z : tableau NumPy 2D (nb_candidats × nb_votants) des satisfactions moyennes
    """
    cand_range = list(range(2, nb_candidats_max + 1))
    vot_range  = list(range(3, nb_votants_max + 1))

    total = len(cand_range) * len(vot_range)
    Z     = np.zeros((len(cand_range), len(vot_range)))

    start = time.perf_counter()
    done  = 0

    for i, nb_c in enumerate(cand_range):
        for j, nb_v in enumerate(vot_range):
            sats = []
            for _ in range(nb_simulations):
                candidats = generateur_candidats(nb_c)
                votants   = generateur_votants(nb_v)
                dist      = distances(candidats, votants)
                tdp       = tableau_des_preferences(candidats, votants)

                res       = scrutin(tdp)
                gagnant_h = res[0][1]
                sats.append(metrique(gagnant_h, dist, votants))

            Z[i, j] = float(np.mean(sats))
            done += 1
            print(f"\r  Progression : {done}/{total}  "
                  f"(candidats={nb_c}, votants={nb_v})  "
                  f"sat={Z[i,j]:.3f}", end="", flush=True)

    print(f"\n  Durée totale : {time.perf_counter() - start:.1f}s")

    # ── Heatmap ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, len(vot_range) // 3), max(5, len(cand_range) // 2)))
    fig.suptitle(f"Satisfaction ({metrique.__name__}) — {scrutin.__name__}",
                 fontsize=13, fontweight='bold')

    im = ax.imshow(Z, origin='lower', aspect='auto', cmap='RdYlGn',
                   vmin=0, vmax=1,
                   extent=[vot_range[0] - 0.5, vot_range[-1] + 0.5,
                            cand_range[0] - 0.5, cand_range[-1] + 0.5])

    plt.colorbar(im, ax=ax, label="Satisfaction moyenne")

    # Annoter chaque cellule si la grille est petite
    if len(cand_range) * len(vot_range) <= 200:
        for i, nb_c in enumerate(cand_range):
            for j, nb_v in enumerate(vot_range):
                val = Z[i, j]
                color = 'black' if 0.35 < val < 0.75 else 'white'
                ax.text(nb_v, nb_c, f"{val:.2f}",
                        ha='center', va='center', fontsize=7, color=color)

    ax.set_xlabel("Nombre de votants",   fontsize=11)
    ax.set_ylabel("Nombre de candidats", fontsize=11)
    ax.set_xticks(vot_range  if len(vot_range)  <= 20 else vot_range[::5])
    ax.set_yticks(cand_range if len(cand_range) <= 20 else cand_range[::2])
    ax.grid(False)
    plt.tight_layout()
    plt.show()

    return Z


def comparer_satisfaction_nb_candidats_votants(scrutins, nb_candidats_max, nb_votants_max,
                                               generateur_votants=distribution_uniforme,
                                               generateur_candidats=distribution_uniforme,
                                               nb_simulations=200,
                                               metrique=satisfaction_relative):
    """
    Compare la satisfaction moyenne de plusieurs scrutins en fonction du nombre
    de candidats et de votants, sur les mêmes paysages électoraux.

    Affiche :
      - Une heatmap par scrutin
      - Une heatmap "meilleur scrutin par cellule"

    Paramètres :
        scrutins             : liste de fonctions de scrutin à comparer
        nb_candidats_max     : nombre de candidats maximum (à partir de 2)
        nb_votants_max       : nombre de votants maximum (à partir de 3)
        generateur_votants   : générateur de positions des votants
        generateur_candidats : générateur de positions des candidats
        nb_simulations       : nombre de simulations par cellule
        metrique             : fonction (gagnant_idx, dist) → float
                               par défaut : satisfaction_relative
                               alternative : satisfaction_réel

    Retourne :
        dict { nom_scrutin → tableau Z (nb_candidats × nb_votants) }
    """
    cand_range  = list(range(2, nb_candidats_max + 1))
    vot_range   = list(range(3, nb_votants_max  + 1))
    nb_scrutins = len(scrutins)
    total_cells = len(cand_range) * len(vot_range)

    Z = {s.__name__: np.zeros((len(cand_range), len(vot_range))) for s in scrutins}

    start = time.perf_counter()
    done  = 0

    for i, nb_c in enumerate(cand_range):
        for j, nb_v in enumerate(vot_range):

            sats = {s.__name__: [] for s in scrutins}

            # Mêmes paysages pour tous les scrutins
            for _ in range(nb_simulations):
                candidats = generateur_candidats(nb_c)
                votants   = generateur_votants(nb_v)
                dist      = distances(candidats, votants)
                tdp       = tableau_des_preferences(candidats, votants)

                for scrutin in scrutins:
                    res       = scrutin(tdp)
                    gagnant_h = res[0][1]
                    sats[scrutin.__name__].append(metrique(gagnant_h, dist, votants))

            for scrutin in scrutins:
                Z[scrutin.__name__][i, j] = float(np.mean(sats[scrutin.__name__]))

            done += 1
            print(f"\r  Progression : {done}/{total_cells}  "
                  f"(candidats={nb_c}, votants={nb_v})", end="", flush=True)

    print(f"\n  Durée totale : {time.perf_counter() - start:.1f}s")

    # ── Plots ─────────────────────────────────────────────────────────────────
    ncols    = min(nb_scrutins + 1, 3)
    nrows    = (nb_scrutins + 1 + ncols - 1) // ncols
    extent   = [vot_range[0] - 0.5,  vot_range[-1]  + 0.5,
                cand_range[0] - 0.5, cand_range[-1] + 0.5]
    annotate = len(cand_range) * len(vot_range) <= 200
    noms     = [s.__name__ for s in scrutins]

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * max(6, len(vot_range) // 3),
                                      nrows * max(4, len(cand_range) // 2)))
    fig.suptitle(f"Comparaison de satisfaction ({metrique.__name__}) — "
                 + ", ".join(noms), fontsize=12, fontweight='bold')
    axes_flat = np.array(axes).flatten()

    # ── Une heatmap par scrutin ───────────────────────────────────────────────
    for k, scrutin in enumerate(scrutins):
        ax  = axes_flat[k]
        nom = scrutin.__name__
        im  = ax.imshow(Z[nom], origin='lower', aspect='auto',
                        cmap='RdYlGn', vmin=0, vmax=1, extent=extent)
        plt.colorbar(im, ax=ax, label="Satisfaction")
        ax.set_title(nom, fontsize=11, fontweight='bold')
        ax.set_xlabel("Votants"); ax.set_ylabel("Candidats")
        ax.set_xticks(vot_range  if len(vot_range)  <= 20 else vot_range[::5])
        ax.set_yticks(cand_range if len(cand_range) <= 20 else cand_range[::2])

        if annotate:
            for i, nb_c in enumerate(cand_range):
                for j, nb_v in enumerate(vot_range):
                    val   = Z[nom][i, j]
                    color = 'black' if 0.35 < val < 0.75 else 'white'
                    ax.text(nb_v, nb_c, f"{val:.2f}",
                            ha='center', va='center', fontsize=7, color=color)

    # ── Heatmap "meilleur scrutin par cellule" ────────────────────────────────
    ax_best    = axes_flat[nb_scrutins]
    Z_stack    = np.stack([Z[n] for n in noms], axis=0)   # (S, C, V)
    Z_best_idx = np.argmax(Z_stack, axis=0)                # (C, V)

    im2 = ax_best.imshow(Z_best_idx, origin='lower', aspect='auto',
                          cmap=plt.cm.get_cmap('tab10', nb_scrutins),
                          vmin=-0.5, vmax=nb_scrutins - 0.5, extent=extent)
    cbar2 = plt.colorbar(im2, ax=ax_best, ticks=range(nb_scrutins))
    cbar2.ax.set_yticklabels(noms, fontsize=8)
    ax_best.set_title("Meilleur scrutin par cellule", fontsize=11, fontweight='bold')
    ax_best.set_xlabel("Votants"); ax_best.set_ylabel("Candidats")
    ax_best.set_xticks(vot_range  if len(vot_range)  <= 20 else vot_range[::5])
    ax_best.set_yticks(cand_range if len(cand_range) <= 20 else cand_range[::2])

    if annotate:
        for i, nb_c in enumerate(cand_range):
            for j, nb_v in enumerate(vot_range):
                ax_best.text(nb_v, nb_c, noms[Z_best_idx[i, j]][:5],
                             ha='center', va='center', fontsize=6,
                             color='white', fontweight='bold')

    for k in range(nb_scrutins + 1, len(axes_flat)):
        axes_flat[k].set_visible(False)

    plt.tight_layout()
    plt.show()

    # ── Synthèse console ──────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  Satisfaction moyenne globale par scrutin")
    print(f"{'─'*55}")
    moyennes = {n: float(np.mean(Z[n])) for n in noms}
    for nom, moy in sorted(moyennes.items(), key=lambda x: x[1], reverse=True):
        barre = "█" * int(moy * 40)
        print(f"  {nom:<22} {moy:.4f}  {barre}")
    print(f"{'═'*55}\n")

    return Z


def gagnant(résultats):
    return résultats[0][1]

def plot_résultats(résultats, labels=None, title="Résultats du scrutin"):
    résultats_triés = sorted(résultats, key=lambda x: x[1])
    scores  = [s for s, _ in résultats_triés]
    indices = [i for _, i in résultats_triés]
    if labels is None:
        labels = [f"{i}" for i in indices]
    total = sum(scores)
    pourcentages = [s / total * 100 for s in scores]
    bars = plt.bar(labels, scores, edgecolor='white', linewidth=2)
    for bar, score, pct in zip(bars, scores, pourcentages):
        plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                 f'{score:.2f}\n({pct:.1f}%)',
                 ha='center', va='top', fontweight='bold', fontsize=10)
    plt.xlabel("Candidats", fontsize=12, fontweight='bold')
    plt.ylabel("Scores",    fontsize=12, fontweight='bold')
    plt.title(title,        fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()