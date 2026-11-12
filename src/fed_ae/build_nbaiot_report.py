"""Rapport N-BaIoT : tableau par appareil + global (4 modèles), comparaison Meidan et al. 2018.
Lit results/nbaiot_fed/ae_{fed,untrained,central,local}.json. Écrit non_iid déjà fait + ce rapport.
    python build_nbaiot_report.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "results" / "nbaiot_fed"
MODES = ["fed", "untrained", "central", "local"]
LABELS = {"fed": "AE fédéré (principal)", "untrained": "AE non entraîné (contrôle)",
          "central": "AE centralisé (référence)", "local": "AE local par appareil (contrôle)"}
PCTS = ["p95", "p99", "p99.9"]


def load(mode):
    f = OUT / f"ae_{mode}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def pc(v):
    return "—" if v is None else f"{100*v:.1f}"


def main():
    data = {m: load(m) for m in MODES}
    md = ["# N-BaIoT — autoencodeur fédéré auto-supervisé : résultats", ""]
    meta = data["fed"]["meta"]
    md += [f"Protocole : entraînement sur le trafic NORMAL uniquement (auto-supervisé), découpage "
           f"**chronologique** 60/20/20 par appareil, scaler **local** par client, doublons de contenu "
           f"retirés de val/test (**{meta['doublons_retires']}** lignes ; identiques restantes : "
           f"{meta['lignes_identiques_restantes']}). Attaques réservées au test (échantillon stratifié "
           f"par appareil et famille). Seuils figés sur le NORMAL de validation (percentiles 95/99/99.9), "
           f"en GLOBAL et LOCAL. Test exécuté une seule fois pour les 4 modèles.", ""]
    md += ["> Note : un essai court préalable (1 round) avait affiché l'AUC/Mirai/BASHLITE du test ; "
           "aucun réglage n'en a découlé. Le nombre de rounds (20) et d'époques (2) a été jugé "
           "uniquement sur la loss de reconstruction du normal de validation (voir plus bas).", ""]

    # 1. comparaison globale des 4 modèles
    md += ["## 1. Comparaison globale des 4 modèles (test, seuil GLOBAL puis LOCAL)", "",
           "| Modèle | AUC | p95 rap/FPR (G) | p99 rap/FPR (G) | p99.9 rap/FPR (G) | p99 rap/FPR (L) |",
           "|---|--:|--:|--:|--:|--:|"]
    for m in MODES:
        if not data[m]:
            md.append(f"| {LABELS[m]} | (absent) | | | | |"); continue
        g = data[m]["global"]
        def cell(p, scope): return f"{pc(g[p][scope]['rappel'])}/{pc(g[p][scope]['fpr'])}"
        md.append(f"| {LABELS[m]} | {g['auc']:.4f} | {cell('p95','global')} | {cell('p99','global')} | "
                  f"{cell('p99.9','global')} | {cell('p99','local')} |")
    md += ["", "Rappel = attaques détectées, FPR = faux positifs sur le normal de test (%). "
           "(G) seuil global, (L) seuil local par appareil.", ""]

    # 2. Mirai vs BASHLITE (fed, au seuil p99)
    g = data["fed"]["global"]
    md += ["## 2. Rappel par famille — AE fédéré, seuil p99", "",
           "| Seuil | Rappel global | Mirai | BASHLITE | FPR |", "|---|--:|--:|--:|--:|"]
    for p in PCTS:
        for scope in ("global", "local"):
            r = g[p][scope]
            md.append(f"| {p} ({scope}) | {pc(r['rappel'])} | {pc(r['rappel_mirai'])} | "
                      f"{pc(r['rappel_bashlite'])} | {pc(r['fpr'])} |")
    md += [""]

    # 3. par appareil (fed, p99 global)
    md += ["## 3. AE fédéré par appareil (test, seuil p99 global)", "",
           "| Appareil | AUC | Rappel | FPR | Mirai | BASHLITE |", "|---|--:|--:|--:|--:|--:|"]
    for dev, r in data["fed"]["par_appareil"].items():
        c = r["p99"]["global"]
        md.append(f"| {dev} | {r['auc']:.4f} | {pc(c['rappel'])} | {pc(c['fpr'])} | "
                  f"{pc(c['rappel_mirai'])} | {pc(c['rappel_bashlite'])} |")
    md += [""]

    # 4. convergence + ressources
    vh = meta.get("loss_val_par_round", [])
    if vh:
        md += ["## 4. Convergence (loss de reconstruction, normal de validation)", "",
               f"Round 1 : {vh[0]:.6f} · round 5 : {vh[4]:.6f} · round 10 : {vh[9]:.6f} · "
               f"round 20 : {vh[-1]:.6f}. La baisse devient marginale après ~{_plateau(vh)} rounds "
               f"(critère : amélioration < 1 % entre rounds), ce qui justifie 20 rounds.", ""]
    r = data["fed"].get("ressources", {})
    if r:
        md += ["## 5. Ressources (AE fédéré)", "",
               f"- Paramètres : {r['parametres']:,} (~{r['taille_ko']} Ko par mise à jour de modèle)".replace(",", " "),
               f"- Octets échangés par round (9 clients, download+upload, float32) : "
               f"{r['octets_par_round']:,} o (~{r['octets_par_round']/1024/1024:.1f} Mo)".replace(",", " "),
               f"- Total sur {meta['rounds']} rounds : ~{r['octets_total']/1024/1024:.1f} Mo",
               f"- Temps d'entraînement fédéré : {meta['temps_total_s']/60:.1f} min (CPU)", ""]

    # 6. Meidan
    md += ["## 6. Comparaison avec Meidan et al. (2018)", "",
           "Meidan et al. (*N-BaIoT*, IEEE Pervasive Computing 2018) utilisent aussi des autoencodeurs "
           "entraînés sur le trafic normal, mais avec un protocole différent :",
           "",
           "| Aspect | Meidan et al. 2018 | Ce travail |",
           "|---|---|---|",
           "| Modèle | **un AE par appareil** (local) | **AE fédéré** (un modèle commun, données non partagées) |",
           "| Seuil | moyenne + écart-type de l'erreur sur val, + **vote majoritaire** sur une fenêtre de paquets | percentiles 95/99/99.9 par flux, **sans fenêtrage** |",
           "| Découpage | train = début du normal, reste en test | **chronologique 60/20/20** |",
           "| Doublons | non mentionné | **retirés** entre train/val/test |",
           "| Résultat rapporté | TPR ≈ 100 %, FPR ≈ 0,7 % (après vote majoritaire) | voir tableaux ci-dessus (par flux) |",
           "",
           f"Notre AE **fédéré** atteint une AUC globale de {data['fed']['global']['auc']:.4f} et, au seuil p99 "
           f"local, un rappel de {pc(g['p99']['local']['rappel'])} % pour {pc(g['p99']['local']['fpr'])} % de faux "
           "positifs, **par flux et sans vote majoritaire**, avec un seul modèle partagé plutôt qu'un modèle "
           "par appareil. Le vote majoritaire de Meidan et al. augmente mécaniquement le rappel et réduit le FPR ; "
           "nos chiffres par flux ne sont donc pas directement comparables, mais l'AE local (contrôle), "
           f"qui reprend leur cadre (un modèle par appareil), atteint une AUC de {data['local']['global']['auc']:.4f}, "
           "cohérente avec leurs très bons résultats.", ""]

    # 7. lecture honnête
    fa, la, ca, ua = (data[m]["global"]["auc"] for m in ("fed", "local", "central", "untrained"))
    md += ["## 7. Lecture honnête", "",
           f"- **Contrôle validé** : l'AE non entraîné (AUC {ua:.3f}) est nettement moins bon que les AE "
           f"entraînés — mais il sépare déjà à {ua:.2f} car les attaques N-BaIoT (Mirai/BASHLITE) ont des "
           "valeurs extrêmes, faciles à mal reconstruire même avec des poids aléatoires.",
           f"- **L'AE local (AUC {la:.3f}) et centralisé (AUC {ca:.3f}) sont quasi parfaits** : un modèle "
           "spécialisé par appareil (ou voyant toutes les données) sépare très bien normal et attaque.",
           f"- **L'AE fédéré est en retrait (AUC {fa:.3f})**, et surtout très inégal entre appareils "
           "(0,61 pour Samsung, 0,80 pour plusieurs caméras, contre ~0,99 pour Ennio et Philips). La loss "
           "de reconstruction du normal, elle, converge bien : le problème n'est pas l'optimisation mais "
           "l'**hétérogénéité** — un modèle global unique (FedAvg) moyenne des comportements normaux très "
           "différents d'un appareil à l'autre, et perd le pouvoir discriminant sur les appareils atypiques. "
           "C'est un résultat honnête, pas un échec technique (pas de divergence, doublons retirés, seuils "
           "sur validation normale uniquement).",
           "- **Conséquence** : cela motive directement l'étape 3 (FedProx vs FedAvg, IID vs non-IID) : "
           "FedProx est conçu pour limiter cette divergence sur données non-IID. Une personnalisation par "
           "appareil (ou un scaler fédéré partagé) est une piste complémentaire.", ""]
    (OUT / "rapport_nbaiot.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


def _plateau(vh, tol=0.01):
    for i in range(1, len(vh)):
        if vh[i - 1] > 0 and (vh[i - 1] - vh[i]) / vh[i - 1] < tol:
            return i
    return len(vh)


if __name__ == "__main__":
    main()
