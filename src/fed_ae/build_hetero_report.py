"""Rapport étape 3 : FedAvg vs FedProx (+ fine-tuning, peu de données, IID vs non-IID).
Lit hetero_variants.json, hetero_lowdata.json, hetero_iid.json. Écrit rapport_hetero.md.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "results" / "nbaiot_fed"
VLAB = {"a_fedavg": "a) FedAvg", "b_fedprox": "b) FedProx (μ*)",
        "c_fedavg_ft": "c) FedAvg + fine-tuning local", "d_fedprox_ft": "d) FedProx + fine-tuning local"}


def L(p):
    f = OUT / p
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def pc(v):
    return "—" if v is None else f"{100*v:.1f}"


def grow(g):
    return (f"{g['auc']:.4f} | {pc(g['p95']['global']['rappel'])}/{pc(g['p95']['global']['fpr'])} | "
            f"{pc(g['p99']['global']['rappel'])}/{pc(g['p99']['global']['fpr'])} | "
            f"{pc(g['p99']['global']['rappel_mirai'])} | {pc(g['p99']['global']['rappel_bashlite'])}")


def main():
    V, Lo, I = L("hetero_variants.json"), L("hetero_lowdata.json"), L("hetero_iid.json")
    et1 = L("ae_fed.json")
    md = ["# N-BaIoT — hétérogénéité : FedAvg vs FedProx (étape 3)", "",
          "Entraînement sur le NORMAL uniquement ; seuils p95/p99/p99.9 sur le normal de validation ; "
          "aucune décision fondée sur les attaques. Variantes a–d : **scaler local** par client. "
          "μ et nb d'époques de fine-tuning choisis sur la loss du normal de validation.", ""]

    if V:
        md += ["## 1. Variantes a–d (global, test)", "",
               "| Variante | AUC | p95 rap/FPR | p99 rap/FPR | Mirai p99 | BASHLITE p99 |",
               "|---|--:|--:|--:|--:|--:|"]
        for k in ("a_fedavg", "b_fedprox", "c_fedavg_ft", "d_fedprox_ft"):
            md.append(f"| {VLAB[k]} | {grow(V[k]['global'])} |")
        b = V["b_fedprox"]
        fte = list(V['c_fedavg_ft'].get('ft_epochs', {}).values())
        md += ["", f"**μ\\*** = {b.get('mu_star')} (choisi sur la loss normal-val ; "
               + ", ".join(f"μ={m}:{v:.5f}" for m, v in b.get("mu_val_loss", {}).items()) + ").",
               f"Fine-tuning local : {min(fte)}–{max(fte)} époques par appareil (early-stopping sur la "
               "loss normal-val).", ""]
        # par appareil : AUC a vs b vs d
        md += ["## 2. AUC par appareil (effet de FedProx et du fine-tuning)", "",
               "| Appareil | FedAvg | FedProx | FedAvg+FT | FedProx+FT |", "|---|--:|--:|--:|--:|"]
        for dev in V["a_fedavg"]["par_appareil"]:
            md.append(f"| {dev} | {V['a_fedavg']['par_appareil'][dev]['auc']:.3f} | "
                      f"{V['b_fedprox']['par_appareil'][dev]['auc']:.3f} | "
                      f"{V['c_fedavg_ft']['par_appareil'][dev]['auc']:.3f} | "
                      f"{V['d_fedprox_ft']['par_appareil'][dev]['auc']:.3f} |")
        md += [""]

    if Lo:
        md += ["## 3. Scénario « peu de données » (plafond normal-train par appareil)", "",
               "Idée : avec peu de normal par appareil, un AE local seul dispose de peu d'exemples ; le "
               "fédéré + fine-tuning met en commun l'apprentissage. **Nuance honnête** : sur N-BaIoT, le "
               "normal est très régulier, donc un AE local atteint déjà une bonne AUC avec 1 000 lignes ; "
               "l'apport du fédéré se voit surtout sur le **rappel à FPR fixe** et pour BASHLITE, pas tant "
               "sur l'AUC.", "",
               "| Plafond | Modèle | AUC | p99 rap/FPR | Mirai | BASHLITE |", "|---|---|--:|--:|--:|--:|"]
        for cap in ("cap_1000", "cap_5000"):
            if cap not in Lo:
                continue
            for k, lab in (("local_seul", "AE local seul"), ("fedavg", "FedAvg"),
                           ("fedavg_ft", "FedAvg + FT"), ("fedprox_ft", "FedProx + FT")):
                g = Lo[cap][k]["global"]
                md.append(f"| {cap.split('_')[1]} | {lab} | {g['auc']:.4f} | "
                          f"{pc(g['p99']['global']['rappel'])}/{pc(g['p99']['global']['fpr'])} | "
                          f"{pc(g['p99']['global']['rappel_mirai'])} | {pc(g['p99']['global']['rappel_bashlite'])} |")
        md += [""]

    if I:
        md += ["## 4. IID vs non-IID — ablation contrôlée (non déployable)", "",
               "⚠️ **Ablation contrôlée, non déployable** : cette expérience utilise un **scaler global** "
               "(ajusté sur le normal de tous les clients réuni) dans les DEUX bras, uniquement pour isoler "
               "l'effet de la répartition des données. Les variantes a–d ci-dessus gardent le scaler local "
               "(méthode déployable). Ici, seule change la répartition : par appareil (non-IID) vs normal "
               "mélangé et redistribué au hasard (IID).", "",
               "| Répartition | AUC | p95 rap/FPR | p99 rap/FPR | Mirai | BASHLITE |", "|---|--:|--:|--:|--:|--:|"]
        for k, lab in (("non_iid", "non-IID (par appareil)"), ("iid", "IID (mélangé)")):
            md.append(f"| {lab} | {grow(I[k]['global'])} |")
        da = I["iid"]["global"]["auc"] - I["non_iid"]["global"]["auc"]
        md += ["", f"Écart d'AUC IID − non-IID : **{da:+.4f}**. Un écart positif net confirme que "
               "l'hétérogénéité (non-IID) dégrade le modèle fédéré, toutes choses égales par ailleurs.", ""]

    # 5. ressources (repris de l'étape 1 ; FedProx = même taille de modèle)
    if et1 and "ressources" in et1:
        r = et1["ressources"]
        md += ["## 5. Ressources (identiques pour FedAvg/FedProx — même modèle)", "",
               f"- {r['parametres']:,} paramètres (~{r['taille_ko']} Ko/màj) ; "
               f"~{r['octets_par_round']/1024/1024:.1f} Mo échangés/round ; "
               f"FedProx ajoute un terme proximal (coût CPU local, 0 octet de plus).".replace(",", " "), ""]

    # 6. lecture honnête (remplie après lecture des chiffres)
    if V and I:
        a_auc = V["a_fedavg"]["global"]["auc"]; b_auc = V["b_fedprox"]["global"]["auc"]
        c_auc = V["c_fedavg_ft"]["global"]["auc"]; d_auc = V["d_fedprox_ft"]["global"]["auc"]
        md += ["## 6. Lecture", "",
               f"- FedAvg {a_auc:.3f} → FedProx {b_auc:.3f} : effet de μ sur l'AUC globale "
               f"({b_auc-a_auc:+.3f}).",
               f"- Fine-tuning local : FedAvg+FT {c_auc:.3f}, FedProx+FT {d_auc:.3f} "
               f"(personnalisation par appareil ; gain {max(c_auc,d_auc)-a_auc:+.3f} vs FedAvg).",
               "- Voir §2 pour l'effet par appareil (notamment les appareils faibles comme Samsung), "
               "§3 pour l'intérêt du fédéré à faible volume, §4 pour l'effet isolé de l'hétérogénéité."]
    (OUT / "rapport_hetero.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
