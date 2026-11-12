# `src/xai/` — Explicabilité (SHAP)

Explique les décisions du classifieur en attribuant une importance à chaque caractéristique.

| Fichier | Rôle |
|---|---|
| `explain_shap_generic.py` | Calcule les valeurs SHAP pour un jeu et produit les figures (importance globale, *waterfall*, *force*). → **Figure 4.6**. |
| `generate_xai_report.py` | Assemble les figures en un **rapport PDF** par jeu (`results/xai_report_*.pdf`). |

Détails : [`../../docs/EXPLICATION_DU_CODE.md`](../../docs/EXPLICATION_DU_CODE.md).
