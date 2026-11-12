"""Étape 1, point 2 : répartition non-IID de N-BaIoT (un client = un appareil).

Compte, par appareil, le nombre de flux normaux (benign), BASHLITE (gafgyt) et Mirai, à partir
des CSV bruts (Raw_NBaIoT_dataset/<appareil>/{benign_traffic.csv, gafgyt_attacks/, mirai_attacks/}).
Comptage rapide par dénombrement des sauts de ligne (sans parser les colonnes).

Sortie : results/nbaiot_fed/non_iid.{csv,md,png}
"""

from __future__ import annotations
import os

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
QS = HERE.parents[1]
RAW = Path(os.environ.get("IOT_DATA", Path(__file__).resolve().parents[2] / "data")) / "Raw_NBaIoT_dataset"
OUT = QS / "results" / "nbaiot_fed"
BLUE, ORANGE, AQUA, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"


def nrows(path: Path) -> int:
    """Nombre de lignes de données (sauts de ligne - 1 pour l'en-tête)."""
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            n += chunk.count(b"\n")
    return max(n - 1, 0)


def count_dir(d: Path) -> int:
    return sum(nrows(f) for f in sorted(d.glob("*.csv"))) if d.is_dir() else 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for dev in sorted(p.name for p in RAW.iterdir() if p.is_dir()):
        ddir = RAW / dev
        benign = ddir / "benign_traffic.csv"
        if not benign.exists():
            continue
        normal = nrows(benign)
        gafgyt = count_dir(ddir / "gafgyt_attacks")
        mirai = count_dir(ddir / "mirai_attacks")
        rows.append(dict(appareil=dev, normal=normal, bashlite_gafgyt=gafgyt, mirai=mirai,
                         total=normal + gafgyt + mirai))
    df = pd.DataFrame(rows)
    df["attaques"] = df.bashlite_gafgyt + df.mirai
    df["pct_attaque"] = (100 * df.attaques / df.total).round(2)
    df["a_mirai"] = df.mirai > 0
    df.to_csv(OUT / "non_iid.csv", index=False)

    tot = df[["normal", "bashlite_gafgyt", "mirai"]].sum()
    md = ["# N-BaIoT — répartition non-IID (un client = un appareil)", "",
          "Comptage des flux bruts par appareil (9 clients fédérés).", "",
          "| Client (appareil) | Normal | BASHLITE (gafgyt) | Mirai | % attaques |",
          "|---|--:|--:|--:|--:|"]
    for _, r in df.iterrows():
        mirai = f"{r.mirai:,}".replace(",", " ") if r.mirai else "**0**"
        md.append(f"| {r.appareil} | {r.normal:,} | {r.bashlite_gafgyt:,} | {mirai} | {r.pct_attaque:.1f} |"
                  .replace(",", " "))
    md.append(f"| **Total** | {tot['normal']:,} | {tot['bashlite_gafgyt']:,} | {tot['mirai']:,} | — |"
              .replace(",", " "))
    md += ["",
           "**Caractère non-IID :**",
           f"- **Familles d'attaque absentes chez certains clients** : {int((~df.a_mirai).sum())} appareils "
           f"({', '.join(df[~df.a_mirai].appareil)}) n'ont **aucun flux Mirai** — ils ne voient que BASHLITE.",
           f"- **Déséquilibre de quantité** : le trafic normal va de {df.normal.min():,} "
           f"({df.loc[df.normal.idxmin(),'appareil']}) à {df.normal.max():,} "
           f"({df.loc[df.normal.idxmax(),'appareil']}) flux, soit un facteur "
           f"{df.normal.max()/df.normal.min():.0f}.".replace(",", " "),
           f"- **Part d'attaques élevée partout** : {df.pct_attaque.min():.1f} % à {df.pct_attaque.max():.1f} %. "
           "L'entraînement auto-supervisé n'utilisera que le trafic NORMAL de chaque client."]
    (OUT / "non_iid.md").write_text("\n".join(md), encoding="utf-8")

    # figure : flux par appareil et par classe (échelle log)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(df))
    ax.barh(y, df.normal, color=BLUE, label="Normal")
    ax.barh(y, df.bashlite_gafgyt, left=df.normal, color=ORANGE, label="BASHLITE (gafgyt)")
    ax.barh(y, df.mirai, left=df.normal + df.bashlite_gafgyt, color=AQUA, label="Mirai")
    ax.set_yticks(y, [a.replace("_", " ")[:26] for a in df.appareil], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Nombre de flux")
    ax.set_title("N-BaIoT — flux par appareil (un client = un appareil)", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.savefig(OUT / "non_iid.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("\n".join(md))


if __name__ == "__main__":
    main()
