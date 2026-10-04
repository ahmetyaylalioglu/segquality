"""Tek sayfalik ozet figur (hocaya mail eki). Tum test seti (n=282) toplamlarindan.
Kaynak: outputs/rapor_fpfn_estdice_basari.csv + outputs/rapor_fpfn_genis_tablo.csv
Tek ornek DEGIL -- tum goruntuler uzerinden Dice tahmini dogrulugu + ASSD.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
acc = pd.read_csv(ROOT / "outputs/rapor_fpfn_estdice_basari.csv")
wide = pd.read_csv(ROOT / "outputs/rapor_fpfn_genis_tablo.csv")

BEST = "G-ikili/SimVQ"
order = ["G-ikili/VQ", "G-ikili/SimVQ", "G-SDF/VQ", "G-SDF/SimVQ"]
acc = acc.set_index("model").loc[order].reset_index()

# hata turu etiketleri (TR) + mantikli siralama (eksik-bolge -> fazla/gurultu)
TR = {
    "heavy_erode": "agir erozyon", "center_only": "sadece merkez", "erode": "erozyon",
    "translate": "oteleme", "partial": "kismi", "remove_blob": "parca silme",
    "dilate": "genisleme", "add_false_positive": "yanlis pozitif ekleme",
    "boundary_noise": "sinir gurultusu",
}
TYPE_ORDER = ["heavy_erode", "center_only", "erode", "translate", "partial",
              "remove_blob", "dilate", "add_false_positive", "boundary_noise"]
wb = wide[wide["model"] == BEST].set_index("hata türü").loc[TYPE_ORDER]

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.26)

# ---- Panel 1: GT-SIZ DICE TAHMINI DOGRULUGU (4 model) ----
ax = fig.add_subplot(gs[0, 0])
x = np.arange(len(order))
w = 0.26
ax.bar(x - w, acc["Pearson r"], w, label="Pearson r", color="#2c7fb8")
ax.bar(x, acc["Spearman ρ"], w, label="Spearman $\\rho$", color="#41b6c4")
ax.bar(x + w, acc["≤0.10 oranı"], w, label="|hata|$\\leq$0.10 oranı", color="#a1dab4")
for i, m in enumerate(order):
    ax.text(i, 0.02, f"MAE={acc['MAE'][i]:.3f}", ha="center", va="bottom",
            fontsize=8, rotation=90, color="#444")
ax.set_xticks(x)
ax.set_xticklabels(order, fontsize=9, rotation=12)
ax.set_ylim(0, 1.05)
ax.set_title("GT'siz Dice tahmini dogrulugu — tum test seti (n=282)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8, loc="lower right")
ax.axhline(0.9, ls=":", c="gray", lw=0.8)
ax.annotate(f"en iyi: {BEST}\nPearson {acc.loc[acc.model==BEST,'Pearson r'].values[0]:.3f} / "
            f"Spearman {acc.loc[acc.model==BEST,'Spearman ρ'].values[0]:.3f}\n"
            f"MAE {acc.loc[acc.model==BEST,'MAE'].values[0]:.3f}",
            xy=(1, 0.52), xytext=(1.6, 0.30), fontsize=8,
            ha="left", color="#b30000",
            arrowprops=dict(arrowstyle="->", color="#b30000", lw=1))

# ---- Panel 2: DICE — girdi vs duzeltilmis (en iyi model, hata turu bazli) ----
ax = fig.add_subplot(gs[0, 1])
labels = [TR[t] for t in TYPE_ORDER]
xp = np.arange(len(labels))
ax.bar(xp - 0.2, wb["Dice girdi"], 0.4, label="girdi (bozuk) Dice", color="#d0d0d0")
ax.bar(xp + 0.2, wb["Dice düzelt"], 0.4, label="duzeltilmis Dice", color="#2c7fb8")
for i, t in enumerate(TYPE_ORDER):
    dd = wb["ΔDice"].iloc[i]
    ax.text(i + 0.2, wb["Dice düzelt"].iloc[i] + 0.01, f"+{dd:.2f}", ha="center",
            va="bottom", fontsize=8, color="#006d2c" if dd > 0.01 else "#999")
ax.set_xticks(xp)
ax.set_xticklabels(labels, fontsize=8, rotation=35, ha="right")
ax.set_ylim(0, 1.08)
ax.set_title(f"Maske duzeltme kazanimi — {BEST}", fontsize=11, fontweight="bold")
ax.legend(fontsize=8, loc="lower left")

# ---- Panel 3: ASSD — girdi vs duzeltilmis (en iyi model, hata turu bazli) ----
ax = fig.add_subplot(gs[1, 0])
ax.bar(xp - 0.2, wb["ASSD girdi"], 0.4, label="girdi (bozuk) ASSD", color="#fdae6b")
ax.bar(xp + 0.2, wb["ASSD düzelt"], 0.4, label="duzeltilmis ASSD", color="#e6550d")
for i, t in enumerate(TYPE_ORDER):
    da = wb["ΔASSD"].iloc[i]
    c = "#006d2c" if da > 0.05 else ("#b30000" if da < -0.02 else "#999")
    ax.text(i, max(wb["ASSD girdi"].iloc[i], wb["ASSD düzelt"].iloc[i]) + 0.2,
            f"{da:+.1f}", ha="center", va="bottom", fontsize=8, color=c)
ax.set_xticks(xp)
ax.set_xticklabels(labels, fontsize=8, rotation=35, ha="right")
ax.set_title(f"Sinir hatasi (ASSD, voxel) — {BEST}  [dusuk = iyi]",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8, loc="upper right")

# ---- Panel 4: est_dice korelasyonu hata turu bazli (en iyi model) ----
ax = fig.add_subplot(gs[1, 1])
rvals = wb["r est_dice"].values
colors = ["#006d2c" if r >= 0.9 else ("#fd8d3c" if r >= 0.8 else "#b30000")
          for r in rvals]
ax.barh(xp, rvals, color=colors)
for i, r in enumerate(rvals):
    ax.text(r - 0.02, i, f"{r:.2f}", ha="right", va="center", fontsize=8,
            color="white", fontweight="bold")
ax.set_yticks(xp)
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0, 1.0)
ax.axvline(0.9, ls=":", c="gray", lw=0.8)
ax.set_title(f"GT'siz Dice tahmini — hata turu bazli guven (Pearson r) — {BEST}",
             fontsize=11, fontweight="bold")
ax.set_xlabel("est_dice vs gercek Dice korelasyonu (r)", fontsize=9)

fig.suptitle(
    "FP/FN cift-bas maske duzeltme — GT'siz Dice tahmini ozeti (tum test seti, 47 hasta, 282 hacim)",
    fontsize=13, fontweight="bold", y=0.985)
out = ROOT / "outputs/ozet_hocaya_fpfn.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print("yazildi:", out)
