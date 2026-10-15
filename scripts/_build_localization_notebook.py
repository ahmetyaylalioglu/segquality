"""Generator for notebooks/hata_lokalizasyon_kiyas.ipynb (Part 1 of the diffusion workstream).

Builds a reviewable notebook that benchmarks the per-voxel error-LOCALIZATION quality of the
existing trained models (recon-diff, FP/FN, cascade) against the true error map, with a voxel
metric (AUROC + AP) and patient-level overlays across degradation levels. Additive, no training.

    python scripts/_build_localization_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "hata_lokalizasyon_kiyas.ipynb"

cells: list[tuple[str, str]] = []


def md(src: str) -> None:
    cells.append(("markdown", src.strip("\n")))


def code(src: str) -> None:
    cells.append(("code", src.strip("\n")))


# ===========================================================================
md(r"""
# Hata lokalizasyonu kıyaslaması — "şurada hata var" (Parça 1)

Global Dice-kestirimi (`est_dice` ρ) *"bu maske iyi mi kötü mü?"* sorusunu yanıtlar. Ama bir
maskenin **nerede** yanlış olduğunu işaretlemek ayrı bir iştir ve global ρ bunu ölçmez. Bu defter,
hâlihazırda eğitilmiş modellerin ürettiği **voxel bazında hata haritalarını**, gerçek hata
haritasına (`bozuk XOR temiz`) karşı ölçer.

Ölçüt: **voxel-AUROC** ve **AP (PR-AUC)**. Hata voksülleri az (dengesiz sınıf) olduğundan AP daha
bilgilendiricidir. Hiçbir şey reconstruction sadakatine göre değil, yalnızca **enjekte edilen
hatayı ne kadar iyi yerelleştirdiğine** göre değerlendirilir. Eğitim yok; yalnızca yeni dosyalar.

Kıyaslanan üç işaretleyici (hepsi mevcut checkpoint'lerden):
- **recon_diff** — `|bozuk − reconstruct(bozuk)|` (yalnız şekil, görüntüsüz; VAR)
- **fpfn** — `max(FP, FN)` tahmin haritaları (görüntü-koşullu; G-VAR, G-VQ)
- **cascade** — `|refine(bozuk) − bozuk|` (Faz-4 referans-maske farkı)

Protokol: test bölünmesi, `seed=42`, `n_per_mask=5`, 3B primitifler.
""")

# ---------------------------------------------------------------- setup
code(r"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')  # Windows OpenMP çifte-yükleme koruması

import sys, logging
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.WARNING)

def find_root(start):
    p = Path(start).resolve()
    for c in [p, *p.parents]:
        if (c / 'scripts').is_dir() and (c / 'models').is_dir():
            return c
    return p

PROJECT_ROOT = find_root(Path.cwd())
for p in (PROJECT_ROOT, PROJECT_ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data import load_config
from data.degradations import dice as _dice
from data.degradations_3d import PRIMITIVES_3D
from data.degradations_partial_3d import PARTIAL_PRIMITIVES_3D
from train import resolve_device, set_seed
import evaluate_correlation_3d as ev3d
import evaluate_correlation_imgcond as evic
import evaluate_correlation_imgcond_fpfn as evfpfn
import evaluate_localization as loc

# train/evaluate_* importları matplotlib'i Agg'e çeker; inline'ı importlardan SONRA kur.
%matplotlib inline

SEED = 42
device = resolve_device('auto')
CKPT = PROJECT_ROOT / 'outputs' / 'checkpoints'
ALL_FNS = {**PRIMITIVES_3D, **PARTIAL_PRIMITIVES_3D}
print('device =', device)
""")

md(r"""
## Değerlendirme kümesi (görüntü + bozuk maske + temiz referans)

FP/FN ve cascade görüntü ister; recon_diff istemez. Tek bir küme kurulur (görüntü+maske),
her satırda temiz referans taşınır — gerçek hata haritası `bozuk XOR temiz`'den çıkar.
""")

code(r"""
cfg = load_config(str(PROJECT_ROOT / 'configs' / 'vae3d_imgcond_fpfn.yaml'))
cfg.setdefault('data', {})['data_source'] = 'precomputed'
set_seed(SEED)
clean_masks, clean_images = evic.load_clean_image_volumes(cfg, 'test', None, SEED)
degrader = ev3d.calibrate_degrader(cfg, clean_masks, SEED)
broken, images, clean_ref, true_dice, is_clean = evfpfn.build_eval_set_fpfn(
    clean_masks, clean_images, degrader, 5, SEED)
is_clean = np.asarray(is_clean, dtype=bool)
print(f'hasta={len(clean_masks)}  hacim={len(broken)} '
      f'({is_clean.sum()} temiz + {(~is_clean).sum()} bozuk)  şekil={broken.shape[1:]}')
""")

md(r"""
## İşaretleyicileri yükle ve voxel-AUROC / AP hesapla

`evaluate_localization` modülünün işaretleyici fonksiyonları ve skorlayıcısı yeniden kullanılır
(defterde tekrar tanım yok). Yalnızca diskte bulunan checkpoint'ler yüklenir.
""")

code(r"""
REG = [
    ('recon_diff/VAR', loc.recon_diff_map, CKPT / 'vae3d_var_best.pt',    'recon'),
    ('fpfn/G-VAR',     loc.fpfn_map,       CKPT / 'vae3d_g_var_best.pt',  'fpfn_var'),
    ('fpfn/G-VQ',      loc.fpfn_map,       CKPT / 'vae3d_g_vq_best.pt',   'fpfn_vq'),
    ('cascade/Faz4',   loc.cascade_map,    CKPT / 'vae3d_h_cascade_best.pt', 'cascade'),
]
import evaluate_correlation_3d_var as evv
import evaluate_correlation_var_fpfn as evvf
import evaluate_correlation_cascade as evc

def _load(kind, path):
    if kind == 'recon':     return evv.load_model(path, device)
    if kind == 'fpfn_var':  return evvf.load_var_fpfn_model(path, device)
    if kind == 'fpfn_vq':   return evfpfn.load_fpfn_model(path, 'binary', 'vq', device)
    if kind == 'cascade':
        cfg_cas = load_config(str(PROJECT_ROOT / 'configs' / 'vae3d_cascade.yaml'))
        cfg_cas.setdefault('data', {})['data_source'] = 'precomputed'
        return evc.load_cascade(cfg_cas, path, device)

localizers = {}
rows = []
for lbl, fn, path, kind in REG:
    if not path.exists():
        print('YOK, atlanıyor:', lbl, path.name); continue
    model = _load(kind, path)
    localizers[lbl] = (fn, model)
    r = loc.score_localizer(fn, model, broken, images, clean_ref, true_dice, is_clean, device)
    row = {'işaretleyici': lbl, 'n': r['n'],
           'AUROC': round(r['auroc_mean'], 3), 'AUROC±': round(r['auroc_std'], 3),
           'AP': round(r['ap_mean'], 3), 'AP±': round(r['ap_std'], 3)}
    for b, bv in r['bands'].items():
        row[f'AUROC {b.split()[0]}'] = round(bv['auroc'], 3)
    rows.append(row)
    print(f"{lbl:18s} AUROC={r['auroc_mean']:.3f}  AP={r['ap_mean']:.3f}  (n={r['n']})")

df_loc = pd.DataFrame(rows).sort_values('AUROC', ascending=False).reset_index(drop=True)
display(df_loc)
print('Not: AUROC yüksek=iyi yerelleştirme; AP sınıf-dengesizliğine karşı daha katı. '
      'Bantlar gerçek-Dice seviyelerinde (kolay→çok zor) ayrıştırma gücünü gösterir.')
""")

md(r"""
## Hasta bazında overlay — hafif → ağır bozulma

Tek hastada üç bozulma seviyesinde her işaretleyicinin ürettiği yumuşak hata haritası, **gerçek
hata** (sarı = `bozuk XOR temiz`) ile yan yana gösterilir. Haritalar bozuk maske üzerine
bindirilir; kırmızı yoğunluk = "burada hata var" tahmini.
""")

code(r"""
PANEL_IDX = 0
gt = clean_masks[PANEL_IDX].astype(bool); img = clean_images[PANEL_IDX]
specs = [('erode', 0.2, 'hafif'), ('center_only', 0.5, 'orta'), ('center_only', 0.75, 'ağır')]
zc = gt.shape[0] // 2
labels = list(localizers)
cols = ['BT görüntü', 'temiz (GT)', 'bozuk', 'gerçek hata'] + labels
fig, axes = plt.subplots(len(specs), len(cols), figsize=(2.25 * len(cols), 2.25 * len(specs)), squeeze=False)
prows = []
for ri, (name, sev, lvl) in enumerate(specs):
    b = np.asarray(ALL_FNS[name](gt, sev, np.random.default_rng(SEED))) > 0
    d_true = float(_dice(gt, b))
    true_err = b ^ gt
    axes[ri, 0].imshow(img[zc], cmap='gray'); axes[ri, 1].imshow(gt[zc], cmap='gray')
    axes[ri, 2].imshow(b[zc], cmap='gray')
    axes[ri, 3].imshow(b[zc], cmap='gray')
    axes[ri, 3].imshow(np.ma.masked_where(~true_err[zc], true_err[zc]), cmap='autumn_r', alpha=0.9, vmin=0, vmax=1)
    prow = {'seviye': lvl, 'bozulma': f'{name}/{sev}', 'gerçek Dice': round(d_true, 3)}
    for ci, lbl in enumerate(labels):
        fn, model = localizers[lbl]
        soft = fn(model, b[None].astype('float32'), img[None].astype('float32'), device)[0]
        from sklearn.metrics import roc_auc_score
        te = true_err.ravel()
        au = roc_auc_score(te, soft.ravel()) if te.any() and not te.all() else float('nan')
        prow[lbl] = round(float(au), 3)
        axes[ri, 4 + ci].imshow(b[zc], cmap='gray')
        axes[ri, 4 + ci].imshow(np.ma.masked_where(soft[zc] < 0.2, soft[zc]), cmap='Reds', alpha=0.85, vmin=0, vmax=1)
        axes[ri, 4 + ci].set_xlabel(f'AUROC={au:.2f}', fontsize=7.5)
    for ci in range(len(cols)):
        axes[ri, ci].set_xticks([]); axes[ri, ci].set_yticks([])
    axes[ri, 0].set_ylabel(f'{lvl}\n{name}/{sev}', fontsize=9)
    axes[ri, 2].set_xlabel(f'Dice={d_true:.2f}', fontsize=8)
    prows.append(prow)
for ci, t in enumerate(cols):
    axes[0, ci].set_title(t, fontsize=8.5)
fig.suptitle(f'Hata lokalizasyonu — hasta {PANEL_IDX}: işaretleyiciler (hafif → ağır)', fontsize=12)
fig.tight_layout()
out = PROJECT_ROOT / 'outputs' / 'rapor_lokalizasyon_panel.png'
fig.savefig(out, dpi=150, bbox_inches='tight'); print('kaydedildi:', out); plt.show()
display(pd.DataFrame(prows))
""")

# ---------------------------------------------------------------- Özet
md(r"""
## Bulguların özeti

Voxel-AUROC/AP tablosu, "nerede hata var" sorusuna hangi sinyalin en iyi cevap verdiğini gösterir.
Bu, global `est_dice` ρ ile **aynı şey değildir**: bir model global skoru güçlü verip
lokalizasyonu zayıf verebilir ya da tersi. Parça 2'de (diffusion denoising-artığı) bu tablodaki
en iyi değeri yenmeyi hedefliyoruz; karar yine bu lokalizasyon metriğiyle verilecek.
""")

code(r"""
print('Lokalizasyon sıralaması (AUROC):')
for _, r in df_loc.iterrows():
    print(f"  {r['işaretleyici']:18s} AUROC={r['AUROC']:.3f}  AP={r['AP']:.3f}")
""")

# ===========================================================================
nb = {
    "cells": [
        {"cell_type": t, "metadata": {},
         **({"source": s.splitlines(keepends=True)} if t == "markdown"
            else {"source": s.splitlines(keepends=True), "outputs": [], "execution_count": None})}
        for t, s in cells
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, ensure_ascii=False, indent=1)
print("wrote", OUT, "|", len(cells), "cells")
