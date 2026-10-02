# Açıklama Notebook'ları

Bu klasör, danışmana sunmak için hazırlanmış **bağımsız çalışan** Jupyter
notebook'larını içerir. Her notebook proje modüllerine ihtiyaç duymadan (gerekli
kodu kendi içine gömerek) çalışır; sadece notebook + veri/checkpoint dosyaları
yeterlidir. Dosyalar yoksa notebook'lar **çökmez**, net Türkçe uyarı basıp
önceden-hesaplanmış sonuçları gösterir.

## Notebook'lar

### 1. `3B_VAE_aciklama_ve_teshis.ipynb` — Standart 3B VAE + Teşhis
Standart (sürekli latent) 3B VAE'nin mimarisi, kayıp fonksiyonu ve **mean-reversion
teşhisi**. Danışmanın gözlemlediği "her hastaya jenerik karaciğer" sorununun üç
ölçümle (cross-reconstruction oranı, latent çeşitliliği, recon-pairwise vs
input-pairwise Dice) gerçek olduğunu gösterir.

### 2. `VQVAE_cozum_ve_karsilastirma.ipynb` — VQ-VAE ile Çözüm (BU ÇALIŞMA)
Mean-reversion sorununun **VQ-VAE** (uzamsal ayrık codebook) ile nasıl çözüldüğünü
gösterir. Bölümler:
- **B1–B2:** Problem özeti + VQ-VAE'nin neden çözdüğü (uzamsal ayrık kod ızgarası).
- **B3–B4:** Gömülü VQ-VAE mimarisi (`VectorQuantizer3D` + `ConvVQVAE3D`, straight-through)
  ve kayıp (`vq_loss` = Dice+BCE + codebook + commitment, KL yok). `use_skip=False` —
  VQ tek başına çözdü, skip connection'a gerek kalmadı.
- **B5:** Canlı demo — sentetik küre VQ-VAE'den geçer, boyut akışı + perplexity.
- **B6:** KANIT — standart VAE ve VQ-VAE yan yana canlı teşhis + iki heatmap.
- **B7–B8:** Sonuç tablosu + yorum (mean-reversion çözüldü, korelasyon korundu).

**Ana sonuç:** recon-pairwise Dice 0.992 → 0.744 (girdi çeşitliliği 0.733 seviyesine),
cross-reconstruction oranı 1.005 → 1.239, kalite korelasyonu ρ 0.851 → 0.862.

### 3. `3B_VAE_liver_aciklama_teshis.ipynb`
Standart VAE teşhis notebook'unun karaciğer-görselli bir varyantı (arşiv).

## Yol Değişkenleri

Her notebook'un başında düzenlenecek yollar vardır. `VQVAE_cozum_ve_karsilastirma.ipynb` için:

```python
VAL_NPZ_PATH  = "outputs/volumes/val.npz"                     # doğrulama hacimleri
VQ_CKPT_PATH  = "outputs/checkpoints/vae3d_vqvae_best.pt"     # eğitilmiş VQ-VAE
STD_CKPT_PATH = "outputs/checkpoints/vae3d_best_latent64.pt"  # standart VAE (karşılaştırma)
K = 20        # teşhis edilecek karaciğer sayısı
SEED = 42     # tekrarlanabilirlik
```

## Gereksinimler

`torch`, `numpy`, `matplotlib`, `scipy`. Proje modülü **gerekmez** — tüm model ve
teşhis kodu notebook içine gömülüdür.
