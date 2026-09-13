# Danışman sorularına yanıt (Zehra Uslu Hoca)

Tüm satır referansları GitHub'daki `main` dalındaki koda aittir:
https://github.com/ahmetyaylalioglu/segquality

---

## 1. Sigmoid + BCEWithLogits çift-sigmoid endişesi (EN ÖNEMLİ)

Hocam endişeniz genel olarak **doğru** — `binary_cross_entropy_with_logits`
kendi içinde sigmoid uyguladığı için, modelin çıkışında ayrıca sigmoid olmamalı,
yoksa çift-sigmoid olur ve eğitim bozulur. Bu kodda **böyle bir hata yoktur**;
çünkü `reconstruct()` içindeki sigmoid **eğitimde kullanılmıyor** — ayrı bir
skor/görselleştirme yoludur. Aşağıda satır satır kanıtı var.

### Modelin çıkışı HAM logit (sigmoidsiz)

- `models/vae3d_vq.py:174` — decoder'ın **son katmanı** çıplak `ConvTranspose3d`;
  `if not is_last` koşulu yüzünden son aşamaya BatchNorm/aktivasyon **eklenmez**
  (kod yorumu: "final stage -> raw logits").
- `models/vae3d_vq.py:222-226` — `forward()` bu ham `logits`'i döndürür.
  **Modelde sigmoid katmanı yoktur.**

### Eğitim BCE'yi HAM logit'e uyguluyor (doğru kullanım)

- `scripts/train_vae3d_vq.py:114` — `logits, vq_term, info = model(x)` → ham logit.
- `scripts/train_vae3d_vq.py:115` → `vq_loss(logits, x, ...)`.
- `losses/vae_losses.py:43` — `bce = F.binary_cross_entropy_with_logits(logits, target)`.
  Girdi ham logit; sigmoid **yalnızca bir kez**, bu fonksiyonun içinde uygulanır.
- `losses/vae_losses.py:44` — `soft_dice_loss(torch.sigmoid(logits), target)`.
  Buradaki sigmoid **sadece Dice terimi için** (Dice olasılık ister), BCE'den
  bağımsız ayrı bir koldur. BCE'ye giren logit hâlâ hamdır.

### `reconstruct()`'taki sigmoid eğitimde ASLA çağrılmaz

- `models/vae3d_vq.py:228-231` — `reconstruct()` = `torch.sigmoid(logits)`.
  Bu metot **yalnızca değerlendirme/görselleştirmede** kullanılır
  (`scripts/evaluate_correlation_3d_vq.py`). Eğitim döngüsü (`run_epoch_vq`)
  `reconstruct()`'ı hiç çağırmaz — doğrudan `model(x)`'in ham logit'ini kullanır.

**Özet:** Eğitimde çıktı → BCEWithLogits zincirinde tek bir sigmoid vardır (BCE'nin
içindeki). `reconstruct()`'taki sigmoid ayrı, eğitim-dışı bir yoldur. **Çift-sigmoid
yoktur, sonuçlar sağlamdır.**

> Not: SDF modelinde (WP-3) sigmoid hiç yoktur; kayıp `F.smooth_l1_loss(pred_sdf,
> target_sdf)`'tir (`scripts/train_vae3d_vq_sdf.py:99`).

---

## 2. VQ-VAE eğitimde giriş bozuk mu, temiz mi? → SADECE TEMİZ

- `scripts/train_vae3d.py:114-124` — eğitim/val yükleyicileri `RAOSVolumes`'tan
  **temiz GT** maskeleri okur.
- `transform=_VolumeTransform()` → `augment_volume` (`train_vae3d.py:77`) =
  `return mask` (no-op; maskeyi değiştirmez).
- `scripts/train_vae3d_vq.py:112-115` — hedef, girişin kendisidir (temiz maskenin
  saf autoencoding'i). Bozma (`SyntheticDegrader`) yalnızca değerlendirmede kullanılır
  (`scripts/evaluate_correlation_3d.py:160-186`).

Yani model eğitimde **hiç bozuk maske görmez**; bozuk maskeler sadece test/korelasyon
aşamasında, skorun gerçek Dice ile örtüşüp örtüşmediğini ölçmek için üretilir.

---

## 3. SDF → binary dönüşümü ve görüntü uzayında karşılaştırma → EVET

SDF tahmini, eşiklenip (köprü: `sdf < 0 → iç`) binary maskeye çevrilir ve
karşılaştırma **binary (görüntü) uzayında** yapılır:

- Köprü: `quality/sdf.py:50-59` — `sdf_to_binary(sdf) = (sdf < 0)`.
- Eğitim val Dice: `scripts/train_vae3d_vq_sdf.py:113-114` —
  `pred_bin = (pred_sdf < 0).float()` → binary Dice.
- Değerlendirme: `models/vae3d_vq_sdf.py:52-58` — `reconstruct` köprüden geçer,
  mevcut `recon_dice` skoru binary üzerinde çalışır.

**Sonuç:** SDF-VQ ρ = 0.844, binary baseline ρ = 0.850 — pratikte eşit, kayıp yok.

**Örnek sonuç görüntüleri** (ekte / `figures/` klasöründe):
- `figures/sdf/sdf_reconstruction_preview.png` — girdi | tahmin edilen SDF haritası
  (±3) | eşiklenmiş binary (`sdf<0`) + self-Dice.
- `figures/sdf/correlation_scatter.png` — SDF ρ=0.844 saçılımı.
- `figures/sdf/binary_vs_sdf_comparison.png` — binary vs SDF (global + bant).
- `figures/vqvae/reconstruction_preview.png` — VQ-VAE recon önizlemesi.
- `figures/vqvae/correlation_scatter.png` — VQ-VAE ρ=0.862 saçılımı.

---

## 4. Straight-through satırı doğru mu? → EVET

`models/vae3d_vq.py:92` — `z_q = z + (z_q - z).detach()`

- **İleri geçiş:** değer `z + (z_q − z) = z_q` (quantize edilmiş kod) olur — sağda
  sadece `z_q` kalır (tespitiniz doğru).
- **Geri geçiş:** `(z_q − z).detach()` gradyansızdır, dolayısıyla gradyan `z_q`'dan
  doğrudan `z`'ye (encoder çıkışına) akar; türevlenemeyen `argmin` atlanır.

Bu, van den Oord et al. (2017) VQ-VAE makalesindeki kanonik straight-through
estimator'ın tam halidir. **Doğru.**

---

## 5. VectorQuantizer3D referansı

`models/vae3d_vq.py:44-102`. Elle yazıldı (MONAI kullanılmadı), şu makaleye göre:
**van den Oord, Vinyals, Kavukcuoglu — "Neural Discrete Representation Learning"
(VQ-VAE), NeurIPS 2017** (arXiv:1711.00937). 2B özellik haritası yerine **3B uzamsal
codebook**'a (`(B, C, 4, 4, 4)` = hacim başına 64 pozisyon) genelleştirildi:

- En yakın komşu (nearest-neighbor) kod araması — `vae3d_vq.py:77-84`.
- Straight-through estimator — `vae3d_vq.py:92` (madde 4).
- Codebook + commitment kaybı — `vae3d_vq.py:87-89` (KL terimi yok).
- Çöküş teşhisi: perplexity + kullanılan kod sayısı — `vae3d_vq.py:97-101`.

---

## 6. Yeni istekler — anlaşıldı, `dev` dalında yapılacak

**(a) codebook_loss'suz eğitim, commitment_cost = 1.**
Gerekçenizi anladım: VQ-VAE burada denoising için kullanılıyor; bozuk görüntü
feature'larının codebook'u kendine çekmesini istemiyoruz — amaç bozuk feature'lardan
*referans* maskeyi elde etmek. `codebook_loss`'u kaldırıp yalnızca commitment terimini
(cost=1) bırakınca kodlar sabit kalır, encoder kodlara *uymaya* zorlanır ama kodları
bozuk girdilere doğru *çekemez*. Bunu `dev` dalında ayrı bir varyant olarak eğitip
ρ ve codebook kullanımını baseline ile karşılaştıracağım (`main` bozulmadan).

İlgili satır: `models/vae3d_vq.py:87-89`
```python
codebook_loss   = F.mse_loss(z_q, z.detach())      # <- kaldırılacak
commitment_loss = F.mse_loss(z, z_q.detach())
vq_loss = codebook_loss + self.commitment_cost * commitment_loss
# yeni: vq_loss = self.commitment_cost * commitment_loss,  commitment_cost = 1
```

**(b) Çok-aday / belirsizlik (WP-4).**
Latent olasılık dağılımı ile temsil edilmediği için, önerdiğiniz gibi en yakın K kodu
uzaklıklarına göre **ters-orantılı ağırlıklı ortalama** ile birleştireceğim:

```
M = ( (1/d0)·M0 + (1/d1)·M1 + ... ) / ( (1/d0) + (1/d1) + ... )
```

Bundan (i) bir konsensüs maske ve (ii) 3B belirsizlik/şüpheli-bölge haritası çıkarıp,
belirsizliğin gerçek hata bölgeleriyle (`bozuk XOR referans`) örtüşüp örtüşmediğini
doğrulayacağım. Bu tamamen inference aşamasında, yeniden eğitim gerektirmez.
