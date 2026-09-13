# SegQuality — İlerleme Raporu (13.09.2026)

Danışman: Zehra Uslu Hoca · Repo: https://github.com/ahmetyaylalioglu/segquality

Bu dosya iki bölümden oluşur:
- **Bölüm A (§1–5):** Önceki danışman sorularına satır-referanslı yanıtlar. Bu
  referanslar `main` dalındaki koda aittir.
- **Bölüm B (§6–9):** Sizden gelen iki yeni isteğin (codebook_loss'suz eğitim + çok-aday
  belirsizlik) **tamamlanmış sonuçları**. Bu bölümdeki kod ve satır referansları `dev`
  dalındaki commit `18c8923`'e aittir (henüz `main`'e merge edilmedi, `main` bozulmadı).

---

## BÖLÜM A — Önceki sorulara yanıtlar

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

---

## BÖLÜM B — Yeni çalışmalar (tamamlandı)

Aşağıdaki her şey `dev` dalı, commit `18c8923`. `main` bozulmadı; mevcut 162 test yeşil
kaldı. Tüm ρ değerleri aynı test protokolüyle (n=282: 47 temiz + 235 bozuk), birincil QC
skoru `recon_dice ρ`, ikincil skor `quant_error ρ` (kuantizasyon mesafesi).

---

## 6. codebook_loss'suz eğitim (commitment_cost = 1) — YAPILDI

İsteğinizi geriye-uyumlu bir bayrak olarak ekledim: `vq_mode`
(`models/vae3d_vq.py`). Default `standard` (eski davranış, byte-aynı); yeni modlar
codebook'u **dondurur** (codebook_loss YOK, EMA YOK). CLI: `--vq-mode`, `--commitment-cost`.

**Donukluğun kanıtı (perplexity ile değil, doğrudan):** codebook gradyanı `None` ve
eğitim öncesi/sonrası ağırlık **bit-aynı** (max|Δ| = 0.0). Perplexity kod *seçimine*
bağlı olduğu için donukluğu kanıtlamaz — bu yüzden gradyan + bit-eşitlik kullandım.

**Sorun ve kök neden — ölçek uyuşmazlığı.** Saf `commitment_only` modunda VQ kaybı
patladı (epoch 1'de 0.73 → epoch 40'ta 964). Bu bir bug DEĞİL, beklenen bir ölçek
sorunu: donuk kodlar init'te `uniform(−1/K, +1/K) ≈ ±0.002` (çok küçük) kalırken,
encoder çıkışı reconstruction için serbestçe büyüyor (ölçülen std ~19, max ~100). O
zaman `commitment = ‖z − z_q‖²` sınırsız ıraksıyor. Reconstruction, decoder'daki
BatchNorm ölçeği sildiği için ayakta kalıyor ama **kuantizasyon-mesafesi QC kanalı
ölüyor** (quant_error ρ: −0.407 → −0.020).

**Çözüm — ölçek-uyumlu donuk codebook.** Kodları encoder ölçeğinde başlatınca
(`commitment_only_scaled`, N(0,1) donuk init) commitment sınırlanıyor ve codebook
**hâlâ donuk** kalıyor (niyetiniz korunur). Her iki QC kanalı da geri geliyor.

### Dört varyant karşılaştırması (test, n=282)

| Varyant | recon_dice ρ | quant_error ρ | val recon Dice | Codebook |
|---|---|---|---|---|
| **standard** (base) | **0.862** | **−0.407** | ~0.87 | Eğitiliyor |
| commit_only (saf, ±0.002 donuk) | 0.771 | −0.020 (ölü) | 0.86 | ~4 kod, VQ patlıyor |
| **commit_scaled** (N(0,1) donuk) | **0.823** | **−0.323** | 0.889 | 4–5 kod, sınırlı |
| commit_cosine (L2-normalize) | 0.461 | −0.263 | **0.928** | 263 kod, sınırlı |

**Öneri:** `commit_scaled` — donuk codebook (niyetiniz) + ölçek düzeltmesi = QC korunur
(ρ 0.823, base 0.862'ye yakın). Teze taşınacak varyant bu.

**EMA neden uygun değil:** EMA commitment'ı sınırlar AMA kodları encoder
feature'larına doğru **hareket ettirir** — tam da kaçınmak istediğimiz şey (bozuk maske
feature'ları codebook'u kendine çekmemeli). Ölçek eşitleme kodları donuk tutarak aynı
sorunu çözer; doğru çözüm EMA değil, ölçek-uyumlu donuk init'tir.

---

## 7. "En iyi reconstruction ≠ en iyi QC" — kritik bulgu

`commit_cosine` (encoder + kodlar L2-normalize) **en iyi reconstruction'ı** verir (val
Dice 0.928), ama **en kötü QC dedektörüdür** (recon_dice ρ 0.461'e çöker). Sebep: model
o kadar iyi yeniden-inşa eder ki **bozuk maskeleri bile** temiz kurar → reconstruction
hatası artık iyi/kötü maskeyi ayırt edemez. Yani "reconstruction-error = kalite"
öncülü, otoencoder fazla güçlü olunca kırılır. **Sonuç:** bu projede reconstruction
kalitesi hedef DEĞİL; QC sinyali *kontrollü* bir darboğaz ister, kusursuz değil.

---

## 8. WP-4 — çok-aday konsensüs + 3B belirsizlik haritası — YAPILDI

Önerdiğiniz mesafe-ağırlıklı formülü uyguladım (inference-only, yeniden eğitim yok;
`scripts/multi_candidate_wp4.py`). Her latent pozisyonda en yakın **k=5** kod → k aday
maske; mesafe-ağırlıklı konsensüs:

```
P = ( (1/d0)·p0 + (1/d1)·p1 + ... ) / ( (1/d0) + (1/d1) + ... )
```

Konsensüs maske = `P > 0.5`. Belirsizlik = her voxel'de ikili entropi
`H(P) = −P·logP − (1−P)·log(1−P)` (adaylar anlaşırsa 0, tartışırsa yüksek).

### Doğrulama (test, n=282)

| Doğrulama | Sonuç |
|---|---|
| voxel AUROC (belirsizlik → bozuk XOR temiz-referans) | **0.723** |
| hacim ρ (ort. belirsizlik, 1−true_Dice), tüm | **0.384** |
| hacim ρ, yalnız bozuk | 0.345 |

Yani belirsizliğin yüksek olduğu voxel'ler gerçekten hatalı bölgelerle örtüşüyor
(AUROC 0.72; 0.5 rastgele olurdu). Belirsizlik ağırlıkla organ **sınırı/yüzeyi**
etrafında yoğunlaşıyor — piksel-mükemmel bir hata haritası değil ama gerçek hata
bölgelerini güvenilir işaretliyor.

STEP 0 (aday çeşitliliği, `scripts/wp4_step0_diversity.py`): runner-up kodlar argmin'den
anlamlı farklı maskelere çözülüyor (Dice(M0,M1)=0.79); anlaşmazlığın ~%46'sı sınır
bandında, gerisi gövde şekli — yani çok-adaylılık gerçek bir çeşitlilik taşıyor.

Görsel: `figures/wp4/wp4_uncertainty.png` — 4 bozuk hacmin aksiyel orta kesiti,
sütunlar: girdi | konsensüs (P>0.5) | belirsizlik (entropi) | hata (girdi XOR referans).

---

## 9. Kod ve dosya haritası (dev, commit 18c8923)

- `models/vae3d_vq.py` — `vq_mode` bayrağı (`standard` | `commitment_only` |
  `commitment_only_scaled` | `commitment_only_cosine`), donuk codebook mantığı.
- `scripts/train_vae3d_vq.py` — `--vq-mode`, `--commitment-cost` CLI.
- `scripts/smoke_commitment_only.py` — donukluk kanıtı (grad None + bit-aynı) + ρ.
- `scripts/multi_candidate_wp4.py` — WP-4 konsensüs + belirsizlik + doğrulama + görsel.
- `scripts/wp4_step0_diversity.py` — aday çeşitliliği ön-teşhisi.
- `outputs/vq_variants_and_wp4.md` — dört varyant + WP-4'ün kalıcı ayrıntılı dökümü.
- `figures/wp4/wp4_uncertainty.png` — belirsizlik görseli.

### Tek cümlelik özet
Donuk codebook isteğinizi ölçek-uyumlu init ile sağlıklı çalışır hale getirdim
(`commit_scaled`, QC korunur, ρ 0.823); "en iyi reconstruction ≠ en iyi QC" tuzağını
deneysel kanıtladım (cosine); ve önerdiğiniz mesafe-ağırlıklı çok-aday yaklaşımıyla
gerçek hata bölgelerini işaretleyen bir belirsizlik haritası ürettim (AUROC 0.72).
