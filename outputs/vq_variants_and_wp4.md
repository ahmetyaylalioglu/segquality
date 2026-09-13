# VQ-VAE Varyant Ablasyonu + WP-4 Belirsizlik Haritası

GT-free segmentasyon kalite kestirimi projesi için VQ-VAE kuantizasyon tasarımının
kalite (QC) sinyaline etkisi ve WP-4 çok-aday belirsizlik sonuçları. Tüm ρ değerleri
test setinde, aynı sentetik bozulma protokolüyle (n=282: 47 temiz + 235 bozuk),
`recon_dice ρ` birincil QC skoru.

## 1. Dört varyant karşılaştırması

| Varyant | recon_dice ρ | quant_error ρ | val recon Dice | Codebook durumu |
|---|---|---|---|---|
| **standard** (base) | **0.862** | **−0.407** | ~0.87 | Eğitiliyor (codebook_loss açık) |
| commit_only (saf, ±0.002 donuk) | 0.771 | −0.020 | 0.86 | ~4 kod, VQ terimi patlıyor (0.7→964) |
| **commit_scaled** (N(0,1) donuk) | **0.823** | **−0.323** | 0.889 | 4–5 kod, VQ terimi sınırlı |
| commit_cosine (L2-normalize) | 0.461 | −0.263 | **0.928** | 263 kod, VQ terimi sınırlı |

Modlar: `vq_mode` bayrağı (`models/vae3d_vq.py`), geriye-uyumlu, default `standard`.
Eğitim: `scripts/train_vae3d_vq.py --vq-mode <mod> --commitment-cost 1.0`.

## 2. Kök neden — ölçek uyuşmazlığı (saf commit_only neden patlıyor)

`codebook_loss = MSE(z_q, z.detach())`, codebook'a giden **tek** gradyan yoludur. EMA
olmadan kaldırılınca codebook rastgele init'te **donar**. Saf modda init
`uniform(−1/K, +1/K) ≈ ±0.002` (çok küçük); encoder çıkışı ise reconstruction için
serbestçe büyür (ölçülen std ~19, max ~100). Sonuç: `commitment = ‖z − z_q‖²`, z_q sıfıra
yakın sabit kalırken sınırsız ıraksar (epoch 1'de 0.73 → epoch 40'ta 964). Reconstruction
BatchNorm ölçeği sildiği için ayakta kalır, ama **quantization-distance QC kanalı ölür**
(quant_error ρ: −0.407 → −0.020).

## 3. Kazanan: commit_scaled

Kodları encoder ölçeğinde (`N(0,1)`) donuk başlatmak, donuk codebook'u feature ölçeğine
oturtarak commitment'ı sınırlar. Codebook **hâlâ donuk** (codebook_loss yok, EMA yok —
danışmanın niyeti korunur), ve **her iki QC kanalı da geri gelir**:
- recon_dice ρ: 0.771 → **0.823** (base 0.862'ye yaklaşır)
- quant_error ρ: −0.020 → **−0.323** (ölü kanaldan anlamlı negatife döner)

## 4. "En iyi reconstruction ≠ en iyi QC" — identity tuzağının deneysel kanıtı

commit_cosine (encoder + kodlar L2-normalize) **en iyi reconstruction'ı** verir
(val Dice 0.928, codebook'un yarısını kullanır: 263/512). **Ama en kötü QC dedektörüdür:**
recon_dice ρ 0.461'e çöker. Sebep: model o kadar iyi yeniden-inşa eder ki **bozuk
maskeleri bile** temiz biçimde geri kurar → reconstruction hatası artık iyi/kötü maskeyi
ayırt edemez. "Reconstruction-error-as-quality" öncülü, otoencoder fazla güçlü olunca
kırılır. **Sonuç:** bu projede reconstruction kalitesi hedef DEĞİLDİR; QC sinyali
*kontrollü* bir darboğaz ister, kusursuz bir darboğaz değil.

## 5. EMA neden uygun değil

EMA commitment'ı sınırlar AMA kodları encoder feature'larına doğru **hareket ettirir** —
bu tam da kaçınılmak istenen şeydir (bozuk maske feature'ları codebook'u kendine
çekmemeli). Ölçek eşitleme (scaled) kodları donuk tutarak aynı sorunu çözer; bu yüzden
doğru çözüm EMA değil, ölçek-uyumlu donuk init'tir.

## 6. WP-4 — çok-aday konsensüs + 3B belirsizlik haritası

Her latent pozisyonda en yakın k=5 kod → k aday maske; mesafe-ağırlıklı konsensüs
`P = Σ_r (1/d_r)·p_r / Σ_r (1/d_r)`; belirsizlik = per-voxel ikili entropi.
Inference-only (`scripts/multi_candidate_wp4.py`), model kodu değişmez.

| Doğrulama | Sonuç (test, n=282) |
|---|---|
| voxel AUROC (belirsizlik → input XOR temiz-referans) | **0.723** |
| hacim ρ (ort. belirsizlik, 1−true_Dice), tüm | **0.384** |
| hacim ρ, yalnız bozuk | 0.345 |

STEP 0 (aday çeşitliliği, `scripts/wp4_step0_diversity.py`): runner-up kodlar argmin'den
anlamlı farklı maskelere çözülüyor (Dice(M0,M1)=0.79, rank ile 0.63'e düşer);
anlaşmazlığın ~%46'sı sınır bandında, gerisi gövde şekli. Ortho görsel:
`figures/wp4/wp4_uncertainty.png` (input | konsensüs | belirsizlik | hata).

## Özet

- **Öneri:** `commit_scaled` — donuk codebook (danışman niyeti) + ölçek düzeltmesi = QC korunur.
- **Kaçın:** cosine (identity tuzağı, QC ölür) ve saf commit_only (quant kanalı ölür, patlar).
- **WP-4:** belirsizlik haritası gerçek hata bölgelerini güvenilir işaretliyor (AUROC 0.72).
