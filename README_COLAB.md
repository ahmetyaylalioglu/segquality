# 3B VAE'yi GitHub + Colab ile Çalıştırma — Sıfırdan Rehber

Bu rehber, kodu hiç GitHub'a koymamış ve hiç Colab kullanmamış biri için
adım adım yazılmıştır. Amaç: 3B karaciğer VAE'sini (Aşama 7) ya kendi
bilgisayarında (RTX 3080 Ti) ya da Google Colab'da eğitmek.

> **Önce yerel dene.** RTX 3080 Ti bu modeli (2.96M parametre, ~31 MB/forward @64³)
> rahat kaldırır. Colab'a yalnızca yerelde bellek (OOM) hatası alırsan veya
> eğitim çok uzun sürerse geç. Yerel çalıştırma için `notebooks/train_3d_local.ipynb`
> kullan; Colab için `notebooks/train_3d_colab.ipynb`.

---

## A. Kodu GitHub'a koyma (ilk kez)

1. **GitHub hesabı aç** (yoksa): [github.com](https://github.com) → Sign up.

2. **Yeni repo oluştur:** Sağ üstteki `+` → **New repository**.
   - Repository name: `segquality`
   - **Private** seç (araştırma kodu, herkese açık olmasın).
   - "Add a README" / ".gitignore" / "license" seçeneklerini **işaretleme**
     (yerelde zaten var; boş repo daha temiz).
   - **Create repository.**

3. **Yereli GitHub'a bağla.** Git Bash aç, proje klasörüne gir
   (`cd /c/Users/.../segquality`), sonra sırayla:

   ```bash
   git init                 # proje henüz git repo'su değilse (bir kez)
   git add .
   git commit -m "3B VAE hazir (Asama 7)"
   git remote add origin https://github.com/KULLANICI/segquality.git   # <-- BUNU DEĞİŞTİR
   git branch -M main
   git push -u origin main
   ```

   > `KULLANICI` yerine kendi GitHub kullanıcı adını yaz. Repo adresini
   > GitHub'daki repo sayfasının yeşil **Code** düğmesinden kopyalayabilirsin.

4. **`.gitignore` kontrolü (ÖNEMLİ).** Büyük dosyalar push'u bozar. Bu repoda
   `.gitignore` şunları zaten hariç tutuyor: `outputs/`, `*.nii.gz`, `*.nii`,
   `*.npz`, `__pycache__/`. Yani ham RAOS verisi, precompute edilmiş hacimler
   ve çıktı görselleri **git'e girmez** — sadece kod gider. `git add .`
   sonrası `git status` ile hiçbir `.nii.gz` / `.npz` dosyasının listelenmediğini
   doğrula.

5. **Kimlik doğrulama (PAT).** GitHub artık `git push`'ta şifre kabul etmiyor;
   **Personal Access Token (PAT)** gerekiyor:
   - GitHub → sağ üst avatar → **Settings** → en altta **Developer settings**
     → **Personal access tokens** → **Tokens (classic)** → **Generate new token**.
   - Kapsam (scope): `repo` yeterli. Süreyi (expiration) uzun tut.
   - Üretilen token'ı kopyala. `git push` şifre sorduğunda, şifre yerine bu
     token'ı yapıştır (bir kez; sonra kaydedilir).
   - **Kolay alternatif:** [GitHub Desktop](https://desktop.github.com) kur —
     token uğraşmadan görsel arayüzle push yapar.

---

## B. Veriyi Colab'a hazırlama

RAOS ham verisi büyük (birçok GB). Colab'a iki yol var:

1. **Önerilen — sadece precompute edilmiş hacimleri yükle (çok küçük).**
   Yerelde 64³ precompute çalıştır (aşağıdaki yerel notebook Hücre 4 ya da
   doğrudan `python scripts/precompute_volumes.py --config configs/vae3d.yaml`).
   Bu, `outputs/volumes/{train,val,test}.npz` üretir — toplam ~10-60 MB.
   Bu `.npz` dosyalarını Google Drive'a yükle:
   ```
   Drive/segquality_data/volumes/train.npz
   Drive/segquality_data/volumes/val.npz
   Drive/segquality_data/volumes/test.npz
   ```
   Colab'da eğitimi `data.data_source: "precomputed"` ile çalıştırırsın —
   ham NIfTI'ye hiç ihtiyaç kalmaz, çok hızlı.

2. **Alternatif — tüm RAOS'u Drive'a yükle.** Sadece 64³'ten farklı bir
   çözünürlük denemek istersen ham veri gerekir. RAOS klasörünü
   (`CancerImages(Set1)` vb.) olduğu gibi Drive'a kopyala ve Colab'da
   `dataset.root`'u o yola işaret et. Yavaş ve büyük; genelde (1) yeterli.

---

## C. Colab'da çalıştırma

1. [colab.research.google.com](https://colab.research.google.com) → **File →
   Upload notebook** → `notebooks/train_3d_colab.ipynb`'yi yükle (ya da GitHub'dan
   aç: **File → Open notebook → GitHub** sekmesi → repo adını yapıştır).

2. **GPU seç:** **Runtime → Change runtime type → Hardware accelerator → GPU**.
   Colab Pro'da **GPU class: Premium** ve mümkünse **A100** (en iyi), yoksa
   **L4** (iyi), en kötü **T4** (idare eder). "High-RAM" varsa aç.

3. **Hücreleri sırayla çalıştır** (notebook'taki markdown açıklamalarını izle):
   nvidia-smi → repo çek + pip install → Drive mount + yol ayarı → precompute+eğitim
   → sonuçları Drive'a indir.

4. **Oturum kesilirse:** Colab oturumu düşerse checkpoint Drive'da kalır (eğitim
   hücresi her epoch Drive'a kaydeder). Yeni oturum aç, ilk hücreleri tekrar
   çalıştır, eğitim hücresi `--resume` ile kaldığı epoch'tan devam eder.

---

## D. Sık sorunlar (troubleshooting)

- **`git push` başarısız — dosya çok büyük.** `.gitignore`'a veri/çıktı eklenmemiş
  demektir. `outputs/`, `*.nii.gz`, `*.nii`, `*.npz` satırlarının `.gitignore`'da
  olduğundan emin ol; sonra `git rm -r --cached outputs` ile yanlışlıkla eklenmişi
  çıkar, tekrar commit + push et.

- **Notebook CUDA görmüyor (yerelde) — yanlış çekirdek seçili.** Bu, kendi
  makinende **en olası ilk takılma noktası**. Sistemde iki Python var: PATH'teki
  düz `python` CPU-only; CUDA yalnızca **sam2** conda ortamında. Notebook yanlış
  çekirdeği kullanırsa `torch.cuda.is_available()` → `False` döner. Çözüm:
  Jupyter/VS Code'da **Kernel** menüsünden **sam2** ortamının çekirdeğini seç.
  Yerel notebook'un Hücre 1'i `sys.executable` yolunu yazdırır ve CUDA yoksa
  bu uyarıyı basar — oradan doğrula.

- **Colab GPU vermiyor.** Runtime type'ı kontrol et (Hardware accelerator = GPU).
  Colab Pro kotan bittiyse bir süre bekle veya daha düşük GPU sınıfı seç.

- **OOM (bellek yetersiz).** `configs/vae3d.yaml`'de `vae3d.input_size`'ı
  `[64,64,64]` → `[32,32,32]` düşür, ya da `train.batch_size`'ı azalt (ör. 8 → 4).

- **`--resume` çalışmıyor.** Checkpoint yolunun gerçekten var olduğundan emin ol.
  Colab'da checkpoint Drive'a kaydedilmeli (yerel `/content` oturumla silinir);
  `--resume` argümanına Drive'daki `vae3d_last_latent64.pt` yolunu ver.
