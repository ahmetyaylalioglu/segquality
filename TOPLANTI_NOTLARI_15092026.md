# Toplantı Notları & Araştırma Task Listesi — 15.09.2026

Danışman toplantısı ham notlarının temizlenmiş / temalara ayrılmış hali + öncelikli task listesi.

---

## Tema A — SDF analizleri
- **A1.** SDF ara-adım görselleştirmesi: girdi maske → tahmin edilen SDF (sürekli, işaretli) → `sdf<0` ikili köprü. Ara adımı göster.
- **A2.** SDF Dice: temiz referans maske ile SDF tahmini arasındaki Dice (→ notebook Bölüm 7'de yapıldı, rapora taşınacak).
- **A3.** Slide-7 scatter (SDF ρ grafiği) yeniden üret / açıkla.
- **A4.** Neden ρ ~0.7? Korelasyonun neden bu bantta kaldığına dair analiz (band-içi Spearman, hangi bantta düşüyor).

## Tema B — Codebook / commitment netleştirme
- **B1.** `vq_mode` toggle (standard / commitment_only / commit_scaled / cosine) davranışını netleştir.
- **B2.** Hangi modu güncel WP-4'te kullanıyoruz? → `commit_scaled` (ρ 0.823, ölçek-uyuşmazlığı çözümü, codebook donuk N(0,1)).
- **B3.** "En iyi rekonstrüksiyon ≠ en iyi QC" (cosine kimlik tuzağı) — tabloyla göster.

## Tema C — Aday (candidate) inceleme  →  **PRIORITY 1: TAMAMLANDI**
`notebooks/wp4_candidate_inspection.ipynb`
- **C1.** Bozuk girdi → en yakın 5 kod → her adayın çıktısını AYRI göster (ortalama almadan). ✔ (Bölüm 3)
- **C2.** Adayların uzaklıkları: lineer mi exponansiyel mi? Grafikle. ✔ (Bölüm 4)
- **C3.** Profil her örnek için aynı mı? Büyük vs küçük kesit. ✔ (Bölüm 5)
- **C4.** İlk-5 değil, herhangi bir rank'a erişim (rank 10/100/K-1). ✔ (Bölüm 6)
- **C5.** Seçili rank kümesinden konsensus + entropi haritası; rastgele karaciğerler alt alta; input | referans | adaylar (tek tek) | konsensus (ortalama) | entropi. ✔ (Bölüm 6.5)
- **C6.** SDF Dice (temiz referans vs tahmin). ✔ (Bölüm 7)
- İlk gözlem: rank 0 iyi eşleşiyor, uzak rank'lar bozuluyor → kod defteri çeşitliliği doğrulandı.

## Tema D — Kalite tahmini (girdi olarak görüntü + maske + entropi)
- **D1. [DANIŞMAN VURGUSU]** Danışman: **maske + input görüntü + entropi (uncertainty)** haritasını **birlikte INPUT** olarak veren bir kalite-tahmin modeli kur. Yani QC skorunu (Dice tahmini) sadece rekonstrüksiyon hatasından değil, bu üç kanaldan öğren.
  - Girdi kanalları: (1) segmentasyon maskesi, (2) orijinal CT/MR görüntüsü, (3) WP-4 entropi haritası.
  - Çıktı: tahmini kalite / Dice.
  - Neden: entropi zaten hatalı bölgeyi işaretliyor (voxel AUROC ~0.72); görüntü+maske ile birleştirince hacim-seviyesi skoru güçlenebilir.

## Tema E — Parametrik şekil öğrenme / NIR  →  **KRİTİK (gelecek yön)**
- **E15.** Parametrik uzayda şekil öğrenme: encoder çıkışında parametrik tahmin + geri ikili maskeye yüzey eşleme.
- **E16.** Parametrik uzayda dağılımlar ("bu girdi için bu şekiller mümkün").
- **E17.** NIR: parametrik uzayda kompakt yüzey temsili.
- **E18.** Codebook anatomik genellemeden spesifiğe gidebilir mi; eğitim setine kıyasla "uydurma" yapmadan.
- **E19.** CNN-encoder parametre öğrenimi topolojik olarak gerçekçi şekiller veriyor mu.
- **E20.** Oku: arxiv.org/pdf/1806.05004.
- **E21. [DANIŞMAN MAİLİ 15.09.2026 — Zehra Uslu]** NIR/INR kaynakları:
  - https://inr4miccai.github.io
  - https://collab.dvb.bayern/spaces/TUMdlma/pages/651275786/Neural+Implicit+Representations+for+Medical+Shapes → özellikle **ImplicitAtlas: Learning Deformable Shape Templates in Medical Imaging** ("yapmak istediğimiz durumla ilgili" — danışman).
  - PDF: *Steerable Anatomical Shape Synthesis with Implicit Neural Representations* (de Wilde ve ark., MICCAI; kod: github.com/MIAGroupUT/steerable-shape-synthesis). Koordinat-tabanlı MLP → SDF; disentangled latent + correlation loss ile steerable (hacim/isthmus/simetri).

## Ertelenen (danışman erteledi)
- **Task 7:** Kaliteye bağlı codebook kullanımı analizi — danışman şimdilik erteledi.

---

## Öncelik Sırası
1. **[TAMAM]** Aday inceleme notebook'u (Tema C).
2. **Tema A** — SDF analizleri (Dice'ı rapora taşı, ρ~0.7 analizi, slide-7).
3. **Tema D** — görüntü+maske+entropi ile kalite-tahmin modeli (danışman vurgusu).
4. **Tema E** — parametrik/NIR yön (kritik, uzun vadeli; arxiv 1806.05004 ile başla).
5. **Tema B** — codebook/commitment toggle dokümante et.
