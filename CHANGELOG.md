# Changelog

Biçim: [Keep a Changelog](https://keepachangelog.com/), sürümleme: SemVer.

## [0.1.0] — 2026-08-02

İlk halka-açık-adayı iskelet: "kurulunca çalışır" + doğrulama-önce disiplini.

### Çekirdek
- Backend dikişi (`tarhan.backend`): array-ops tek modül, f64 truth-path,
  `solve_tridiag`; hızlandırma backend'leri v0.2 kapısında.
- `tarhan.physics`: 22 fonksiyon, her biri dürüstlük katmanlı
  (first-principles/textbook/empirical) — 15+ formül physics_verify
  oracle'ından geçmiş durumda.
- Numerics: Bernoulli/SG akıları, FE difüzyon + Cottrell, Grünwald
  yarı-integral, convergence_rates + Richardson, Levich konveksiyonu,
  voltametri (Nernst + Butler-Volmer tam-CV), **stiff zaman-entegrasyonu**
  (`numerics/transient.py` — scipy-delege BDF/Radau/LSODA, analitik-Jacobian
  öncelikli, f64 truth-path, sessiz degrade yasak; gelecek transient
  elektrokimya/EIS için primitif).
- Modeller (Layer-3): 1D SOFC hücre gerilimi (O'Hayre §6.2 birebir);
  0D PEMFC polarizasyon eğrisi (`models/pemfc0d`, Spiegel/FuelCellStore seti —
  oracle-doğrulamalı kayıp-merdiveni MONTAJI, yeni formül yok; Kim
  m·exp(n·i) biçimi parametre-zorunlu); transient kronoamperometri
  (`models/chronoamp1d` — method-of-lines difüzyon + BDF, transient
  primitifinin ilk domain uygulaması; Cottrell'e uzamsal O(h²));
  **1D pn-diyot Gummel/Scharfetter-Gummel drift-diffusion** (SRH'li) —
  V_bi µV-düzeyi, ideality 1.000-1.002, DEVSIM'le 1e-3-altı mutabakat,
  Sze ψ_bi−2kT/q düzeltmesi Richardson limitinde 4e-5.

### Doğrulama (validation/CATALOG.md)
- Layer-0 kataloğu: rank 0-14 TAMAM — orijinal rank-12 hedefi (parametrik
  PEMFC V(i)) dahil (yalnız Sod bilinçli kapsam-dışı);
  151 PASS + 4 strict-xfail (devsim'li lokal ölçüm; CI devsim'siz skip'ler).
- **Xfail'ler İKİ AYRI TÜRDÜR** (dil 2026-07-15 bağımsız review'da düzeltildi —
  eskiden dördü birden "belgelenmiş kaynak-tutarsızlığı" sayılıyordu, bu abartıydı):
    * **Kanıtlanmış kaynak hatası (2):** kitabın kendi basılı girdileri basılı
      cevabını üretmiyor, aritmetikle gösterildi — O'Hayre Örn. 5.1 (D=0.1-vs-0.2),
      Hu Örn. 4-2 (A² yerine 1e-8).
    * **Teyit edilmemiş provenans (2):** katalog aktarımımız ile korelasyon
      uyuşmuyor, ama hangisinin yanlış olduğu basılı kaynaktan HENÜZ teyit
      edilmedi — Springer λ(0.3), Pierret ε_r. Bunlar "kaynak yanlış" diye
      OKUNMAMALIDIR. Emsal: rank-12'nin (0.085, 1.1) sabitleri de kaynak hatası
      sanılıyordu; birincil-kaynak kontrolü hatanın BİZDE olduğunu gösterdi
      (Spiegel'in α₁/k'si). Kalan ikisi için aynı araştırma açık kalem.
- **Rank-12 ATIF DÜZELTMESİ + PROVENANS ÇÖZÜMÜ (2026-07-09, kaynak-araştırması):**
  pemfc0d parametre seti (i0=10^-6.912, α=0.5, R=0.19, i_L=1.4) Kim/Barbir değil
  **Spiegel (2008)/FuelCellStore**. Dolaşımdaki (0.085, 1.1) Kim'in m,n'i DEĞİL —
  Spiegel'in α₁/k'sidir (ayrı i_L ile). Eski strict-xfail → geçen provenans testi
  (gerçek Kim A/cm² sabitleri m≈3e-5 V, n≈8; DOI 10.1149/1.2050072).
- Çapraz-oracle: DEVSIM 2.10 (opsiyonel `[oracle]` extra); ayrıca **SUNDIALS
  CVODE cvRoberts** basılı tablosu Robertson stiff yeteneğinin çapraz-kod pini
  (transient yeteneği kaynaktan doğrulanmış tolerans setiyle).

### Araçlar
- `tarhan demo` (Cottrell) + `tarhan demo --case diode` — öz-denetimli,
  headless PNG; CI'da her push'ta koşar.
- `tarhan-mcp` — FastMCP sunucusu (opsiyonel `[mcp]` extra): 9 araç
  (diyot I-V/band, CV, Nicholson, SOFC eğrisi, PEMFC V(i) eğrisi, kayıp
  merdiveni, formül kataloğu), girdi-korumalı, çıktı-desimatlı.

### Yetenek doğrulamaları (katalog ötesi)
- Robertson stiff kinetiği (`layer0/numerics/test_robertson_stiff.py`, 10 test):
  korunum makine-hassasiyeti (16 dekad), 3 bağımsız yöntem ~1e-9, SUNDIALS
  çapraz-kod pin; dürüst bulgu — SUNDIALS basılı tablosu gevşek-tol demo'su,
  t≥4e9'da kendi değerleri ~%1–33 kayar (yüksek-hassasiyet Radau ile teyitli).
- Transient kronoamperometri (`layer0/electrochem/test_chronoamp_transient.py`,
  5 test): transient/BDF primitifinin ilk domain uygulaması — yüzey akısı Cottrell
  analitiğine uzamsal **O(h²)** (ölçülen 2.000), profil erf'e ≤1e-4, implicit
  avantaj: BDF'in kabul ettiği 483 adım vs explicit CFL'in gerektirdiği 8889 →
  **ölçülen 18.4×** (2026-07-19 düzeltmesi: eskiden "≥100×" yazıyordu ve testi boştu
  — `nsteps` çıktı-noktası sayısıydı; `count_steps=True` ile gerçek adım ölçülür).
  rank-0'ın implicit yolu.
- pn1d MMS uzamsal-mertebe (`layer0/semiconductor/test_pn1d_mms_order.py`, 5 test):
  (a) izole `_poisson_newton` ve `_continuity_solve` operatörleri ikisi de temiz
  **O(h²)** (6 grid, 2.000) — bunlar ÜRETİM fonksiyonlarını doğrudan çağırır;
  (b) üretim konvansiyonlarını yeniden üreten TEST-YEREL bir (ψ,n,p) ayrıklaştırma,
  kuplajlı ve makine-hassasiyetli çözüldüğünde (coupled-Newton = scipy.optimize.root)
  yine **O(h²)** (2.000). Üretimden DOĞRUDAN paylaşılan tek parça `bernoulli`'dir;
  Laplacian/diverjans/işaretler test-içinde yeniden ifade edilir → "motorun
  ayrıklaştırması" demek fazla güçlüydü (2026-07-15 review düzeltmesi).
  **Kapsam (2026-07-15 review'da dürüstçe daraltıldı):** (b)'deki kuplajlı rezidüel
  bir TEST KURGUSUDUR — `solve_bias`'ta kuplajlı çözüm yolu YOKTUR (Gummel koşar) ve
  kaynak-enjeksiyonunun motorda karşılığı yoktur. Rezidüelin motorun denge çözümünde
  ‖F‖~1e-14 vermesi konvansiyon eşleşmesine GÜÇLÜ KANIT ama kimlik ispatı değildir
  (tek durum). Yani ölçülen: ayrıklaştırmanın mertebesi; ÖLÇÜLMEYEN: üretimdeki
  Gummel solve'un mertebesi (gürültü tabanı orada durmaya devam eder).
  Mimari: rezidüel bizim, Newton scipy'a delege (transient/BDF ile aynı çizgi).

### Bilinen açık kalemler
- **AYRIKLAŞTIRMA mertebesi** MMS ile O(h²) doğrulandı (izole operatörler üretim
  fonksiyonlarında; kuplajlı sistem test-içi rezidüelde). **ÜRETİMDEKİ Gummel
  solve'un mertebesi hâlâ ölçülmedi** — `solve_bias`'ta kuplajlı çözüm yolu yok ve
  J-öz-yakınsaması ~1e-4-bağıl Gummel gürültü tabanına çarpıyor. Bunu kapatmak
  gerçek bir coupled-Newton MOTOR MODU ister (2026-07-15 review'da netleşti).
- **İki xfail'in provenansı teyit edilmemiş** (Springer λ(0.3), Pierret ε_r):
  hatanın kaynakta mı bizim katalog aktarımımızda mı olduğu bilinmiyor. Kim
  emsalindeki gibi birincil-kaynak araştırması gerekiyor.
- `test_rank10_sg_flux.py`'deki %1 asimptot sınırı türetilmemiş gevşek bir sanity
  bound (güvenli ama basılı-sayıya/ölçülmüş-mertebeye bağlı değil).
- GUI yok (karar gereği v0.2'de, kernel-oracle-yeşili sonrası).
- ~~LICENSE/CITATION yasal isim~~ → **kapandı (2026-07-31)**: CITATION.cff ve
  pyproject.toml `authors` alanları dolduruldu, ikisi senkron.
- Zenodo concept-DOI hâlâ alınmadı: ilk halka açık sürümde alınıp CITATION.cff'e
  eklenecek (concept-DOI, version-DOI değil).
