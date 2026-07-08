# Changelog

Biçim: [Keep a Changelog](https://keepachangelog.com/), sürümleme: SemVer.

## [0.1.0.dev0] — yayımlanmadı

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
  0D PEMFC polarizasyon eğrisi (`models/pemfc0d`, Kim-1995/Barbir seti —
  oracle-doğrulamalı kayıp-merdiveni MONTAJI, yeni formül yok; Kim
  m·exp(n·i) biçimi parametre-zorunlu);
  **1D pn-diyot Gummel/Scharfetter-Gummel drift-diffusion** (SRH'li) —
  V_bi µV-düzeyi, ideality 1.000-1.002, DEVSIM'le 1e-3-altı mutabakat,
  Sze ψ_bi−2kT/q düzeltmesi Richardson limitinde 4e-5.

### Doğrulama (validation/CATALOG.md)
- Layer-0 kataloğu: rank 0-14 TAMAM — orijinal rank-12 hedefi (Kim/Barbir
  parametrik PEMFC V(i)) dahil (yalnız Sod bilinçli kapsam-dışı);
  129 PASS + 5 strict-xfail (devsim'li lokal ölçüm; CI devsim'siz skip'ler).
  Xfail'lerin 4'ü belgelenmiş kaynak-tutarsızlığı (Springer λ(0.3),
  O'Hayre D-çelişkisi, Hu A²-kayması, Pierret ε_r), 5.'si dolaşımdaki
  Kim m·exp(n·i) sabit-aktarımı (0.085, 1.1) — aynı i_L fiziğiyle 5.3×
  tutarsız, kaynak-sayfa teyidi açık kalem.
- Çapraz-oracle: DEVSIM 2.10 (opsiyonel `[oracle]` extra); ayrıca **SUNDIALS
  CVODE cvRoberts** basılı tablosu Robertson stiff yeteneğinin çapraz-kod pini
  (transient yeteneği kaynaktan doğrulanmış tolerans setiyle).

### Araçlar
- `tarhan demo` (Cottrell) + `tarhan demo --case diode` — öz-denetimli,
  headless PNG; CI'da her push'ta koşar.
- `tarhan-mcp` — FastMCP sunucusu (opsiyonel `[mcp]` extra): 8 araç
  (diyot I-V/band, CV, Nicholson, SOFC, kayıp merdiveni, formül kataloğu),
  girdi-korumalı, çıktı-desimatlı.

### Yetenek doğrulamaları (katalog ötesi)
- Robertson stiff kinetiği (`layer0/numerics/test_robertson_stiff.py`, 10 test):
  korunum makine-hassasiyeti (16 dekad), 3 bağımsız yöntem ~1e-9, SUNDIALS
  çapraz-kod pin; dürüst bulgu — SUNDIALS basılı tablosu gevşek-tol demo'su,
  t≥4e9'da kendi değerleri ~%1–33 kayar (yüksek-hassasiyet Radau ile teyitli).

### Bilinen açık kalemler
- DD çözücüsü için temiz yakınsama-mertebesi çalışması coupled-Newton/MMS
  ister (J-öz-yakınsaması Gummel gürültü tabanıyla ~1e-4-bağılda sınırlı).
- GUI yok (karar gereği v0.2'de, kernel-oracle-yeşili sonrası).
- LICENSE/CITATION yasal isim: yayın öncesi TODO(owner).
