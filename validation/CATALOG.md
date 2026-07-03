# TARHAN Layer-0 Doğrulama Kataloğu

Disiplin: kitaptan öğren → algoritmayı KENDİ kodumuzla yeniden yaz (kaynak kodu
asla kopyalanmaz) → basılı sayı/tablolanmış sabitle karşılaştır → sapmayı logla.
Tam plan + kitap envanteri: Obsidian wiki, "TARHAN — Textbook Simulation Catalog".
Prototipler `../tarhan-validation/` klasöründe geliştirildi; kanonik hali buradaki
pytest suite'idir (`pytest` ile koşar, CI her push'ta işletir).

| # | Vaka | Kaynak | Beklenen | Durum | Test |
|---|------|--------|----------|-------|------|
| 0 | Cottrell kronoamperometri, explicit FD | Britz & Strutwolf 4e (2016) | G=1/√(πT) | ✅ maks %0.0155, mertebe p=1.97 | `layer0/electrochem/test_rank00_cottrell_fd.py` |
| 1 | Nafion λ(a), κ(λ,T) | Springer et al., JES 138 (1991) | λ(1)=14.00; κ(14,80°C)=0.124 S/cm | ✅ 5/6 basılı rakam birebir; λ(0.3) aktarımı `xfail` (kaynak teyidi açık) | `layer0/fuelcell/test_rank01_springer_membrane.py` |
| 2 | P+N step junction | Hu, Modern Semiconductor Devices, Örnek 4-1 (açık PDF) | φ_bi≈1 V; W=0.12 μm; x_P≈1.2 Å | ✅ 3/3 basılı değer | `layer0/semiconductor/test_rank02_hu_junction.py` |
| 3 | 1D difüzyon MMS tam çözüm | Linge & Langtangen §3.6.5 (CC BY 4.0) | ‖e‖∞ ~1e-14, grid'den bağımsız | ✅ ≤8e-15 (3 grid) | `layer0/numerics/test_rank03_mms_exact.py` |
| 4 | Cottrell yarı-integrali M=1 | Oldham, Myland & Bond (2012) | sabit; mertebe 0.5 (tekillik) | ✅ p=0.4999; Richardson M_ext=1.000000; FD↔analitik <%0.3 | `layer0/electrochem/test_rank04_semiintegral.py` |
| 5 | Hu Örnek 4-2: C-V doping çıkarımı | Hu Böl. 4 | N_l=6e15, N_h=1.8e18 cm⁻³ | ⏳ Hu PDF birim-konvansiyonu transkripsiyonu şart (kör hesap 1e8 sapıyor) | — |
| 6 | PEMFC kayıp-merdiveni | O'Hayre 3e, Böl. 2-5 işlenmiş örnekler | kutulu basılı cevaplar | ⏳ baskı transkripsiyonu (Internet Archive borrow) | — |
| 7 | Gözlenen yakınsama mertebeleri | Linge & Langtangen §1.1.4, §3.6.6 | central/CN→2, FE→1 | ✅ 2.0000 / 1.0000 / 2.0002 | `layer0/numerics/test_rank07_convergence_rates.py` |
| 8 | Reversible CV tepe ψ_p | Compton ⊕ Britz ⊕ Bard&Faulkner | 0.4463 (3 kitap çapraz) | ⏳ sıradaki çözücü-hedefi | — |
| 9 | Pierret junction + Shockley eğimleri | Pierret Böl. 5-6 | V_bi=0.716 V; 60/120 mV/dekad | ⏳ | — |
| 10 | SG vs central-difference akı | Farrell (WIAS 2263); Selberherr | SG sıfır: U_T·ln10 TAM | ✅ 8.9e-16; CD 18/11'de yanlış + yanlış işaret | `layer0/numerics/test_rank10_sg_flux.py` |
| 11 | Levich RDE limit akımı | Newman & Thomas-Alyea 3e | 0.620 sabiti | ⏳ | — |
| 12 | Tam PEMFC polarizasyon eğrisi | Barbir 2e / Kim 1995 | V(i) tablosu | ⏳ | — |

## Kurallar

- **Kod kopyalanmaz** — lisans durumu ne olursa olsun (Britz Fortran: lisanssız;
  nanoHUB: CC BY-NC-SA; Kulikovsky: license:null; Langtangen kitabı CC BY ama repo
  lisansı doğrulanamadı). Algoritma basılı metinden yeniden yazılır, köken loglanır.
- **Sabitler vaka girdisidir** — Hu (ε_r=12, kT/q=0.026) vs Pierret (11.7, 0.0259,
  n_i=1e10) vs Sze (n_i=9.65e9): motor hiçbirini hardcode etmez.
- **Keyfî tolerans yok** — basılı hane sayısına, ölçülmüş yakınsama mertebesine
  veya Richardson-ekstrapole limite bağlanır (rank-4 dersi).
- **Baskı kaydı** — her vakanın hangi baskıdan/transkripsiyondan geldiği yazılır;
  şüpheli aktarım `xfail(strict)` ile açık kalem yapılır (rank-1 λ(0.3) örneği).
