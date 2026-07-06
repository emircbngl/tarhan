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
| 5 | Hu Örnek 4-2: C-V doping çıkarımı (TERS-problem) | Hu Böl. 4 s.99-100 (açık PDF, görsel-okundu) | N_l=6e15, N_h=1.8e18 (+duyarlılık 0.78V→1.8e17) | ✅ üçü de birebir; 2 formül oracle-VERIFIED. **1e8 gizemi çözüldü: kitabın basılı yerine-koyması A² yerine 1e-8 kullanıyor** (literal girdiler 5.9e23 verir — fiziksel saçma; öz-tutarlı eğim 2e31). Girdi-tutarsızlığı strict-xfail — muhtemel errata | `layer0/semiconductor/test_rank05_hu_cv_extraction.py` |
| 6 | PEMFC kayıp-merdiveni | O'Hayre 3e, Böl. 2-5 işlenmiş örnekler (kullanıcı EPUB'undan transkripsiyon) | Örn. 2.3/3.3/4.1/5.1 kutulu cevapları | ✅ E⁰_DMFC=1.199 V; 0.059 V/dekad + 6-dekad 0.36 V; R_ionic 0.01/0.005 Ω → η 0.15/0.10 V; c_O2=5.8, D_eff=0.0506, j_L=2.26, η_conc=0.22 V. 4 formül oracle-VERIFIED. Kitap-içi D=0.1-vs-0.2 tutarsızlığı `xfail` (muhtemel errata) | `layer0/fuelcell/test_rank06_ohayre_losses.py` |
| 7 | Gözlenen yakınsama mertebeleri | Linge & Langtangen §1.1.4, §3.6.6 | central/CN→2, FE→1 | ✅ 2.0000 / 1.0000 / 2.0002 | `layer0/numerics/test_rank07_convergence_rates.py` |
| 8 | Reversible CV tepe ψ_p | Compton ⊕ Britz ⊕ Bard&Faulkner | 0.4463 (3 kitap çapraz); θ_p=−1.109; Δθ_yarım=2.20 | ✅ J_p=0.44636 (3 çözünürlükte kararlı); θ_p=−1.109; 2.2023. Boyutlu Randles–Ševčík oracle 2/2 VERIFIED | `layer0/electrochem/test_rank08_cv_peak.py` |
| 9 | Pierret junction + Shockley eğimleri | Pierret Böl. 5-6 | V_bi=0.716 V; W=0.972 μm; 60/120 mV/dekad | ✅ V_bi=0.7156→0.716; W=0.9716→0.972 (ε_r=11.8'i sabitler — katalog aktarımı 11.7 `xfail` ile belgeli); eğimler 59.64/119.27 (nominal %1 içinde). Shockley oracle 3/3 VERIFIED | `layer0/semiconductor/test_rank09_pierret_shockley.py` |
| 10 | SG vs central-difference akı | Farrell (WIAS 2263); Selberherr | SG sıfır: U_T·ln10 TAM | ✅ 8.9e-16; CD 18/11'de yanlış + yanlış işaret | `layer0/numerics/test_rank10_sg_flux.py` |
| 11 | Levich RDE limit akımı (İLK konveksiyon terimi) | Newman & Thomas-Alyea 3e; Cochran a=0.51023 | 0.620 sabiti <%0.5; j_L∝√ω | ✅ iki bağımsız yol (Γ'sız kuadratür + FD/Thomas) 5e-7 mutabakat → 0.620450; FD mertebe ≈2; √ω ölçeklemesi tam; boyutlu form oracle 2/2 VERIFIED. **Rank-11 dersi (physics_learn'de): elle bileşik-sabit hesabı 0.620469 vermişti — çift-yol yakaladı; sabitler makine-değerlendirilir** | `layer0/electrochem/test_rank11_levich.py` |
| 12 | Tam PEMFC polarizasyon eğrisi | Kim et al., JES 142, 2670 (1995) — Barbir 2e Böl.3 sunar; parametre seti dolaşımdaki "Barbir-tarzı" set (wiki kataloğunda kayıtlı) | Nitel çapalar: ~1.0 V OCV-yakını, ~0.6 V @ ≈1 A/cm², i_L=1.4 roll-off | ✅ MONTAJ (yeni formül yok — üç kayıp terimi rank-6'da oracle-doğrulamalı): V(1 mA/cm²)=0.997, V(1.0)=0.582, roll-off dikleşmesi ölçülü; Kim-biçimi Tafel+ohmik çekirdeği merdivenle ≤1e-12 özdeş (log10↔ln taban-değişimi kanıtlı). **Dolaşımdaki m·exp(n·i) sabitleri (0.085, 1.1) aynı i_L fiziğiyle 5.3× tutarsız → strict-xfail** (Kim-1995 tipik m~3e-5 V, n~8 cm²/A; kaynak-sayfa teyidi açık kalem) | `layer0/fuelcell/test_rank12_pemfc_polarization.py` |
| 12″ | (kaynak-ikamesi, tarihçe) Uçtan-uca 1D SOFC hücre gerilimi — rank-12 ilk turda Barbir yerine elde-var O'Hayre ile kapatılmıştı | O'Hayre 3e §6.2, Tablo 6.4 | ASR=0.176 Ω·cm²; η_ohm=0.088 V; η_cat=0.158 V; V=0.754 V | ✅ basılı zincir birebir; 2 formül oracle-VERIFIED; birim reçetesi belgelendi | `layer0/fuelcell/test_rank12_sofc_1d_cell.py` |
| 13 | Solar hücre FF/V_oc çapaları (çift-yol: Green 1981 vs tam maksimizasyon) | Green, Solid-State Electronics 24 (1981); PVEducation | FF₀ formülü ~1e-4 doğruluk sınıfı | ✅ çift-yol mutabakatı ≤1.4e-4 (v_oc 10.5-30); V_oc oracle 2/2 + FF₀ 3/3 VERIFIED | `layer0/semiconductor/test_rank13_solar_ff_voc.py` |
| 14 | Nicholson (1965) ΔEp-ψ çalışma tablosu, quasi-reversible CV (BV Robin sınırı) | Nicholson, Anal. Chem. 37, 1351 — birincil-kaynak PDF transkripsiyonu | Tablo I: 13 (ψ, ΔEp·n) çifti | ✅ 13/13 çift ±2 mV (TARHAN değerleri grid-yakınsamış; artık fark 1965 tablo granülaritesi); BV(ψ=500)→Nernst limiti 0.08 mV | `layer0/electrochem/test_rank14_nicholson.py` |

## Kurallar

- **Kod kopyalanmaz** — lisans durumu ne olursa olsun (Britz Fortran: lisanssız;
  nanoHUB: CC BY-NC-SA; Kulikovsky: license:null; Langtangen kitabı CC BY ama repo
  lisansı doğrulanamadı). Algoritma basılı metinden yeniden yazılır, köken loglanır.
- **Sabitler vaka girdisidir** — Hu (ε_r=12, kT/q=0.026) vs Pierret (ε_r=11.8 — katalog aktarımı 11.7 idi, W=0.972 μm çapası 11.8'i sabitledi [rank-9 xfail]; 0.0259,
  n_i=1e10) vs Sze (n_i=9.65e9): motor hiçbirini hardcode etmez.
- **Keyfî tolerans yok** — basılı hane sayısına, ölçülmüş yakınsama mertebesine
  veya Richardson-ekstrapole limite bağlanır (rank-4 dersi).
- **Baskı kaydı** — her vakanın hangi baskıdan/transkripsiyondan geldiği yazılır;
  şüpheli aktarım `xfail(strict)` ile açık kalem yapılır (rank-1 λ(0.3) örneği).
