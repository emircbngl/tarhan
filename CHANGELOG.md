# Changelog

Biçim: [Keep a Changelog](https://keepachangelog.com/), sürümleme: SemVer.

## [Unreleased] — 0.3.0.dev0

Nothing here is released. `CITATION.cff` still describes v0.2.0 because that is
the last version with a Zenodo archive; these entries move into a dated section
when a release is cut.

### Added
- `run solve` reaches 2D steady, via a generated axis-aligned rectangular pn
  diode (`models/diode2d_mesh.py`). Eight scalars determine the mesh exactly,
  so a run records its device as eight numbers rather than 625 coordinates.
  **This is not mesh generation** — one shape, no refinement, no curved
  boundaries. Validated against the 1D solver on the same node positions:
  max|Δψ̂| 2.28e-13, terminal-current ratio 1.000000 at 0.3 V and 0.4 V.
- `run sweep`, under one fixed solver contract; varying a solver term is
  refused, because a column that is not comparable is not a column.
- `--device`, a flat TOML/JSON of device overrides, refused by name when a key
  is not part of the capability's device.
- `candidate list|show|screen` and `run solve --candidate`. A property is a
  value, a unit, a basis and a doubt; uncertainty can make a threshold
  `undecided`, which is not a soft fail. **No material database ships.**
- Canonical units per solver-facing property, converted on load — uncertainty
  through the same conversion — and refused otherwise.
- A scheduled cross-oracle workflow (`oracle.yml`) that fails rather than skips
  when DEVSIM is absent.
- Run artifacts carry a build id over the package source, per-file checksums,
  a schema version, and the real argv.

### Fixed
- `--candidate-id` without `--candidate` solved default material while
  recording the named candidate: a falsified provenance record.
- Units were checked only for emptiness, so `mu_n = 0.1 m^2/Vs` silently
  solved a material 1e4 times slower, and `mu_n = 3 kg` was accepted.
- `PNDiode1D` did not validate `gamma`; `gamma < 1` made `build_grid` loop
  forever.
- Two candidates with equal nominal values shared one artifact directory, the
  second overwriting the first.
- A `--device` file could overwrite a candidate's material values while
  provenance still named the candidate.
- `write_run` was not atomic: a crash could corrupt the previous artifact, and
  a re-run without field data left a stale `fields.npz` that the new manifest
  then checksummed.
- `valid_range` was stored and never consulted; it is now enforced, and a range
  over a condition no run can supply is refused rather than kept as decoration.
- `--quiet` still wrote progress lines to stderr.
- `--tol nan` was reported as non-convergence rather than as bad input.
- `compare runs` silently dropped metrics present on only one side.
- A sweep with no `--vary` produced a one-point table; a large grid was
  materialised before any solve began.
- `read_run` kept parsing artifacts from a newer schema than it understands.
- Duplicate JSON keys in candidate and device files silently kept the last.
  (The device half of this was claimed here before it was true, and fixed in
  the following commit — the entry is left standing as a reminder that a
  changelog is a claim like any other.)
- `hardness = 50 cm` passed a `hardness >= 1 m` threshold: properties outside
  the unit table were compared as bare numbers. They now require an identical
  unit string, since there is no table to convert through.
- A very small `h0` walked toward hundreds of millions of grid nodes without
  ever stalling, so the progress guard never fired. The node count is now
  estimated from the geometric series before anything is allocated, in 1D and
  in 2D, and `ny` is bounded with it.
- A candidate-driven run recorded only an id and a short fingerprint, so the
  evidence lived in the user's file; moving it left the run unable to say what
  it rested on. A checksummed `candidate.lock.json` now travels with the result.
- `candidate show` emitted identity and fingerprint as stderr notes, so
  `--format json --quiet` lost them.
- A failed write left a hidden staging directory behind; cleanup covered only
  the rename.
- A current-schema manifest with its checksum map removed was accepted as
  "legacy" rather than refused as damaged.
- `compare runs` named one-sided metrics only on stderr, so the JSON rows
  carried just the intersection.
- A device override in `run sweep` still reported `converged`: the
  reference-device check was added to `solve` and not to `sweep`. It is
  computed from the resolved device now, so `--vary mu_n=…` counts too.
- The envelope was one interval per input, so equilibrium — the
  best-validated point in either capability — reported
  `outside-validated-range`. It is a union of intervals now.
- The envelope reached `--format json` as prose. It is structured data.
- `run sweep` built its own three-key provenance, so a sweep point overwrote a
  solve of the same problem with a poorer record. Both use one producer now.
- The 1D capability had no envelope at all, and a device or candidate override
  no longer counts as "inside the evidence".
- `screen()` defaulted to the UNSAFE `require_conditions=False`; only the CLI
  passed the safe value. Safe-by-default means the default is safe.
- `--at bias_v=nan` was accepted, and every range comparison against NaN is
  False, so an out-of-range property could screen clean.
- The `envelope` was absent from `capabilities list/show`, so a machine-
  readable limit no machine format carried.
- A solve outside a capability's validated range wrote `status=converged`
  with no signal; capabilities now carry a machine-readable `envelope` and the
  run records `converged-outside-validated-range`.
- Deleting one entry from a manifest's checksum map disabled that file's check,
  so a forged `metrics.json` read back as `verified`. Verification now requires
  the recorded set and the directory to match exactly.
- `--vary bias_v=nan` was accepted, failed as a solver error, and emitted a
  bare `NaN` into the JSON stdout.
- An unconditioned screen of a ranged property returned PASS; the warning was a
  stderr note that `--quiet` dropped. It is `undecided` now.
- A bare threshold against a property with no canonical unit compared raw
  numbers; it needs an explicit matching unit.
- `gamma=1e308` raised `OverflowError` and exited 5 instead of refusing input.
- `valid_range` was enforced at solve time only; `candidate screen --at
  bias_v=...` now screens under a condition, and an unconditioned screen of
  ranged properties says so.

## [0.2.0] — 2026-08-07

**Why 0.2.0, when this release breaks two published signatures and changes the
licence.** SemVer's rule for `0.y.z` is that the public API is not stable and
anything may change — which is exactly what "pre-alpha" means where the README,
the CLI and the MCP manifest all say it. A major version would have made the
version string claim a maturity the project claims nowhere else. So the number
stays small and the breaks stay loud: they are the first section below, each
with the reason that forced it.

### Changed — API breaks

- **`assemble_continuity` now requires a `carrier` argument** (`"electron"` or
  `"hole"`); there is no default. The default was the bug: the two equations
  differ only by the sign of ψ inside the Scharfetter–Gummel exponent, so
  calling it for the wrong carrier does not raise, does not warn, and converges
  — to the wrong answer. Measured against the equilibrium null space, the
  residual is 9.4e-16 with the right carrier and 8.9e-01 with the wrong one. A
  silent wrong answer is worse than a broken caller, so the caller breaks.
- **`build_mesh`'s `weight_atol` became `shape_rtol`**, and every geometric
  tolerance in the mesh builder is now *relative* — to the longest edge of the
  triangle for degeneracy, to the mesh extent for edge length, to the edge
  length for the Voronoi facet. The absolute tolerances were calibrated on unit
  meshes and rejected real device meshes, where a cell is ~1e-5 cm and every
  legitimate area sits under the old absolute floor. `shape_rtol` is validated
  on entry: non-finite or non-positive is refused rather than propagated as NaN
  comparisons that quietly answer "not degenerate".
- **Licence: AGPL-3.0-or-later → Apache-2.0** (`LICENSE`, `NOTICE`, MCP
  metadata). This changes the terms the published artifact may be used under.
  Anyone who took v0.1.0 under AGPL keeps it under AGPL; this release is the
  first one carrying Apache-2.0, which is the more permissive of the two, so no
  existing user loses a right they had.

### Added — 2D, as far as it honestly goes

- `numerics/mesh.py`: box-method (finite-volume / Voronoi) edge geometry for 2D
  triangular meshes — edge length, Voronoi facet, and the positivity guard,
  which is the Delaunay condition written as `cot α + cot β ≥ 0`. Geometry only;
  mesh *generation* stays out of scope — a mesh is read, never made.
  `EdgeGeometry` also carries per-triangle `facet_shares`, so an edge on a
  material interface can be weighted per region.
- `numerics/assemble.py`: the edge loop — Scharfetter–Gummel continuity and
  Poisson assembly over the box mesh, Dirichlet contacts, `subdomain` support
  for a carrier that exists in only part of the device, and `node_volumes`.
  `edge_coef` is checked finite and non-negative, because a negative
  coefficient destroys the M-matrix property that carrier positivity rests on.
- `backend.solve_sparse`: the sparse seam, taking COO triplets rather than an
  assembled matrix, so the backend boundary does not force a matrix type on its
  caller.
- `models/pn2d.py`: `PNDiode2D` — equilibrium and biased solves, `iv_sweep`,
  `contact_current`. Contacts are validated non-empty, in range, disjoint, and
  integer-valued *before* casting: a node listed in two contacts made the answer
  depend on dictionary insertion order (node 0 landed at −13.82 or +5.50 thermal
  volts on the same device, purely by swapping two dictionary keys),
  and `0.9` quietly becoming node `0` reports currents for an electrode that is
  not where the caller thinks it is.
- Validation stages **2D-0, 2D-1, 2D-2 and 2D-3′**, each against DEVSIM on
  DEVSIM's own meshes: ψ to 1.8e-15 for a 1D problem run through the 2D
  machinery; max |Δψ| 2.24e-16 V at equilibrium; diode I_n ratio 1.00000 with
  ideality 1.0119–1.0134 against DEVSIM's own 1.0114–1.0126; contact charge
  ratio 1.000000000. **2D-3 and 2D-4 are BLOCKED** — no AC or circuit layer, and
  a reference mesh that is not Delaunay — with reasons recorded in
  `docs/DESIGN-2D.md` §5 rather than left to folklore.
- `validation/layer0/test_docs_match_the_code.py`: Layer-0 applied to the prose.
  It fails when a live section denies a capability the code has, when a
  documented source path does not exist, when a stage marked DONE has no test
  behind it, or when a BLOCKED stage does not say why. It exists because that
  class of defect appeared twice in review, and it found a stale claim on its
  first run. It cannot catch a wrong *number*, and says so in its own docstring.
- `tools/job.py`: a job runner, so an agent waits on a file contract instead of
  guessing how long a command takes.

### Fixed — two 1D bugs, both found from the 2D side

- **`pn1d._contact_densities` lost the minority carrier to catastrophic
  cancellation.** The contact densities came from the quadratic root, whose
  minority branch subtracts two nearly equal numbers, so `n·p/δ²` came out
  1.110223 instead of 1 — uniformly, across sixteen orders of magnitude of
  doping. That uniformity was the clue: a precision floor would track the
  doping, and this did not. The majority carrier is now formed by addition and
  the minority as `δ²/majority`, which never subtracts.
- **The mesh tolerances were absolute** (see the API break above), found by
  handing the 1D-derived builder a real device mesh, where it rejected valid
  triangles.
- `diode_iv` counts its bias points *before* building the sweep, so a request
  over the 60-point cap is rejected without materialising the list. Measured on
  the rejected path: 112 MB / 1198 ms at `v_step=1e-7` before, 1 KB / 0.1 ms
  after. These are MCP tools, so the step size is caller-supplied.
- `diode_iv`'s grid never steps past the requested `v_stop` when the step does
  not divide the interval; the last point is `v_stop` itself.
- The MCP test's skip guard named the parent package instead of the module the
  code imports (`mcp.server.fastmcp`). An environment carrying mcp 2.x — which
  removed that module, and is why `pyproject` pins `mcp>=1.1,<2` — did not skip,
  then failed inside `build_server` telling the reader to install an extra that
  was already installed. Caught by fresh-clone verification, which installs only
  `[dev]` and therefore has no pin protecting it.
- Two remaining Turkish error strings in `diode_iv` are now English, matching
  the rest of the tool surface an agent reads.
- MCP licence metadata and instructions aligned with the Apache-2.0 source
  licence.

### Changed

- `tarhan demo` completes without opening a window when stdout is not a
  terminal, so it cannot block CI or an agent shell.
- The GPU verdict is stated as a verdict about *scale*, not about GPUs: MLX is
  ~50× slower than NumPy at the 1D working size (n=2e4), and that measurement is
  explicitly not generalised to 2D or 3D. It has **not** been re-measured since
  2D landed, and `AGENTS.md` now says that instead of promising to.

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
- ~~Zenodo concept-DOI hâlâ alınmadı~~ → **kapandı (2026-08-02)**: GitHub-Zenodo
  entegrasyonu açıldı, v0.1.0 Release'i arşivlendi. Concept-DOI
  **10.5281/zenodo.21761218** (her zaman en son sürüme çözülür) CITATION.cff'e
  `doi` alanı olarak, v0.1.0'ın sürüm-DOI'si 10.5281/zenodo.21761219 ise
  `identifiers` altına yazıldı. README'de rozet var.
