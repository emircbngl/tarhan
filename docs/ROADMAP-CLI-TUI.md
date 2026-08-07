# TARHAN — CLI, Terminal UI ve Uygulama Yol Haritası

**Durum:** Tasarım belgesi — uygulanmış özellik listesinden ayrı tutulur.  
**Amaç:** TARHAN'ın fizik motorunu araştırmacılar, otomasyonlar ve ileride bir uygulama için tek, güvenilir çalışma yüzeyi haline getirmek.

## 1. Ürün ilkesi

TARHAN'ın arayüzü bir komut listesi değildir. Bir kullanıcının malzeme adayından karşılaştırılabilir sonuca gitmesini sağlayan, her adımın fiziksel kapsamını ve doğrulama durumunu açık eden bir yüzeydir.

```text
candidate → material model → device → scenario → run → result → comparison → report
```

Üç terim kesin biçimde ayrılır:

| Terim | Anlamı |
|---|---|
| **Komut** | Kullanıcının yaptığı eylem: `run solve`, `candidate rank` gibi. |
| **Capability / ability** | Motorun gerçekten yapabildiği fizik: ör. 1D pn drift-diffusion. |
| **Artifact** | Tekrar açılabilen ve izlenebilen bilimsel kayıt: aday, cihaz, senaryo, run, sonuç. |

Bir komutun görünmesi, arkasındaki fiziğin doğrulanmış olduğu anlamına gelmez. Bu yüzden capability durumları her yerde görünür:

```text
validated     Bağımsız oracle / Layer-0 testi ile sınırı ölçülmüş
experimental  Çalışır; fakat henüz iddia edilen doğrulama seviyesinde değildir
blocked       Bilinen bir önkoşul yüzünden bilinçli olarak çalıştırılmaz
planned       Arayüz ve veri modeli için öngörülmüş; motor henüz yok
```

Bu model, 2D, 3D ve olası "4D" çalışmalarını komut ağacını şişirmeden taşır. Boyut, komutun adı değil capability'nin niteliğidir.

## 2. Kullanıcı rolleri ve başarı ölçütleri

| Rol | Ana ihtiyaç | Başarılı deneyim |
|---|---|---|
| Araştırmacı | Adayı cihaz senaryosunda denemek | Sonucun hangi varsayımlarla üretildiğini tekrar açabilmek |
| Model geliştiricisi | Yeni fizik / numerik eklemek | Doğrulama merdiveninin hangi basamağında olduğunu görebilmek |
| Otomasyon / agent | Deterministik batch çalışma | JSON çıktıyı ve hata durumlarını metin ayıklamadan tüketebilmek |
| Kullanıcı / karar verici | Aday veya run kıyaslamak | Aynı koşulda kıyas yapılmadığında bunu açıkça görmek |

Başarı ölçütü yalnızca "bir sayı üretti" değildir. Her sonuç için şu beş sorunun cevaplanması gerekir:

1. Hangi capability bu sonucu üretti?
2. Hangi girişler ve birimler kullanıldı?
3. Hangi solver, tolerans, mesh ve sürüm kullanıldı?
4. Sonuç hangi doğrulama kapsamındadır?
5. Başka bir aday/run ile gerçekten kıyaslanabilir mi?

## 3. Capability kataloğu

Capability kimlikleri alan adlarıyla, boyutla ve problem türüyle tanımlanır:

```text
electrochemistry.cottrell.1d
semiconductor.pn.drift-diffusion.1d
semiconductor.pn.drift-diffusion.2d
semiconductor.mos.electrostatics.2d
semiconductor.mos.iv.2d
semiconductor.mos.ac.2d
semiconductor.device.3d
```

Her capability kaydı şunları taşır:

```text
id, status, dimension, model family
required inputs, produced fields/metrics
physical and numerical limits
validation evidence and measured error
blocked/planned reason and unlock condition
```

Örnek kullanıcı yüzeyi:

```text
$ tarhan capabilities show semiconductor.mos.iv.2d

status: blocked
reason: Referans MOSFET mesh'i box yönteminin Delaunay önkoşulunu ihlal ediyor.
needs: Delaunay oracle mesh veya ayrı bir element-temelli ayrıklaştırma.
does not mean: "MOSFET desteği yok".
```

`capabilities` komutu yardım metninden daha önemlidir: kullanıcıya gelecekteki vaatleri değil, bugünkü fiziksel sınırı anlatır.

## 4. Hedef CLI ağacı

```text
tarhan
├── about
├── capabilities
│   ├── list
│   ├── show <capability-id>
│   └── doctor
│
├── demo
│   ├── cottrell
│   ├── diode
│   └── pn2d                     # capability açıldığında
│
├── candidate
│   ├── new
│   ├── import
│   ├── generate
│   ├── validate
│   ├── screen
│   ├── rank
│   ├── show <candidate-id>
│   ├── list
│   └── compare <candidate-id...>
│
├── device
│   ├── new
│   ├── inspect
│   ├── compile
│   └── mesh
│       ├── inspect
│       └── validate
│
├── run
│   ├── solve
│   ├── sweep
│   ├── resume
│   ├── show <run-id>
│   ├── status <run-id>
│   └── list
│
├── campaign
│   ├── new
│   ├── run
│   ├── status
│   ├── compare
│   └── report
│
├── compare
│   ├── runs <run-id...>
│   ├── candidates <candidate-id...>
│   ├── oracle <run-id>
│   └── regress <baseline> <candidate>
│
├── validate
│   ├── catalog
│   ├── capability <capability-id>
│   ├── artifact <path>
│   └── reproduce <case-id>
│
└── export
    ├── result
    ├── report
    ├── csv
    └── json
```

`demo` öğrenme ve hızlı güven testi içindir. Bilimsel çalışma hattı `candidate → device → run → compare`dır.

## 5. Artifact modeli

### 5.1 Candidate

Bir aday serbest metin ya da tek bir skor değildir.

```text
Candidate
├── identity
│   ├── composition / structure / dimensionality
│   └── canonical identifier
├── properties
│   ├── band gap, mobility, εr, effective mass, thermal limits…
│   ├── unit and uncertainty
│   └── temperature, orientation and doping validity range
├── provenance
│   ├── source / DOI / dataset / model revision
│   └── measured | computed | inferred
├── applicability
│   ├── usable device models
│   └── missing required parameters
└── validation state
```

`candidate generate` arama uzayı üretir. `screen` sert eşiklerle eler. `rank`, tek ve açıklamasız bir “en iyi” puan yerine Pareto cephesi, ağırlıklar ve belirsizlik etkisini verir.

### 5.2 Device, scenario, run

| Artifact | Sorumluluğu |
|---|---|
| Device | Geometri, kontaklar, malzeme eşlemesi, mesh referansı |
| Scenario | Bias, sıcaklık, sweep, çözüm talebi |
| Run | Çözülmüş input, capability, sürüm, solver ayarı, durum |
| Result | Alanlar, metrikler, yakınsama bilgisi, uyarılar |
| Comparison | Ortak sözleşme, metrik farkları, kıyaslanamazlık sebebi |

Bir run klasörü en az şu yapıyı üretir:

```text
runs/<run-id>/
├── manifest.json        # capability, sürüm, komut, durum
├── input.lock.toml      # çözülmüş inputlar
├── provenance.json      # aday/device/scenario kökeni
├── metrics.json
├── fields.npz
├── stdout.log
└── report.md
```

Run kimliği, mümkün olduğunda giriş ve çözücü sözleşmesinin içerik hash'inden türetilir. Aynı input farklı isimli iki “gizemli” sonucu üretmez.

### 5.3 Kıyas sözleşmesi

İki sonuç yalnızca aşağıdakiler açık ise karşılaştırılır:

```text
aynı model ailesi ve capability seviyesi
aynı veya açıkça dönüştürülmüş birim
aynı senaryo / bias / sıcaklık
aynı mesh veya raporlanmış mesh farkı
aynı solver toleransı ve metrik tanımı
```

Bu koşullar sağlanmıyorsa `compare` bir sıralama uydurmaz. Bunun yerine örneğin `not comparable: different mesh and tolerance` üretir.

## 6. CLI çıktı sözleşmesi

İnsan kullanıcısı ile otomasyon hiçbir zaman aynı stdout'u paylaşmaz.

```text
tarhan run solve device.toml
tarhan run solve device.toml --format json
tarhan run solve device.toml --output runs/
```

| Mod | stdout | stderr |
|---|---|---|
| İnsan / TTY | Kısa sonuç özeti ve tablo | İlerleme, uyarı, örs feedback'i |
| JSON | Yalnızca geçerli JSON | İlerleme ve teşhis |
| CSV | Yalnızca tablo verisi | İlerleme ve teşhis |
| Quiet | Yalnızca istenen sonuç | Kritik hata hariç boş |

Küresel bayraklar:

```text
--format table|json|csv|markdown
--output <directory-or-file>
--color auto|always|never
--progress auto|always|never
--quiet
--seed <integer>
--dry-run
```

Önerilen hata sınıfları:

```text
0  success
2  user input / schema / unit error
3  unavailable or blocked capability
4  solver did not converge; partial artifact exists
5  internal error
```

Bu ayrım, agentların ve CI'ın serbest metin hata mesajı ayrıştırmasını önler.

## 7. Terminal tabanlı UI (TUI) planı

### 7.1 Konumlandırma

TUI, CLI'ın yerine geçmez. Aynı artifact ve capability katmanının etkileşimli gezginidir. Başlatma, batch çalıştırma ve otomasyon hâlâ normal CLI üzerinden yapılabilir.

```text
Core physics API
      ├── CLI adapter            # script/batch/JSON
      └── TUI adapter            # inspect, choose, monitor, compare
```

TUI için ilk amaç “terminalde uygulama yapmak” değil; karmaşıklaştırmaya başlamış run/campaign durumunu görünür kılmaktır.

### 7.2 TUI ekranları

```text
tarhan ui
├── Forge / Home
│   ├── active capabilities
│   ├── recent runs
│   └── explicit blocked items
├── Candidates
│   ├── filters and Pareto view
│   ├── provenance inspector
│   └── compare tray
├── Device inspector
│   ├── material/contact/mesh summary
│   └── prerequisite warnings
├── Run monitor
│   ├── queue and current stage
│   ├── residual / convergence history
│   └── cancel, resume, open artifact
├── Comparison
│   ├── comparability contract
│   ├── metric deltas
│   └── oracle/regression result
└── Validation
    ├── capability ladder
    └── evidence and known limits
```

İlk sürüm klavye-öncelikli olmalı:

```text
j/k veya ↑/↓       hareket
Enter               aç / seç
c                   compare tray'e ekle
r                   seçili artifact'i run et
v                   validation kanıtını aç
q                   çık
?                   tuş yardımını aç
```

Mouse desteği sonradan gelir; komut paleti ve klavye akışı önce gelir.

### 7.3 Örs ve çekiç feedback'i

Örs TARHAN'ın kimlik işareti, çekiç ise yalnızca iş sürerkenki hareketli feedback'tir.

```text
  [hammer frame]  assembling mesh
  [hammer frame]  solving Poisson
  [spark frame]   converged — 24 iterations
```

Kurallar:

- Sadece TTY/TUI'de gösterilir.
- JSON/CSV stdout'una asla yazılmaz.
- Ekranı sürekli yenileyip log geçmişini silmez.
- Her fiziksel aşamada anlamlı durum değişikliği taşır; dekorasyon için dönmez.
- Unicode desteklenmediğinde 7-bit ASCII sürümüne düşer.

### 7.4 TUI aşamaları

| Aşama | Kapsam | Çıkış ölçütü |
|---|---|---|
| TUI-0 | CLI feedback katmanı: renk, ilerleme, örs frame'leri | Pipe/JSON çıktısına hiç karışmaması |
| TUI-1 | Run monitor ve artifact listesi | Çalışan run'ı, logu ve sonucu kaybetmeden izlemek |
| TUI-2 | Candidate explorer ve compare tray | Kıyaslanamaz kayıtların sebebini göstermek |
| TUI-3 | Campaign dashboard | Birden fazla run'ın durumunu ve regresyonunu izlemek |
| TUI-4 | Görsel alan/mesh tarayıcısı | Büyük alan verisini terminalde okunabilir özetlemek; ağır görselleştirme değil |

TUI-0 için hafif bir terminal render katmanı yeterlidir. Tam ekran uygulama ve çoklu input akışı gerçekten gerektiğinde Textual benzeri bir framework değerlendirilir; framework seçimi capability/artifact sözleşmesinden önce yapılmaz.

## 8. Sonraki aşama: TARHAN App

### 8.1 Uygulamanın görevi

Uygulama yeni bir fizik motoru veya CLI'ın ikinci bir implementasyonu değildir. Yerel veya uzak TARHAN run servisinin görsel istemcisidir.

```text
Physics core
   ↓
stable Python service/API
   ├── CLI
   ├── TUI
   └── TARHAN App
```

Bu ayrım zorunludur: hesaplama, validation ve artifact üretimi tek yerde kalır. Uygulama yalnızca bunları başlatır, izler, karşılaştırır ve sunar.

### 8.2 App bilgi mimarisi

```text
Workspace
├── Overview
├── Candidates
│   ├── candidate detail
│   ├── provenance
│   └── comparison board
├── Devices
│   ├── device editor
│   └── mesh/contacts inspector
├── Runs
│   ├── run detail
│   ├── convergence / warnings
│   └── result explorer
├── Campaigns
│   ├── queue
│   ├── parameter sweep
│   └── regression view
└── Validation
    ├── capability map
    ├── oracle evidence
    └── limits / blocked work
```

İlk App sürümü yerel-öncelikli olmalı: kullanıcının bilgisayarında artifact klasörünü ve yerel çalıştırıcıyı kullanır. Bulut kuyrukları, ortak workspace, canlı ekip paylaşımı ve uzaktan hesaplama daha sonra gelir.

### 8.3 App aşamaları

| Aşama | Kapsam | Bilinçli olarak dışarıda bırakılanlar |
|---|---|---|
| App-0 | Tasarım sistemi, read-only artifact explorer | Run başlatma, hesaplama, kullanıcı hesabı |
| App-1 | Local run launcher ve run monitor | Uzak hesap, ortak proje |
| App-2 | Candidate/device editor ve karşılaştırma panosu | Gizli otomatik fizik ayarları |
| App-3 | Alan grafikleri, I–V/C–V çizimleri, mesh görünümü | Tarayıcı içinde solver implementasyonu |
| App-4 | Campaign queue ve regresyon raporları | Dağıtık scheduler zorunluluğu |
| App-5 | İsteğe bağlı remote worker / paylaşım | Varsayılan bulut bağımlılığı |

### 8.4 App tasarım ilkeleri

- Her grafik altında capability, doğrulama seviyesi ve run kimliği görünür olmalı.
- “Güzel ama doğrulanmamış” sonuçlar doğrulanmış sonuçlarla aynı güven dilini kullanmamalı.
- Kullanıcıya önemli solver toleransları ve mesh varsayımları gizlenmemeli.
- Uygulama bir sonuç değiştiğinde hangi input veya sürümün değiştiğini göstermeli.
- Export edilen rapor, CLI'ın ürettiği manifest ile aynı kaynağa dayanmalı.

## 9. Bağımlı uygulama sırası

```text
P0  Capability registry + CLI çıktı/hata sözleşmesi
 │
P1  Artifact schema + run manifest + provenance
 │
P2  Device/scenario derleme ve güvenilir tek-run akışı
 │
P3  Candidate generate/validate/screen/rank/compare
 │
├──────── TUI-0: terminal feedback ────────┐
├──────── TUI-1: run/artifact explorer ────┤
│                                           ↓
P4  Campaign, oracle comparison, regression     App-0: read-only explorer
 │                                                ↓
P5  Validated 2D capability'lerini CLI'a açma   App-1: local run monitor
 │                                                ↓
P6  3D/ileri capability'ler, yalnızca hazırsa   App-2+: editors and campaign views
```

Candidate üretimi, TUI ve App P1'den önce başlamamalıdır. Aksi hâlde adaylar provenance'sız JSON'lara, UI ise fizik motorundan kopuk ikinci bir ürün yüzeyine dönüşür.

## 10. Bilinçli olarak yapılmayacaklar

- Henüz olmayan 2D/3D/4D fiziği için çalışıyormuş gibi komut açmak.
- Capability sınırlarını başarı mesajlarının altında saklamak.
- JSON/CSV tüketicilerinin stdout'una logo veya animasyon karıştırmak.
- Uygulama içinde CLI'dan farklı solver kuralları uygulamak.
- Tek skorlu ve kaynak/belirsizliksiz “en iyi malzeme” sıralaması vermek.
- Önce büyük bir GUI/TUI kurup artifact ve validation sözleşmesini sonradan eklemek.

## 11. İlk uygulanabilir dilim

İlk gerçek ürün dilimi şudur:

```text
tarhan capabilities list
tarhan run solve <scenario> --format json --output runs/
tarhan run show <run-id>
tarhan compare runs <run-a> <run-b>
tarhan ui                 # yalnızca run monitor / artifact explorer
```

Bu dilim, aday üretiminden önce bile doğru temeli kurar: sonuç saklanır, hangi fiziğin kullanıldığı görünür, otomasyon çıktıyı tüketir ve terminal kullanıcısı çalışan işi takip eder.

