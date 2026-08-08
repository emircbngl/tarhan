# P0 — Capability registry + CLI çıktı/hata sözleşmesi

**Durum:** onay bekliyor. Kapsam: yol haritası (`docs/ROADMAP-CLI-TUI.md`) §3, §6,
§9'un P0 satırı ve §7.4'ün TUI-0 satırı. `run solve` ve sonrası P1'dir, bu
dilimde YOK.

## Kararlar (kullanıcı onaylı, 2026-08-07)

- **"4D" = 3D + zaman.** Yani yeni bir uzaysal boyut değil, ayrı bir eksen.
  Capability kimliği boyutu ve zamanı AYRI taşır. Gerekçe: `chronoamp1d`
  transient, `pn1d` steady-state, ve bugünkü şemada ikisi de "1d" — şema var olan
  bir farkı söyleyemiyor. `.4d`'yi aynı yuvaya eklemek bu karışıklığı kalıcı
  yapardı.
- **TUI-0 çıktı sözleşmesiyle birlikte gelir**, çünkü asıl değeri "JSON/CSV
  stdout'una asla bulaşmaz" kuralını en baştan makineye kontrol ettirmek.
- **TUI-1 ve sonrası ertelendi.** Gerekçe tarih değil ölçüt: izlenecek gecikme
  yok. Tüm test paketi ~12 s, tek bir 1D çözüm milisaniyeler. Run monitor, tek
  bir run ~30 saniyeyi aştığı gün hak eder.

## Adımlar

- [x] **1. Capability kayıt şeması.** `src/tarhan/capabilities.py`. `dimension:
      int` ve `time` ayrı; kimlik türetiliyor (`domain.family.{n}d.time`), alan
      olarak yok — dolayısıyla elle yazılamaz ve kendi alanlarıyla çelişemez.
      6/6 geçersiz kayıt reddedildi (kanıtsız `validated`, `does_not_mean`'siz
      `blocked`, dimension=4, dimension=True, `time='4d'`, modül adı veren
      `planned`).
      **Bulgu:** zaman ekseni `steady|transient` olarak dar kaldı. 2D-3 (MOS
      C–V) küçük-sinyal **AC** ve ikisi de onu ifade edemiyor; `ac` eklendi.
      Yol haritası §3 zaten `semiconductor.mos.ac.2d` öngörüyormuş.
- [x] **2. Bugünkü capability'ler.** `src/tarhan/capability_registry.py`,
      6 `validated` kayıt. Her `measured` dizesi testin docstring'inden ya da
      assertion'ından alındı, README'den DEĞİL — README yuvarlıyor (0.57 µV →
      "0.6 µV", %0.0155 → "%0.016") ve yuvarlanmış sayıyı kesin görünümlü bir
      iddiaya çevirmek tam da reddettiğimiz şey.
- [x] **3. BLOCKED ve PLANNED kayıtları.** 2D-3 → `mos.capacitance.2d.ac`,
      2D-4 → `mosfet.drift-diffusion.2d.steady`, ikisi de `blocked` ve
      `does_not_mean` zorunlu tutuldu. 3D ve 3D+zaman (`3d.transient` = yol
      haritasının "4D"si) `planned`.
- [x] **4. §5 tablosuna karşı çivileyen test.**
      `validation/layer0/test_capability_registry.py`, 32 test.
      **Mutasyonla kanıtlandı:** registry'deki `3.350171660e-12` değeri
      `...999e-12` yapıldığında test düştü, geri alınınca geçti. Geçtiğini
      görmek yetmez; ısırdığını görmek gerekiyordu.
- [x] **5. `tarhan capabilities list` / `show <id>`.** `src/tarhan/cli.py`.
      `show`, kayıt tam olarak yazdırıldıktan sonra çalıştırılamaz capability
      için **3 ile çıkar** — yazdırmak sorulan sorunun cevabı, çıkış kodu ise
      aynı cevabın betiğin paragraf ayrıştırmadan okuyabileceği hâli.
- [x] **6. CLI çıktı sözleşmesi.** `src/tarhan/cliout.py`. `--format
      table|json|csv`, `--quiet`, `--color auto|always|never`. İlerleme ve not
      **her modda** stderr'e — insan modunda bile. Bu gerekenden katı, bilerek:
      "sadece etkileşimli durumda" izni verildiği an bir `--format json`
      tüketicisi açıklayamadığı bir yarım-parse ile düşer.
- [x] **7. Hata sınıfları.** 0/2/3/4/5 tanımlandı ve teste çivilendi. Bugün
      bağlı olanlar: 0, 2, 3, 5. **4 (yakınsamadı) henüz çağrılmıyor** — `run
      solve`'a ait, o da P1. Numaralandırma, bir şey ona bağlanmadan önce
      sabitlensin diye şimdi konuldu.
- [x] **8. TUI-0.** `Feedback` sınıfı: yalnız stderr TTY iken, asla stdout'a,
      ekranı silmeden, her gerçek aşamada bir satır. Glyph kodlanamıyorsa ASCII'ye
      düşer — `demo`'yu bir zamanlar Windows'ta salt kozmetik sebeple çökerten
      hatanın aynısı olmasın diye.

## 3–10 arası kalemler (2026-08-07)

- [x] **3. Registry boşluğu.** `1d.transient` ve `2d.transient` eklendi. Eksik
      bir basamak yalnızca bilgilendirmemekle kalmaz, **yanlış bilgilendirir** —
      boşluk bir sıralama gibi okunur.
- [x] **4. 1D transient cihaz.** `pn1d` zaman türevi kazandı. Anahtar gözlem:
      `n,p` durum değişkeniyken yük `n−p−N` ψ içermez, Poisson **lineerdir**,
      index-1 DAE bir ODE'ye iner. Sabit nokta 1.59e-13 / 3.69e-13; %5 bozulma
      1.553e-1 → 7.61e-8; t0 = 4.794e-13 s, sertlik 5.37e3, BDF 425 adım.
- [x] **5. 2D transient.** Operatör yeniden yazılmadı, `assemble_continuity`'nin
      kısıtsız rezidüeli kullanıldı. Sabit nokta 2.60e-17 / 4.60e-16; lineer
      Poisson = Newton, 1.3e-16 bağıl (makine hassasiyeti).
      **Öğrenilen:** sabit-nokta testi **her iki işaret için de** geçiyor;
      işareti yalnız gevşeme testi belirliyor.
- [x] **6. 3D steady — BLOKLU, ölçülmüş gerekçeyle.** Çevrel-merkezli dual
      well-centered tetrahedronda tam (facet √2/3, hacim (1/6)ΣA·L), ama çevrel
      merkez dışarıdaysa **tam iki kat** şişiyor — ve DEVSIM'in kendi 3D
      diyodunda 6701'in 2555'i (%38.1) o durumda. 2D-4'ü blokladı diye anılan
      oran %0.27'ydi.
- [x] **7. 4D (3D + zaman).** Tamamen 6'ya bağlı; `planned`, blokçusu adıyla.
- [x] **8. Artifact şeması.** `src/tarhan/artifact.py`. Kimlik **içeriktir**:
      capability + çözülmüş girdi + solver sözleşmesinin hash'i. **Zaman damgası
      hash'e girmiyor** — girseydi her koşu inşaen benzersiz olur, "aynı problem
      aynı yere düşer" özelliği uygulanmış görünürken yanlış olurdu.
- [x] **9. `run solve`.** Bloklu/planned capability iş yapılmadan **3** ile
      reddediliyor; yakınsamama **4** ile ve **artifact yazılmadan** — yarım bir
      durum, koşunun hak ettiğinden fazlasını iddia ederdi. `EXIT_NO_CONVERGENCE`
      nihayet çağrı yeri kazandı; `cliout.py` ve `AGENTS.md`'deki "henüz yok"
      cümleleri de birlikte düzeltildi.
- [x] **10. `run show` + `compare runs`.** Kıyas sözleşmesi tutmuyorsa `compare`
      **sıralama uydurmuyor**: hangi terimin farklı olduğunu söyleyip 2 ile
      çıkıyor, ve reddi JSON'da da makine-okur biçimde veriyor.

## İnceleme (3–10)

**Sayılar.** Test 387 → 407 → 420. Registry 12 kayıt: 9 validated, 3 blocked.
Beş dilim ayrı ayrı yayınlandı, her biri CI yeşil.

**Bu turun asıl dersi bir sayıyı yayınlamamak oldu.** 3D için işaretli dual
kurulumu "kenarların %26'sı negatif" diye bir engel üretti ve onu rapor etmenin
eşiğindeydim. Aynı kurulum mesh hacmini yalnız %66 üretiyordu — yani kendi
tutarlılık kontrolünü geçemiyordu. Geri çektim. Tam olarak üretmesi gereken bir
büyüklüğü üretemeyen ölçüm, hiçbir şeyin kanıtı değildir.

**Üç iddiam yanlış çıktı ve üçü de kayda geçti:** "doğrudan LU geçerli değil"
(varsayımsal 128³ mesh'i 1417 düğümlük gerçek oracle'la karıştırmışım),
"referans mesh yok" (baştan beri oradaydı), ve yukarıdaki %26. Var olmayan bir
engel, hiç not olmamasından kötüdür — ilerleyebilecek işi durdurur.

**Bir de tolerans dersi:** CI kırmızı döndü çünkü eşiği kendi makinemin
ölçtüğü sayıya göre kurmuştum (8.62e-12 → eşik 1e-10); ubuntu 1.624e-10 verdi.
Eşik, ölçülen sayıya değil **gerçek bir kusurun üreteceği büyüklüğe** göre
kurulur.

**Bilinçli olarak yapılmayanlar.** 3D ve 4D bloklu. `run solve` yalnız
`pn1d.1d.steady`'ye bağlı — 2D steady doğrulanmış ama CLI'a bağlanmamış, ve hata
mesajı hangi yarısının eksik olduğunu söylüyor ("kanıtlanmış olmak" ile
"bağlanmış olmak" farklı olgular). Transient yolda yer değiştirme akımı, SRH ve
bias dalga formu yok. `campaign`, `candidate` ve TUI-1+ hâlâ yol haritasında.

## TUI-1 hazırda bekliyor (bağlı değil)

`src/tarhan/forge.py` — canlı ilerleme ekranı. **Hiçbir komuta bağlı değil**,
bilerek: ölçüt hâlâ karşılanmadı (tek run ~30 sn'yi aşmalı; bugün 1D çözüm
milisaniyeler). Amaç, o gün geldiğinde bağlamanın tasarım işi değil mekanik iş
olması.

- Çalışırken **tek satır**: `╱▂▄▂ SOLVE  detay  (12.4s · 25% of solve · stage 3/4)`.
  Örs sabit, çekiç vuruyor, kıvılcım yalnız temas karesinde.
- Biterken **bir kez** örs bloğu + aşama listesi + gerçek süreler.
- Ekran **hiç** silinmiyor (`ESC[2J` sıfır, testle çivili); scrollback yaşıyor.
- İlerleme = tamamlanan aşama / toplam, artı **yalnızca çağıranın bildiği**
  `within`. Zamanın ilerlemeyi sürdüğü hiçbir yol yok, bu da testle çivili.
- TTY yoksa aşama başına düz satır; `--quiet` tamamen susturuyor.
- `cliout.Feedback` **kaldırıldı**: çağrı yeri yoktu ve `Forge` onun yaptığı her
  şeyi yapıyor. İkisini tutmak "ilerleme nasıl gösterilir"e iki kaynak bırakırdı.

`validation/layer0/test_forge.py` (14 test) iki gerçek kusur yakaladı:
`" · ".join(...)` ayracı ASCII moduna sızıyordu (Windows'ta çökme sınıfı), ve
`pty` modül düzeyinde import ediliyordu — Windows'ta bütün dosyanın toplanmasını
patlatırdı. İkincisini test yazarken değil, testi CI gözüyle okurken gördüm.

## Terminal UI — üç kalem (2026-08-07)

- [x] **Sol alttaki gösterge okunmuyordu.** Önce braille (U+2800, hücre başına
      2×4 nokta) denendi ve **ölçülerek reddedildi**: dört nokta-satırında dolgu
      silüet dokuya dönüşüyor, seyreltince de adlandırılamaz hâle geliyor.
      Braille çizgi-grafikte iyi, küçük dolu figürde değil. Çalışan şey hareket:
      sabit bir örse (`▄▆▄`) soldan yaklaşan çekiç, temas karesinde kıvılcım.
      Alan **sabit yedi hücre**, böylece çekiç ilerlerken metin kaymıyor.
- [x] **`READY TO FORGE` alt başlığı**, wordmark'ın hemen altında — bloğun
      altında değil, ki markanın parçası gibi okunsun. Yalnız dinlenme hâlinde;
      animasyonlu kareler taşımıyor.
- [x] **`tarhan capabilities doctor`** — kurulumdan sonra çalıştırılacak komut.
      Örs döverken çubuk **gerçek kontrolleri** sayıyor (numpy/scipy/matplotlib
      import, registry yükleme, her kanıt dosyasının varlığı, opsiyonel DEVSIM),
      bitince `READY TO FORGE`. Eksik varsa 3 ile çıkıyor ve neyin eksik
      olduğunu söylüyor.
      **Kısıt:** `pip install` sırasında animasyon oynatılamaz — wheel kurulumu
      hiçbir kod çalıştırmaz. Bu, isteğin yapılabilir en yakın karşılığı.

**Yol boyunca yakalanan kusur:** doktor komutu JSON sözleşmesini kırıyordu.
DEVSIM import edilirken C seviyesinden fd 1'e banner basıyor;
`contextlib.redirect_stdout` bunu göremez, dolayısıyla `--format json`
çıktısının ortasına düşüp `json.loads`'ı patlatıyordu. `os.dup2` ile tanıtıcı
seviyesinde stderr'e yönlendirildi ve teste bağlandı.

**Prototip artık ikiz değil.** `/private/tmp/tarhan-terminal-prototype/` altındaki
demo kendi kopyasını taşıyordu ve bir gün içinde paketten ayrıştı — son elle
taşıma sessizce başarısız olup demoyu eski animasyonu gösterir hâlde bıraktı.
Şimdi gerçek `tarhan.forge`'u import eden ince bir sürücü. Tek uygulama.

## Bu dilimde ek olarak yapılanlar

- `AGENTS.md`: yeni komut ve çıkış kodu sözleşmesi belgelendi. Ayrıca oradaki
  "152 tests" ifadesi düzeltildi — gerçek sayı 338. Doküman testi bunu
  yakalayamaz, kendi docstring'inde yanlış SAYIYI yakalayamayacağını söylüyor.
- `README.md`: Quickstart'a üç satır.

## İnceleme

**Ne yapıldı.** Capability registry (10 kayıt), §5'e karşı çivileyen test,
`capabilities list/show`, çıktı sözleşmesi, hata sınıfları, TUI-0.
59 yeni test: 279 → **338 passed, 4 xfailed**.

**Yol boyunca ortaya çıkan iki tasarım hatası, ikisi de kod yazarken.**
1. Zaman ekseni `steady|transient` olarak dar kaldı; 2D-3 küçük-sinyal AC ve
   ikisi de onu ifade edemiyordu. `ac` eklendi. Bir registry, bir şeyin NEDEN
   bloklandığını söyleyemiyorsa kendi amacını baltalar.
2. `--color always` ile tablo başlığının kalınlaşması testi düşürdü. İncelenince
   yanlış olan testin öncülüydü: `table` zaten insan biçimi. Savunulması gereken
   değişmez daha dar — **ayrıştırılacak bir akış asla escape taşımaz**. Test o
   iddiaya çevrildi, kod değil.

**Kanıtlanan, iddia edilmeyen.** §5 çivileme testi mutasyonla sınandı. JSON
saflığı gerçek alt süreçte gerçek borularla ölçüldü — StringIO ile yapılan bir
kontrol, komut doğrudan terminale yazsa bile geçerdi. `EXIT_INTERNAL` yolu
zorlanmış bir istisnayla tetiklendi.

**Bilinçli olarak yapılmayanlar.** `run solve`, run manifest, artifact şeması,
`compare`, candidate üretimi, TUI-1+, App, MCP yüzeyinde 2D, PyPI. `--output`,
`--seed`, `--dry-run` bayrakları da yok: `run solve` olmadan ölü bayak olurlardı.
`EXIT_NO_CONVERGENCE` tanımlı ama çağrısız, ve bu yukarıda açıkça yazıyor.

## Bu dilimde bilinçli olarak YOK

`run solve`, run manifest, artifact şeması, `compare`, candidate üretimi, TUI-1+,
App. Hepsi P1 ve sonrası. Yol haritası §9 bunu açıkça sıralıyor; §11'in "ilk
dilim" listesi ise P0+P1+TUI-1'i tek torbaya koyuyor — o listeyi olduğu gibi
uygulamak bağımlılık sırasını çiğnerdi.
