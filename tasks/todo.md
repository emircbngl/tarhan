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
